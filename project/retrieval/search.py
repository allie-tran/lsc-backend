import asyncio
import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import AsyncGenerator, Callable, List, Optional

from configs import FILTER_FIELDS, MAX_IMAGES_PER_EVENT, MERGE_EVENTS
from database.main import group_collection, image_collection
from database.models import GeneralRequestModel, Response
from database.requests import get_es, get_request
from database.utils import get_relevant_fields
from fastapi import HTTPException
from pydantic import InstanceOf
from query_parse.es_utils import get_conditional_time_filters, get_location_filters
from query_parse.extract_info import (
    Query,
    create_es_query,
    create_query,
    modify_es_query,
)
from query_parse.question import detect_question, question_to_retrieval
from query_parse.types.elasticsearch import (
    ESBoolQuery,
    ESEmbedding,
    LocationInfo,
    MSearchQuery,
    TimeInfo,
)
from query_parse.types.lifelog import DateTuple, Mode, TimeCondition
from query_parse.types.options import FunctionWithArgs, SearchPipeline
from query_parse.types.requests import (
    GeneralQueryRequest,
    MapRequest,
    Step,
    TimelineDateRequest,
)
from query_parse.visual import encode_image
from question_answering.text import answer_text_only, get_specific_description
from question_answering.video import answer_visual_with_text
from results.models import (
    AnswerListResult,
    AnswerResult,
    AsyncioTaskResult,
    Event,
    EventResults,
    GenericEventResults,
    Image,
    PartialEvent,
    TimelineGroup,
    TimelineResult,
)
from results.utils import (
    RelevantFields,
    create_event_label,
    limit_images_per_event,
    merge_events,
    merge_scenes_and_images,
)
from rich import print

from retrieval.async_utils import merge_generators
from retrieval.search_utils import (
    get_raw_search_results,
    get_search_request,
    merge_msearch_with_main_results,
    organize_by_relevant_fields,
    process_es_results,
    process_search_results,
    send_multiple_search_request,
    send_search_request,
)

logger = logging.getLogger(__name__)


async def streaming_manager(request: GeneralQueryRequest):
    """
    Managing the streaming of the search results
    """
    cached_responses = GeneralRequestModel(request=request)
    if cached_responses.finished:
        print("Cached responses found")
        # req = GeneralRequestModel.model_validate(cached_responses)
        # for response in req.responses:
        #     response.oid = req.oid
        #     data = response.model_dump_json()
        #     yield f"data: {data}\n\n"
        # print("[blue]ALl Done[/blue]")
        # print("-" * 50)
        # cached_responses.mark_finished()
        # yield "data: END\n\n"
        # return

    try:
        search_function = get_search_function(request, single_query, two_queries)
        async for response in search_function:
            cached_responses.add(response)
            data = response.model_dump_json(by_alias=True)
            yield f"data: {data}\n\n"

    except asyncio.CancelledError:
        print("Client disconnected")

    except Exception as e:
        print("[red]Error[/red]", e)
        yield "data: ERROR\n\n"
        raise (e)

    print("[blue]ALl Done[/blue]")
    print("-" * 50)
    cached_responses.mark_finished()
    yield "data: END\n\n"


def get_search_function(
    request: GeneralQueryRequest, single_query: Callable, two_queries: Callable
) -> AsyncGenerator:
    search_function = None
    before, main, after = request.before, request.main, request.after
    size = request.pipeline.size if request.pipeline else 200
    match (before, main, after):
        case ("", main, ""):
            search_function = single_query(request.main, request.pipeline)
        case (before, main, ""):
            search_function = two_queries(
                main,
                before or "",
                TimeCondition(condition="before", time_limit_str=request.before_time),
                size=size,
            )
        case ("", main, after):
            search_function = two_queries(
                main,
                after or "",
                TimeCondition(condition="after", time_limit_str=request.after_time),
                size=size,
            )
        case (before, main, after):
            raise NotImplementedError("Triplet is not implemented yet")
        case _:
            raise ValueError("Invalid query")
    return search_function


# ============================= #
# Easy Peasy Part: one query only
# ============================= #
async def simple_search(
    main_query: ESBoolQuery,
    size: int,
    tag: str = "",
) -> AsyncioTaskResult[EventResults]:
    """
    Search a single query without any fancy stuff
    """
    all_results: List[EventResults] = []
    for mode in [Mode.event, Mode.image]:
        request = get_search_request(main_query, size, mode)
        es_response = await send_search_request(request)
        es_results = process_es_results(es_response, mode=mode)
        # Give some label to the results
        if es_results:
            print(f"[green]Found {len(es_results.events)} matches for {mode}[/green]")
            es_results.min_score = main_query.min_score
            es_results.max_score = main_query.max_score
            all_results.append(es_results)

    match len(all_results):
        case 0:
            return AsyncioTaskResult(task_type="search", tag=tag, results=None)
        case 1:
            results = create_event_label(all_results[0])
            return AsyncioTaskResult(task_type="search", tag=tag, results=results)
        case _:
            results = merge_scenes_and_images(all_results[0], all_results[1])
            results = create_event_label(results)
            return AsyncioTaskResult(task_type="search", tag=tag, results=results)


async def single_query(
    text: str,
    pipeline: Optional[SearchPipeline] = None,
):
    """
    Search (and answer) a single query
    """

    if not pipeline:
        pipeline = SearchPipeline()

    step = Step(step=1, total=2)
    # ============================= #
    # 1. Query Parser (no skipping but modifiable)
    # ============================= #
    output = await pipeline.query_parser.async_execute(
        [
            FunctionWithArgs(
                function=detect_question, args=[text], output_name="is_question"
            ),
            FunctionWithArgs(
                function=question_to_retrieval,
                args=[text],
                use_previous_output=True,
                output_name="search_text",
                is_async=True,
            ),
            FunctionWithArgs(
                function=create_query,
                use_previous_output=True,
                output_name="query",
                is_async=True,
            ),
            FunctionWithArgs(  # no skipping
                function=create_es_query,
                use_previous_output=True,
                kwargs={"ignore_limit_score": False},
                output_name="es_query",
                is_async=True,
            ),
        ]
    )

    if output["is_question"]:
        step.total = 4

    configs = output["query"].print_info()
    print("[blue]Query[/blue]", configs)
    pipeline.query_parser.add_output(configs)

    # ============================= #
    # 2. Search (Field extractor can be skipped)
    # ============================= #
    # a. Check if we need to extract the relevant fields
    field_extractor = pipeline.field_extractor
    to_extract_field = not field_extractor.executed and not field_extractor.skipped
    async_tasks = get_search_tasks(
        output["es_query"],
        pipeline.size,
        text,
        "single",
        to_extract_field,
    )
    # ----------------------------- #
    # b. Start the async tasks
    results = None
    relevant_fields = RelevantFields()

    for future in asyncio.as_completed(async_tasks):
        res = await future
        if res.task_type == "search":
            results = res.results
            step.step += 1
            yield Response(
                type="images",
                response=process_search_results(results),
                progress=step.progress(),
                es_id=output["query"].oid,
            )
        elif res.task_type == "llm":
            relevant_fields = res.results
            field_extractor.add_output(relevant_fields.model_dump())

    if results is None:
        print("[red]No results found[/red]")
        return

    # ============================= #
    # 3. Processing the results
    # ============================= #
    # a. Organize the results by relevant fields
    pipeline.field_organizer.default_output = {"results": results}
    results = pipeline.field_organizer.execute(
        [
            FunctionWithArgs(
                function=organize_by_relevant_fields,
                args=[results, relevant_fields.relevant_fields],
                output_name="results",
            )
        ]
    )["results"]

    # ----------------------------- #
    # b. Merge the events
    pipeline.event_merger.default_output = {"results": results}
    results = pipeline.event_merger.execute(
        [
            FunctionWithArgs(
                function=merge_events,
                args=[results, relevant_fields],
                output_name="results",
            )
        ]
    )["results"]

    # ----------------------------- #
    # c. Limit the images
    pipeline.image_limiter.default_output = {"results": results}
    results = pipeline.image_limiter.execute(
        [
            FunctionWithArgs(
                function=limit_images_per_event,
                args=[results, text, pipeline.image_limiter.output["max_images"]],
                output_name="results",
            )
        ]
    )["results"]

    # ----------------------------- #
    # d. Check if anything changed
    # Not actually part of the pipeline
    unchanged = all(
        p.skipped
        for p in [
            pipeline.field_organizer,
            pipeline.event_merger,
            pipeline.image_limiter,
        ]
    )
    if not unchanged:
        print("[blue]Some changes detected[/blue]")
        results = create_event_label(results, relevant_fields.relevant_fields)

    step.step += 1
    yield Response(
        progress=step.progress(),
        type="modified",
        response=process_search_results(results),
    )
    yield Response(
        progress=step.progress(), type="pipeline", response=pipeline.export()
    )

    # ============================= #
    # 4. Answer the question
    # ============================= #
    if not output["is_question"]:
        return
    k = min(pipeline.top_k, len(results.events))

    print(f"[yellow]Answering the question for {k} events...[/yellow]")
    all_answers = AnswerListResult()
    async for answers in get_answer_tasks(
        text, results, relevant_fields.relevant_fields, k
    ):
        for answer in answers:
            all_answers.add_answer(answer)

        step.step += 1
        step.total += 1

        yield Response(
            progress=step.progress(), type="answers", response=all_answers.answers
        )

    if not all_answers:
        yield Response(progress=step.progress(), type="answers", response=[])


def get_search_tasks(
    main_query: ESBoolQuery,
    size: int,
    text: str,
    tag: str = "",
    filter_fields: bool = FILTER_FIELDS,
) -> List[asyncio.Task]:
    tasks = [simple_search(main_query, size, f"{tag}_event")]
    # Starting the async tasks
    if filter_fields and text:
        tasks.append(get_relevant_fields(text, tag))

    async_tasks = [asyncio.create_task(task) for task in tasks]
    return async_tasks


async def get_answer_tasks(
    text: str,
    results: EventResults,
    relevant_fields: List[str],
    k: int = 10,
) -> AsyncGenerator[List[AnswerResult], None]:
    textual_descriptions = []
    for event in results.events[:k]:
        textual_descriptions.append(get_specific_description(event, relevant_fields))

    if not textual_descriptions:
        return

    print("[green]Textual description sample[/green]", textual_descriptions[0])
    k = min(k, len(results.events))

    async_tasks: Sequence = [
        answer_text_only(text, textual_descriptions, k),
        answer_visual_with_text(text, textual_descriptions, results, k),
    ]

    async for task in merge_generators(*async_tasks):
        yield task


# ============================= #
# Level 2: Two queries
# ============================= #
async def add_conditional_filters_to_query(
    conditional_query: Query,
    main_results: EventResults,
    condition: TimeCondition,
) -> MSearchQuery:
    """
    Add the conditional filters to the query
    """
    es_query = await create_es_query(conditional_query)
    filters = get_conditional_time_filters(main_results, condition)

    msearch_queries = []
    for cond_filter in filters:
        clone_query = es_query.model_copy(deep=True)
        clone_query.filter.append(cond_filter)
        msearch_queries.append(clone_query)

    return MSearchQuery(queries=msearch_queries)


async def two_queries(
    main_text: str, conditional_text: str, condition: TimeCondition, size: int
):
    """
    Search for two related queries based on the time condition
    """
    is_question = detect_question(main_text)
    if is_question:
        search_text = await question_to_retrieval(main_text, is_question)
    else:
        search_text = main_text

    query = await create_query(search_text, is_question=is_question)
    es_query = await create_es_query(query, ignore_limit_score=False)
    conditional = await create_query(conditional_text, is_question=is_question)
    conditional_es_query = await create_es_query(conditional, ignore_limit_score=False)

    tasks = get_search_tasks(es_query, size, main_text, "main")
    tasks += get_search_tasks(
        conditional_es_query, size, conditional_text, "conditional"
    )

    # Starting the async tasks
    main_results, conditional_results, relevant_fields, conditional_relevant_fields = (
        None,
        None,
        None,
        None,
    )
    main_query, conditional_query = None, None

    for future in asyncio.as_completed(tasks):
        res: AsyncioTaskResult = await future
        task_type = res.task_type
        tag = res.tag

        if task_type == "search":
            assert isinstance(
                res.results, GenericEventResults
            ), "Results should be EventResults"
            if tag == "main":
                main_results = res.results
            elif tag == "conditional":
                conditional_results = res.results
        elif task_type == "llm":
            assert isinstance(res.results, list), "Results should be a list"
            if tag == "main":
                relevant_fields = res.results
            elif tag == "conditional":
                conditional_relevant_fields = res.results

    if (
        main_results is None
        or conditional_results is None
        or conditional_query is None
        or main_query is None
    ):
        print("[red]No results found[/red]")
        return

    # Add the conditional filters
    msearch_query = await add_conditional_filters_to_query(
        conditional_query, main_results, condition
    )

    # Send the search request
    print("[green]Sending the multi-search request...[/green]")
    msearch_results = await send_multiple_search_request(msearch_query)

    if not msearch_results:
        print("[red]No results found[/red]")
        return

    # Merge the two results
    merged_results = merge_msearch_with_main_results(
        main_results, msearch_results, condition
    )
    print("[green]Merged results[/green]", len(merged_results.events))
    yield {"type": "raw", "results": merged_results}

    # ============================= #
    # Processing...
    # ============================= #
    def apply_msearch(func: Callable, *args, **kwargs):
        return [func(res, *args, **kwargs) if res else None for res in msearch_results]

    changed = False
    if FILTER_FIELDS:
        if main_text and relevant_fields:
            main_results = organize_by_relevant_fields(main_results, relevant_fields)
        # if conditional_text and conditional_relevant_fields:
        #     conditional_results = organize_by_relevant_fields(
        #         conditional_results, conditional_relevant_fields
        #     )
        msearch_results = apply_msearch(
            organize_by_relevant_fields, conditional_relevant_fields
        )
        changed = True

    if MERGE_EVENTS:
        main_results = merge_events(main_results)  # TODO! add the relevant fields
        # conditional_results = merge_events(conditional_results)
        msearch_results = apply_msearch(merge_events)
        changed = True

    if MAX_IMAGES_PER_EVENT:
        print(f"[blue]Limiting images to {MAX_IMAGES_PER_EVENT}[/blue]")
        main_results = limit_images_per_event(
            main_results, main_text, MAX_IMAGES_PER_EVENT
        )
        # conditional_results = limit_images_per_event(
        #     conditional_results, conditional_text, MAX_IMAGES_PER_EVENT
        # )
        msearch_results = apply_msearch(
            limit_images_per_event, conditional_text, MAX_IMAGES_PER_EVENT
        )
        changed = True

    if changed:
        merged_results = merge_msearch_with_main_results(
            main_results, msearch_results, condition
        )
        # Give a different label:
        merged_results = create_event_label(merged_results)
        # Send the modified results
        yield {"type": "modified", "results": merged_results}

    # Answer the question
    if not is_question:
        return
    print("[yellow]Answering the question...[/yellow]")
    k = min(10, len(merged_results.events))
    textual_descriptions = []
    if main_results.relevant_fields or conditional_results.relevant_fields:
        for event in merged_results.events[:k]:
            main_description = get_specific_description(
                event.main, main_results.relevant_fields
            )
            conditional_description = get_specific_description(
                event.conditional, conditional_results.relevant_fields
            )
            textual_descriptions.append(
                f"{main_description}. About {condition.time_limit_str} {condition.condition} that, {conditional_description}"
            )
    if not textual_descriptions:
        return
    print("[green]Textual description sample[/green]", textual_descriptions[0])

    all_answers: AnswerListResult = AnswerListResult()

    async for new_answers in get_answer_tasks(
        main_text, main_results, main_results.relevant_fields, k
    ):
        for answer in new_answers:
            all_answers.add_answer(answer)
        yield {"type": "answers", "answers": all_answers}


async def search_location_again(request: MapRequest) -> Optional[List[Event]]:
    query_doc = get_es(request.es_id)
    if not query_doc:
        print("[red]Query not found[/red]")
        return []
    query_doc["oid"] = query_doc.pop("_id")
    query = Query.model_validate(query_doc)
    location, center = request.location, request.center
    print("Location", location)
    print("Center", center)

    assert location, "Location should not be empty"
    location = location.lower()

    location_info = LocationInfo(locations=[location], from_center=center)
    lcoation_filters = get_location_filters(location_info)
    # Modify the query
    new_query = await modify_es_query(query, location=location_info, extra_filters=lcoation_filters, mode=Mode.event)
    if not new_query:
        print("[red]New query not found[/red]")
        return []
    results = await simple_search(new_query, size=20, tag="location")
    if not results.results:
        print("[red]No results found[/red]")
        return []
    return results.results.events


async def search_from_location(
    request: MapRequest,
) -> Optional[List[InstanceOf[Event]]]:
    """
    Search from the location
    """
    # Do another search with different location filter
    if request.es_id:
        print("[green]Searching from location again...[/green]")
        return await search_location_again(request)

    # Just filter the results based on the location
    # Find main cached request with oid
    main_request = get_request(request.oid)
    if not main_request:
        print("[red]Main request not found[/red]")
        raise HTTPException(status_code=404, detail="I don't know how you got here")

    # Filter the results based on the location
    location = request.location
    results = main_request["responses"][0]["response"]

    assert results, "Results should not be empty"
    assert location, "Location should not be empty"

    location = location.lower()

    new_results = []
    for event in results:
        loc = event["main"]["location"].lower()
        if location in loc:
            new_results.append(event["main"])

    if not new_results:
        print(f"[red]No results found for location {location}[/red]")
        return None

    return [PartialEvent(**event) for event in new_results]


async def search_from_time(request: TimelineDateRequest) -> TimelineResult:
    """
    Filter down the search from the timeline view
    """
    all_images = []

    # Filter the results based on the time
    date = datetime.strptime(request.date, "%d-%m-%Y")
    start_time = date.replace(hour=0, minute=0, second=0)

    # Do another search with different time filter
    all_images = await try_search_again(request, date)
    print("[green]Found images[/green]", len(all_images))

    # Just filter the results based on the time
    filtered = image_collection.find({"image": {"$in": all_images}}).sort("time", 1)
    filtered = list(filtered)
    all_groups = set([image["group"] for image in filtered])

    # Find time info
    group_docs = group_collection.find(
        {
            "group": {"$in": list(all_groups)},
        }
    ).sort("time", 1)

    group_info = {group["group"]: group for group in group_docs}

    # Group by group -> scene -> images
    groups = defaultdict(dict)
    for image in filtered:
        key = image["image"]
        group = image["group"]
        if key not in groups[group]:
            groups[group][key] = []
        groups[group][key].append(
            Image(
                src=image["image"],
                aspect_ratio=image["aspect_ratio"],
                hash_code=image["hash_code"],
            )
        )

    timeline_groups = []
    for group in groups:
        scenes = groups[group].values()
        timeline_groups.append(
            TimelineGroup(
                group=group,
                location=group_info[group]["location"],
                location_info=group_info[group]["location_info"],
                time_info=[group_info[group]["time_info"]],
                scenes=list(scenes),
            )
        )
    print("[green]Found groups[/green]", len(timeline_groups))
    return TimelineResult(result=timeline_groups, date=start_time)


async def try_search_again(request, date) -> List[str]:
    if not request.es_id:
        return []

    # Do another search with different time filter
    query_doc = get_es(request.es_id)
    if not query_doc:
        print("[red]Query not found[/red]")
        return []

    query_doc["oid"] = query_doc.pop("_id")
    query = Query.model_validate(query_doc)
    time_info = TimeInfo(
        dates=[DateTuple(year=date.year, month=date.month, day=date.day)]
    )
    new_query = await modify_es_query(query, time=time_info, mode=Mode.image)
    if new_query:
        search_request = get_search_request(new_query, size=20, mode=Mode.image)
        print(search_request.min_score)
        results = await get_raw_search_results(search_request)
        all_images = results.results or []
        return all_images

    return []


async def search_similar_events(image: str) -> Optional[EventResults]:
    """
    Search for similar events
    """
    image_feat = encode_image(image)
    if not image_feat:
        return None

    # Find the similar events
    es = ESBoolQuery()
    es.must.append(ESEmbedding(embedding=image_feat.tolist()))

    result = await simple_search(es, size=200, tag="similar")
    if not result.results or not result.results.events:
        print("[red]No similar events found[/red]")
        return None

    return result.results
