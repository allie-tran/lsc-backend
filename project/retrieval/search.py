import asyncio
import heapq
import json
import logging
from collections.abc import Sequence
from typing import Any, AsyncGenerator, Callable, List

from configs import FILTER_FIELDS, MAX_IMAGES_PER_EVENT, MERGE_EVENTS
from database.models import ResponseOrError, SearchRequestModel
from database.utils import get_relevant_fields
from fastapi import HTTPException
from query_parse.es_utils import get_conditional_time_filters
from query_parse.extract_info import Query, create_es_query, create_query
from query_parse.question import detect_question, question_to_retrieval
from query_parse.types import ESSearchRequest
from query_parse.types.elasticsearch import ESBoolQuery, MSearchQuery
from query_parse.types.lifelog import TimeCondition
from query_parse.types.options import FunctionWithArgs, SearchPipeline
from query_parse.types.requests import GeneralQueryRequest
from question_answering.text import answer_text_only, get_specific_description
from question_answering.video import answert_visual_with_text
from results.models import (
    AsyncioTaskResult,
    DoubletEvent,
    EventResults,
    GenericEventResults,
    ReturnResults,
    TripletEvent,
)
from results.utils import create_event_label, limit_images_per_event, merge_events
from rich import print

from retrieval.search_utils import (
    merge_msearch_with_main_results,
    organize_by_relevant_fields,
    send_multiple_search_request,
    send_search_request,
)

logger = logging.getLogger(__name__)


async def streaming_manager(request: GeneralQueryRequest):
    """
    Managing the streaming of the search results
    """
    cached_responses = SearchRequestModel(request=request)
    try:
        search_function = None
        before, main, after = request.before, request.main, request.after
        match (before, main, after):
            case ("", main, ""):
                search_function = single_query(request.main, request.pipeline)
                print("[green]Single query[/green]")
            case (before, main, ""):
                search_function = two_queries(
                    main,
                    before or "",
                    TimeCondition(
                        condition="before", time_limit_str=request.before_time
                    ),
                    size=request.size,
                )
                print("[green]Two queries[/green]")
            case ("", main, after):
                search_function = two_queries(
                    main,
                    after or "",
                    TimeCondition(condition="after", time_limit_str=request.after_time),
                    size=request.size,
                )
                print("[green]Two queries[/green]")
            case (before, main, after):
                raise NotImplementedError("Triplet is not implemented yet")
            case _:
                raise ValueError("Invalid query")

        async for response in search_function:
            if response["type"] in ["raw", "modified"]:
                results: EventResults = response["results"]

                # This is the search results
                if not results:
                    cached_responses.add(
                        ResponseOrError(
                            success=False, status_code=404, response="No results found"
                        )
                    )
                    raise HTTPException(status_code=404, detail="No results found")
                assert not isinstance(results, str)

                # Format into EventTriplets
                triplet_results = []
                for main in results.events:
                    triplet = TripletEvent(main=main)

                    if isinstance(main, DoubletEvent):
                        if main.condition.condition == "before":
                            triplet.before = main.conditional
                        else:
                            triplet.after = main.conditional

                    triplet_results.append(triplet)
                # Yield the results
                result_repr = ReturnResults(
                    result_list=triplet_results, scroll_id=results.scroll_id
                )

                response = ResponseOrError(
                    success=True, response=result_repr, type=response["type"]
                )

                cached_responses.add(response)

                yield "data: " + response.model_dump_json(
                    exclude_none=True, exclude_unset=True
                ) + "\n\n"
            elif response["type"] == "pipeline":
                response = ResponseOrError(success=True, **response)
                yield "data: " + response.model_dump_json() + "\n\n"
            else:
                cached_responses.add(ResponseOrError(success=True, **response))
                yield "data: " + json.dumps(response) + "\n\n"

    except asyncio.CancelledError:
        print("Client disconnected")

    except Exception as e:
        yield "data: ERROR\n\n"
        print("[red]Error[/red]", e)
        cached_responses.add(ResponseOrError(success=False, response=str(e)))
        # pass
        raise (e)

    print("[blue]ALl Done[/blue]")
    print("-" * 50)
    cached_responses.mark_finished()
    yield "data: END\n\n"


# ============================= #
# Easy Peasy Part: one query only
# ============================= #
async def simple_search(
    query: Query, main_query: ESBoolQuery, size: int, tag: str = ""
) -> AsyncioTaskResult:
    """
    Search a single query without any fancy stuff
    """
    async_results = AsyncioTaskResult(
        results=None, tag=tag, task_type="search", query=query
    )

    print(f"[blue]Min score {main_query.min_score:.2f}[/blue]")
    search_request = ESSearchRequest(
        original_text=query.original_text,
        query=main_query.to_query(),
        sort_field="start_timestamp",
        min_score=main_query.min_score,
        size=size,
    )
    results = await send_search_request(search_request)

    if results is None:
        print("[red]No results found[/red]")
        return async_results

    async_results.results = results
    print(f"[green]Found {len(results)} events[/green]")
    results.min_score = main_query.min_score
    results.max_score = main_query.max_score

    # Give some label to the results
    results = create_event_label(results)

    # Just send the raw results first (unmerged, with full info)
    print("[green]Sending raw results...[/green]")
    return async_results


async def single_query(
    text: str,
    pipeline: SearchPipeline = SearchPipeline(),
):
    """
    Search (and answer) a single query
    """
    if not pipeline:
        pipeline = SearchPipeline()

    # ============================= #
    # 1. Query Parser (no skipping but modifiable)
    # ============================= #
    output = await pipeline.query_parser.async_execute(
        [
            FunctionWithArgs(  # no skipping
                function=detect_question, args=[text], output_name="is_question"
            ),
            FunctionWithArgs(  # no skipping
                function=question_to_retrieval,
                args=[text],
                use_previous_output=True,
                output_name="search_text",
                is_async=True,
            ),
            FunctionWithArgs(  # no skipping
                function=create_query,
                use_previous_output=True,
                output_name="query",
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

    pipeline.query_parser.add_output(
        {
            # "visual": output["query"].visualinfo,
            # "time": output["query"].timeinfo,
            # "location": output["query"].locationinfo,
            **output["query"].query_parts,
        }
    )

    # ============================= #
    # 2. Search (Field extractor can be skipped)
    # ============================= #
    # a. Check if we need to extract the relevant fields
    field_extractor = pipeline.field_extractor
    to_extract_field = not field_extractor.executed and not field_extractor.skipped

    async_tasks = get_search_tasks(
        output["query"],
        output["es_query"],
        pipeline.size,
        text,
        "single",
        to_extract_field,
    )

    # ----------------------------- #
    # b. Start the async tasks
    results = None
    relevant_fields = None

    for future in asyncio.as_completed(async_tasks):
        res = await future
        if res.task_type == "search":
            results = res.results
            yield {"type": "raw", "results": results}
        elif res.task_type == "llm":
            relevant_fields = res.results
            field_extractor.add_output({"relevant_fields": relevant_fields})

    if results is None:
        print("[red]No results found[/red]")
        return

    # ============================= #
    # 3. Processing the results
    # ============================= #
    # a. Split the fields into readable format
    split_outputs = {"merge_by": [], "sort_by": [], "relevant_fields": []}

    if relevant_fields:
        for field in relevant_fields:
            if "groupby" in field:
                split_outputs["merge_by"].append(field)
            elif "sortby" in field:
                split_outputs["sort_by"].append(field)
            else:
                split_outputs["relevant_fields"].append(field)
        field_extractor.add_output(split_outputs)

    # a. Organize the results by relevant fields
    pipeline.field_organizer.default_output = {"results": results}
    results = pipeline.field_organizer.execute(
        [
            FunctionWithArgs(
                function=organize_by_relevant_fields,
                args=[results, split_outputs["relevant_fields"]],
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
                args=[results, set(split_outputs["merge_by"])],
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
    changed = any(
        p.skipped
        for p in [
            pipeline.field_organizer,
            pipeline.event_merger,
            pipeline.image_limiter,
        ]
    )
    if changed:
        results = create_event_label(results)
    yield {"type": "modified", "results": results}

    # ============================= #
    # 4. Answer the question
    # ============================= #
    k = min(pipeline.top_k, len(results.events))

    yield {"type": "pipeline", "response": pipeline.export()}

    print("[yellow]Answering the question...[/yellow]")
    textual_descriptions = []
    if results.relevant_fields:
        print("[green]Relevant fields[/green]", results.relevant_fields)
        for event in results.events[:k]:
            textual_descriptions.append(
                get_specific_description(event, results.relevant_fields)
            )
    if not textual_descriptions:
        return
    print("[green]Textual description sample[/green]", textual_descriptions[0])

    all_answers = []
    async for answers in get_answer_tasks(text, textual_descriptions, results, k):
        # extend the answers, skip if it's already there,
        # replace if the new answer includes the old one
        for answer in answers:
            if answer in all_answers:
                continue
            for i, a in enumerate(all_answers):
                if a in answer:
                    all_answers[i] = answer
                    break
            else:
                all_answers.append(answer)
        yield {"type": "answers", "response": all_answers}


def get_search_tasks(
    query: Query,
    main_query: ESBoolQuery,
    size: int,
    text: str,
    tag: str = "",
    filter_fields: bool = FILTER_FIELDS,
) -> List[asyncio.Task]:
    # Starting the async tasks
    async_tasks: Sequence = [
        asyncio.create_task(simple_search(query, main_query, size, tag))
    ]
    if filter_fields and text:
        task = asyncio.create_task(get_relevant_fields(text, tag))
        async_tasks.append(task)
    return async_tasks


async def merge_generators(*generators: AsyncGenerator) -> AsyncGenerator[Any, None]:
    priority_queue = []
    next_idx = 0

    async def add_to_queue(generator, idx):
        nonlocal next_idx
        async for value in generator:
            heapq.heappush(priority_queue, (next_idx, value, idx))
            next_idx += 1

    tasks = [add_to_queue(generator, idx) for idx, generator in enumerate(generators)]
    await asyncio.gather(*tasks)

    while priority_queue:
        _, value, _ = heapq.heappop(priority_queue)
        yield value


async def get_answer_tasks(
    text: str,
    textual_descriptions: List[str],
    events: GenericEventResults,
    k: int = 10,
) -> AsyncGenerator[List[str], None]:
    async_tasks: Sequence = [
        answer_text_only(text, textual_descriptions, len(events.events)),
        answert_visual_with_text(text, textual_descriptions, events, k),
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
    es_query = await conditional_query.to_elasticsearch()
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

    query = Query(search_text, is_question=is_question)
    es_query = await query.to_elasticsearch(ignore_limit_score=False)
    conditional_query = Query(conditional_text, is_question=is_question)
    conditional_es_query = await conditional_query.to_elasticsearch()

    tasks = get_search_tasks(query, es_query, size, main_text, "main")
    tasks += get_search_tasks(
        conditional_query, conditional_es_query, size, conditional_text, "conditional"
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
                main_query = res.query
            elif tag == "conditional":
                conditional_results = res.results
                conditional_query = res.query
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
        main_results = merge_events(main_results, set()) # TODO! add the relevant fields
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

    all_answers = []
    async for answers in get_answer_tasks(
        main_text, textual_descriptions, merged_results, k
    ):
        for answer in answers:
            if answer in all_answers:
                continue
            for i, a in enumerate(all_answers):
                if a in answer:
                    all_answers[i] = answer
                    break
            else:
                all_answers.append(answer)
        yield {"type": "answers", "answers": all_answers}
