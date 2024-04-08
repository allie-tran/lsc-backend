import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Literal, Optional, Tuple

from fastapi import HTTPException

from configs import (DERIVABLE_FIELDS, FILTER_FIELDS, MAX_IMAGES_PER_EVENT,
                     MERGE_EVENTS)
from database.utils import convert_to_events, get_relevant_fields

from query_parse.extract_info import Query
from query_parse.question import detect_question, question_to_retrieval
from query_parse.types import ESSearchRequest
from query_parse.types.requests import GeneralQueryRequest
from question_answering.text import answer_text_only, get_specific_description
from question_answering.video import answer_visual_only
from results.models import EventResults, ReturnResults, TripletEvent
from results.utils import (create_event_label, deriving_fields,
                           limit_images_per_event, merge_events)
from rich import print

from retrieval.search_utils import send_search_request

logger = logging.getLogger(__name__)

# ============================= #
# Easy Peasy Part: one query only
# ============================= #


async def async_query(request: GeneralQueryRequest):
    """
    Streaming response
    """
    try:
        # First yield in search single is the results
        async for response in single_query(request.main, request.size):
            # results is a dict
            if response["type"] in ["raw", "modified"]:

                results = response["results"]
                if not results:
                    raise HTTPException(status_code=404, detail="No results found")

                assert not isinstance(results, str)
                # Format into EventTriplets
                triplet_results = [TripletEvent(main=main) for main in results.events]

                # Return the results
                result_repr = ReturnResults(
                    result_list=triplet_results,
                    scroll_id=results.scroll_id
                ).model_dump_json()

                # yield the results
                res = {"type": response["type"], "results": result_repr}
                yield "data: " + json.dumps(res) + "\n\n"
            else:
                yield "data: " + json.dumps(response) + "\n\n"

    except asyncio.CancelledError:
        print("Client disconnected")

    except Exception as e:
        # pass
        raise(e)

    yield "data: END\n\n"

async def simple_search(
    text_query: str, isQuestion: bool, size: int
) -> Tuple[Optional[EventResults], Literal["search"]]:
    """
    Search a single query without any fancy stuff
    """
    query = Query(text_query, is_question=isQuestion)
    main_query = await query.to_elasticsearch(ignore_limit_score=False)

    print(f"[blue]Min score {main_query.min_score:.2f}[/blue]")
    search_request = ESSearchRequest(
        original_text=query.original_text,
        query=main_query.to_query(),
        sort_field="start_timestamp",
        min_score=main_query.min_score,
        size=size
    )
    print(search_request.query)
    results = await send_search_request(search_request)

    if results is None:
        return None, "search"

    print(f"[green]Found {len(results)} events[/green]")
    results.min_score = main_query.min_score
    results.max_score = main_query.max_score

    # Give some label to the results
    results = create_event_label(results)

    # Just send the raw results first (unmerged, with full info)
    print("[green]Sending raw results...[/green]")
    return results, "search"


async def single_query(
    text: str,
    size: int,
):
    """
    Search (and answer) a single query
    """
    is_question = detect_question(text)
    if is_question:
        search_text = await question_to_retrieval(text)
    else:
        search_text = text

    # Starting the async tasks
    async_tasks: Sequence = [
        asyncio.create_task(simple_search(search_text, is_question, size))
    ]

    if FILTER_FIELDS and text:
        task = asyncio.create_task(get_relevant_fields(text))
        async_tasks.append(task)

    # Return the results when any of the tasks are done
    results = None
    relevant_fields = None
    for future in asyncio.as_completed(async_tasks):
        async_res, task_type = await future
        if task_type == "search":
            results = async_res
            yield {"type": "raw", "results": results}
        elif task_type == "llm":
            relevant_fields = async_res

    if results is None:
        print("[red]No results found[/red]")
        return

    changed = False
    # Get the relevant fields
    if FILTER_FIELDS and text and relevant_fields:
        scene_ids = [event.scene for event in results.events]
        results.relevant_fields = relevant_fields
        results.events = convert_to_events(scene_ids, relevant_fields)
        derivable_fields = set(relevant_fields) & set(DERIVABLE_FIELDS)
        if derivable_fields:
            print(f"[green]Deriving {derivable_fields} [/green]")
            new_events = deriving_fields(results.events, list(derivable_fields))
            results.events = new_events
        changed = True

    # Merge the results
    if MERGE_EVENTS:
        results = merge_events(results)
        changed = True

    if MAX_IMAGES_PER_EVENT:
        print(f"[blue]Limiting images to {MAX_IMAGES_PER_EVENT}[/blue]")
        results = limit_images_per_event(results, search_text, MAX_IMAGES_PER_EVENT)
        changed = True

    if changed:
        # Give a different label:
        results = create_event_label(results)
        # Send the modified results
        yield {"type": "modified", "results": results}

    if not is_question:
        return

    # Answer the question
    print("[yellow]Answering the question...[/yellow]")
    k = min(10, len(results.events))
    textual_descriptions = []
    if results.relevant_fields:
        print("[green]Relevant fields[/green]", results.relevant_fields)
        for i, event in enumerate(results.events[:k]):
            textual_descriptions.append(
                get_specific_description(event, results.relevant_fields)
            )
    print("[green]Textual description sample[/green]", textual_descriptions[0])
    answers = await answer_text_only(text, textual_descriptions, len(results.events))
    if len(answers) > 0:
        yield {"type": "answers", "answers": answers}
        # return # TODO!
    else:
        print("[red]No answers found (LLM)[/red]")

    # Answer with images
    print("[yellow]Answering with images...[/yellow]")
    async for answer in answer_visual_only(text, textual_descriptions, results, k):
        if answer == "Timeout":
            break
        if answer is None:
            continue
        answers.append(answer)
        yield {"type": "answers", "answers": answers}

    print("[blue]Done[/blue]")

# ============================= #
# Level 2: Multiple queries
# ============================= #

# pass for now TODO!
