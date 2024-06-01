import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Callable, List, Optional, Sequence, Union

import requests
from configs import DERIVABLE_FIELDS, ES_URL
from database.utils import convert_to_events
from elastic_transport import ObjectApiResponse
from fastapi import HTTPException
from query_parse.types.elasticsearch import ESSearchRequest, MSearchQuery
from query_parse.types.lifelog import TimeCondition
from query_parse.types.requests import GeneralQueryRequest
from requests import Response
from results.models import (
    DoubletEvent,
    DoubletEventResults,
    EventResults,
    GenericEventResults,
    TripletEvent,
)
from results.utils import create_event_label, deriving_fields
from rich import print

logger = logging.getLogger(__name__)

json_headers = {"Content-Type": "application/json"}


def handle_failed_request(
    response: Union[Response, ObjectApiResponse], query: Any
) -> None:
    logger.error(f"There was an error with the request: ({response.status_code})")
    logger.error(response.text)
    with open("request.log", "a") as f:
        f.write(response.text + "\n")
        f.write(json.dumps(query) + "\n")


def delete_scroll_id(scroll_id) -> None:
    response = requests.delete(
        f"{ES_URL}/_search/scroll",
        data=json.dumps({"scroll_id": scroll_id}),
        headers=json_headers,
    )
    if response.status_code != 200:
        handle_failed_request(response, {"scroll_id": scroll_id})


# def get_scroll_request(query: ESScroll) -> Optional[ESScroll]:
#     """
#     Get the next scroll request and update the query
#     """
#     response = requests.post(
#         f"{ES_URL}/_search/scroll",
#         data=json.dumps({"scroll": "5m", "scroll_id": query.scroll_id}),
#         headers=json_headers,
#     )
#     if response.status_code != 200:
#         handle_failed_request(response, query)
#         return None

#     response_json = response.json()  # Convert to json as dict formatted
#     events = [d["_source"] for d in response_json["hits"]["hits"]]
#     scores = [d["_score"] for d in response_json["hits"]["hits"]]

#     query.add_results(EventResults(events=events, scores=scores))

#     new_scroll_id = response_json["_scroll_id"]
#     if new_scroll_id:
#         # Delete the old scroll_id
#         delete_scroll_id(query.scroll_id)
#         query.scroll_id = new_scroll_id

#     return query

# es = AsyncElasticsearch([ES_URL])

# async def es_search(query: dict):
#     """
#     Send a new search request
#     """
#     start = time()
#     response_json = await es.search(
#         **query,
#     )
#     print("Time taken", time() - start)
#     return response_json


async def send_search_request(query: ESSearchRequest) -> Optional[EventResults]:
    """
    Send a new search request
    """
    json_query = query.to_query()

    # Use normal search
    response = requests.post(
        f"{ES_URL}/_search{'?scroll=5m' if query.scroll else ''}",
        data=json.dumps(json_query),
        headers=json_headers,
    )
    if response.status_code != 200:
        handle_failed_request(response, json_query)
        return None
    response_json = response.json()  # Convert to json as dict formatted

    scroll_id = ""
    if query.scroll:
        scroll_id = response_json["_scroll_id"]

    # Get the event ids and scores
    event_ids = [d["_source"]["scene"] for d in response_json["hits"]["hits"]]
    if len(event_ids) == 0:
        print("[red]No events found![/red]")
        return None
    scores = [d["_score"] for d in response_json["hits"]["hits"]]

    # Get the relevant fields and form the EventResults object
    if query.test:
        if isinstance(scores[0], float):
            max_score = scores[0]
        else:
            max_score = 1.0
        min_score = 0.0
        aggs = response_json["aggregations"]
        min_score = aggs["score_stats"]["std_deviation_bounds"]["upper"]
        # It is a test query, so we don't need to return the events
        return EventResults(
            events=[],
            scores=[],
            min_score=min_score,
            max_score=max_score,
        )

    # Just get everything for now
    events = convert_to_events(event_ids)
    result = EventResults(events=events, scores=scores, scroll_id=scroll_id)
    return result


async def send_multiple_search_request(
    queries: MSearchQuery,
) -> Sequence[Optional[EventResults]]:

    response = requests.get(
        f"{ES_URL}/_msearch",
        data=queries.to_query(),
        headers={"Content-Type": "application/x-ndjson"},
    )

    if response.status_code != 200:
        handle_failed_request(response, queries.to_query())
        return []

    response_json = response.json()  # Convert to json as dict formatted
    list_events: List[Optional[EventResults]] = []
    for res in response_json["responses"]:
        try:
            event_ids = [d["_source"]["scene"] for d in res["hits"]["hits"]]
            if len(event_ids) == 0:
                list_events.append(None)
                continue
            events = EventResults(
                events=convert_to_events(event_ids),
                scores=[d["_score"] for d in res["hits"]["hits"]],
            )
            events = create_event_label(events)
            list_events.append(events)
        except KeyError:
            print("KeyError", res)
            list_events.append(None)

    return list_events


def merge_msearch_with_main_results(
    main_results: EventResults,
    msearch_results: Sequence[EventResults | None],
    time_condition: TimeCondition,
    merge_msearch: bool = False,
) -> DoubletEventResults:
    """
    Merge the msearch results with the main results
    Strategy:
    - For each event in the main results, get the first event in the msearch results
    - Create a doublet event with the main event as the main and the msearch event as the conditional
    - Sort the doublets by the score

    If merge_msearch is True, then any main event that has the same msearch event will be merged
    Strategy:
    - Keep track of the msearch events that have been used using a dictionary
    - For each event in the main results, get the first event in the msearch results
    - If the msearch event has been used, then find the event that has the same msearch event
    - Create a doublet event with the main event as the main and the msearch event as the conditional
    """
    events = main_results.events
    scores = main_results.scores

    doublets = []
    doublet_scores = []
    used_msearch = {}

    assert len(events) == len(msearch_results)

    for i, msearch_result in enumerate(msearch_results):
        if (
            msearch_result is None
            or not msearch_result.events
            or not msearch_result.events[0].images
        ):
            continue

        if merge_msearch:
            scene = msearch_result.events[0].scene
            # Check if the msearch event has been used
            if scene in used_msearch:
                # Find the event that has the same msearch event
                j = used_msearch[scene]
                doublets[j].main.merge_with_one(
                    events[i], [doublet_scores[j], scores[i]]
                )
                continue
            used_msearch[scene] = i

        doublets.append(
            DoubletEvent(
                main=events[i],
                conditional=msearch_result.events[0],
                condition=time_condition,
            )
        )
        doublet_scores.append(scores[i] + msearch_result.scores[0])

    # Sort the doublets by the score
    doublets, doublet_scores = zip(
        *sorted(zip(doublets, doublet_scores), key=lambda x: x[1], reverse=True)
    )

    return DoubletEventResults(events=doublets, scores=doublet_scores)


def organize_by_relevant_fields(results, relevant_fields) -> EventResults:
    scene_ids = [event.scene for event in results.events]
    results.relevant_fields = relevant_fields
    results.events = convert_to_events(scene_ids, relevant_fields)
    derivable_fields = set(relevant_fields) & set(DERIVABLE_FIELDS)
    if derivable_fields:
        new_events = deriving_fields(results.events, list(derivable_fields))
        results.events = new_events
    return results


def process_search_results(results: GenericEventResults) -> List[TripletEvent]:
    # This is the search results
    if not results:
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
    return triplet_results


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

