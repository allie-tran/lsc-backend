import asyncio
import json
import logging
from time import time
from typing import Any, List, Optional, Union

import requests
from configs import ES_URL
from database.utils import convert_to_events
from elastic_transport import ObjectApiResponse
from elasticsearch import AsyncElasticsearch
from query_parse.types.elasticsearch import ESSearchRequest, MSearchQuery
from requests import Response
from results.models import EventResults
from rich import print

logger = logging.getLogger(__name__)

json_headers = {"Content-Type": "application/json"}


def handle_failed_request(
    response: Union[Response, ObjectApiResponse], query: Any
) -> None:
    logger.error(f"There was an error with the request: ({response.status_code})")
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
        max_score = scores[0]
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


def send_multiple_search_request(queries: MSearchQuery) -> Optional[MSearchQuery]:
    response = requests.post(
        f"{ES_URL}/_msearch",
        data=json.dumps(queries.to_query()),
        headers=json_headers,
    )
    if response.status_code != 200:
        handle_failed_request(response, queries)
        return None

    response_json = response.json()  # Convert to json as dict formatted

    list_events: List[Optional[EventResults]] = []
    for res in response_json["responses"]:
        try:
            events = EventResults(
                events=[d["_source"] for d in res["hits"]["hits"]],
                scores=[d["_score"] for d in res["hits"]["hits"]],
            )
            list_events.append(events)
        except KeyError as e:
            print("KeyError", res)
            list_events.append(None)

    # queries.extend_results(list_events)
    return queries
