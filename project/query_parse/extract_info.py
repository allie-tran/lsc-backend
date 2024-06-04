from collections import defaultdict
from typing import Dict, Optional, Self, Sequence

from database.main import es_collection
from myeachtra.dependencies import ObjectId
from openai import BaseModel
from pydantic import SkipValidation, model_validator
from retrieval.search_utils import send_search_request
from rich import print as rprint

from query_parse.es_utils import (
    get_location_filters,
    get_temporal_filters,
    get_visual_filters,
)
from query_parse.location import search_for_locations
from query_parse.question import parse_query
from query_parse.time import TimeTagger, search_for_time
from query_parse.types import (
    ESBoolQuery,
    ESCombineFilters,
    ESSearchRequest,
    LocationInfo,
    TimeInfo,
    VisualInfo,
)
from query_parse.types.lifelog import Mode
from query_parse.visual import search_for_visual

time_tagger = TimeTagger()


class Query(BaseModel):
    original_text: str = ""
    is_question: bool = False
    query_parts: Optional[Dict[str, str]] = {}

    time: TimeInfo = TimeInfo()
    location: LocationInfo = LocationInfo()
    visual: VisualInfo = VisualInfo()

    oid: Optional[SkipValidation[ObjectId]] = None
    extracted: bool = False

    location_queries: Sequence[ESCombineFilters] = []
    temporal_queries: Sequence[ESCombineFilters] = []
    visual_queries: Sequence[ESCombineFilters] = []
    es: Optional[ESBoolQuery] = None

    @model_validator(mode="after")
    def insert_request(self) -> Self:
        if not self.oid:
            inserted = es_collection.insert_one(self.model_dump(exclude={"responses"}))
            self.oid = inserted.inserted_id
        return self

    def mark_extracted(self, es: ESBoolQuery):
        self.extracted = True
        self.es = es
        es_collection.update_one(
            {"_id": self.oid}, {"$set": {"extracted": True}}, upsert=True
        )

    def print_info(self):
        return {
            "time": self.time.export(),
            "location": self.location.export(),
            "visual": self.visual.export(),
        }


async def extract_info(text: str, is_question: bool) -> Query:
    query_parts = await parse_query(text)

    extracted_parts = None
    if query_parts:
        extracted_parts = query_parts
        rprint("Extracted Parts:", extracted_parts)

    # =================================================================== #
    # LOCATIONS
    # =================================================================== #
    parsed = defaultdict(lambda: [])  # deprecated
    clean_query, locationinfo, loc_visualisation = search_for_locations(
        query_parts["location"].lower(), parsed
    )

    # =================================================================== #
    # TIME
    # =================================================================== #
    time_str = ""
    if "time" in query_parts and "date" in query_parts:
        time_str = f"{query_parts['time']} {query_parts['date']}"
    elif "time" in query_parts:
        time_str = query_parts["time"]
    elif "date" in query_parts:
        time_str = query_parts["date"]

    if not time_str:
        time_str = clean_query
    clean_query, timeinfo, time_visualisation = search_for_time(
        time_tagger, time_str.lower()
    )

    # =================================================================== #
    # VISUALISATION
    # =================================================================== #
    query_visualisation = loc_visualisation
    query_visualisation.update(time_visualisation)

    # =================================================================== #
    # VISUAL INFO
    # =================================================================== #
    visualinfo = search_for_visual(query_parts["visual"])
    if visualinfo:
        print("Visual Text:", visualinfo.text)
    else:
        print("No Visual Text.")

    # =================================================================== #
    # END
    # =================================================================== #
    query = Query(
        original_text=text,
        query_parts=extracted_parts,
        is_question=is_question,
        time=timeinfo,
        location=locationinfo,
        visual=visualinfo,
    )
    rprint(query)
    return query


async def create_query(search_text: str, is_question: bool) -> Query:
    return await extract_info(search_text, is_question)


def time_to_filters(
    query: Query, overwrite: bool = False, mode: Mode = Mode.event
) -> Sequence[ESCombineFilters]:
    if not query.temporal_queries or overwrite:
        query.temporal_queries, _ = get_temporal_filters(query.time, mode)
    return query.temporal_queries


def location_to_filters(
    query: Query, overwrite: bool = False
) -> Sequence[ESCombineFilters]:
    if not query.location_queries or overwrite:
        query.location_queries = get_location_filters(query.location)
    return query.location_queries


def text_to_visual(query: Query, overwrite: bool = False) -> Sequence[ESCombineFilters]:
    if not query.visual_queries or overwrite:
        query.visual_queries = get_visual_filters(query.visual)
    return query.visual_queries


async def create_es_query(
    query: Query,
    ignore_limit_score: bool = False,
    overwrite: bool = False,
    mode: Mode = Mode.event,
) -> ESBoolQuery:
    """
    Convert a query to an Elasticsearch query
    """
    if not query.es or overwrite:
        # Get the filters
        time, date, timestamp, duration, weekday = time_to_filters(query, mode=mode)
        place, _, region = location_to_filters(query)  # type: ignore
        embedding, ocr, _ = text_to_visual(query)  # type: ignore

        min_score = 0.0
        max_score = 1.0
        es = ESBoolQuery()

        # Should queries:
        if place:
            es.should.append(place)
            min_score += 0.01
        # if place_type:
        #     es.should.append(place_type)
        #     min_score += 0.003
        if duration:
            es.should.append(duration)
            min_score += 0.05
        if ocr:
            es.should.append(ocr)
            min_score += 0.01
        # if concepts:
        #     es.should.append(concepts)
        #     min_score += 0.05

        if embedding:
            es.should.append(embedding)

        # Filter queries (no scores)
        es.filter.append(time)
        es.filter.append(date)
        es.filter.append(timestamp)
        es.filter.append(region)
        es.filter.append(weekday)

        if ignore_limit_score:
            max_score = 1.0

        # gauge the max score for normalisation
        # this is done by sending a request with size=1
        elif embedding:
            test_request = ESSearchRequest(
                query=embedding.to_query(), size=1, test=True, mode=mode
            )
            test_results = await send_search_request(test_request)
            if test_results and test_results.hits.max_score:
                max_score = min_score + test_results.hits.max_score
                min_score = min_score + test_results.hits.max_score / 2

        elif es.should:
            test_request = ESSearchRequest(
                query=es.should.to_query(), size=1, test=True, mode=mode
            )
            test_results = await send_search_request(test_request)
            assert not isinstance(test_results, list)
            if test_results and test_results.hits.max_score:
                max_score = test_results.hits.max_score
                min_score = min(min_score, max_score / 2)

        es.min_score = min_score
        es.max_score = max_score
        query.es = es
        query.mark_extracted(es)

    return query.es


async def modify_es_query(
    old_query: Query,
    *,
    location: Optional[LocationInfo] = None,
    time: Optional[TimeInfo] = None,
    visual: Optional[VisualInfo] = None,
    extra_filters: Optional[Sequence[ESCombineFilters]] = None,
    mode: Mode = Mode.event,
) -> Optional[ESBoolQuery]:
    """
    Modify an existing query with new location, time and visual information
    """
    changed = False
    query = old_query.model_copy(deep=True)

    if location:
        query.location = location
        query.location_queries = location_to_filters(query, overwrite=True)
        changed = True

    if time:
        query.time = time
        query.temporal_queries = time_to_filters(query, overwrite=True, mode=mode)
        changed = True

    if visual:
        query.visual = visual
        query.visual_queries = text_to_visual(query, overwrite=True)
        changed = True

    if changed:
        query.es = await create_es_query(
            query, ignore_limit_score=True, overwrite=True, mode=mode
        )

    if query.es and extra_filters:
        for extra_filter in extra_filters:
            query.es.filter.append(extra_filter)

    return query.es
