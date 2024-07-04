from collections import defaultdict
from typing import Optional, Self, Sequence

from database.main import es_collection
from myeachtra.dependencies import ObjectId
from openai import BaseModel
from pydantic import SkipValidation, model_validator
from retrieval.cuvs import search
from retrieval.search_utils import send_search_request

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
from query_parse.types.elasticsearch import ESEmbedding, ESFilter
from query_parse.types.lifelog import Mode, SingleQuery
from query_parse.visual import search_for_visual

time_tagger = TimeTagger()


class Query(BaseModel):
    query_parts: Optional[SingleQuery] = None

    time: TimeInfo = TimeInfo()
    location: LocationInfo = LocationInfo()
    visual: VisualInfo = VisualInfo()

    location_queries: Sequence[ESCombineFilters] = []
    temporal_queries: Sequence[ESCombineFilters] = []
    visual_queries: Sequence[ESCombineFilters] = []

    es: Optional[ESBoolQuery] = None

    def print_info(self):
        return {
            "time": self.time.export(),
            "location": self.location.export(),
            "visual": self.visual.export(),
        }


class ComboQuery(BaseModel):
    original_text: str = ""
    is_question: bool = False

    main: Optional[Query] = None
    before: Optional[Query] = None
    after: Optional[Query] = None
    must_not: Optional[Query] = None

    extracted: bool = False
    oid: Optional[SkipValidation[ObjectId]] = None
    es: Optional[ESBoolQuery] = None

    @model_validator(mode="after")
    def insert_request(self) -> Self:
        if not self.oid:
            inserted = es_collection.insert_one(self.model_dump(exclude={"responses"}))
            self.oid = inserted.inserted_id
        return self

    def print_info(self):
        return {
            "main": self.main.print_info() if self.main else None,
            "before": self.before.print_info() if self.before else None,
            "after": self.after.print_info() if self.after else None,
            "must_not": self.must_not.print_info() if self.must_not else None,
        }

    def mark_extracted(self, es: ESBoolQuery):
        self.extracted = True
        self.es = es
        es_collection.update_one(
            {"_id": self.oid}, {"$set": {"extracted": True}}, upsert=True
        )


async def extract_info(text: str, is_question: bool) -> ComboQuery:
    query_parts = await parse_query(text, is_question)

    def extract_part(part: SingleQuery | None) -> Query | None:
        if not part:
            return
        # =================================================================== #
        # LOCATIONS
        # =================================================================== #
        parsed = defaultdict(lambda: [])  # deprecated
        clean_query, locationinfo, loc_visualisation = search_for_locations(
            part.location.lower(), parsed
        )

        # =================================================================== #
        # TIME
        # =================================================================== #
        time_str = ""
        if part.time and part.date:
            time_str = f"{part.time} {part.date}"
        elif part.time:
            time_str = part.time
        elif part.date:
            time_str = part.date

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
        visualinfo = search_for_visual(text)

        # =================================================================== #
        # END
        # =================================================================== #
        return Query(
            query_parts=part,
            time=timeinfo,
            location=locationinfo,
            visual=visualinfo,
        )

    return ComboQuery(
        main=extract_part(query_parts.main),
        before=extract_part(query_parts.before),
        after=extract_part(query_parts.after),
        must_not=extract_part(query_parts.must_not),
        original_text=text,
        is_question=is_question,
    )


async def create_query(search_text: str, is_question: bool) -> ComboQuery:
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

        # Important
        es = ESBoolQuery() if not query.es else query.es

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
            # New
            assert isinstance(embedding, ESEmbedding), "Embedding is not an ESEmbedding"
            _, relevant_images = search(embedding.embedding, top_k=1000)

            if mode == Mode.event:
                field = "images"
            else:
                field = "image_path"

            # Hack for the tour
            if not "marklin" in query.visual.text.lower():
                es.should.append(
                    ESFilter(field=field, value=relevant_images.tolist(), boost=0.5)
                )

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

    return query.es


async def create_es_combo_query(
    query: ComboQuery,
    ignore_limit_score: bool = False,
    overwrite: bool = False,
    mode: Mode = Mode.event,
) -> ESBoolQuery:
    es = ESBoolQuery()
    # Create the main query
    if query.main:
        query.main.es = await create_es_query(
            query.main, ignore_limit_score, overwrite, mode
        )
        es = query.main.es.model_copy(deep=True)

    if query.must_not:
        query.must_not.es = await create_es_query(
            query.must_not, ignore_limit_score, overwrite, mode
        )
        es.must_not.append(query.must_not.es)

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
    overwrite: bool = False,
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

    if changed or overwrite:
        query.es = await create_es_query(
            query, ignore_limit_score=True, overwrite=True, mode=mode
        )

    if query.es and extra_filters:
        for extra_filter in extra_filters:
            query.es.filter.append(extra_filter)

    return query.es
