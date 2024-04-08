from typing import Optional, Sequence

from results.models import Visualisation
from retrieval.search_utils import send_search_request

from query_parse.es_utils import (
    get_location_filters,
    get_temporal_filters,
    get_visual_filters,
)
from query_parse.location import search_for_locations
from query_parse.question import parse_query
from query_parse.time import TimeTagger, add_time, search_for_time
from query_parse.types import (
    ESBoolQuery,
    ESCombineFilters,
    ESSearchRequest,
    LocationInfo,
    TimeInfo,
    VisualInfo,
)
from query_parse.types.elasticsearch import MSearchQuery
from query_parse.types.lifelog import TimeCondition
from query_parse.utils import parse_tags
from query_parse.visual import search_for_visual

time_tagger = TimeTagger()


class Query:
    def __init__(
        self,
        text,
        is_question,
        *,
        gps_bounds: Optional[Sequence] = None,
    ):
        self.is_question = is_question
        self.original_text = text

        # Time info
        self.timeinfo = TimeInfo()
        self.temporal_queries = []

        # Location info
        self.locationinfo = LocationInfo(gps_bounds=gps_bounds)
        self.location_queries = []

        # Visual info
        self.visualinfo = VisualInfo()
        self.visual_queries = []

        # Visualisation info
        self.query_visualisation = Visualisation()

        # For Elasticsearch
        self.es: Optional[ESBoolQuery] = None

        # Set to True if the info has been extracted
        self.extracted = False
        self.query_parts = {}

    async def extract_info(self, text: str, shared_timeinfo: Optional[TimeInfo] = None):
        # Preprocess the text
        text, parsed = parse_tags(text.lower())
        query_parts = await parse_query(text)
        if query_parts:
            self.query_parts = query_parts
        print("Query Parts:", query_parts.items())

        # =================================================================== #
        # LOCATIONS
        # =================================================================== #
        clean_query, locationinfo, loc_visualisation = search_for_locations(
            query_parts["location"], parsed
        )

        # =================================================================== #
        # TIME
        # =================================================================== #
        time = f"{query_parts['time']} {query_parts['date']}".strip()
        if not time:
            time = clean_query

        clean_query, timeinfo, time_visualisation = search_for_time(time_tagger, time)

        # =================================================================== #
        # VISUALISATION
        # =================================================================== #
        query_visualisation = loc_visualisation
        query_visualisation.update(time_visualisation)

        if shared_timeinfo:
            timeinfo = add_time(timeinfo, shared_timeinfo)

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
        self.visualinfo = visualinfo
        self.timeinfo = timeinfo
        self.locationinfo = locationinfo
        self.query_visualisation = query_visualisation
        self.extracted = True

    def get_info(self) -> dict:
        return {
            "query_visualisation": "",  # TODO!
            "country_to_visualise": self.query_visualisation.map_countries,
            "place_to_visualise": self.query_visualisation.map_locations,
        }

    def time_to_filters(self) -> Sequence[ESCombineFilters]:
        if not self.temporal_queries:
            self.temporal_queries, query_visualisation = get_temporal_filters(
                self.timeinfo
            )
            if query_visualisation:
                for value in query_visualisation.values():
                    self.query_visualisation.time_hints.extend(value)
        return self.temporal_queries

    def location_to_filters(self) -> Sequence[ESCombineFilters]:
        if not self.location_queries:
            self.location_queries = get_location_filters(self.locationinfo)
        return self.location_queries

    def text_to_visual(self) -> Sequence[ESCombineFilters]:
        if not self.visual_queries:
            self.visual_queries = get_visual_filters(self.visualinfo)
        return self.visual_queries

    async def to_elasticsearch(self, ignore_limit_score: bool = False) -> ESBoolQuery:
        """
        Convert a query to an Elasticsearch query
        """
        if not self.es:
            if not self.extracted:
                await self.extract_info(self.original_text)
            # Get the filters
            time, date, timestamp, duration, weekday = self.time_to_filters()
            place, place_type, region = self.location_to_filters()  # type: ignore
            embedding, ocr, concepts = self.text_to_visual()  # type: ignore

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
                    query=embedding.to_query(), size=1, test=True
                )
                test_results = await send_search_request(test_request)
                if test_results is not None:
                    max_score = min_score + test_results.max_score
                    min_score = min_score + test_results.min_score

            elif es.should:
                test_request = ESSearchRequest(
                    query=es.should.to_query(), size=1, test=True
                )
                test_results = await send_search_request(test_request)
                if test_results is not None and test_results.max_score > 0.0:
                    max_score = test_results.max_score
                    min_score = min(min_score, max_score / 2)

            es.min_score = min_score
            es.max_score = max_score
            self.es = es

        return self.es


class TemporalQueries:
    """
    This class is used for searching for two related queries
    For example: "I went to the beach and then the park"
    """

    def __init__(self, queries: Sequence[Query], conditions: Sequence[TimeCondition]):
        self.queries = queries
        self.conditions = conditions

    async def extract_info(self):
        for query in self.queries:
            await query.extract_info(query.original_text)

    def get_info(self) -> dict:
        return {}

    async def to_elasticsearch(self):
        """
        Convert a query to an Elasticsearch query
        """
        msearch_queries = []
        for query in self.queries:
            es_query = await query.to_elasticsearch()
            msearch_queries.append(es_query)

        return MSearchQuery(queries=msearch_queries)


def create_query(search_text: str, is_question: bool) -> Query:
    return Query(search_text, is_question)


async def create_es_query(
    query: Query, ignore_limit_score: bool = False
) -> ESBoolQuery:
    return await query.to_elasticsearch(ignore_limit_score)
