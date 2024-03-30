from typing import Optional, Sequence

from retrieval.search_utils import send_search_request

from query_parse.es_utils import (
    get_location_filters,
    get_temporal_filters,
    get_visual_filters,
)
from query_parse.location import search_for_locations
from query_parse.time import TimeTagger, add_time, search_for_time
from query_parse.types import (
    ESBoolQuery,
    ESCombineFilters,
    ESSearchRequest,
    LocationInfo,
    TimeInfo,
    VisualInfo,
    Visualisation,
)
from query_parse.utils import parse_tags
from query_parse.visual import search_for_visual

time_tagger = TimeTagger()


class Query:
    def __init__(
        self, text, shared_filters=None, gps_bounds: Optional[Sequence] = None
    ):
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

        # Extract the information from the text
        self.extract_info(text, shared_filters)

        # For Elasticsearch
        self.es: Optional[ESBoolQuery] = None
        self.cached = False
        self.scroll_id = ""
        self.es_filters = []
        self.es_should = []
        self.max_score = 1.0
        self.min_score = 0.0
        self.normalise_score = lambda x: (x - self.min_score) / (
            self.max_score - self.min_score
        )

    def extract_info(self, text: str, shared_timeinfo: Optional[TimeInfo] = None):

        # Preprocess the text
        text = text.strip(". \n").lower()
        text, parsed = parse_tags(text)
        print("Text query:", text)

        # =================================================================== #
        # LOCATIONS
        # =================================================================== #
        clean_query, locationinfo, loc_visualisation = search_for_locations(
            text, parsed
        )

        # =================================================================== #
        # TIME
        # =================================================================== #
        clean_query, timeinfo, time_visualisation = search_for_time(
            time_tagger, clean_query
        )

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
        visualinfo = search_for_visual(clean_query)

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
                for key, value in query_visualisation.items():
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

    def to_elasticsearch(self, ignore_limit_score: bool = False) -> ESBoolQuery:
        """
        Convert a query to an Elasticsearch query
        """
        if not self.es:
            scroll_id = None
            main_query = {}

            # Get the filters
            time, date, duration, weekday = self.time_to_filters()
            place, place_type, region = self.location_to_filters()
            embedding, ocr, concepts = self.text_to_visual()

            min_score = 0.0
            max_score = 1.0
            test_query = None  # This is for gauging the max score
            es = ESBoolQuery()

            # Should queries:
            if place:
                es.should.append(place)
                min_score += 0.01
            if place_type:
                es.should.append(place_type)
                min_score += 0.003
            if duration:
                es.should.append(duration)
                min_score += 0.05
            if ocr:
                es.should.append(ocr)
                min_score += 0.01
            if concepts:
                es.should.append(concepts)
                min_score += 0.05

            if embedding:
                es.should.append(embedding)

            # Filter queries (no scores)
            es.filter.append(time)
            es.filter.append(date)
            es.filter.append(region)
            es.filter.append(weekday)

            if ignore_limit_score:
                # if embedding:
                #     min_score += CLIP_MIN_SCORE - 0.15
                max_score = 1.0

            # gauge the max score for normalisation
            # this is done by sending a request with size=1
            elif embedding:
                test_request = ESSearchRequest(query=embedding.to_query(), size=1)
                test_results = send_search_request(test_request)
                if test_results:
                    max_score = min_score + test_results.max_score
                    min_score = min_score + test_results.min_score

            elif es.should:
                test_request = ESSearchRequest(query=es.should.to_query(), size=1)
                test_results = send_search_request(test_request)
                if test_results and test_results.max_score > 0.0:
                    max_score = test_results.max_score
                    min_score = min(min_score, max_score / 2)

            es.min_score = min_score
            es.max_score = max_score
            self.es = es

        return self.es
