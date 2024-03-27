from collections.abc import Iterable, Sequence
from typing import Optional

from project.images.nlp_utils.es_utils import get_location_filters, get_temporal_filters

from ..nlp_utils.common import *
from ..nlp_utils.time import TimeTagger, add_time, search_for_time
from ..nlp_utils.types import (
    ESCombineFilters,
    ESEmbedding,
    ESQuery,
    TimeInfo,
    Visualisation,
)
from ..nlp_utils.utils import (
    choose_countries_for_map,
    get_visual_text,
    parse_tags,
    postprocess_countries,
    remove_keywords,
)

time_tagger = TimeTagger()


def is_enabled(location: str, disabled: list[str]) -> bool:
    """
    Check if the location is OK to be used
    """
    if location in ["", "the house", "restaurant"] + disabled:
        return False
    return True


def search(wordset: Iterable[str], text: str, disabled: list[str] = []) -> list[str]:
    """
    Search for keywords in the text
    """
    results = []
    text = " " + text + " "
    for keyword in wordset:
        if is_enabled(keyword, disabled):
            if re.search(r"\b" + re.escape(keyword) + r"\b", text, re.IGNORECASE):
                results.append(keyword)
    return results


def search_possible_location(text: str, disabled: list[str] = []) -> list[str]:
    """
    Search for possible locations in the text
    Based on partial matches of the location names
    """
    results = []
    for location in LOCATIONS:
        if is_enabled(location, disabled):
            for i, extra in enumerate(LOCATIONS[location]):
                if is_enabled(extra, disabled):
                    if re.search(r"\b" + re.escape(extra) + r"\b", text, re.IGNORECASE):
                        if location not in results:
                            results.append(location)
                        break
    return results


class Query:
    def __init__(self, text, shared_filters=None):
        self.original_text = text

        # Time info
        self.temporal_filters = []

        # Location info
        self.location_queries = []
        self.location_filters = []

        # Visualisation info
        self.query_visualisation = Visualisation()

        # Visual info
        self.clip_embedding = None

        # Extract the information from the text
        self.extract_info(text, shared_filters)

        # For Elasticsearch
        self.es: ESQuery | None = None
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
        def search_words(wordset, disabled=[]):
            return search(wordset, text, disabled)

        # Preprocess the text
        text = text.strip(". \n").lower()
        text, parsed = parse_tags(text)
        print("Text query:", text)

        # =================================================================== #
        # LOCATIONS
        # =================================================================== #
        locations = search_words(LOCATIONS, parsed["disabled_locations"])
        text = remove_keywords(text, locations)

        # self.location_infos = search_words(
        #     LOCATION_INFOS, self.parsed["disabled_locations"]
        # )

        regions = search_words(REGIONS, parsed["disabled_regions"])
        text = remove_keywords(text, regions)
        regions = postprocess_countries(regions)

        # =================================================================== #
        # TIME
        # =================================================================== #
        clean_query, timeinfo = search_for_time(time_tagger, text)

        # =================================================================== #
        # VISUALISATION
        # =================================================================== #
        # Locations
        query_visualisation = Visualisation()
        if locations:
            query_visualisation.locations = locations
            query_visualisation.map_locations = [
                GPS_NORMAL_CASE[location] for location in locations
            ]
        else:
            query_visualisation.suggested_locations = search_possible_location(
                text, parsed["disabled_locations"]
            )

        # Regions
        if regions:
            query_visualisation.regions = regions
            query_visualisation.map_countries = choose_countries_for_map(regions)

        # Time
        for key, value in timeinfo.original_texts.items():
            if value:
                self.query_visualisation.time_hints.extend(f"{key}: {value}")

        if shared_timeinfo:
            timeinfo = add_time(timeinfo, shared_timeinfo)

        # =================================================================== #
        # VISUAL INFO
        # =================================================================== #
        visual_text = get_visual_text(text, clean_query)

        if visual_text:
            print("Visual Text:", visual_text)
        else:
            print("No Visual Text.")

        # =================================================================== #
        # END
        # =================================================================== #
        self.timeinfo = timeinfo
        self.visual_text = visual_text
        self.visualisation = query_visualisation
        self.locations = locations
        self.regions = regions

    def get_info(self) -> dict:
        return {
            "query_visualisation": "",  # TODO!
            "country_to_visualise": self.query_visualisation.map_countries,
            "place_to_visualise": self.query_visualisation.map_locations,
        }

    def time_to_filters(self) -> Sequence[ESCombineFilters]:
        if not self.temporal_filters:
            self.temporal_filters, query_visualisation = get_temporal_filters(
                self.timeinfo
            )
            if query_visualisation:
                for key, value in query_visualisation.items():
                    self.query_visualisation.time_hints.extend(value)
        return self.temporal_filters

    def location_to_filters(self) -> Sequence[ESCombineFilters]:
        if not self.location_filters:
            self.location_filters = get_location_filters(self.locations, self.regions)
        return self.location_filters

    def embed_visual(self) -> ESEmbedding:
        if not self.embedding:
            self.embedding = ESEmbedding(embedding=[1, 2, 3])
        return self.embedding

    def to_elasticsearch(self):

        pass
