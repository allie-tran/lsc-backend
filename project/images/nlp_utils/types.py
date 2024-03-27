from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Optional, Union

from nltk import defaultdict
from pydantic import BaseModel


class RegexInterval(BaseModel):
    start: int
    end: int
    text: str
    tag: Optional[str] = ""


class DateTuple(BaseModel):
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None


class Visualisation(BaseModel):
    """
    A class to represent info to visualise in the frontend for a search result
    """

    # QUERY ANALYSIS
    locations: List[str] = []
    suggested_locations: List[str] = []
    regions: List[str] = []
    time_hints: List[str] = []

    # MAP VISUALISATION
    map_locations: List[tuple[float, float]] = []
    map_countries: List[dict] = []

    # Don't know what this is
    location_infos: List[Dict[str, Any]] = []


class ESRequest(BaseModel):
    """
    A class to represent a request to Elasticsearch
    """

    index: str
    endpoint: str  # TODO!


class ESQuery(BaseModel):
    """
    A class to represent a query in Elasticsearch
    """

    def to_query(self) -> dict:
        raise NotImplementedError


class ESSearch(ESQuery):
    """
    A class to represent a search query in Elasticsearch
    """

    query: dict
    size: int = 200
    includes: List[str] = []
    sort_field: str = "timestamp"
    sort_order: str = "desc"

    to_agg: Optional[bool] = False
    min_score: Optional[float] = 0.0

    def to_query(self) -> dict:
        return {
            "query": self.query,
            "size": self.size,
            "sort": [{self.sort_field: self.sort_order}],
            "_source": {"includes": self.includes},
        }


class ESScroll(ESQuery):
    """
    A class to represent a scroll in Elasticsearch
    """

    scroll_id: str
    scroll: str = "1m"  # For how long should the scroll be kept alive

    def to_query(self) -> dict:
        return {"scroll": self.scroll, "scroll_id": self.scroll_id}

    def replace_scroll_id(self, scroll_id: str):
        # delete the old scroll_id
        # TODO!
        self.scroll_id = scroll_id


class ESGeoDistance(ESQuery):
    """
    A class to represent a geo distance query in Elasticsearch
    """

    lat: float
    lon: float
    distance: str
    pivot: str = "0.5m"
    boost: float = 1.0

    def to_query(self) -> dict:
        return {
            "geo_distance": {
                "distance": self.distance,
                "gps": [self.lon, self.lat],
                "boost": self.boost,
            }
        }


class ESMatch(BaseModel):
    """
    A class to represent a match query in Elasticsearch
    """

    field: str
    query: str
    boost: float = 1.0

    def to_query(self) -> dict:
        return {
            "match": {
                self.field: {
                    "query": self.query,
                    "boost": self.boost,
                }
            }
        }


class LocationInfo(BaseModel):
    """
    A class to represent location information in a search result
    """

    location: str
    region: str
    country: str
    gps: tuple[float, float] = (0.0, 0.0)
    original_texts: dict[str, list[str]] = defaultdict(list)


class TimeInfo(BaseModel):
    """
    A class to represent time information in a search result
    """

    time: tuple[int, int] = (0, 0)  # seconds since midnight
    duration: int | None = None  # seconds
    weekdays: List[int] = []
    dates: List[DateTuple] = []

    original_texts: dict[str, list[str]] = defaultdict(
        list
    )  # original texts for time info


class ESSubQuery(BaseModel):
    query: Union[Dict[str, Any], List[Dict[str, Any]]]


class ESRangeFilter(ESQuery):
    """
    A class to represent a range filter in Elasticsearch
    """

    field: str
    start: int
    end: int
    boost: float = 1.0

    def to_query(self) -> dict:
        return {
            "range": {
                self.field: {
                    "gte": self.start,
                    "lte": self.end,
                    "boost": self.boost,
                }
            }
        }


class ESFilter(ESQuery):
    """
    A class to represent a filter in Elasticsearch
    """

    field: str
    value: Any
    operator: str = "eq"
    boost: float = 1.0

    def to_query(self) -> dict:
        return {"term": {self.field: {"value": self.value, "boost": self.boost}}}


class ESOrFilters(BaseModel):
    """
    A class to represent a list of OR filters in Elasticsearch
    """

    name: str = "Unnamed"
    filters: Sequence[Union[ESRangeFilter, ESFilter, "ESOrFilters"]]
    minimum_should_match: int = 1

    def to_query(self) -> dict:
        return {
            "bool": {
                "should": [filter.to_query() for filter in self.filters],
                "minimum_should_match": self.minimum_should_match,
            }
        }


class ESAndFilters(ESQuery):
    """
    A class to represent a list of AND filters in Elasticsearch
    """

    name: str = "Unnamed"
    filters: Sequence[Union[ESRangeFilter, ESFilter, ESOrFilters]]
    minimum_should_match: int = 1

    def to_query(self) -> dict:
        return {
            "bool": {
                "must": [filter.to_query() for filter in self.filters],
                "minimum_should_match": self.minimum_should_match,
            }
        }


ESCombineFilters = Union[ESOrFilters, ESAndFilters]


class ESEmbedding(ESQuery):
    """
    A class to represent an embedding in Elasticsearch
    """

    embedding: List[float]
    field: Optional[str] = "clip_vector"
    model: Optional[str] = "exact"
    similarity: Optional[str] = "cosine"
    candidates: Optional[int] = 1000

    def to_query(self) -> dict:
        return {
            "elastiknn_nearest_neighbors": {
                "field": self.field,
                "vec": {"values": self.embedding},
                "model": self.model,
                "similarity": self.similarity,
                "candidates": self.candidates,
            }
        }


class ESBoolQuery(ESQuery):
    """
    A class to represent a boolean query in Elasticsearch
    """

    # These are defined after processing the query
    must: List[ESCombineFilters] = []
    must_not: List[ESCombineFilters] = []
    should: List[ESCombineFilters] = []
    filter: List[ESCombineFilters] = []

    minimum_should_match: Optional[int] = 1

    # These are defined after getting the results from Elasticsearch
    cached: Optional[bool] = False
    cache_key: Optional[str] = None

    # For pagination
    scroll_id: Optional[str] = None

    # Function for normalizing the scores
    normalize: Optional[Callable] = lambda x: x
