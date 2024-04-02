"""
All classes and functions related to Elasticsearch
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from configs import DEFAULT_SIZE, SCENE_INDEX
from nltk import defaultdict
from pydantic import BaseModel

from .lifelog import DateTuple

# ====================== #
# ELASTICSEARCH
# ====================== #


class ESQuery(BaseModel):
    """
    A class to represent a query in Elasticsearch
    """

    name: str = "Unnamed"
    is_empty: bool = False

    def to_query(self) -> Optional[Union[Dict, List[Dict]]]:
        raise NotImplementedError

    def __bool__(self):
        return not self.is_empty


class ESSearchRequest(ESQuery):
    """
    A class to represent a search query in Elasticsearch
    """

    test: bool = False
    original_text: Optional[str] = None
    index: str = SCENE_INDEX
    main_field: str = "scene"
    query: Any
    size: int = DEFAULT_SIZE
    includes: List[str] = [main_field]
    sort_field: Optional[str] = None
    sort_order: str = "desc"

    scroll: Optional[bool] = False
    to_agg: Optional[bool] = False
    min_score: Optional[float] = 0.0

    def to_query(self) -> dict:
        sort = ["_score"]
        if self.sort_field:
            sort += [{self.sort_field: self.sort_order}]

        query = {
            "query": self.query,
            "size": self.size,
            "sort": sort,
            "_source": {"includes": self.includes},
        }

        if self.to_agg:
            query["aggs"] = {
                "score_stats": {"extended_stats": {"script": "_score", "sigma": 1.8}}
            }

        if self.min_score:
            query["min_score"] = self.min_score
        return query


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


class GPS(BaseModel):
    lat: float
    lon: float

    def to_dict(self) -> dict:
        return {"lat": self.lat, "lon": self.lon}

    def to_list(self) -> list:
        return [self.lon, self.lat]


class ESGeoBoundingBox(ESQuery):
    """
    A class to represent a geo bounding box query in Elasticsearch
    """

    top_left: GPS
    bottom_right: GPS
    boost: float = 1.0

    def to_query(self) -> dict:
        return {
            "geo_bounding_box": {
                "gps": {
                    "top_left": self.top_left.to_dict(),
                    "bottom_right": self.bottom_right.to_dict(),
                }
            }
        }


class ESMatch(ESQuery):
    """
    A class to represent a match query in Elasticsearch
    """

    field: str
    query: Optional[str] = None
    boost: float = 1.0

    def to_query(self) -> dict:
        assert self.query is not None
        return {
            "match": {
                self.field: {
                    "query": self.query,
                    "boost": self.boost,
                }
            }
        }

    def __bool_(self):
        return bool(self.query)


class ESFuzzyMatch(BaseModel):
    """
    A class to represent a fuzzy match query in Elasticsearch
    """

    field: str
    query: Optional[str] = None
    fuzziness: str = "AUTO"
    boost: float = 1.0

    def to_query(self) -> dict:
        assert self.query is not None
        return {
            "fuzzy": {
                self.field: {
                    "value": self.query,
                    "fuzziness": self.fuzziness,
                    "boost": self.boost,
                }
            }
        }

    def __bool__(self):
        return bool(self.query)


class LocationInfo(BaseModel):
    """
    A class to represent location information in a query
    """

    locations: List[str] = []
    regions: List[str] = []
    location_types: List[str] = []
    gps_bounds: Optional[Sequence] = None
    original_texts: Dict[str, List[str]] = defaultdict(list)

    def __bool__(self):
        return any([self.locations, self.regions, self.location_types])


class TimeInfo(BaseModel):
    """
    A class to represent time information in a query
    """

    time: Tuple[int, int] = (0, 0)  # seconds since midnight
    duration: Optional[int] = None  # seconds
    weekdays: List[int] = []
    dates: List[DateTuple] = []

    original_texts: Dict[str, List[str]] = defaultdict(
        list
    )  # original texts for time info

    def __bool__(self):
        return any([self.time, self.duration, self.weekdays, self.dates])


class VisualInfo(BaseModel):
    """
    A class to represent visual information in a query
    """

    text: str = ""
    concepts: List[str] = []
    OCR: List[str] = []

    def __bool__(self):
        return bool(self.text)


class ESRangeFilter(ESQuery):
    """
    A class to represent a range filter in Elasticsearch
    """

    field: str
    start: int
    end: int
    boost: float = 1.0

    def to_query(self):
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


class ESListQuery(ESQuery):
    """
    A class to represent a list filter in Elasticsearch
    """

    queries: list = []
    name: str = "Unnamed"

    def append(self, query: Any):
        # Making sure that the query is not empty
        if not query:
            return
        self.queries.append(query)
        self.is_empty = False

    def to_query(self) -> Optional[Union[Dict, List[Dict]]]:
        if len(self.queries) == 0:
            return None
        if len(self.queries) == 1:
            return self.queries[0].to_query()
        return [query.to_query() for query in self.queries]  # type: ignore

    def __bool__(self):
        return len(self.queries) > 0


class ESOrFilters(ESListQuery):
    """
    A class to represent a list of OR filters in Elasticsearch
    """

    name: str = "Unnamed"
    minimum_should_match: int = 1

    def to_query(self):
        # get the query from ESListQuery
        list_query = super().to_query()

        if list_query is None:
            return None

        return {
            "bool": {
                "should": list_query,
                "minimum_should_match": self.minimum_should_match,
            }
        }


class ESAndFilters(ESListQuery):
    """
    A class to represent a list of AND filters in Elasticsearch
    """

    pass


class ESEmbedding(ESQuery):
    """
    A class to represent an embedding in Elasticsearch
    """

    embedding: Optional[List[float]] = None
    field: Optional[str] = "clip_vector"
    model: Optional[str] = "exact"
    similarity: Optional[str] = "cosine"
    candidates: Optional[int] = 1000

    def to_query(self) -> dict:
        if not self.embedding:
            raise ValueError("Embedding is empty")
        return {
            "elastiknn_nearest_neighbors": {
                "field": self.field,
                "vec": {"values": self.embedding},
                "model": self.model,
                "similarity": self.similarity,
                "candidates": self.candidates,
            }
        }

    def __bool__(self):
        return self.embedding is not None


ESCombineFilters = Union[
    ESOrFilters,
    ESAndFilters,
    ESEmbedding,
    ESFilter,
    ESRangeFilter,
    ESMatch,
    ESFuzzyMatch,
]


class ESBoolQuery(ESQuery):
    """
    A class to represent a boolean query in Elasticsearch
    This is the MAIN query class that is fed to Elasticsearch
    Everything is combined here
    """

    # These are defined after processing the query
    must: ESAndFilters = ESAndFilters()
    must_not: ESAndFilters = ESAndFilters()
    should: ESOrFilters = ESOrFilters()
    filter: ESAndFilters = ESAndFilters()

    minimum_should_match: Optional[int] = 1

    # These are defined after getting the results from Elasticsearch
    cached: Optional[bool] = False
    cache_key: Optional[str] = None

    # For pagination
    to_scroll: Optional[bool] = False
    scroll_id: Optional[str] = None

    # Function for normalizing the scores
    min_score: float = 0.0
    max_score: float = 1.0
    normalize: Optional[Callable] = lambda x: x

    # Results
    aggregations: Optional[dict] = None

    def to_query(self) -> dict:
        query = {}
        if self.must:
            query["must"] = self.must.to_query()
        if self.must_not:
            query["must_not"] = self.must_not.to_query()
        if self.should:
            query["should"] = self.should.to_query()
            query["minimum_should_match"] = self.minimum_should_match
        if self.filter:
            query["filter"] = self.filter.to_query()

        return {"bool": query}


class MSearchQuery(BaseModel):
    """
    A class to represent a multi search query in Elasticsearch
    """

    queries: List[ESBoolQuery]

    def to_query(self) -> List[Dict]:
        return [query.to_query() for query in self.queries]

    def add_query(self, query: ESBoolQuery):
        self.queries.append(query)
