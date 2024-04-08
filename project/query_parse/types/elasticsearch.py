"""
All classes and functions related to Elasticsearch
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from configs import DEFAULT_SIZE, SCENE_INDEX
from nltk import defaultdict
from pydantic import BaseModel, field_validator

from rich import print
from .lifelog import DateTuple

# ====================== #
# ELASTICSEARCH
# ====================== #


class ESQuery(BaseModel):
    """
    A class to represent a query in Elasticsearch
    """

    name: str = "Unnamed"

    def to_query(self) -> Optional[Union[Dict, List[Dict]]]:
        raise NotImplementedError

    def __bool__(self):
        raise NotImplementedError


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

    def __bool__(self):
        return bool(self.query)

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

        if self.test:
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

    def __bool__(self):
        return bool(self.lat) and bool(self.lon)

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

    @field_validator("lat")
    @classmethod
    def check_lat(cls, lat):
        if isinstance(lat, str):
            lat = float(lat)
        assert -90 <= lat <= 90, "Latitude should be between -90 and 90"
        return lat

    @field_validator("lon")
    @classmethod
    def check_lon(cls, lon):
        if isinstance(lon, str):
            lon = float(lon)
        assert -180 <= lon <= 180, "Longitude should be between -180 and 180"
        return lon

    def __bool__(self):
        return bool(self.lat) and bool(self.lon)

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

    def __bool__(self):
        return bool(self.top_left) and bool(self.bottom_right)

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
        assert bool(self.query), "Query is empty"
        return {
            "match": {
                self.field: {
                    "query": self.query,
                    "boost": self.boost,
                }
            }
        }

    def __bool__(self):
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
    timestamps: List[Tuple[int, str]] = []
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

    def __bool__(self):
        return self.start is not None and self.end is not None


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

    def __bool__(self):
        return self.value is not None


class ESListQuery(ESQuery):
    """
    A class to represent a list filter in Elasticsearch
    It could be nested (although very complex)
    We build the queries from the bottom up (root is the last query)
    """

    queries: list = []
    name: str = "Unnamed"
    logical_operator: str = "none"
    has_child: bool = False

    @field_validator("queries")
    @classmethod
    def check_queries(cls, queries):
        # append the queries and not filter out the empty ones
        return [query for query in queries if query]

    def append(self, child: Any):
        # Making sure that the query is not empty
        if not child:
            return

        self.has_child = True
        # case 1: if it's a normal query then just append
        if not isinstance(child, ESListQuery):
            self.queries.append(child)
            return

        # case 2: if it's a list query
        # parent is always empty here because we are building from the bottom up

        # if query is the same kind
        if child.logical_operator == self.logical_operator:
            self.queries.extend(child.queries)
            return

        # if query is different kind
        self.queries.append(child)
        return


    def to_query(self) -> Optional[Union[Dict, List[Dict]]]:
        # case 1: it has children -> call them recursively
        if self.has_child:
            queries = []
            for child in self.queries:
                # if it's a normal query, just append
                if not isinstance(child, ESListQuery):
                    queries.append(child.to_query())
                    continue

                # if it's a list query (calling it recursively)
                child_query = child.to_query()

                # all queries are the same kind
                if isinstance(child_query, list):
                    query_list = child_query
                    # There is no way they are the same kind
                    # because of the way we append queries
                    if child.logical_operator == "and":
                        queries.append({"bool": {"must": query_list}})
                    elif child.logical_operator == "or":
                        queries.append({"bool": {"should": child_query}})

                # the query is nested/only one query
                elif isinstance(child_query, dict):
                    queries.append(child_query)

        # case 2: it has no children (leaf node)
        else:
            if len(self.queries) == 1:
                queries = self.queries[0].to_query()
            else:
                queries = [query.to_query() for query in self.queries]

        return queries

    def __bool__(self):
        return len(self.queries) > 0


class ESOrFilters(ESListQuery):
    """
    A class to represent a list of OR filters in Elasticsearch
    """
    minimum_should_match: int = 1
    logical_operator: str = "or"


class ESAndFilters(ESListQuery):
    """
    A class to represent a list of AND filters in Elasticsearch
    """
    logical_operator: str = "and"


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
    filter: ESAndFilters = ESAndFilters()
    should: ESOrFilters = ESOrFilters()

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
        if self.filter:
            query["filter"] = self.filter.to_query()
        if self.should:
            query["should"] = self.should.to_query()
            query["minimum_should_match"] = self.minimum_should_match

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
