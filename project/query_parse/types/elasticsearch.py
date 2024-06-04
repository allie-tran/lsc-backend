"""
All classes and functions related to Elasticsearch
"""

import json
from typing import Any, Callable, Dict, List, Literal, Optional, Self, Sequence, Tuple, Union

from configs import DEFAULT_SIZE, IMAGE_INDEX, SCENE_INDEX
from nltk import defaultdict
from pydantic import BaseModel, field_validator, model_validator

from .lifelog import DateTuple, Mode

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
    mode: Mode = Mode.event
    query: Any

    sort_field: Optional[str] = None
    sort_order: str = "desc"

    scroll: Optional[bool] = False
    to_agg: Optional[bool] = False
    min_score: Optional[float] = 0.0

    index: str = SCENE_INDEX
    main_field: str = "scene"
    size: int = DEFAULT_SIZE
    includes: List[str] = [main_field]

    @model_validator(mode="after")
    def get_fields_based_on_mode(self) -> Self:
        if self.mode == "image":
            self.main_field = "image_path"
            self.includes = [self.main_field]
            self.index = IMAGE_INDEX
            self.size *= 5
            if self.sort_field in ["start_timestamp", "end_timestamp"]:
                self.sort_field = "timestamp"
            if self.sort_field in ["start_time", "end_time"]:
                self.sort_field = "time"
        return self

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


class ESFuzzyMatch(ESMatch):
    """
    A class to represent a fuzzy match query in Elasticsearch
    """

    fuzziness: str = "AUTO"

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
    from_center: Optional[GPS] = None

    def __bool__(self):
        return any([self.locations, self.regions, self.location_types])

    def export(self) -> str:
        info_dict = self.model_dump(
            exclude_defaults=True,
            exclude_none=True,
            exclude_unset=True,
            exclude={"original_texts"},
        )
        locations = info_dict.get("locations")
        regions = info_dict.get("regions")
        location_types = info_dict.get("location_types")

        res = []

        if locations:
            locations = ", ".join(locations)
            res.append(locations)
        if regions:
            regions = ", ".join(regions)
            res.append(regions)
        if location_types:
            location_types = ", ".join(location_types)
            res.append(location_types)

        return ", ".join(res)


class TimeInfo(BaseModel):
    """
    A class to represent time information in a query
    """

    time: Tuple[int, int] = (0, 86400)  # seconds from midnight
    timestamps: List[Tuple[int, str]] = []
    duration: Optional[int] = None  # seconds
    weekdays: List[str] = []
    dates: List[DateTuple] = []

    original_texts: Dict[str, List[str]] = defaultdict(
        list
    )  # original texts for time info

    def __bool__(self):
        return any([self.time, self.duration, self.weekdays, self.dates])

    @staticmethod
    def seconds_to_time(seconds: int) -> str:
        """Preferred time format: HH:MM AM/PM"""
        hours, remainder = divmod(seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}"

    def export(self) -> str:
        info_dict = self.model_dump(
            exclude_defaults=True,
            exclude_none=True,
            exclude_unset=True,
            exclude={"original_texts"},
        )
        time = info_dict.get("time")
        duration = info_dict.get("duration")
        weekdays = info_dict.get("weekdays")
        dates = info_dict.get("dates")

        res = []

        if time:
            time = f"{self.seconds_to_time(time[0])} to {self.seconds_to_time(time[1])}"
            res.append(time)
        if duration:
            duration = f"{duration} seconds"
            res.append(duration)
        if weekdays:
            weekdays = ", ".join(weekdays)
            res.append(weekdays)
        if dates:
            dates = ", ".join([DateTuple(**date).export() for date in dates])
            res.append(dates)

        return ", ".join(res)


class VisualInfo(BaseModel):
    """
    A class to represent visual information in a query
    """

    text: str = ""
    concepts: List[str] = []
    OCR: List[str] = []

    def __bool__(self):
        return bool(self.text)

    def export(self) -> str:
        return self.text


class ESRangeFilter(ESQuery):
    """
    A class to represent a range filter in Elasticsearch
    """

    field: str
    start: int | str | float
    end: int | str | float
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
            if isinstance(child, ESQuery):
                self.queries.append(child)
            else:
                print(child)
                raise ValueError("Child is not a query")
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


# For LSC24, exclude the year 2015, 2016 and 2018
MUST_NOT = ESAndFilters(
    queries=[
        ESFilter(field="year", value=2015),
        ESFilter(field="year", value=2016),
        ESFilter(field="year", value=2018),
    ]
)


class ESBoolQuery(ESQuery):
    """
    A class to represent a boolean query in Elasticsearch
    This is the MAIN query class that is fed to Elasticsearch
    Everything is combined here
    """
    # These are defined after processing the query
    must: ESAndFilters = ESAndFilters()
    must_not: ESAndFilters = MUST_NOT
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
            try:
                query["filter"] = self.filter.to_query()
            except AttributeError as e:
                print(self.filter)
                raise (e)
        if self.should:
            query["should"] = self.should.to_query()
            query["minimum_should_match"] = self.minimum_should_match

        return {"bool": query}

    def __bool__(self):
        return bool(self.must or self.must_not or self.filter or self.should)


class MSearchQuery(BaseModel):
    """
    A class to represent a multi search query in Elasticsearch
    """

    queries: List[ESBoolQuery]
    index: str = SCENE_INDEX
    includes: List[str] = ["scene"]

    def to_query(self) -> str:
        body = []
        for query in self.queries:
            body.append({"index": self.index})
            body.append(
                {
                    "query": query.to_query(),
                    "size": 1,
                    "_source": {"includes": self.includes},
                }
            )

        return "\n".join(json.dumps(b) for b in body) + "\n"

    def add_query(self, query: ESBoolQuery):
        self.queries.append(query)

    def __bool__(self):
        return bool(self.queries)
