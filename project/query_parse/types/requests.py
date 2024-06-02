# ====================== #
# REQUESTS
# ====================== #

from datetime import datetime
from typing import List, Optional, Union

from pydantic import BaseModel, SkipValidation, field_serializer, field_validator
from pydantic.alias_generators import to_camel
from myeachtra.dependencies import ObjectId
from query_parse.types.elasticsearch import GPS
from query_parse.types.options import SearchPipeline


class TemplateRequest(BaseModel):
    session_id: Optional[str] = None

    class Config:
        alias_generator = to_camel
        populate_by_name = True
        str_strip_whitespace = True

    def find_one(self):
        self_dict = self.model_dump(
            exclude_defaults=True,
            exclude_none=True,
            exclude_unset=True,
            exclude={"session_id", "_id"},
        )

        # Turn dict in to "request.name", "request.finished", etc
        criteria = {"finished": True, "name": self.__class__.__name__}
        for key, value in self_dict.items():
            criteria[f"request.{key}"] = value

        return criteria


class GeneralQueryRequest(TemplateRequest):
    session_id: Optional[str] = None
    main: str

    # Optional temporal queries
    before: str = ""
    before_time: str = "1h"

    after: str = ""
    after_time: str = "1h"

    # Optional spatial queries
    gps_bounds: Optional[List[float]] = None

    # Miscs
    pipeline: Optional["SearchPipeline"] = None
    exclude_images: Optional[List[str]] = None


class TimelineRequest(TemplateRequest):
    session_id: Optional[str] = None
    image: str
    # requestid
    oid: Optional[SkipValidation[ObjectId]] = None
    # esid
    es_id: Optional[SkipValidation[ObjectId]] = None


class TimelineDateRequest(TemplateRequest):
    session_id: Optional[str] = None
    date: str
    # requestid
    oid: Optional[SkipValidation[ObjectId]] = None
    # esid
    es_id: Optional[SkipValidation[ObjectId]] = None

    @field_validator("date")
    def check_date(cls, value):
        # make sure date is inthe right format "dd-mm-yyyy"
        if isinstance(value, str):
            try:
                value = value.replace("/", "-")
                datetime.strptime(value, "%d-%m-%Y")
                return value
            except ValueError:
                pass
        if isinstance(value, datetime):
            return value
        raise ValueError("Invalid date format, should be 'dd-mm-yyyy'")


# ====================== #
# MAP REQUESTS
# ====================== #
class MapRequest(TemplateRequest):
    location: str
    center: GPS
    # requestid
    oid: Optional[SkipValidation[ObjectId]] = None
    # esid
    es_id: Optional[SkipValidation[ObjectId]] = None

    @field_serializer("oid")
    @classmethod
    def serialize_oid(cls, v):
        return str(v)

# ====================== #
# USER OPTIONS & RESPONSES
# ====================== #
class VisualSimilarityRequest(TemplateRequest):
    image: str

class SortRequest(TemplateRequest):
    session_id: Optional[str] = None
    sort: str
    order: str = "asc"
    size: int = 10

AnyRequest = Union[
    GeneralQueryRequest, TimelineRequest, TimelineDateRequest, SortRequest,
    MapRequest
]
