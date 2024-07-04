# ====================== #
# REQUESTS
# ====================== #

from datetime import datetime
from enum import Enum
from typing import List, Optional, Union

from myeachtra.dependencies import CamelCaseModel, ObjectId
from pydantic import (
    PositiveInt,
    SkipValidation,
    field_serializer,
    field_validator,
    model_validator,
)
from query_parse.types.elasticsearch import GPS
from query_parse.types.options import SearchPipeline


class Step(CamelCaseModel):
    step: PositiveInt
    total: PositiveInt

    @model_validator(mode="after")
    def check_step(self):
        if self.step > self.total:
            raise ValueError("Step cannot be greater than total")
        return self

    def progress(self) -> int:
        return int((self.step / self.total) * 100)


class TemplateRequest(CamelCaseModel):
    session_id: Optional[str] = None

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


class Task(str, Enum):
    AD_HOC = "AD-HOC"
    QA = "QA"
    KIS = "KIS"
    NONE = ""


class GeneralQueryRequest(TemplateRequest):
    task_type: Task
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
    location: Optional[str] = None
    center: Optional[GPS] = None
    group: Optional[str] = None
    image: Optional[str] = None
    scene: Optional[str] = None

    # requestid
    oid: Optional[SkipValidation[ObjectId]] = None
    # esid
    es_id: Optional[SkipValidation[ObjectId]] = None

    @field_serializer("oid")
    @classmethod
    def serialize_oid(cls, v):
        return str(v)

    @model_validator(mode="after")
    def check_validity(self):
        # If location is provided, then ok
        if self.location:
            return self

        # If any of the group, image, scene is provided, then ok
        if any([self.image, self.scene, self.group]):
            return self

        raise ValueError(
            "At least one of the location, group, image or scene should be provided"
        )


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


class AnswerThisRequest(TemplateRequest):
    image: str
    question: str
    relevant_fields: Optional[List[str]] = None


AnyRequest = Union[
    GeneralQueryRequest, TimelineRequest, TimelineDateRequest, SortRequest, MapRequest
]
