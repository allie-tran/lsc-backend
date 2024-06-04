from datetime import datetime
from typing import Any, Generic, List, Optional, Self, TypeVar

from configs import CACHE
from myeachtra.dependencies import ObjectId, CamelCaseModel
from pydantic import (
    Field,
    SkipValidation,
    computed_field,
    field_serializer,
    model_validator,
)
from query_parse.types.requests import AnyRequest

from database.main import request_collection
from database.requests import find_request


class TimeStampModel(CamelCaseModel):
    """
    For caching the streaming response for the timeline
    """

    timestamp: datetime = Field(default_factory=datetime.now)


RequestT = TypeVar("RequestT", bound=AnyRequest)
ResponseT = TypeVar("ResponseT")


class Response(CamelCaseModel):
    progress: Optional[int] = None
    type: str = ""
    response: Any

    # For caching
    oid: Optional[SkipValidation[ObjectId]] = None
    index: Optional[int] = None
    es_id: Optional[SkipValidation[ObjectId]] = None

    @field_serializer("oid", "es_id")
    def serialize_oid(self, v: ObjectId) -> str:
        return str(v)

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(
            exclude_unset=True, exclude_defaults=True, exclude_none=True, **kwargs
        )


class GeneralRequestModel(TimeStampModel, Generic[RequestT, ResponseT]):
    finished: bool = False
    request: RequestT
    responses: List[Response] = []
    oid: Optional[SkipValidation[ObjectId]] = None

    @field_serializer("oid")
    def serialize_oid(self, v: ObjectId) -> str:
        return str(v)

    @computed_field
    def name(self) -> str:
        return self.request.__class__.__name__

    @model_validator(mode="after")
    def insert_request(self) -> Self:
        if not CACHE:
            return self
        if not self.oid:
            # Find the request in the database
            existing_request = find_request(self.request)
            if existing_request:
                print("Found cached request")
                self.responses = [
                    Response(**res) for res in existing_request["responses"]
                ]
                self.oid = existing_request["_id"]
                self.finished = existing_request["finished"]
            else:
                inserted = request_collection.insert_one(
                    self.model_dump(exclude={"responses"})
                )
                self.oid = inserted.inserted_id
        return self

    def add(self, response: Response) -> Response:
        if not CACHE:
            return response
        response.oid = self.oid
        response.index = len(self.responses) - 1
        request_collection.update_one(
            {"_id": self.oid},
            {"$push": {"responses": response.model_dump(exclude={"oid"})}},
            upsert=True,
        )
        self.responses.append(response)
        return response

    def mark_finished(self):
        if not CACHE:
            return
        self.finished = True
        request_collection.update_one(
            {"_id": self.oid}, {"$set": {"finished": True}}, upsert=True
        )


# ====================== #
# FOURSQUARE
# ====================== #
class TranslatedName(CamelCaseModel):
    name: str
    language: str


class FourSquareLocation(CamelCaseModel):
    formatted_address: str
    country: str


class FourSquareIcon(CamelCaseModel):
    prefix: str
    suffix: str


class FourSquareCategory(CamelCaseModel):
    id: int
    name: str
    icon: FourSquareIcon


class BasicFourSquarePlace(CamelCaseModel):
    fsq_id: str
    name: str
    categories: List[FourSquareCategory] = []


class RelatedPlaces(CamelCaseModel):
    children: List[BasicFourSquarePlace] = []
    parent: Optional[BasicFourSquarePlace] = None


class FourSquarePlace(BasicFourSquarePlace):
    name_translated: List[TranslatedName] = []
    location: FourSquareLocation
    related_places: Optional[RelatedPlaces] = None


# ====================== #
