from datetime import datetime
from typing import Any, Dict, ForwardRef, Generic, List, Optional, Self, TypeVar

from myeachtra.dependencies import ObjectId
from pydantic import (
    BaseModel,
    Field,
    SkipValidation,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from query_parse.types.requests import AnyRequest

from database.main import request_collection
from database.requests import find_request


class TimeStampModel(BaseModel):
    """
    For caching the streaming response for the timeline
    """

    timestamp: datetime = Field(default_factory=datetime.now)


RequestT = TypeVar("RequestT", bound=AnyRequest)
ResponseT = TypeVar("ResponseT")


class Response(BaseModel):
    progress: Optional[int] = None
    type: str = ""
    response: Any

    # For caching
    oid: Optional[SkipValidation[ObjectId]] = None
    index: Optional[int] = None

    @field_serializer("oid")
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
        self.finished = True
        request_collection.update_one(
            {"_id": self.oid}, {"$set": {"finished": True}}, upsert=True
        )


# ====================== #
# FOURSQUARE
# ====================== #
class TranslatedName(BaseModel):
    name: str
    language: str


class FourSquareLocation(BaseModel):
    formatted_address: str
    country: str


class FourSquareIcon(BaseModel):
    prefix: str
    suffix: str


class FourSquareCategory(BaseModel):
    id: int
    name: str
    icon: FourSquareIcon


class BasicFourSquarePlace(BaseModel):
    fsq_id: str
    name: str
    categories: List[FourSquareCategory] = []


class RelatedPlaces(BaseModel):
    children: List[BasicFourSquarePlace] = []
    parent: Optional[BasicFourSquarePlace] = None


class FourSquarePlace(BasicFourSquarePlace):
    name_translated: List[TranslatedName] = []
    location: FourSquareLocation
    related_places: Optional[RelatedPlaces] = None


Icon = ForwardRef("Icon")


class LocationInfoResult(BaseModel):
    location: str = ""
    location_info: str = ""
    fsq_info: FourSquarePlace | Dict = {}
    icon: Any = None
    related_events: List[Any] = []

    @field_validator("location", "location_info", mode="before")
    @classmethod
    def check_location(cls, v: Any) -> str:
        if isinstance(v, float):
            return ""
        return str(v)
