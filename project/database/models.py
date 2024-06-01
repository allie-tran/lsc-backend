from datetime import datetime
from typing import (
    Annotated,
    Any,
    Dict,
    ForwardRef,
    Generic,
    List,
    Optional,
    Self,
    TypeVar,
)

from bson import ObjectId as _ObjectId
from fastapi.responses import StreamingResponse
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    SkipValidation,
    computed_field,
    field_validator,
    model_validator,
)
from query_parse.types.requests import (
    AnyRequest,
    GeneralQueryRequest,
    TimelineDateRequest,
    TimelineRequest,
)
from results.models import ReturnResults, TimelineResult

from database.main import request_collection
from database.requests import find_request


def check_object_id(value: str) -> str:
    if not _ObjectId.is_valid(value):
        raise ValueError("Invalid ObjectId")
    return value


ObjectId = Annotated[
    str,
    Field(..., alias="_id", description="MongoDB ObjectId"),
    AfterValidator(check_object_id),
]


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

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(
            exclude_unset=True, exclude_defaults=True, exclude_none=True, **kwargs
        )


class GeneralRequestModel(TimeStampModel, Generic[RequestT, ResponseT]):
    finished: bool = False
    request: RequestT
    responses: List[Response] = []
    oid: Optional[SkipValidation[ObjectId]] = None

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

    def add(self, response: Response):
        self.responses.append(response)
        request_collection.update_one(
            {"_id": self.oid},
            {"$push": {"responses": response.model_dump()}},
            upsert=True,
        )

    def mark_finished(self):
        self.finished = True
        request_collection.update_one(
            {"_id": self.oid}, {"$set": {"finished": True}}, upsert=True
        )

    def get_full_response(self):
        raise NotImplementedError


async def streaming_helper(responses: List[Response]):
    for response in responses:
        yield f"data: {response.model_dump_json()}\n\n"
    yield "data: END\n\n"


class SearchRequestModel(GeneralRequestModel[GeneralQueryRequest, str | ReturnResults]):
    async def get_full_response(self) -> StreamingResponse:
        return StreamingResponse(
            streaming_helper(self.responses),
            media_type="text/event-stream",
        )


class TimelineRequestModel(
    GeneralRequestModel[TimelineRequest | TimelineDateRequest, TimelineResult]
):
    def get_full_response(self) -> Optional[TimelineResult]:
        res = self.responses[-1]
        return TimelineResult.model_validate(res.response)


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

    @field_validator("location", "location_info", mode="before")
    @classmethod
    def check_location(cls, v: Any) -> str:
        if isinstance(v, float):
            return ""
        return str(v)
