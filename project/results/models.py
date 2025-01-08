from datetime import datetime
from typing import Any, Dict, Generic, List, Literal, Optional, Self, Sequence, TypeVar, Union

from database.models import FourSquarePlace
from myeachtra.dependencies import CamelCaseModel
from pydantic import (
    ConfigDict,
    Field,
    InstanceOf,
    SkipValidation,
    computed_field,
    field_validator,
    model_validator,
)
from query_parse.types.elasticsearch import GPS
from query_parse.types.lifelog import RelevantFields, TimeCondition
from query_parse.utils import (
    extend_no_duplicates,
    extend_with_count,
    merge_list,
    merge_str,
)


class ResponseModel(CamelCaseModel):
    pass


class Visualisation(CamelCaseModel):
    """
    A class to represent info to visualise in the frontend for a search result
    """

    # QUERY ANALYSIS
    locations: List[str] = []
    suggested_locations: List[str] = []
    regions: List[str] = []
    time_hints: List[str] = []

    # MAP VISUALISATION
    map_locations: List[List[Union[float, int]]] = []
    map_countries: List[dict] = []

    # Don't know what this is
    location_infos: List[Dict[str, Any]] = []

    def update(self, other: "Visualisation"):
        for key, value in other.model_dump().items():
            if key in self.model_dump():
                self.model_dump()[key].extend(value)

    def to_dict(self) -> dict:
        return self.model_dump()


class Icon(CamelCaseModel):
    """
    An icon for the map
    """

    type: Literal["foursquare", "material"]
    name: str
    prefix: str = ""
    suffix: str = ""

    @model_validator(mode="after")
    def check_consistency(self):
        if self.type == "foursquare":
            if not self.prefix or not self.suffix:
                raise ValueError("Foursquare icons need a prefix and suffix")
        return self


GeneralIcon = Icon(type="material", name="place")


class Marker(CamelCaseModel):
    """
    A marker for the map
    """

    model_config = ConfigDict(coerce_numbers_to_str=True)
    location: str
    location_info: str
    points: List[GPS] = Field(default_factory=list, exclude=True)
    icon: Optional[Icon] = GeneralIcon
    center: Optional[GPS] = None

    @field_validator("location", "location_info")
    @classmethod
    def check_location(cls, v: str | float) -> str:
        if isinstance(v, float):
            return str(v)
        return str(v)


    @field_validator("points")
    @classmethod
    def check_points(cls, v: Sequence[GPS | None]) -> List[GPS]:
        return [x for x in v if x]

    @model_validator(mode="after")
    def get_center(self) -> Self:
        """
        Get the center of the GPS
        """
        if not self.center:
            lats = [x.lat for x in self.points]
            lons = [x.lon for x in self.points]
            if lats and lons:
                self.center = GPS(lat=sum(lats) / len(lats), lon=sum(lons) / len(lons))
        return self

    def __bool__(self):
        return bool(self.points)


class Image(CamelCaseModel):
    src: str
    aspect_ratio: float = 16 / 9
    hash_code: str = ""


class Event(CamelCaseModel):
    # IDs
    group: str = ""
    scene: str = ""

    # Data
    name: str = ""
    images: List[Image] = []

    start_time: datetime = Field(..., exclude=True)
    end_time: datetime = Field(..., exclude=True)
    duration: float = 30 # 30 seconds

    location: str = ""
    location_info: str = ""

    # For map visualisation
    gps: List[GPS] = Field(default_factory=list, exclude=True)
    markers: List[Marker] = []
    orphans: List[GPS] = []

    region: List[str] = []
    country: str = ""

    ocr: List[str] = Field(default_factory=list, exclude=True)
    description: str = ""

    image_scores: List[float] = Field(default_factory=list, exclude=True)
    icon: Optional[Icon] = Icon(type="material", name="place")

    count: int = 1

    # # Extra
    # time: Optional[datetime] = None
    # minute: Optional[str] = None
    # hour: Optional[str] = None
    # day: Optional[str] = None
    # date: Optional[str] = None
    # week: Optional[str | int] = None
    # weekday: Optional[str] = None
    # month: Optional[str] = None
    # year: Optional[str | int] = None
    # city: Optional[str] = None
    # days: Optional[int] = None
    # hours: Optional[int] = None
    # weeks: Optional[int] = None
    # place: Optional[str] = None
    # place_info: Optional[str] = None

    def custom_iter(self):
        return iter([self])

    @field_validator("start_time", "end_time")
    @classmethod
    def check_time(cls, v: Union[str, datetime]) -> datetime:
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

    @field_validator("ocr")
    @classmethod
    def check_ocr(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return v.split(",")
        return v

    @field_validator("location")
    @classmethod
    def check_location(cls, v: str) -> str:
        if v == "---":
            return ""
        return v

    def __bool__(self):
        return bool(self.images)

    def dict(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["start_time"] = data["start_time"].isoformat()
        data["end_time"] = data["end_time"].isoformat()
        return data

    @computed_field
    def center(self) -> Optional[GPS]:
        """
        Get the center of the GPS
        """
        if not self.markers:
            return None

        lats = [x.lat for marker in self.markers for x in marker.points]
        lons = [x.lon for marker in self.markers for x in marker.points]
        if lats and lons:
            return GPS(lat=sum(lats) / len(lats), lon=sum(lons) / len(lons))

    @computed_field
    def score(self) -> float:
        if self.image_scores:
            return max(self.image_scores)
        return 0.0

    def merge_with_one(self, other: "Event", scores: List[float]):
        score, other_score = scores
        # IDS
        self.group = merge_str(self.group, other.group)
        self.scene = merge_str(self.scene, other.scene)

        # DATA
        self.name = self.name or other.name

        # Score
        if not self.image_scores:
            self.image_scores = [score] * len(self.images)

        self.images.extend(other.images)
        self.image_scores.extend([other_score] * len(other.images))

        # Time
        self.start_time = min(self.start_time, other.start_time)
        self.end_time = max(self.end_time, other.end_time)
        self.duration = (self.end_time - self.start_time).total_seconds()

        # Location
        self.location = merge_str(self.location, other.location, " and ")
        self.location_info = merge_str(self.location_info, other.location_info, " and ")

        # Map
        self.markers.extend(other.markers)

        # Region
        self.region = merge_list(self.region, other.region)
        self.country = merge_str(self.country, other.country)

        # OCR
        self.ocr = merge_list(self.ocr, other.ocr)

    def merge_with_many(self, score: float, others: List["Event"], scores: List[float]):
        # Usualy the case if that the scores are descreasing
        # So we can just take the first one
        if len(others) == 1:
            return self.merge_with_one(others[0], [score, scores[0]])

        # DATA
        self.name = self.name or others[0].name

        self.image_scores = [score] * len(self.images)
        for i, other in enumerate(others):
            self.images.extend(other.images)
            self.image_scores.extend([scores[i]] * len(other.images))

        # Time
        self.start_time = min([self.start_time] + [x.start_time for x in others])
        self.end_time = max([self.end_time] + [x.end_time for x in others])
        self.duration = (self.end_time - self.start_time).total_seconds()

        # Location
        self.location = " and ".join(
            extend_no_duplicates([self.location], [x.location for x in others])
        )
        self.location_info = " and ".join(
            extend_no_duplicates(
                [self.location_info], [x.location_info for x in others]
            )
        )

        # Map
        for other in others:
            self.markers.extend(other.markers)

        # Region
        self.region = extend_no_duplicates(
            self.region, [x for y in others for x in y.region]
        )
        self.country = ", ".join(
            extend_no_duplicates([self.country], [x.country for x in others])
        )

        # OCR
        self.ocr = extend_with_count(self.ocr, [x.ocr for x in others])

    def copy_to_derived_event(self) -> "DerivedEvent":
        data = self.model_dump()
        # add all the excluded fields
        for name, field in self.model_fields.items():
            if field.exclude:
                data[name] = getattr(self, name)

        return DerivedEvent(**data)


class DerivedEvent(Event):
    """
    An event with derived fields
    """

    model_config = ConfigDict(extra="allow")

fakedate = datetime(2020, 1, 1)

class Results(CamelCaseModel):
    pass



class ImageResults(Results):
    images: List[str]
    scores: List[float] = Field(default_factory=list, exclude=True)


EventT = TypeVar("EventT")

class DoubletEvent(CamelCaseModel):
    main: Event
    conditional: Event
    condition: TimeCondition

    def custom_iter(self):
        return iter([self.main, self.conditional])


class TripletEvent(CamelCaseModel):
    main: Event
    before: Optional[Event] = None
    after: Optional[Event] = None

    def custom_iter(self):
        # return only the ones that are not None
        return iter([x for x in [self.main, self.before, self.after] if x])

class GenericEventResults(Results, Generic[EventT]):
    events: List[EventT]
    scores: List[float]
    scroll_id: str = ""
    min_score: float = 0.0
    max_score: float = 1.0
    normalized: bool = False

    relevant_fields: List[str] = []

    def __bool__(self):
        return bool(self.events)

    def __len__(self):
        return len(self.events)


OriginalEventResults = GenericEventResults[Event]
DerivedEventResults = GenericEventResults[DerivedEvent]

EventResults = GenericEventResults[InstanceOf[Event]]

DoubletEventResults = GenericEventResults[DoubletEvent]
TripletEventResults = GenericEventResults[TripletEvent]


class EvidenceLink(CamelCaseModel):
    event_id: int


class AnswerResultWithEvent(CamelCaseModel):
    text: str
    evidence: List[Event] = []
    explanation: List[str] = []

class AnswerResult(CamelCaseModel, revalidate_instances="always"):
    text: str
    evidence: List[int] = []
    explanation: List[str] = []
    time: datetime = Field(default_factory=datetime.now, exclude=True)

    @field_validator("evidence")
    def sort_evidence(cls, v: List[int] | List[EvidenceLink]) -> List[int]:
        new_v: List[int] = []
        for x in v:
            if isinstance(x, EvidenceLink):
                new_v.append(x.event_id)
            else:
                new_v.append(x)
        # Sort and remove duplicates
        return list(set(sorted(new_v)))

    @field_validator("explanation")
    def sort_explanation(cls, v: List[str]) -> List[str]:
        # remove explanations that are substrings of others
        new_v = []
        for i in range(len(v)):
            if not any(v[i] in x for x in v[i + 1 :]):
                new_v.append(v[i])
        return new_v


class AnswerListResult(CamelCaseModel, revalidate_instances="always"):

    answers: Dict[str, AnswerResult] = {}

    def add_answer(self, answer: AnswerResult):
        if answer.text not in self.answers:
            self.answers[answer.text] = answer
            return

        current_answer = self.answers[answer.text]
        current_answer.evidence.extend(answer.evidence)
        current_answer.explanation.extend(answer.explanation)

        current_answer = AnswerResult.model_validate(current_answer)
        self.answers[answer.text] = current_answer

    def export(self) -> List[AnswerResult]:
        sorted_answers = sorted(
            self.answers.values(), key=lambda x: x.time
        )
        return sorted_answers


class TimelineScene(CamelCaseModel):
    scene: str
    images: List[Image]

class TimelineGroup(CamelCaseModel):
    """
    A group of events
    """

    group: str
    scenes: List[TimelineScene]
    time_info: List[str]
    location: str
    location_info: str


class HighlightItem(CamelCaseModel):
    """
    A highlight item
    """

    group: str
    scene: str
    image: str


class TimelineResult(CamelCaseModel):
    """
    Wrapper for the results
    """

    date: datetime
    result: List[TimelineGroup]
    highlight: Optional[HighlightItem] = None


fakedate = datetime(2020, 1, 1)


class PartialEvent(DerivedEvent):
    """
    A partial event
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="allow", coerce_numbers_to_str=True
    )
    start_time: SkipValidation[datetime] = Field(default=fakedate, exclude=True)
    end_time: SkipValidation[datetime] = Field(default=fakedate, exclude=True)

    @field_validator("location", "location_info")
    @classmethod
    def check_location(cls, v: str | float) -> str:
        if isinstance(v, float):
            return str(v)
        return str(v)


class LocationInfoResult(Marker):
    fsq_info: FourSquarePlace | Dict = {}
    related_events: List[PartialEvent] = []
    address: str = ""

ResultT = TypeVar("ResultT", EventResults, RelevantFields, List[str])

class AsyncioTaskResult(CamelCaseModel, Generic[ResultT]):
    results: Optional[ResultT] = None
    tag: str = ""
    task_type: Literal["search"] | Literal["llm"]
    query: Optional[Any] = None
