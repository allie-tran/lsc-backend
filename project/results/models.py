from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator
from query_parse.utils import (
    extend_no_duplicates,
    extend_with_count,
    merge_list,
    merge_str,
)


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
    map_locations: List[List[float]] = []
    map_countries: List[dict] = []

    # Don't know what this is
    location_infos: List[Dict[str, Any]] = []

    def update(self, other: "Visualisation"):
        for key, value in other.dict().items():
            if key in self.dict():
                self.dict()[key].extend(value)

    def to_dict(self) -> dict:
        return self.dict()


class Image(BaseModel):
    # IDs
    group: str = ""
    scene: str = ""

    # Data
    image: str
    time: datetime
    location: str = ""
    location_info: str = ""

    region: List[str] = []
    country: str = ""

    ocr: List[str] = []
    description: str = ""

    @field_validator("time")
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


class Event(BaseModel):
    # IDs
    group: str = ""
    scene: str = ""

    # Data
    name: str = ""
    images: List[str] = []

    start_time: datetime
    end_time: datetime

    location: str = ""
    location_info: str = ""

    region: List[str] = []
    country: str = ""

    ocr: List[str] = []
    description: str = ""

    image_scores: List[float] = []

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
        data = super().dict(*args, **kwargs)
        data["start_time"] = data["start_time"].isoformat()
        data["end_time"] = data["end_time"].isoformat()
        return data

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
        # skip duration

        # Location
        self.location = merge_str(self.location, other.location, " and ")
        self.location_info = merge_str(self.location_info, other.location_info, " and ")

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

        # IDS
        self.group = "+".join(
            extend_no_duplicates([self.group], [x.group for x in others])
        )
        self.scene = "+".join(
            extend_no_duplicates([self.scene], [x.scene for x in others])
        )

        # DATA
        self.name = self.name or others[0].name

        self.image_scores = [score] * len(self.images)
        for i, other in enumerate(others):
            self.images.extend(other.images)
            self.image_scores.extend([scores[i]] * len(other.images))

        # Time
        self.start_time = min([self.start_time] + [x.start_time for x in others])
        self.end_time = max([self.end_time] + [x.end_time for x in others])

        # Location
        self.location = " and ".join(
            extend_no_duplicates([self.location], [x.location for x in others])
        )
        self.location_info = " and ".join(
            extend_no_duplicates(
                [self.location_info], [x.location_info for x in others]
            )
        )

        # Region
        self.region = extend_no_duplicates(
            self.region, [x for y in others for x in y.region]
        )
        self.country = "+".join(
            extend_no_duplicates([self.country], [x.country for x in others])
        )

        # OCR
        self.ocr = extend_with_count(self.ocr, [x.ocr for x in others])


class DerivedEvent(Event):
    """
    An event with derived fields
    """

    model_config = ConfigDict(extra="allow")


class Results(BaseModel):
    pass


class ImageResults(Results):
    images: List[str]
    scores: List[float]


class EventResults(Results):
    events: Union[List[Event], List[DerivedEvent]]
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


class TripletEvent(BaseModel):
    main: Event
    before: Optional[Event] = None
    after: Optional[Event] = None


class ReturnResults(BaseModel):
    """
    Wrapper for the results
    """

    scroll_id: Optional[str] = None
    result_list: List[TripletEvent] = []
    visualisation: Optional[Visualisation] = None
    answers: Optional[List[str]] = None

    # =========================================== #
    # These are old variables that I don't remember
    # what they are for anymore

    # info
    # scores


class TimelineGroup(BaseModel):
    """
    A group of events
    """

    group: str
    scenes: List[List[str]]
    time_info: List[str]
    location: str
    location_info: str


class TimelineResult(BaseModel):
    """
    Wrapper for the results
    """

    name: str
    result: List[TimelineGroup]
