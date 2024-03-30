from datetime import datetime
from typing import List, Optional, Union

from pydantic import BaseModel, field_validator


class Event(BaseModel):
    # IDs
    group: Optional[str] = ""
    scene: Optional[str] = ""

    # Data
    name: Optional[str] = ""
    images: Optional[List[str]] = None

    start_time: Optional[Union[str, datetime]] = ""
    end_time: Optional[Union[str, datetime]] = ""

    location: Optional[str] = "---"
    location_info: Optional[str] = None

    region: Optional[List[str]] = None
    country: Optional[str] = None

    ocr: List[str] = []
    description: Optional[str] = None

    @field_validator("start_time", "end_time")
    @classmethod
    def check_time(cls, v: Union[str, datetime]) -> str:
        if isinstance(v, str):
            return v
        return v.strftime("%Y-%m-%dT%H:%M:%S")

    @field_validator("ocr")
    @classmethod
    def check_ocr(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return v.split(",")
        return v


class Results(BaseModel):
    pass


class ImageResults(Results):
    images: List[str]
    scores: List[float]


class EventResults(Results):
    events: List[Event]
    scores: List[float]
    scroll_id: str = ""
    min_score: float = 0.0
    max_score: float = 1.0
    normalized: bool = False

    def extend(self, other: "EventResults"):
        if not self.normalized or not other.normalized:
            raise ValueError("Both results must be normalized before joining")
        self.events.extend(other.events)
        self.scores.extend(other.scores)

    def normalize_score(self):
        if self.normalized:
            return
        self.scores = [
            (x - self.min_score) / (self.max_score - self.min_score)
            for x in self.scores
        ]
        self.normalized = True

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

    # =========================================== #
    # These are old variables that I don't remember
    # what they are for anymore

    # info
    # scores
