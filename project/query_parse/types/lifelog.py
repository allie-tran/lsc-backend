# ====================== #
# PROCESSING
# ====================== #

from enum import Enum
from typing import List, Literal, Optional, Self, Tuple
from query_parse.types.eating import EatingActivity, EatingLocationPosition, EatingLocationPositionExplanation, EatingLocationSetting, EatingLocationSettingExplanation, FoodGroup, Mood, MoodExplanation, SocialContact, SocialContactExplanation

from myeachtra.dependencies import CamelCaseModel
from pydantic import field_validator, model_validator

" POS tagging, NER, and other NLP related types "
Tags = Tuple[str, str]


class Mode(str, Enum):
    event = "event"
    image = "image"


class RegexInterval(CamelCaseModel):
    start: int
    end: int
    text: str
    tag: Optional[str] = ""


class DateTuple(CamelCaseModel):
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def validate_date(cls, v):
        if isinstance(v, tuple or list) and len(v) == 3:
            return {"year": v[0], "month": v[1], "day": v[2]}
        return v

    @field_validator("year", "month", "day")
    def year_validator(cls, v) -> Optional[int]:
        if v is None or isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            v = int(v)
        raise ValueError(f"Invalid date value: {v}")

    def export(self):
        """
        Export the date tuple to a string
        Format: DD/MM/YYYY, DD/MM, YYYY
        """
        date = "/".join(
            [str(v) for v in [self.day, self.month, self.year] if v is not None]
        )
        return date


class TimeCondition(CamelCaseModel):
    condition: Literal["before", "after"]
    time_limit_str: str = "1h"
    time_limit_float: float = 1.0

    def switch(self):
        if self.condition == "before":
            self.condition = "after"
        else:
            self.condition = "before"

    def __iseq__(self, other) -> bool:
        return self.condition == other.condition


class TimeGap(CamelCaseModel):
    unit: str
    value: float


class LocationGap(CamelCaseModel):
    unit: str
    value: float

    @field_validator("value")
    def validate_value(cls, v):
        if isinstance(v, str) and v.isdigit():
            return float(v)
        return v


class MaxGap(CamelCaseModel):
    time_gap: Optional[TimeGap] = None
    gps_gap: Optional[LocationGap] = None

    @field_validator("time_gap")
    def validate_time_gap(cls, v):
        if v is not None:
            if v.unit not in ["hour", "minute", "day", "week", "month", "year"]:
                return None
        return v

    @field_validator("gps_gap")
    def validate_gps_gap(cls, v):
        if v is not None:
            if v.unit not in ["meter", "km"]:
                return None
        return v


class SortBy(CamelCaseModel):
    field: str
    order: Literal["asc", "desc"]

class AggregatedAnswer(CamelCaseModel):
    field: Optional[str] = None
    # operation: Literal["count", "sum", "average", "argmax", "argmin", "max", "min"]
    operation: Optional[str] = None


class RelevantFields(CamelCaseModel):
    relevant_fields: List[str] = []
    merge_by: List[str] = []
    sort_by: List[SortBy] = []
    max_gap: Optional[MaxGap] = None
    aggregated_answer: Optional[AggregatedAnswer] = None

    @field_validator("merge_by", mode="before")
    def validate_merge_by(cls, v) -> List[str]:
        values = []
        for value in v:
            if isinstance(value, str):
                values.append(value)
        values = set(values)
        return list(values)

    @field_validator("relevant_fields", "merge_by")
    @classmethod
    def change_place_to_location(cls, v) -> List[str]:
        if "place" in v:
            v.remove("place")
            v.append("location")
        if "place_info" in v:
            v.remove("place_info")
            v.append("location_info")
        return v

    @model_validator(mode="after")
    def change_fields(self) -> Self:
        self.relevant_fields += self.merge_by
        self.relevant_fields += [sort.field for sort in self.sort_by]
        if self.aggregated_answer and self.aggregated_answer.field:
            self.relevant_fields.append(self.aggregated_answer.field)

        if self.max_gap is None:
            self.max_gap = MaxGap()
        if self.max_gap.time_gap is not None:
            self.relevant_fields.append("start_time")
        if self.max_gap.gps_gap is not None:
            self.relevant_fields.append("center")

        self.relevant_fields = list(set(self.relevant_fields))
        return self


class EatingFilters(CamelCaseModel):
    patient_id: List[str] = []
    date: List[str] = []
    mood: List[Mood] = []
    location_setting: List[EatingLocationSetting] = []
    location_position: List[EatingLocationPosition] = []
    social_contact: List[SocialContact] = []
    eating_activity: List[EatingActivity] = []
    food: List[FoodGroup] = []

    def __bool__(self) -> bool:
        return any(
            [
                self.date,
                self.patient_id,
                self.mood,
                self.location_setting,
                self.location_position,
                self.social_contact,
                self.eating_activity,
                self.food,
            ]
        )

    def iter_fields(self):
        return {
            "date": self.date,
            "patient_id": self.patient_id,
            "mood": self.mood,
            "location_setting": self.location_setting,
            "location_position": self.location_position,
            "social_contact": self.social_contact,
            "eating_activity": self.eating_activity,
            "food": self.food,
        }.items()

    def format(self) -> str:
        explanation = []
        if self.mood:
            strs = [MoodExplanation[m] for m in self.mood]
            explanation.append(f"Mood: {', or '.join(strs)}")
        if self.location_setting:
            strs = [EatingLocationSettingExplanation[l] for l in self.location_setting]
            explanation.append(f"Location setting: {', or '.join(strs)}")
        if self.location_position:
            strs = [EatingLocationPositionExplanation[l] for l in self.location_position]
            explanation.append(f"Location position: {', or'.join(strs)}")
        if self.social_contact:
            strs = [SocialContactExplanation[l] for l in self.social_contact]
            explanation.append(f"Social contact: {', or'.join(strs)}")
        if self.eating_activity:
            explanation.append(f"Eating activity: {', or'.join(self.eating_activity)}")
        if self.food:
            explanation.append(f"Food: {', or'.join(self.food)}")
        return "\n".join(explanation)

class SingleQuery(CamelCaseModel):
    visual: str = ""
    location: str = ""
    time: str = ""
    date: str = ""
    filters: EatingFilters = EatingFilters()

    def __bool__(self) -> bool:
        return any([self.visual, self.location, self.time, self.date])

class ParsedQuery(CamelCaseModel):
    main: SingleQuery
    after: SingleQuery | None = None
    before: SingleQuery | None = None
    must_not: SingleQuery | None = None
