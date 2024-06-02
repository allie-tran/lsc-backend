# ====================== #
# PROCESSING
# ====================== #

from typing import List, Literal, Optional, Self, Tuple

from pydantic import BaseModel, field_validator, model_validator  # noqa: E0611

" POS tagging, NER, and other NLP related types "
Tags = Tuple[str, str]


class RegexInterval(BaseModel):
    start: int
    end: int
    text: str
    tag: Optional[str] = ""


class DateTuple(BaseModel):
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
        date = "/".join([str(v) for v in [self.day, self.month, self.year] if v is not None])
        return date

class TimeCondition(BaseModel):
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


class TimeGap(BaseModel):
    unit: str
    value: float


class LocationGap(BaseModel):
    unit: str
    value: float


class MaxGap(BaseModel):
    time_gap: Optional[TimeGap] = None
    gps_gap: Optional[LocationGap] = None

    @field_validator("time_gap")
    def validate_time_gap(cls, v):
        if v is not None:
            if v.unit not in ["none", "hour", "minute", "day", "week", "month", "year"]:
                return None
        return v

    @field_validator("gps_gap")
    def validate_gps_gap(cls, v):
        if v is not None:
            if v.unit not in ["none", "meter", "km"]:
                return None
        return v


class SortBy(BaseModel):
    field: str
    order: Literal["asc", "desc"]


class RelevantFields(BaseModel):
    relevant_fields: List[str] = []
    merge_by: List[str] = []
    sort_by: List[SortBy] = []
    max_gap: MaxGap = MaxGap()

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
        if self.max_gap.time_gap is not None:
            self.relevant_fields.append("start_time")
        if self.max_gap.gps_gap is not None:
            self.relevant_fields.append("center")
        self.relevant_fields = list(set(self.relevant_fields))
        return self
