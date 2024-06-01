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

    @field_validator("year", "month", "day")
    def year_validator(cls, v):
        if v is None:
            return v
        if isinstance(v, str) and v.isdigit():
            v = int(v)
        return v


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
    value: int


class LocationGap(BaseModel):
    unit: str
    value: int


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

    @model_validator(mode="after")
    def add_to_fields(self) -> Self:
        self.relevant_fields += self.merge_by
        self.relevant_fields += [sort.field for sort in self.sort_by]
        if self.max_gap.time_gap is not None:
            self.relevant_fields.append("start_time")
        if self.max_gap.gps_gap is not None:
            self.relevant_fields.append("center")
        self.relevant_fields = list(set(self.relevant_fields))
        return self
