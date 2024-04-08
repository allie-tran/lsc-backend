# ====================== #
# PROCESSING
# ====================== #

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, field_validator  # noqa: E0611

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

