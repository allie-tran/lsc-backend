# ====================== #
# PROCESSING
# ====================== #

from typing import Optional, Tuple

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

