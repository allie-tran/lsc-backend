# ====================== #
# PROCESSING
# ====================== #

from typing import Optional, Tuple

from pydantic import BaseModel  # noqa: E0611

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
