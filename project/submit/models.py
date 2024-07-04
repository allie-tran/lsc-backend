from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic.alias_generators import to_camel


class CamelCaseModel(BaseModel):
    class Config:
        alias_generator = to_camel
        populate_by_name = True


class AnswerItem(CamelCaseModel):
    text: str = ""
    media_item_name: str = ""
    media_item_collection_name: Optional[str] = None
    start: int = 0
    end: int = 0

    @model_validator(mode="after")
    def check_consistency(self):
        # only ONE of the following ways are allowed
        # 1. Only text
        # 2. Only media item name, optional collection name
        # 3. Media item name, start, end, optional collection name (for video)
        if not self.text and not self.media_item_name:
            raise ValidationError("Either text or media item name is required")
        if self.text and self.media_item_name:
            raise ValidationError("Only one of text or media item name is allowed")
        if self.media_item_name and (self.start or self.end):
            if not self.start or not self.end:
                raise ValidationError("Both start and end are required for media item")
        return self

    @field_validator("media_item_name")
    @classmethod
    def check_media_item_name(cls, v: str) -> str:
        # Remove the extension and parent directory (if any)
        return v.split("/")[-1].split(".")[0]

    @field_validator("start", "end")
    @classmethod
    def check_start_end(cls, v: int) -> int:
        return max(0, v)

    @field_validator("text")
    @classmethod
    def check_text(cls, v: str) -> str:
        return v.strip()


class AnswerSet(CamelCaseModel):
    task_id: str = ""
    task_name: str = ""
    answers: List[AnswerItem] = []


class DRESSubmitRequest(CamelCaseModel):
    answer_sets: List[AnswerSet] = []

    @field_validator("answer_sets")
    @classmethod
    def check_answer_sets(cls, v: list[AnswerSet]) -> list[AnswerSet]:
        non_empty = [x for x in v if x.answers]
        if len(non_empty):
            return non_empty
        raise ValidationError("At least one answer set is required")


class DRESSubmitResponse(CamelCaseModel):
    status: bool
    submission: str
    description: str


class SubmitAnswerRequest(CamelCaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    evaluation_id: str
    session_id: str = Field(..., serialization_alias="session")
    query_type: str 
    answer: str


class SubmitAnswerResponse(CamelCaseModel):
    severity: Literal["info", "warning", "error", "success"]
    message: str
    verdict: Literal["CORRECT", "INCORRECT", "INVALID", "ERROR", "INDETERMINATE"]
