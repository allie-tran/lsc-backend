from typing import List, Literal
from pydantic import BaseModel, Field


class HitsTotal(BaseModel):
    value: int = Field(..., description="Total number of matching documents")
    relation: Literal["eq", "gte"] = Field(..., description="Relation between value and total number of documents")

class HitObject(BaseModel):
    index: str = Field(..., description="Index name")
    id: str = Field(...,  description="Document ID")
    score: float = Field(..., description="Document score")
    source: dict = Field(..., description="Document source")

    class Config:
        alias_generator = lambda x: f"_{x}"
        populate_by_name = True

class Hits(BaseModel):
    total: HitsTotal
    max_score: float | None = None
    hits: List[HitObject]


class ESResponse(BaseModel):
    scroll_id: str = Field(default="", alias="_scroll_id")
    hits: Hits
    aggregations: dict | None = None
