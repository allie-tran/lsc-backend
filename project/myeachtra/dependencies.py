from typing import Annotated
from pydantic import AfterValidator, Field, SerializeAsAny, WrapSerializer
from bson import ObjectId as _ObjectId

def check_object_id(value: str) -> str:
    if not _ObjectId.is_valid(value):
        raise ValueError("Invalid ObjectId")
    return value

ObjectId = Annotated[str,
    Field(..., alias="_id", description="MongoDB ObjectId"),
    AfterValidator(check_object_id),
]

