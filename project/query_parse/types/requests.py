# ====================== #
# REQUESTS
# ====================== #

from typing import List, Optional

from pydantic import BaseModel  # noqa: E0611


class GeneralQueryRequest(BaseModel):
    session_id: Optional[str] = None
    main: str

    # Optional temporal queries
    before: Optional[str] = None
    before_time: Optional[str] = None

    after: Optional[str] = None
    after_time: Optional[str] = None

    # Optional spatial queries
    gps_bounds: Optional[List[float]] = None

    # Miscs
    size: int = 200
    # share_info: bool = False

class TimelineRequest(BaseModel):
    session_id: Optional[str] = None
    image: str

class GeneralQuestionRequest(GeneralQueryRequest):
    pass
