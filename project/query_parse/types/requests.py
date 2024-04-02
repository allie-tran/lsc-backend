# ====================== #
# REQUESTS
# ====================== #

from typing import List, Optional

from pydantic import BaseModel  # noqa: E0611


class GeneralQueryRequest(BaseModel):
    main: str

    # Optional temporal queries
    before: Optional[str] = None
    before_time: Optional[str] = None

    after: Optional[str] = None
    after_time: Optional[str] = None

    # Optional spatial queries
    gps_bounds: Optional[List[float]] = None

    # Miscs
    # size: int = 200
    # share_info: bool = False


class GeneralQuestionRequest(GeneralQueryRequest):
    pass
