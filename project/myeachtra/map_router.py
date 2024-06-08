from fastapi import APIRouter, HTTPException

from fastapi import HTTPException
from rich import print

from database.utils import get_location_info
from query_parse.types.requests import (
    MapRequest,
)
from results.models import LocationInfoResult
from retrieval.search import search_from_location


map_router = APIRouter()

@map_router.post(
    "",
    description="Get location info",
    response_model=LocationInfoResult,
    status_code=200,
)
async def location(request: MapRequest):  # type: ignore
    """
    Get location info
    """
    info = get_location_info(request)
    if not info:
        print(f"Location not found: {request.location}")
        raise HTTPException(status_code=404, detail="No results found")

    location, info = info
    request.location = location
    if not request.center:
        request.center = info["gps"]
    if "center" not in info:
        info["center"] = request.center

    print("Requesting related events", request)

    related_events = await search_from_location(request)
    if related_events:
        info["related_events"] = related_events

    res = LocationInfoResult.model_validate(info, from_attributes=True)
    return res
