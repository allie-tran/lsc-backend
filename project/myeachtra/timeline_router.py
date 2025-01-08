from database.models import GeneralRequestModel, Response
from fastapi import APIRouter, Depends, HTTPException
from myeachtra.auth_models import get_user
from query_parse.types.requests import TimelineDateRequest, TimelineRequest
from results.models import TimelineResult
from retrieval.search import search_from_time
from retrieval.timeline import get_timeline, get_timeline_for_date

timeline_router = APIRouter()


@timeline_router.post(
    "",
    description="Get the timeline of an image",
    response_model=TimelineResult,
    status_code=200,
    dependencies=[Depends(get_user)],
)
async def timeline(request: TimelineRequest):
    """
    Timeline endpoint
    """
    # cached_request = GeneralRequestModel(request=request)
    # if cached_request.finished:
    #     return cached_request.responses[-1].response

    print("Getting timeline")
    result = get_timeline(request.image, request.data)
    if not result:
        raise HTTPException(status_code=404, detail="No results found")
    # cached_request.add(Response(response=result, type="timeline"))
    # cached_request.mark_finished()
    return result


@timeline_router.post(
    "/date",
    description="Get the timeline of a date",
    response_model=TimelineResult,
    status_code=200,
    dependencies=[Depends(get_user)],
)
async def timeline_date(request: TimelineDateRequest):
    """
    Get the timeline of a date
    """
    cached_request = GeneralRequestModel(request=request)
    if cached_request.finished:
        return cached_request.responses[-1]

    date = request.date
    result = get_timeline_for_date(date, request.data)
    # result = await search_from_time(request)

    cached_request.add(Response(response=result, type="timeline"))
    return result


@timeline_router.post(
    "/relevant-only",
    description="Get the timeline of a date",
    response_model=TimelineResult,
    status_code=200,
    dependencies=[Depends(get_user)],
)
async def timeline_relevant_only(request: TimelineDateRequest):
    """
    Get the timeline of a date
    """
    cached_request = GeneralRequestModel(request=request)
    if cached_request.finished:
        return cached_request.responses[-1]

    result = await search_from_time(request)

    cached_request.add(Response(response=result, type="timeline"))
    return result
