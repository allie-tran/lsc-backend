import json
import logging
from uuid import uuid4

import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from rich import print

from configs import DEV_MODE, REDIS_HOST, REDIS_PORT
from database.encode_blurhash import batch_encode
from database.models import GeneralRequestModel, LocationInfoResult, Response
from database.utils import get_location_info
from query_parse.types.requests import (
    GeneralQueryRequest,
    MapRequest,
    TimelineDateRequest,
    TimelineRequest,
)
from results.models import TimelineResult
from retrieval.search import search_from_location, streaming_manager
from retrieval.timeline import get_more_scenes, get_timeline, get_timeline_for_date
from submit.router import submit_router

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
load_dotenv(".env")

app = FastAPI()
origins = ["http://localhost", "http://localhost:3001"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(submit_router, prefix="/submit")


@app.post(
    "/search",
    description="Send a search request. Returns a token to be used to stream the results",
)
async def search(request: GeneralQueryRequest):
    if not request.session_id and not DEV_MODE:
        raise HTTPException(status_code=401, detail="Please log in")

    # Save to redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    token = uuid4().hex
    message = request.model_dump_json()
    r.set(token, message)
    print("Got search request!")
    return {"searchToken": token}


@app.get(
    "/get-stream-results/{session_id}/{token}",
    description="Stream the search results",
    status_code=200,
)
async def get_stream_results(session_id: str, token: str):
    if not session_id and not DEV_MODE:
        raise HTTPException(status_code=401, detail="Please log in")

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    message = r.get(token)

    # Delete the message from redis after getting it
    r.delete(token)

    if not message:
        raise HTTPException(status_code=404, detail="No results found")

    print("Starting search")
    request_body = json.loads(message.decode("utf-8"))  # type: ignore
    request = GeneralQueryRequest(**request_body)

    return StreamingResponse(streaming_manager(request), media_type="text/event-stream")


@app.post(
    "/timeline",
    description="Get the timeline of an image",
    response_model=TimelineResult,
    status_code=200,
)
async def timeline(request: TimelineRequest):
    """
    Timeline endpoint
    """
    if not request.session_id and not DEV_MODE:
        raise HTTPException(status_code=401, detail="Please log in")

    cached_request = GeneralRequestModel(request=request)
    if cached_request.finished:
        return cached_request.responses[-1]

    result = get_timeline(request.image)
    cached_request.add(Response(response=result, type="timeline"))
    cached_request.mark_finished()
    return result


@app.get(
    "/timeline_more/{group_id}/{direction}",
    description="Get more scenes for the timeline",
    response_model=TimelineResult,
    status_code=200,
)
async def timeline_more(group_id: str, direction: str):
    """
    Get more scenes for the timeline
    """
    result = get_more_scenes(group_id, direction)
    if not result:
        raise HTTPException(status_code=404, detail="No results found")
    return result


@app.post(
    "/timeline-date",
    description="Get the timeline of a date",
    response_model=TimelineResult,
    status_code=200,
)
async def timeline_date(request: TimelineDateRequest):
    """
    Get the timeline of a date
    """
    if not request.session_id and not DEV_MODE:
        raise HTTPException(status_code=401, detail="Please log in")

    cached_request = GeneralRequestModel(request=request)
    if cached_request.finished:
        return cached_request.responses[-1]

    date = request.date
    result = get_timeline_for_date(date)
    cached_request.add(Response(response=result, type="timeline"))
    return result


@app.get("/health", description="Health check endpoint", status_code=200)
async def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}


@app.get(
    "/create-request-database",
    description="Create the request database",
    status_code=200,
)
async def create_request_database():
    """
    Create the request database
    """
    return {"message": "ok"}


@app.post(
    "/location",
    description="Get location info",
    response_model=LocationInfoResult,
    status_code=200,
)
async def location(request: MapRequest):  # type: ignore
    """
    Get location info
    """
    info = get_location_info(request.location, request.center)
    if not info:
        print(f"Location not found: {request.location}")
        raise HTTPException(status_code=404, detail="No results found")

    related_events = search_from_location(request)
    if related_events:
        info["related_events"] = related_events

    res = LocationInfoResult.model_validate(info)
    print(res)
    return res


@app.get(
    "/encode-blurhash",
    description="Encode images",
    status_code=200,
)
async def encode():
    """
    Encode images using blurhash so that they can be displayed
    in place of the actual image when loading
    """
    batch_encode()
    return {"message": "ok"}
