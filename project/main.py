import json
import logging
from uuid import uuid4

import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from configs import DEV_MODE
from query_parse.types.requests import GeneralQueryRequest, TimelineRequest
from results.models import TimelineResult
from retrieval.search import async_query
from retrieval.timeline import get_timeline
from configs import REDIS_HOST, REDIS_PORT

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
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    message = r.get(token)

    # Delete the message from redis after getting it
    r.delete(token)

    if not message:
        raise HTTPException(status_code=404, detail="No results found")

    print("Starting search")
    request_body = json.loads(message.decode("utf-8"))  # type: ignore
    print(request_body)
    request = GeneralQueryRequest(**request_body)
    return StreamingResponse(async_query(request.main), media_type="text/event-stream")


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

    result = get_timeline(request.image)
    if not result:
        raise HTTPException(status_code=404, detail="No results found")
    return result


