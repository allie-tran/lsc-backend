import json
import logging
from typing import List
from uuid import uuid4

import redis
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from rich import print

from configs import DEV_MODE, REDIS_HOST, REDIS_PORT
from database.encode_blurhash import batch_encode
from database.utils import get_full_data, get_unique_values
from myeachtra.timeline_router import timeline_router
from myeachtra.map_router import map_router
from myeachtra.auth_models import get_user, verify_user
from query_parse.types.requests import AnswerThisRequest, ChoicesRequest, Data, GeneralQueryRequest, ImageInfoRequest, LoginRequest, LoginResponse
from results.models import AnswerResultWithEvent
from retrieval.graph import get_vegalite, to_csv
from retrieval.search import answer_single_event, streaming_manager
from submit.router import submit_router

logging.basicConfig(level=logging.DEBUG)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
load_dotenv(".env")

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
    "https://n-2mbzycnxd-allie-trans-projects.vercel.app",
    "https://mysceal.computing.dcu.ie",
    "vercel.app",
    "mysceal.computing.dcu.ie",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
# app.add_middleware(GZipMiddleware, minimum_size=1000)
app.include_router(submit_router, prefix="/submit")
app.include_router(timeline_router, prefix="/timeline")
app.include_router(map_router, prefix="/location")

@app.post("/login", description="Login endpoint")
async def login(request: LoginRequest):
    """
    Login endpoint
    """
    verified = verify_user(request)
    return LoginResponse(session_id=verified)


@app.post(
    "/search",
    description="Send a search request. Returns a token to be used to stream the results",
    dependencies=[Depends(get_user)],
)
async def search(request: GeneralQueryRequest):
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
        raise HTTPException(status_code=404, detail="Token not found")

    print("Starting search")
    request_body = json.loads(message.decode("utf-8"))  # type: ignore
    request = GeneralQueryRequest(**request_body)

    return StreamingResponse(streaming_manager(request), media_type="text/event-stream")


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


@app.post(
    "/answer-this",
    description="Answer a question on a scene",
    status_code=200,
    response_model=List[AnswerResultWithEvent],
)
async def answer_this(request: AnswerThisRequest):
    """
    Answer a question on a scene
    """
    answers = []
    async for answer in answer_single_event(request):
        answers.append(answer)
    return answers


@app.post(
    "/query_to_csv",
    description="Given a query, return the results in CSV format",
    status_code=200,
    response_model=str,
)
async def query_to_csv(query: GeneralQueryRequest):
    """
    Given a query, return the results in CSV format
    """
    csv = await to_csv(query.main, query.data)
    return csv.to_csv(index=False)


@app.get("/health", description="Health check endpoint", status_code=200)
async def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}


@app.post(
    "/query_to_vegalite",
    description="Given a query, return the results in Vega-Lite format",
    status_code=200,
    response_model=dict,
)
async def query_to_vegalite(query: GeneralQueryRequest):
    """
    Given a query, return the results in Vega-Lite format
    """
    data = await get_vegalite(query.main, query.data)
    return data

@app.post(
    "/image-dicts",
    description="Get all information about the images in the database",
    status_code=200,
)
async def get_image_dicts(request: ImageInfoRequest):
    """
    Get all information about the images in the database
    """
    return get_full_data(request.images, request.data)

@app.post(
    "/choices",
    description="Get the choices for a dropdown",
    status_code=200,
)
async def get_choices(request: ChoicesRequest):
    """
    Get the choices for a dropdown
    """
    match (request.data, request.field):
        case Data.Deakin, "patientId":
            return get_unique_values(request.data, "patient.id")
        case _:
            raise HTTPException(status_code=404, detail=f"{request.field} not found for {request.data}")

