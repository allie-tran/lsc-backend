import asyncio
import json
import logging
import queue
from threading import Thread
from uuid import uuid4

import redis
from configs import DEV_MODE
from django.core.exceptions import EmptyResultSet
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from myeachtra.settings import REDIS_HOST, REDIS_PORT
from query_parse.types.requests import GeneralQueryRequest
from rest_framework.decorators import api_view
from results.models import ReturnResults, TripletEvent

from retrieval.search import single_query
from retrieval.timeline import get_timeline

logger = logging.getLogger(__name__)
# set DEBUG level
logger.setLevel(logging.DEBUG)


async def async_query(request: GeneralQueryRequest):
    """
    Streaming response
    """
    try:
        text, size = request.main, request.size

        # First yield in search single is the results
        async for response in single_query(text, size):
            # results is a dict
            if response["type"] in ["raw", "modified"]:
                results = response["results"]
                if not results:
                    raise EmptyResultSet

                # Format into EventTriplets
                triplet_results = [TripletEvent(main=main) for main in results.events]

                # Return the results
                result_repr = ReturnResults(
                    result_list=triplet_results,
                    scroll_id=results.scroll_id,
                ).model_dump_json()

                # yield the results
                res = {"type": response["type"], "results": result_repr}
                yield "data: " + json.dumps(res) + "\n\n"
            else:
                yield "data: " + json.dumps(response) + "\n\n"

        yield "data: END\n\n"

    except asyncio.CancelledError:
        print("Client disconnected")

    except Exception as e:
        print("Error", e)

    # signal the end of request
    yield "data: END\n\n"


def async_to_sync(async_gen):
    """
    Convert async generator to sync generator
    """
    q = queue.Queue()

    def run_async_gen():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def push_to_queue():
            async for item in async_gen:
                q.put(item)
            q.put(None)

        loop.run_until_complete(push_to_queue())
        loop.close()

    Thread(target=run_async_gen, daemon=True).start()

    def sync_generator():
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    return sync_generator()


@csrf_exempt
@api_view(["POST"])
def search(request) -> HttpResponse:
    """
    Search endpoint
    """
    # Coerce the messages into the right format of GeneralQueryRequest
    body = json.loads(request.body.decode("utf-8"))
    session_id = body.get("session_id", "")

    if not session_id and not DEV_MODE:
        return HttpResponse("Please log in".encode(), status=401)

    # Save to redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    token = uuid4().hex
    message = json.dumps(body)
    r.set(token, message)

    print("Got search request!")
    print(body)
    return JsonResponse({"searchToken": token})


@csrf_exempt
def get_stream_search(request):
    """
    Get stream search endpoint
    """
    # session_id = request.GET.get("sessionID", "")
    token = request.GET.get("searchToken", "")

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    message = r.get(token)

    # Delete the message from redis after getting it
    r.delete(token)

    if not message:
        return HttpResponse("Invalid token".encode(), status=404)

    print("Starting search")
    request_body = json.loads(message.decode("utf-8"))  # type: ignore

    request = GeneralQueryRequest(**request_body)
    # Streaming response
    return StreamingHttpResponse(
        async_to_sync(async_query(request)), content_type="text/event-stream"
    )


@csrf_exempt
@api_view(["POST"])
def timeline(request) -> HttpResponse:
    """
    Timeline endpoint
    """
    # Coerce the messages into the right format of GeneralQueryRequest
    body = json.loads(request.body.decode("utf-8"))
    session_id = body.get("session_id", "")

    if not session_id and not DEV_MODE:
        return HttpResponse("Please log in".encode(), status=401)

    image = body["image"]
    result = get_timeline(image)
    if not result:
        return JsonResponse({"error": "No results found"}, status=404)
    return JsonResponse(result.model_dump_json())
