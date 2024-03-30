import json
import logging

from django.core.exceptions import EmptyResultSet
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from query_parse.types import GeneralQueryRequest
from rest_framework.decorators import api_view
from results.models import ReturnResults, TripletEvent

from retrieval.search import process_single, search_single

logger = logging.getLogger(__name__)
# set DEBUG level
logger.setLevel(logging.DEBUG)


@csrf_exempt
@api_view(["POST"])
def search(request) -> JsonResponse:
    """
    Search endpoint
    """
    # Coerce the messages into the right format of GeneralQueryRequest
    request = GeneralQueryRequest(**json.loads(request.body.decode("utf-8")))

    # Search single (starting only)
    query = process_single(request.main)
    results = search_single(query)

    if not results:
        raise EmptyResultSet

    print(f"Results: {len(results)}")
    # Format into EventTriplets
    triplet_results = [TripletEvent(main=main) for main in results.events]

    # Return the results
    result_repr = ReturnResults(
        result_list=triplet_results, scroll_id=results.scroll_id
    )

    return JsonResponse(result_repr.dict())
