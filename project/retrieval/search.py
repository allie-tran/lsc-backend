import logging
from typing import Optional

from query_parse.extract_info import Query
from query_parse.types import ESSearchRequest, EventResults

from retrieval.search_utils import send_search_request

logger = logging.getLogger(__name__)

# ============================= #
# Easy Peasy Part: one query only
# ============================= #


def process_single(text_query: str) -> Query:
    """
    Process a single query
    """
    query = Query(text_query)
    return query


def search_single(query: Query) -> Optional[EventResults]:
    """
    Search a single query
    """
    main_query = query.to_elasticsearch(ignore_limit_score=True)
    logger.info("Min score: %s", main_query.min_score)
    search_request = ESSearchRequest(
        query=main_query.to_query(),
        sort_field="start_timestamp",
        min_score=main_query.min_score,
    )
    results = send_search_request(search_request)
    if results is not None:
        logger.info(f"Found {len(results)} events")
        # Overwriting the min_score and max_score
        # based on the test run in to_elasticsearch()
        results.min_score = main_query.min_score
        results.max_score = main_query.max_score
    return results


# ============================= #
# Level 2: Multiple queries
# ============================= #

# pass for now TODO!

# ============================= #
# Level 3: Question Answering
# ============================= #


def process_question_answer(text_query: str) -> Query:
    """
    Process a question answering query
    """
    query = Query(text_query)
    return query
