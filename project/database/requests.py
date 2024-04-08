"""
Database for keeping track of requests and their responses
"""

from typing import Optional

from configs import EXPIRE_TIME
from query_parse.types.requests import AnyRequest

from database.main import request_collection


def create_new_collection():
    """
    Create a new collection for requests
    """
    request_collection.drop()
    request_collection.create_index("session_id")
    request_collection.create_index("timestamp", expireAfterSeconds=EXPIRE_TIME)


def find_request(request: AnyRequest) -> Optional[dict]:
    """
    Find the request in the database
    First, check the same request name
    """
    criteria = request.find_one()
    if criteria:
        # Get the lastest request
        result = request_collection.find(criteria).sort("timestamp", -1).limit(1)
        for res in result:
            print("Found request")
            return res
