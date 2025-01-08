"""
Database for keeping track of requests and their responses
"""

from typing import Optional

from bson import ObjectId
from configs import EXPIRE_TIME
from query_parse.types.requests import AnyRequest, Data

from database.main import es_collection, get_db, request_collection


def create_new_collection(data: Data = Data.LSC23):
    """
    Create a new collection for requests
    """
    db = get_db(data)
    request_collection(db).drop()
    request_collection(db).create_index("session_id")
    request_collection(db).create_index("timestamp", expireAfterSeconds=EXPIRE_TIME)


def find_request(request: AnyRequest) -> Optional[dict]:
    """
    Find the request in the database
    First, check the same request name
    """
    create_new_collection(request.data)
    db = get_db(request.data)
    criteria = request.find_one()
    if criteria:
        # Get the lastest request
        result = request_collection(db).find(criteria).sort("timestamp", -1).limit(1)
        for res in result:
            print("Found request", res["_id"])
            return res


def get_request(oid: Optional[str], data: Data = Data.LSC23) -> Optional[dict]:
    """
    Get the request by its oid
    """
    if oid:
        db = get_db(data)
        return request_collection(db).find_one({"_id": ObjectId(oid), "finished": True})


def get_es(oid: Optional[str], data: Data = Data.LSC23) -> Optional[dict]:
    if oid:
        db = get_db(data)
        return es_collection(db).find_one({"_id": ObjectId(oid), "extracted": True})
