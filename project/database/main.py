from pymongo import MongoClient

from query_parse.types.requests import Data

client = MongoClient("localhost", 27017)

LSC_DB = client["LSC_new"]
DEAKIN_DB = client["sherlock"]

def get_db(db_name: Data):
    match db_name:
        case Data.LSC23:
            return LSC_DB
        case Data.Deakin:
            return DEAKIN_DB

scene_collection = lambda db: db["scenes"]
image_collection = lambda db: db["images"]
group_collection = lambda db: db["groups"]

location_collection = lambda db: db["locations"]

user_collection = lambda db: db["users"]
request_collection = lambda db: db["requests"]
es_collection = lambda db: db["es"]
