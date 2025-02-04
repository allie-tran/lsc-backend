from pymongo import MongoClient

from query_parse.types.requests import Data

client = MongoClient("localhost", 27017)

LSC_DB = client["LSC24_new"]
DEAKIN_DB = client["sherlock"]

def get_db(db_name: Data):
    match db_name:
        case Data.LSC23:
            return LSC_DB
        case Data.Deakin:
            return DEAKIN_DB

def create_text_index(db, collection_name):
    # For captions field
    db[collection_name].create_index([("captions", "text")])

try:
    create_text_index(LSC_DB, "images")
    create_text_index(DEAKIN_DB, "images")
except Exception as e:
    print(e)

scene_collection = lambda db: db["scenes"]
image_collection = lambda db: db["images"]
group_collection = lambda db: db["groups"]

location_collection = lambda db: db["locations"]

user_collection = lambda db: db["users"]
request_collection = lambda db: db["requests"]
es_collection = lambda db: db["es"]

user_collection = client["MyEachtra"].users
