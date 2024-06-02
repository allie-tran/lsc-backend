from pymongo import MongoClient

client = MongoClient("localhost", 27017)
db = client["LSC24"]

scene_collection = db["scenes"]
image_collection = db["images"]
group_collection = db["groups"]
location_collection = db["locations"]

user_collection = db["users"]
request_collection = db["requests"]
es_collection = db["es"]
