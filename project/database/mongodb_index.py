import json
import os

from dotenv import load_dotenv
from pymongo import ASCENDING, IndexModel, MongoClient

load_dotenv()

client = MongoClient("localhost", 27017)
db = client["LSC24"]

info_path = f"{os.getenv('FILES_DIRECTORY')}/scene_dict.json"

collection = db["scenes"]
collection.drop()

collection.create_indexes(
    [
        IndexModel([("scene", ASCENDING)], unique=True, name="scene_unique"),
        IndexModel([("start_timestamp", ASCENDING)]),
        IndexModel([("location", ASCENDING)]),
    ]
)
# Index into mongodb

for scene, info in json.load(open(info_path)).items():
    info["scene"] = scene
    collection.insert_one(info)
