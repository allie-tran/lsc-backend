from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from pymongo import ASCENDING, IndexModel, MongoClient
from query_parse.types.elasticsearch import GPS
from query_parse.types.requests import MapRequest
from results.models import Icon


from database.models import FourSquarePlace
from database.utils import get_location_info

load_dotenv()

# |%%--%%| <prxsFfbytD|GXUqk7DgNt>
client = MongoClient("localhost", 27017)
db = client["LSC24_new"]

scene_collection = db["scenes"]
image_collection = db["images"]
group_collection = db["groups"]
location_collection = db["locations"]
#|%%--%%| <GXUqk7DgNt|McAUJo2wHU>

# Create indexes
image_collection.create_indexes(
    [
        IndexModel([("image", ASCENDING)], unique=True, name="image_unique"),
        IndexModel([("scene", ASCENDING)]),
        IndexModel([("group", ASCENDING)]),
        IndexModel([("time", ASCENDING)]),
        IndexModel([("location", ASCENDING)]),
    ]
)

scene_collection.create_indexes(
    [
        IndexModel([("scene", ASCENDING)], unique=True, name="scene_unique"),
        IndexModel([("group", ASCENDING)]),
        IndexModel([("start_timestamp", ASCENDING)]),
        IndexModel([("location", ASCENDING)]),
    ]
)

group_collection.create_indexes(
    [
        IndexModel([("group", ASCENDING)], unique=True, name="group_unique"),
        IndexModel([("start_timestamp", ASCENDING)]),
        IndexModel([("location", ASCENDING)]),
    ]
)

location_collection.create_indexes(
    [
        IndexModel([("location", ASCENDING)], name="location"),
        IndexModel([("gps", ASCENDING)]),
    ]
)
#|%%--%%| <McAUJo2wHU|XCNQKQ3Fxe>
# Copy data from old database to new database
old_db = client["LSC24"]
old_scene_collection = old_db["scenes"]
old_image_collection = old_db["images"]
old_group_collection = old_db["groups"]

#|%%--%%| <XCNQKQ3Fxe|KrXJH7dxFm>
# Copy scene, group data from old database to new database
image_collection.delete_many({})
scene_collection.delete_many({})
group_collection.delete_many({})
count = old_scene_collection.count_documents({})
from tqdm.auto import tqdm
for scene in tqdm(old_scene_collection.find(), total=count):
    scene_collection.insert_one(scene)

count = old_group_collection.count_documents({})
for group in tqdm(old_group_collection.find(), total=count):
    group_collection.insert_one(group)

count = old_image_collection.count_documents({})
for image in tqdm(old_image_collection.find(), total=count):
    image_collection.insert_one(image)
# |%%--%%| <KrXJH7dxFm|pDwTO7Z7Pm>
class LocationDocument(BaseModel):
    class Config:
        coerce_numbers_to_str = True
        str_strip_whitespace = True

    location: str
    location_info: str = ""

    all_gps: List[GPS] = []
    gps: GPS
    fsq_id: Optional[str] = None
    fsq_info: Optional[FourSquarePlace] = None
    icon: Optional[Icon] = None

    images: List[str] = []
    scenes: List[str] = []
    groups: List[str] = []


# Copy scenes
# location_collection.delete_many({})
# scene_collection.delete_many({})
count = old_scene_collection.count_documents({})
from tqdm.auto import tqdm
for scene in tqdm(old_scene_collection.find(), total=count):
    if scene_collection.find_one({"scene": scene["scene"]}):
        continue
    scene_collection.insert_one(scene)
    images = [img["src"] for img in scene["images"]]
    if scene["location"] and scene["location"] != "---":
        gps = scene["gps"]
        if gps:
            all_lats = [p["lat"] for p in gps]
            all_lons = [p["lon"] for p in gps]
            if len(all_lats) > 0 and len(all_lons) > 0:
                center = GPS(
                    lat=sum(all_lats) / len(all_lats),
                    lon=sum(all_lons) / len(all_lons),
                )
            else:
                continue
        else:
            continue

        # create location if not exists
        location = get_location_info(
            MapRequest(
                location=scene["location"],
                center=center,
            )
        )
        assert location, f"Location not found for {scene['location']}"
        _, data = location

        doc = location_collection.find_one({"fsq_id": data["fsq_id"]})
        if doc:
            all_gps = doc["all_gps"]
            all_gps.extend(scene["gps"])
            try:
                all_lats = [p["lat"] for p in all_gps]
                all_lons = [p["lon"] for p in all_gps]
            except TypeError:
                print(all_gps)
                raise
            if len(all_lats) > 0 and len(all_lons) > 0:
                new_gps = GPS(
                    lat=sum(all_lats) / len(all_lats),
                    lon=sum(all_lons) / len(all_lons),
                )
                # insert images to location
                location_collection.update_one(
                    {"location": data["location"]},
                    {
                        "$push": {"images": {"$each": images}},
                        "$push": {"all_gps": {"$each": scene["gps"]}},
                        "$addToSet": {"scenes": scene["scene"]},
                        "$addToSet": {"groups": scene["group"]},
                        "$set": {"gps": new_gps.model_dump()},
                    },
                )
        else:
            data["gps"] = center.model_dump()
            try:
                doc = LocationDocument(
                    **data,
                    images=images,
                    scenes=[scene["scene"]],
                    groups=[scene["group"]],
                    all_gps=scene["gps"],
                )
                location_collection.insert_one(doc.model_dump())
            except ValidationError as e:
                print("Error", scene["scene"], scene["location"])
                continue

        # Add location fsq_id, icon,
        scene_collection.update_one(
            {"scene": scene["scene"]},
            {"$set": {"fsq_id": data["fsq_id"], "icon": data["icon"]}},
        )

        image_collection.update_many(
            {"scene": scene["scene"]},
            {"$set": {"fsq_id": data["fsq_id"], "icon": data["icon"]}},
        )

        group_collection.update_many(
            {"group": scene["group"]},
            {"$set": {"fsq_id": data["fsq_id"], "icon": data["icon"]}},
        )

#|%%--%%| <pDwTO7Z7Pm|bl58VQ5yGi>
# Error 2015-02-28/b00000378_21i6bq_20150228_092736e.jpg Starbucks
# Error 2015-02-28/b00000377_21i6bq_20150228_092630e.jpg
# Error 2015-03-07/b00000748_21i6bq_20150307_140206e.jpg McDonald's
#|%%--%%| <bl58VQ5yGi|vtdqQ7lWdX>
