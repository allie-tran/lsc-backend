import json
import os
from collections import defaultdict
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from pymongo import ASCENDING, IndexModel, MongoClient

load_dotenv()

# |%%--%%| <ukWc1wcVNW|rp7W9iwksc>

client = MongoClient("localhost", 27017)
db = client["LSC24"]


# |%%--%%| <rp7W9iwksc|T6TKqNGAPU>
r"""°°°
# Scene information
°°°"""
# |%%--%%| <T6TKqNGAPU|wUswCs5kTP>
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


def create_time_info(start_time: datetime, end_time: datetime) -> str:
    start = start_time.strftime("%I:%M %p")
    end = end_time.strftime("%I:%M %p")
    if start_time.date() == end_time.date():
        return f"{start}"
    return f"{start} - {end}"


# Index into mongodb
for scene, info in json.load(open(info_path)).items():

    # Change string to datetime
    for field in ["start_time", "end_time"]:
        info[field] = datetime.strptime(info[field], "%Y/%m/%d %H:%M:%S%z")

    info["scene"] = scene
    info["time_info"] = create_time_info(info["start_time"], info["end_time"])
    collection.insert_one(info)

# |%%--%%| <wUswCs5kTP|kcFR8Sqc2t>
r"""°°°
# Group information
°°°"""
# |%%--%%| <kcFR8Sqc2t|lpYPDwcsGv>
info_path = f"{os.getenv('FILES_DIRECTORY')}/group_segments.json"

collection = db["groups"]
collection.drop()

collection.create_indexes(
    [
        IndexModel([("group", ASCENDING)], unique=True, name="group_unique"),
        IndexModel([("location", ASCENDING)]),
    ]
)

# |%%--%%| <lpYPDwcsGv|ouVTlPYMA3>

# Index into mongodb
for group, info in json.load(open(info_path)).items():
    scene_ids = []
    images = []
    start_time = None
    end_time = None
    if "scenes" not in info:
        print(f"Group {group} has no scenes")
        print(info)
        continue
    for scene in info["scenes"]:
        db_scene = db["scenes"].find_one({"scene": scene})
        if db_scene:
            scene_ids.append(db_scene["_id"])
            if start_time is None:
                start_time = db_scene["start_time"]
            end_time = db_scene["end_time"]

    time_info = ""
    if start_time and end_time:
        time_info = create_time_info(start_time, end_time)

    collection.insert_one(
        {
            "group": group,
            "location": info["location"],
            "location_info": info["location_info"],
            "scenes": scene_ids,
            "start_time": start_time,
            "end_time": end_time,
            "time_info": time_info,
        }
    )

# |%%--%%| <ouVTlPYMA3|3Ko9jHfhW4>
r"""°°°
# Image information
°°°"""
# |%%--%%| <3Ko9jHfhW4|0i1qVr870L>

parent_directory = os.getenv("FILES_DIRECTORY")
parent_directory = os.path.dirname(parent_directory)
info_path = f"{parent_directory}/info_dict.json"

collection = db["images"]
collection.drop()
for image, info in json.load(open(info_path)).items():
    for field in ["time", "utc_time"]:
        info[field] = datetime.strptime(info[field], "%Y/%m/%d %H:%M:%S%z")

    info["image"] = image
    del info["image_path"]

    collection.insert_one(info)


# |%%--%%| <0i1qVr870L|sQmjGodDOo>
r"""°°°
# Location information
°°°"""
# |%%--%%| <sQmjGodDOo|UxDpkE4DQV>

parent_directory = os.path.dirname(parent_directory)
locations = pd.read_csv(f"{parent_directory}/both_years.csv")

# |%%--%%| <UxDpkE4DQV|ZxifuHnyhx>

# Group by location name
locations.columns
locations.iloc[0]

location_info = defaultdict(
    lambda: {
        "latitude": [],
        "longitude": [],
        "location": None,
        "location_info": None,
        "images": [],
        "parent": "",
    }
)

print(locations["checkin"].unique())

for _, row in locations.iterrows():  # type: ignore
    location = row["checkin"]
    location_info[location]["latitude"].append(row["new_lat"])
    location_info[location]["longitude"].append(row["new_lng"])
    location_info[location]["location"] = location
    location_info[location]["location_info"] = row["location_info"]
    if row["parent"] != "None":
        location_info[location]["parent"] = row["parent"]

location_info["Sandyford View"]

# |%%--%%| <ZxifuHnyhx|Wl4lOWpa8o>


collection = db["locations"]
collection.drop()
# Insert into mongodb
for location, info in location_info.items():
    if "images" in info:
        del info["images"]
    info["latitude"] = sum(info["latitude"]) / len(info["latitude"])
    info["longitude"] = sum(info["longitude"]) / len(info["longitude"])
    collection.insert_one(info)

# Link with parents
for location, info in location_info.items():
    parent = info["parent"]
    parent_id = collection.find_one({"location": parent})
    if parent_id:
        parent_id = parent_id["_id"]
    collection.update_one({"location": location}, {"$set": {"parent_id": parent_id}})
