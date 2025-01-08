import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional

import pandas as pd
from configs import IMAGE_DIRECTORY
from dotenv import load_dotenv
from pymongo import ASCENDING, IndexModel, MongoClient
from query_parse.types.elasticsearch import GPS
from results.models import Icon, Marker

load_dotenv()

# |%%--%%| <ukWc1wcVNW|rp7W9iwksc>

client = MongoClient("localhost", 27017)
db = client["LSC24"]


# |%%--%%| <rp7W9iwksc|T6TKqNGAPU>
r"""°°°
# Scene information
°°°"""
# |%%--%%| <T6TKqNGAPU|kIFopye8AN>
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


def get_aspect_ratio(image_path: str) -> float:
    from PIL import Image, ImageOps

    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        assert img is not None, f"Image {image_path} is None"
        width, height = img.size
        return width / height


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
    aspect_ratios = []
    for image in info["images"]:
        aspect_ratios.append(get_aspect_ratio(f"{IMAGE_DIRECTORY}/{image}"))
    images = [
        {"src": image, "aspect_ratio": aspect_ratio}
        for image, aspect_ratio in zip(info["images"], aspect_ratios)
    ]
    info["images"] = images
    info["time_info"] = create_time_info(info["start_time"], info["end_time"])
    collection.insert_one(info)

# |%%--%%| <kIFopye8AN|59dWoVCJA2>
location_collection = db["locations"]


def calculate_distance(point1: GPS, point2: GPS) -> float:
    """
    Calculate the distance between two points
    """
    return ((point1.lat - point2.lat) ** 2 + (point1.lon - point2.lon) ** 2) ** 0.5


def get_location_info(location: str, center: GPS) -> Optional[dict]:
    """
    Get the location info
    If there is only one location with the same name, return that
    If there are multiple locations with the same name, return the one closest to the center
    """
    locations = location_collection.find({"location": location})
    locations = list(locations)
    if len(locations) == 1:
        return locations[0]["fsq_info"]
    elif len(locations) > 1:
        if not center:
            return locations[0]["fsq_info"]
        threshold = 0.01  # 0.1 degrees is about 11 km
        closest_location = None
        closest_distance = float("inf")
        for loc in locations:
            distance = calculate_distance(center, loc["gps"])
            if distance < threshold:
                return loc["fsq_info"]
            if distance < closest_distance:
                closest_location = loc
                closest_distance = distance
        if closest_location:
            return closest_location["fsq_info"]


# |%%--%%| <59dWoVCJA2|bDjeyDHhkr>

collection = db["scenes"]

count = collection.count_documents({})
from tqdm.auto import tqdm

# Add location icon to the scene information
for scene in tqdm(collection.find(), total=count):
    if "icon" in scene:
        continue
    location = scene["location"]
    loc_info: str = scene["location_info"]
    gps = [GPS(**x) for x in scene["gps"]]

    marker = Marker(location=location, location_info=loc_info, points=gps)
    icon = get_icon(marker)

    info = get_location_info(location, marker.center)  # type: ignore
    if info:
        scene["fsq_info"] = info
    scene["icon"] = icon.model_dump()

    collection.update_one({"scene": scene["scene"]}, {"$set": scene})

# |%%--%%| <bDjeyDHhkr|aoWL1O2dQu>

collection = db["scenes"]
image_collection = db["images"]

image_to_hash = {}
for image in image_collection.find():
    image_to_hash[image["image"]] = image["hash_code"]

# add hash_code from image to scene
size = collection.count_documents({})
for scene in tqdm(collection.find(), total=size):
    images = scene["images"]
    new_images = []
    for image in images:
        hash_code = image_to_hash.get(image["src"], "")
        new_images.append({**image, "hash_code": hash_code})
    collection.update_one(
        {"scene": scene["scene"]}, {"$set": {"images": new_images}}
    )



# |%%--%%| <aoWL1O2dQu|kcFR8Sqc2t>
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

    for scene, _ in info["scenes"]:
        db_scene = db["scenes"].find_one({"scene": scene})

        if db_scene:
            scene_ids.append(db_scene["_id"])
            if start_time is None:
                start_time = db_scene["start_time"]
            end_time = db_scene["end_time"]

    if not scene_ids:
        print(info["scenes"])
        print(f"Group {group} has no scenes")
        continue

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

parent_directory = os.getenv("FILES_DIRECTORY") or "."
parent_directory = os.path.dirname(parent_directory)
info_path = f"{parent_directory}/info_dict.json"

collection = db["images"]
collection.drop()
for image, info in json.load(open(info_path)).items():
    for field in ["time", "utc_time"]:
        info[field] = datetime.strptime(info[field], "%Y/%m/%d %H:%M:%S%z")

    info["image"] = image
    del info["image_path"]
    info["aspect_ratio"] = get_aspect_ratio(f"{IMAGE_DIRECTORY}/{image}")
    collection.insert_one(info)


# |%%--%%| <0i1qVr870L|sQmjGodDOo>
r"""°°°
# Location information
°°°"""
# |%%--%%| <sQmjGodDOo|ry5N2lY1LV>

parent_directory = os.path.dirname(parent_directory)
locations = pd.read_csv(f"{parent_directory}/both_years.csv")

# |%%--%%| <ry5N2lY1LV|gq8jxVDy25>
headers = {
    "accept": "application/json",
    "Authorization": "fsq3FiNdnT37XfaIzPGR6qyZlVXJLnF78FfrpUsB0foOH+I=",
}

from joblib import Memory

import requests

memory = Memory("cachedir")


@memory.cache
def search_fourspace(location, lat, lng) -> str:
    url = (
        f"https://api.foursquare.com/v3/places/search?query={location}&ll={lat}%2C{lng}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error in {location}")
        return ""
    data = response.json()["results"]
    if not data:
        print(f"No data for {location}")
        return ""
    return data[0]["fsq_id"]


# Get the location info
@memory.cache
def get_info(fsq_id: str) -> dict:
    url = f"https://api.foursquare.com/v3/places/{fsq_id}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error in {fsq_id}")
        return {}
    data = response.json()
    return data


# |%%--%%| <gq8jxVDy25|n7wNZrb3wN>

# Group by location name
locations.columns
locations.iloc[0]

location_info: defaultdict[str, dict] = defaultdict(
    lambda: {
        "latitude": [],
        "longitude": [],
        "location": None,
        "location_info": None,
        "images": [],
        "parent": "",
    }
)


from tqdm.auto import tqdm

for row in tqdm(locations.itertuples()):
    row = row._asdict()  # type: ignore
    location: str = row["checkin"]
    if not row["stop"] or location in ["Inside"]:
        continue
    location_info[location]["latitude"].append(row["new_lat"])
    location_info[location]["longitude"].append(row["new_lng"])
    location_info[location]["location"] = location
    location_info[location]["location_info"] = row["location_info"]
    if row["parent"] != "None":
        location_info[location]["parent"] = row["parent"]

location_info["Sandyford View"]

# |%%--%%| <n7wNZrb3wN|RCiY3fmccA>
collection = db["locations"]
collection.drop()

for location, info in tqdm(location_info.items()):
    lats = [float(lat) for lat in location_info[location]["latitude"] if lat]
    lngs = [float(lng) for lng in location_info[location]["longitude"] if lng]
    center = {
        "lat": sum(lats) / len(lats),
        "lon": sum(lngs) / len(lngs),
    }
    info["center"] = center

    # Get the foursquare id
    fsq_id = search_fourspace(location, center["lat"], center["lon"])

    if fsq_id:
        fsq_info = get_info(fsq_id)
        info["fsq_info"] = fsq_info
        info["fsq_id"] = fsq_id
    else:
        info["fsq_info"] = {}
        info["fsq_id"] = None

    if "images" in info:
        del info["images"]

    collection.insert_one(info)

# |%%--%%| <RCiY3fmccA|oQU54K2SA3>
# Link with parents
for location, info in location_info.items():
    parent = info["parent"]
    parent_id = collection.find_one({"location": parent})
    if parent_id:
        parent_id = parent_id["_id"]
    collection.update_one({"location": location}, {"$set": {"parent_id": parent_id}})
# |%%--%%| <oQU54K2SA3|BxykxSXjWC>


def get_icon_from_fsq(info: dict) -> Optional[Icon]:
    if info and info["categories"]:
        return Icon(
            type="foursquare",
            **info["categories"][0]["icon"],
            name=info["categories"][0]["name"],
        )
    return None


def get_icon_from_location_name(location: str, location_info: str) -> Optional[Icon]:
    match (location.lower(), location_info.lower()):
        case ("home", _):
            return Icon(type="material", prefix="home", name="Home")
        case ("work", _):
            return Icon(type="material", prefix="work", name="Work")
        case (x, _) if "dcu" in x:
            return Icon(type="material", prefix="work", name="Work")
        case (_, x) if "restaurant" in x or "cafe" in x:
            return Icon(type="material", prefix="restaurant", name="Restaurant")
        case (_, x) if "bar" in x:
            return Icon(type="material", prefix="local_bar", name="Bar")
        case (_, x) if "hotel" in x:
            return Icon(type="material", prefix="hotel", name="Hotel")
        case (_, x) if "museum" in x:
            return Icon(type="material", prefix="museum", name="Museum")
        case (_, x) if "park" in x:
            return Icon(type="material", prefix="park", name="Park")
        case (_, x) if "school" in x:
            return Icon(type="material", prefix="school", name="School")
        case (_, x) if "hospital" in x:
            return Icon(type="material", prefix="local_hospital", name="Hospital")
        case (_, x) if "store" in x:
            return Icon(type="material", prefix="store", name="Store")
        case (_, x) if "gym" in x:
            return Icon(type="material", prefix="fitness_center", name="Gym")
        case (_, x) if "pharmacy" in x:
            return Icon(type="material", prefix="local_pharmacy", name="Pharmacy")
        case (_, x) if "bank" in x:
            return Icon(type="material", prefix="local_atm", name="Bank")
        case (_, x) if "library" in x:
            return Icon(type="material", prefix="local_library", name="Library")
        case (_, x) if "university" in x:
            return Icon(type="material", prefix="school", name="University")
        case (_, x) if "airport" in x:
            return Icon(type="material", prefix="local_airport", name="Airport")
        case (_, x) if "train" in x:
            return Icon(type="material", prefix="train", name="Train station")
        case (_, x) if "bus" in x:
            return Icon(type="material", prefix="directions_bus", name="Bus station")
        case (_, x) if "subway" in x:
            return Icon(
                type="material", prefix="directions_subway", name="Subway station"
            )
        case (_, x) if "taxi" in x:
            return Icon(type="material", prefix="local_taxi", name="Taxi stand")
        case (_, x) if "car" in x:
            return Icon(type="material", prefix="local_car_wash", name="Car wash")
        case (_, x) if "gas" in x:
            return Icon(type="material", prefix="local_gas_station", name="Gas station")
        case (_, x) if "parking" in x:
            return Icon(type="material", prefix="local_parking", name="Parking")
        case (_, x) if "church" in x:
            return Icon(type="material", prefix="church", name="Church")
        case (_, x) if "mosque" in x:
            return Icon(type="material", prefix="mosque", name="Mosque")
        case (_, x) if "synagogue" in x:
            return Icon(type="material", prefix="synagogue", name="Synagogue")
        case (_, x) if "temple" in x:
            return Icon(type="material", prefix="temple", name="Temple")
        case (_, x) if "cemetery" in x:
            return Icon(type="material", prefix="cemetery", name="Cemetery")
        case (_, x) if "beach" in x:
            return Icon(type="material", prefix="beach_access", name="Beach")
        case (_, x) if "mountain" in x:
            return Icon(type="material", prefix="terrain", name="Mountain")
        case (_, x) if "lake" in x:
            return Icon(type="material", prefix="water", name="Lake")
        case (_, x) if "river" in x:
            return Icon(type="material", prefix="water", name="River")
        case (_, x) if "sea" in x:
            return Icon(type="material", prefix="water", name="Sea")
        case (_, x) if "ocean" in x:
            return Icon(type="material", prefix="water", name="Ocean")
        case (_, x) if "pool" in x:
            return Icon(type="material", prefix="pool", name="Pool")
        case (_, x) if "stadium" in x:
            return Icon(type="material", prefix="stadium", name="Stadium")
        case (_, x) if "theater" in x:
            return Icon(type="material", prefix="theaters", name="Theater")
        case _:
            return Icon(type="material", prefix="location_on", name="Location icon")


def get_icon(marker: Marker) -> Optional[Icon]:
    """
    Get the icon for the marker
    """
    fsq_info = get_location_info(marker.location, marker.center)  # type: ignore
    if fsq_info and fsq_info["fsq_info"]:
        return get_icon_from_fsq(fsq_info["fsq_info"])
    return get_icon_from_location_name(marker.location, marker.location_info)


# |%%--%%| <BxykxSXjWC|3HEqWHy5Oz>
from typing import Any, Dict

location_collection = db["locations"]

for location in tqdm(location_collection.find()):
    if "icon" in location:
        continue
    loc: Dict[str, Any] = location  # type: ignore
    name = loc["location"]
    info = loc["location_info"]
    fsq_info = loc["fsq_info"]

    if not isinstance(info, str):
        info = ""
    if not name:
        name = ""

    icon = get_icon_from_fsq(fsq_info)
    if not icon:
        icon = get_icon_from_location_name(name, info)
    loc["icon"] = icon.model_dump() if icon else None
    location_collection.update_one({"_id": loc["_id"]}, {"$set": loc})

# |%%--%%| <3HEqWHy5Oz|DybSVGw3TH>
