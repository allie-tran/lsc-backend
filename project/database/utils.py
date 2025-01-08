import os
from typing import List, Literal, Optional, Tuple

from configs import ESSENTIAL_FIELDS, IMAGE_ESSENTIAL_FIELDS
from geopy.geocoders import Nominatim
from llm import gpt_llm_model
from llm.prompt.organize import RELEVANT_FIELDS_PROMPT
from myeachtra.dependencies import memory
from pydantic import ValidationError
from query_parse.types.elasticsearch import GPS
from query_parse.types.requests import MapRequest
from query_parse.utils import extend_no_duplicates
from results.models import AsyncioTaskResult, Event, Icon, Image, Marker
from results.utils import RelevantFields
from retrieval.async_utils import async_timer, timer
from rich import print as rprint

import requests
from database.main import image_collection, location_collection, scene_collection


def to_event(image: dict) -> Event:
    if "start_time" in image:
        return Event(**image)
    try:
        image["start_time"] = image.pop("time")
        image["end_time"] = image["start_time"]
        image["images"] = [
            Image(
                src=image.pop("image"),
                aspect_ratio=image.pop("aspect_ratio"),
                hash_code=image.pop("hash_code"),
            )
        ]
        image["gps"] = [GPS(**image.pop("gps"))]
        if "icon" in image:
            image["icon"] = Icon(**image.pop("icon"))
        return Event(**image)
    except KeyError as e:
        print(image)
        raise e



@timer("convert_to_events")
def convert_to_events(
    key_list: List[str],
    relevant_fields: Optional[List[str]] = None,
    key: Literal["scene", "image"] = "scene",
) -> List[Event]:
    """
    Convert a list of event ids to a list of Event objects
    """
    collection = scene_collection if key == "scene" else image_collection
    fields = ESSENTIAL_FIELDS if key == "scene" else IMAGE_ESSENTIAL_FIELDS

    documents = []
    index = {key: i for i, key in enumerate(key_list)}
    if relevant_fields:
        projection = extend_no_duplicates(relevant_fields, fields)
        try:
            documents = collection.find({key: {"$in": key_list}}, projection=projection)
        except Exception as e:
            rprint("[red]Error in convert_to_events[/red]", e)

    if not documents:
        documents = collection.find({key: {"$in": key_list}})

    # Sort the documents based on the order of the event_list
    documents = sorted(documents, key=lambda doc: index[doc[key]])

    events = []
    if key == "scene":
        events = [Event(**doc) for doc in documents]
    else:
        events = [to_event(doc) for doc in documents]

    for event in events:
        event.markers, event.orphans = calculate_markers(event)
    return events


def segments_to_events(
    segments: List[Tuple[int, int]],
    scores,
    photo_ids: List[str],
    relevant_fields: Optional[List[str]] = None,
) -> List[Event]:
    """
    Convert the segments to events
    """
    images = []
    for start, end in segments:
        images.extend(photo_ids[start:end])
    documents = []
    if relevant_fields:
        projection = extend_no_duplicates(relevant_fields, ESSENTIAL_FIELDS)
        try:
            documents = image_collection.find(
                {"image": {"$in": images}}, projection=projection
            )
        except Exception as e:
            rprint("[red]Error in convert_to_events[/red]", e)

    if not documents:
        documents = image_collection.find({"image": {"$in": images}})

    image_to_doc = {doc["image"]: doc for doc in documents}
    events = []
    for (start, end), score in zip(segments, scores):
        event_images = [photo for photo in photo_ids[start:end] if photo in image_to_doc]
        if not event_images:
            continue
        event = to_event(image_to_doc[event_images[0]])
        if len(event_images) > 1:
            rest = [
                to_event(image_to_doc[photo_id])
                for photo_id in event_images[1:]
            ]
            event.merge_with_many(score, rest, [score] * len(rest))
        events.append(event)

    for event in events:
        event.markers, event.orphans = calculate_markers(event)

    return events


def calculate_markers(event: Event) -> Tuple[List[Marker], List[GPS]]:
    """
    Calculate the markers for one event, straight from the mongo document
    """
    if event.location:
        markers = [
            Marker(
                location=event.location,
                points=event.gps,
                location_info=event.location_info,
                icon=event.icon,
            )
        ]
        return markers, []
    return [], event.gps


@async_timer("get_relevant_fields")
async def get_relevant_fields(query: str, tag: str) -> AsyncioTaskResult:
    """
    Get the relevant fields from the query
    """
    prompt = RELEVANT_FIELDS_PROMPT.format(query=query)
    data = {}
    while True:
        try:
            data = await gpt_llm_model.generate_from_text(prompt)
            if data:
                rprint("Relevant Fields", data)
                break
        except ValidationError as e:
            rprint(e)

    relevant_fields = RelevantFields.model_validate(data)
    return AsyncioTaskResult(results=relevant_fields, tag=tag, task_type="llm")


headers = {
    "accept": "application/json",
    "Authorization": os.getenv("FOURSQUARE_API_KEY"),
}


@memory.cache
def search_fourspace(location: str, lat: float, lng: float) -> str:
    lat = round(lat, 6)
    lng = round(lng, 6)
    url = (
        f"https://api.foursquare.com/v3/places/search?query={location}&ll={lat}%2C{lng}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        rprint(f"Error in {location}")
        return ""
    data = response.json()["results"]
    if not data:
        rprint(f"No data for {location}")
        return ""
    return data[0]["fsq_id"]


# Get the location info
@memory.cache
def get_info(fsq_id: str) -> dict:
    url = f"https://api.foursquare.com/v3/places/{fsq_id}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        rprint(f"Error in {fsq_id}")
        return {}
    data = response.json()
    return data


def calculate_distance(point1: GPS, point2: GPS) -> float:
    """
    Calculate the distance between two points
    """
    return ((point1.lat - point2.lat) ** 2 + (point1.lon - point2.lon) ** 2) ** 0.5


def get_location_name(request: MapRequest) -> Tuple[str, Optional[GPS]]:
    location = request.location
    if location == "---":
        location = None

    if location and request.center:
        return location, request.center

    if request.image:
        doc = image_collection.find_one({"image.src": request.image})
        if doc:
            return doc["location"], GPS(**doc["gps"])

    scene = scene_collection.find_one(
        {
            "$or": [
                {"images": {"$elemMatch": {"src": request.image}}},
                {"scene": request.scene},
                {"group": request.group},
            ]
        }
    )
    if scene:
        points = [GPS(**point) for point in scene["gps"]]
        if len(points) == 1:
            return scene["location"], points[0]
        if points:
            return scene["location"], GPS(
                lat=sum(point.lat for point in points) / len(points),
                lon=sum(point.lon for point in points) / len(points),
            )
        else:
            return scene["location"], request.center

    raise ValueError("No location found")


def get_image_center(image: str) -> Optional[GPS]:
    doc = image_collection.find_one({"image.src": image})
    if doc:
        return GPS(**doc["gps"])
    return None


def get_location_info(request: MapRequest) -> Optional[Tuple[str, dict]]:
    """
    Get the location info
    If there is only one location with the same name, return that
    If there are multiple locations with the same name, then check other parameters
    """
    location, center = get_location_name(request)
    if not location or location == "---":
        return "---", {
            "location": location,
            "location_info": "",
            "fsq_id": "",
            "fsq_info": {},
            "gps": center.model_dump() if center else request.center,
            "icon": None,
        }

    locations = location_collection.find({"location": location})
    locations = list(locations)

    info = None
    location_info = ""

    if len(locations) == 1:
        return location, locations[0]
    elif len(locations) > 1:
        if not center:
            for loc in locations:
                if loc["gps"]:
                    return location, loc
        else:
            threshold = 0.01  # 0.1 degrees is about 11 km
            closest_location = None
            closest_distance = float("inf")
            for loc in locations:
                distance = calculate_distance(center, loc["gps"])
                if distance < threshold:
                    return loc
                if distance < closest_distance:
                    closest_location = loc
                    closest_distance = distance
            if closest_location:
                return location, closest_location

    if not center:
        return None
    # Search for the location
    id = search_fourspace(location, center.lat, center.lon)
    if not id:
        return None
    info = get_info(id)
    location_info = ""
    if info["categories"]:
        location_info = info["categories"][0]["name"]

    icon = get_icon_from_fsq(info)
    if not icon:
        icon = get_icon_from_location_name(location, location_info)

    data = {
        "location": location,
        "location_info": location_info,
        "fsq_id": id,
        "fsq_info": info,
        "gps": center.model_dump(),
        "icon": icon.model_dump() if icon else None,
    }

    location_collection.insert_one(
        data,
    )
    return location, data


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


def get_all_images_with_location(location: str) -> List[str]:
    images = image_collection.find({"location": location})
    return [image["image"] for image in images]


def get_all_images_from_same_scene(image: str) -> List[Image]:
    scene = scene_collection.find_one({"images": {"$elemMatch": {"src": image}}})
    if scene:
        return [Image(**img) for img in scene["images"]]
    return []


geolocator = Nominatim(user_agent="myeachtra")


def reverse_geomapping(center: GPS) -> Optional[str]:
    location = geolocator.reverse((center.lat, center.lon), exactly_one=True, language="en")  # type: ignore
    print(location)
    if location:
        return location.address  # type: ignore
    return None
