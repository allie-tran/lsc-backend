from datetime import datetime
from typing import List, Optional

from database.main import group_collection, image_collection, scene_collection
from results.models import HighlightItem, TimelineGroup, TimelineResult


def get_timeline(image: str) -> Optional[TimelineResult]:
    """
    For a given image, get the timeline of the group it belongs to
    The start of timeline is the start of the first group of the day,
    or the last group of the previous day if it lasts over midnight

    The end of timeline is the end of the last group of the day,
    or the first group of the next day if it lasts over midnight
    """
    image_info = image_collection.find_one({"image": image})
    if not image_info:
        return None

    highlight = HighlightItem(**image_info)
    image_date = image_info["time"]
    start_time = image_date.replace(hour=0, minute=0, second=0)
    end_time = image_date.replace(hour=23, minute=59, second=59)

    # Get all groups of the same day
    group_ids = group_collection.find(
        {
            "$or": [
                {"start_time": {"$gte": start_time, "$lte": end_time}},
                {"end_time": {"$gte": start_time, "$lte": end_time}},
            ]
        },
        {"group": 1},
    )
    group_range_ids = [group["group"] for group in group_ids]
    results = get_scene_for_group_ids(group_range_ids)
    return TimelineResult(date=start_time, result=results, highlight=highlight)


def get_scene_for_group_ids(group_range_ids: List[str]) -> List[TimelineGroup]:
    grouped_results = scene_collection.aggregate(
        [
            {"$match": {"group": {"$in": group_range_ids}}},
            {
                "$group": {
                    "_id": "$group",
                    "group": {"$first": "$group"},
                    "scenes": {"$push": "$images"},
                    "time_info": {"$push": "$time_info"},
                    "location": {"$first": "$location"},
                    "location_info": {"$first": "$location_info"},
                }
            },
            {"$sort": {"group": 1}},
        ]
    )

    results = []
    for group in grouped_results:
        group_obj = TimelineGroup(**group)
        results.append(group_obj)

    return results


def get_timeline_for_date(str_date: str) -> Optional[TimelineResult]:
    """
    Get all scenes for a given date
    """
    date = datetime.strptime(str_date, "%d-%m-%Y")
    start_time = date.replace(hour=0, minute=0, second=0)
    end_time = date.replace(hour=23, minute=59, second=59)

    # Get all groups of the same day
    group_ids = group_collection.find(
        {
            "$or": [
                {"start_time": {"$gte": start_time, "$lte": end_time}},
                {"end_time": {"$gte": start_time, "$lte": end_time}},
            ]
        },
        {"group": 1},
    )
    group_range_ids = [group["group"] for group in group_ids]
    results = get_scene_for_group_ids(group_range_ids)
    return TimelineResult(date=start_time, result=results)


def get_more_scenes(
    group_id: str, direction: str = "before"
) -> Optional[TimelineResult]:
    """
    Get more scenes before or after the given group id
    """
    group_info = group_collection.find_one({"group": group_id})
    if not group_info:
        return None

    group_date = group_info["start_time"]
    if direction == "before":
        group_ids = group_collection.find(
            {"end_time": {"$lt": group_date}}, {"group": 1}
        )
    else:
        group_ids = group_collection.find(
            {"start_time": {"$gt": group_date}}, {"group": 1}
        )

    group_range_ids = [group["group"] for group in group_ids]
    results = get_scene_for_group_ids(group_range_ids)
    return TimelineResult(date=group_date, result=results)
