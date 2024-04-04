from typing import Any, Dict, Optional

from database.utils import image_collection, scene_collection
from results.models import TimelineGroup, TimelineResult


def get_timeline(image: str) -> Optional[TimelineResult]:
    """
    For a given image, get the timeline of the group it belongs to
    Return the group before that, the group after that and the group itself
    """
    # Get the group of the image (grouped by location)
    group_id = image_collection.find_one({"image": image})
    if not group_id:
        return None

    name = group_id["time"].strftime("%d %b %Y")
    group_id_int = int(group_id["group"].split("G_")[-1])

    # Get the scenes of the group
    group_range_ids = [
        f"G_{group_id_int-1}",
        f"G_{group_id_int}",
        f"G_{group_id_int+1}",
    ]

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
        ]
    )

    results = []
    for group in grouped_results:
        group_obj = TimelineGroup(**group)
        results.append(group_obj)

    return TimelineResult(result=results, name=name)


# def get_more_scenes(group_id, direction="top"):
#     group_id = int(group_id.split("G_")[-1])
#     group_results = []
#     if direction == "bottom":
#         group_range = range(group_id + 1, group_id + 3)
#     else:
#         group_range = range(group_id - 2, group_id)
#     line = 0
#     space = 0
#     group_range = [f"G_{index}" for index in group_range]
#     for group in group_range:
#         if group in groups:
#             scenes = []
#             for scene_name, images in groups[group]["scenes"]:
#                 scenes.append((scene_name, images, time_info[scene_name]))
#             if scenes:
#                 space += 1
#                 line += (len(scenes) - 1) // 4 + 1
#                 group_results.append(
#                     (
#                         group,
#                         [
#                             groups[group]["location"],
#                             str(groups[group]["location_info"]),
#                         ],
#                         scenes,
#                     )
#                 )
#     return group_results, line, space
