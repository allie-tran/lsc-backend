import json
from collections import defaultdict
from datetime import datetime
from typing import List

from configs import FILES_DIRECTORY
from query_parse.utils import cache
from database.utils import image_collection, scene_collection

time_info = json.load(open(f"{FILES_DIRECTORY}/backend/time_info.json"))


def get_dict(image: str) -> dict:
    if "/" not in image:
        image = f"{image[:6]}/{image[6:8]}/{image}"
    info = image_collection.find_one({"image": image})
    assert info, f"Image {image} not found"
    return info


@cache
def get_date_info(image: str) -> str:
    time = datetime.strptime(get_dict(image)["time"], "%Y/%m/%d %H:%M:%S%z")
    return time.strftime("%A, %d %B %Y")


def get_location(image: str) -> str:
    return get_dict(image)["location"]


def get_gps(images) -> List[dict]:
    if images:
        if isinstance(images[0], tuple):  # images with weights
            images = [image[0] for image in images]
        if isinstance(images[0], str):
            images = image_collection.find(
                {"image": {"$in": images}}, {"gps": 1, "_id": 0}
            )
        sorted_by_time = [
            image["gps"] for image in sorted(images, key=lambda x: x["time"])
        ]
        return sorted_by_time
    return []


def group_scene_results(results, group_factor="group", query_info=[]):
    size = len(results)
    if size == 0:
        return [], []

    # group the results by group_factor
    grouped_results = defaultdict(lambda: [])
    for result in results:
        assert group_factor in result[0], f"{group_factor} not in {result[0]}"
        group = result[0][group_factor]
        grouped_results[group].append(result)

    # sort the scene in each group by their scene_id
    for group in grouped_results:
        grouped_results[group] = sorted(
            grouped_results[group], key=lambda x: x[0]["scene"]
        )

    # split the group if number of images of the children scenes is too large
    cut_off = 120
    new_grouped_results = []
    current_len = 0
    for group in grouped_results:
        new_group = []
        for scene, score in grouped_results[group]:
            num_images = 0
            images = scene_collection.find_one({"scene": scene["scene"]})
            if images:
                num_images = len(images)
            if (
                scene["location_info"] not in ["Car", "Airplane"]
                and num_images + current_len > cut_off
                and new_group
            ):
                new_grouped_results.append(new_group)
                current_len = 0
                new_group = []
            new_group.append((scene, score))
            current_len += num_images
        if new_group:
            new_grouped_results.append(new_group)

    # sort the groups by the highest score of their scene
    new_grouped_results = sorted(
        new_grouped_results,
        key=lambda group: max([score for scene, score in group]),
        reverse=True,
    )

    results_with_info = []
    scores = []
    for scenes_with_scores in new_grouped_results:
        score = max([s for scene, s in scenes_with_scores])
        scenes = [res[0] for res in scenes_with_scores]
        new_scenes = []
        ocr = []
        for scene in scenes:
            if isinstance(scene["start_time"], str):
                scene["start_time"] = datetime.strptime(
                    scene["start_time"], "%Y/%m/%d %H:%M:%S%z"
                )
                scene["end_time"] = datetime.strptime(
                    scene["end_time"], "%Y/%m/%d %H:%M:%S%z"
                )
            for text in scene["ocr"]:
                if text and text not in ocr:
                    ocr.append(text)
            new_scenes.append(scene)
        scenes = new_scenes
        best_scene = scenes[0][group_factor]
        scenes = sorted(scenes, key=lambda x: x["start_time"])
        scores.append(score)
        images = [image for scene in scenes for image in scene["images"]]
        new_scene = {
            "current": images,
            "start_time": scenes[0]["start_time"],
            "end_time": scenes[-1]["end_time"],
            "location": get_display_info(scenes, best_scene, query_info),
            "original_location": scenes[0]["location"],
            "ocr": ocr,
        }
        for key in scenes[0].keys():
            if key not in new_scene:
                new_scene[key] = scenes[0][key]
        results_with_info.append(new_scene)
    return results_with_info, scores


def get_display_info(scenes, best_scene, query_info):
    location = scenes[0]["country"]
    if scenes[0]["location"] != "---":
        to_show = []
        if "regions" in query_info:
            to_show = [
                region
                for region in scenes[0]["region"]
                if region.lower() in query_info["regions"]
                and region != scenes[0]["country"]
            ]
        if to_show:
            location = (
                scenes[0]["location"] + f", " + ", ".join(to_show) + f" ({location})"
            )
        else:
            location = scenes[0]["location"] + f" ({location})"

    return [
        location,
        datetime.strftime(scenes[0]["start_time"], "%A, %d/%m/%Y"),
        time_info[best_scene],
    ]
