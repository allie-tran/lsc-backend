import json
import os
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
import geopy.distance
import requests
from ..nlp_utils.common import cache, basic_dict, FILES_DIRECTORY

all_images = list(basic_dict.keys())
groups = json.load(open(f"{FILES_DIRECTORY}/group_segments.json"))
scene_segments = {}
for group_name in groups:
    for scene_name, images in groups[group_name]["scenes"]:
        assert "S_" in scene_name, f"{scene_name} is not a valid scene id"
        scene_segments[scene_name] = images
time_info = json.load(open(f"{FILES_DIRECTORY}/backend/time_info.json"))

def get_dict(image):
    if "/" not in image:
        image = f"{image[:6]}/{image[6:8]}/{image}"
    return basic_dict[image]

@cache
def get_info(image):
    time = datetime.strptime(get_dict(image)["time"], "%Y/%m/%d %H:%M:%S%z")
    return time.strftime("%A, %d %B %Y, %I:%M%p")

@cache
def get_date_info(image):
    time = datetime.strptime(get_dict(image)["time"], "%Y/%m/%d %H:%M:%S%z")
    return time.strftime("%A, %d %B %Y")

def get_location(image):
    return get_dict(image)["location"]




def distance(lt1, ln1, lt2, ln2):
    return (geopy.distance.distance([lt1, ln1], [lt2, ln2]).km)


def not_noise(last_gps, current_gps):
    """Assume 30secs"""
    print(distance(last_gps["lat"], last_gps["lon"],
                   current_gps["lat"], current_gps["lon"]))
    return distance(last_gps["lat"], last_gps["lon"],
                    current_gps["lat"], current_gps["lon"]) < 0.03


def filter_sorted_gps(gps_points):
    if gps_points:
        points = [gps_points[0]]
        for point in gps_points[1:]:
            if not_noise(points[-1], point):
                points.append(point)
        print(len(gps_points), len(points))
        return points
    return []


def get_gps(images):
    if images:
        if isinstance(images[0], tuple): #images with weights
            images = [image[0] for image in images]
        if isinstance(images[0], str):
            images = [get_dict(image) for image in images]
        sorted_by_time = [image["gps"] for image in sorted(
            images, key=lambda x: x["time"])]
        return sorted_by_time
    return []

def delete_scroll_ids(scroll_ids):
    response = requests.delete(
            f"http://localhost:9200/_search/scroll", headers={"Content-Type": "application/json"}, data=json.dumps({"scroll_id": scroll_ids}))
    assert response.status_code == 200, f"Wrong request ({response.status_code})"

def post_request(json_query, index="lsc2019_combined_text_bow", scroll=False):
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"http://localhost:9200/{index}/_search{'?scroll=5m' if scroll else ''}", headers=headers, data=json_query)
    if response.status_code == 200:
        # stt = "Success"
        response_json = response.json()  # Convert to json as dict formatted
        id_images = [[d["_source"], d["_score"]]
                     for d in response_json["hits"]["hits"]]
        scroll_id = response_json["_scroll_id"] if scroll else None
    else:
        print(f'Response status {response.status_code}')
        print(response.text)
        id_images = []
        scroll_id = None

    if not id_images:
        with open("request.log", "a") as f:
            f.write(json_query + '\n')
        # print(json_query)
        print(f'Empty results. Output in request.log')
    return id_images, scroll_id


def post_mrequest(json_query, index="lsc2019_combined_text_bow"):
    headers = {"Content-Type": "application/x-ndjson"}
    response = requests.post(
        f"http://localhost:9200/{index}/_msearch", headers=headers, data=json_query)
    if response.status_code == 200:
        # stt = "Success"
        response_json = response.json()  # Convert to json as dict formatted
        id_images = []
        for res in response_json["responses"]:
            try:
                id_images.append([[d["_source"], d["_score"]]
                            for d in res["hits"]["hits"]])
            except KeyError as e:
                print(res)
                id_images.append([])
    else:
        print(f'Response status {response.status_code}')
        id_images = []

    # with open('request.log', 'w') as f:
        # f.write(json_query + '\n')
    return id_images

def get_min_event(images, event_type="group"):
    return np.argmin([int(image[event_type].split('_')[-1])
                      for image in images])


def get_max_event(images, event_type="group"):
    return np.argmax([int(image[event_type].split('_')[-1])
                      for image in images])


def find_place_in_available_group(regrouped_results, group, start_time, end_time, group_time=2):
    if regrouped_results:
        for i, regroup in enumerate(regrouped_results):
            # if regroup["group"] == group:
                # earlier
            if end_time < regroup["start_time"] and regroup["end_time"] - start_time < timedelta(hours=group_time):
                return i, start_time, regroup["end_time"]
            if start_time > regroup["end_time"] and end_time - regroup["start_time"] < timedelta(hours=group_time):
                return i, regroup["start_time"], end_time
    return -1, start_time, end_time


def group_scene_results(results, factor="group", group_more_by=0):
    size = len(results)
    if size == 0:
        return [], []
    if factor == "group":
        grouped_results = defaultdict(lambda: [])
        count = 0
        for result in results:
            group = result[0]["group"]
            grouped_results[group].append(result)

        # if factor == "scene":
        results_with_info = []
        scores = []
        for scenes_with_scores in grouped_results.values():
            score = scenes_with_scores[0][1]
            scores.append(score)
            scenes = [res[0] for res in scenes_with_scores]
            new_scenes = []
            for scene in scenes:
                scene["start_time"] = datetime.strptime(scene["start_time"], "%Y/%m/%d %H:%M:%S%z")
                scene["end_time"] = datetime.strptime(scene["end_time"], "%Y/%m/%d %H:%M:%S%z")
                new_scenes.append(scene)
            scenes = new_scenes
            scenes = sorted(scenes, key=lambda x: x["start_time"])
            images = []
            results_with_info.append({
                "current": [image for scene in scenes for image in scene["images"]],
                "start_time": scenes[0]["start_time"],
                "end_time": scenes[-1]["end_time"],
                "location": scenes[0]["location"] + "\n" + \
                            datetime.strftime(scenes[0]["start_time"], "%Y/%m/%d"),
                "group": scenes[0]["group"],
                "scene": scenes[0]["scene"]})
            
    return results_with_info, scores

        


def format_single_result(results, factor="dummy", group_more_by=0):
    size = len(results)
    if size == 0:
        return [], []
    count = 0
    results_with_info = []
    scores = []
    for result, score in results:
        scores.append(score)
        results_with_info.append({
            "current": [result["image_path"]],
            "start_time": result["time"],
            "end_time": result["time"],
            "location": result["location"],
            "group": result["group"],
            "scene": result["scene"]})
    return results_with_info, scores

def add_full_scene(scene_id, images):
    new_scene = []
    max_padding = 2
    current_padding = 0
    if len(images[0]) == 2:
        threshold = max(0.05, 1/len(images))
        for img, score in images:
            if score >= threshold:
                new_scene.append((img, score, True))
                current_padding = 0
            elif score > 0:
                current_padding += 1
                if current_padding < max_padding:
                    new_scene.append((img, score, False))
    elif len(images[0]) == 3:
        return images
    else:
        full_scene = scene_segments[scene_id]
        for img in full_scene:
            if img in images:
                new_scene.append((img, 1.0, True))
                current_padding = 0
            else:
                current_padding += 1
                if current_padding < max_padding:
                    new_scene.append((img, 0.0, False))
    return new_scene
    

def group_results(results, factor="group", group_more_by=0):
    size = len(results)
    if size == 0:
        return [], []
    grouped_results = defaultdict(lambda: [])
    count = 0
    for result in results:
        group = result[0][factor]
        grouped_results[group].append(result)

    # IMAGECLEF
    # if factor == "scene":
    results_with_info = []
    scores = []
    for images_with_scores in grouped_results.values():
        score = images_with_scores[0][1]
        scores.append(score)
        images = [res[0] for res in images_with_scores]
        start_time, end_time = get_time_of_group(images)
        results_with_info.append({
            "current": [image["image_path"] for image in images][:5],
            "start_time": start_time,
            "end_time": end_time,
            "location": images[0]["location"] + "\n" + \
                        datetime.strftime(start_time, "%Y/%m/%d"),
            "group": images[0]["group"],
            "scene": images[0]["scene"]})

    if results_with_info and group_more_by:
        final_results = [results_with_info[0]]
        final_scores = [scores[0]]
        for result, score in zip(results_with_info[1:], scores[1:]):
            ind, start_time, end_time = find_place_in_available_group(final_results, result["group"],
                                                     result["start_time"], result["end_time"], group_more_by)
            if ind == -1:
                final_results.append(result)
                final_scores.append(score)
            else:
                final_results[ind]["current"].extend(result["current"][:5])
                final_results[ind]["start_time"] = start_time
                final_results[ind]["end_time"] = end_time
    else:
        final_results = results_with_info
        final_scores = scores
    return final_results, final_scores

def get_time_of_group(images, field="time"):
    times = [datetime.strptime(
        image[field], "%Y/%m/%d %H:%M:%S%z") for image in images]
    start_time = min(times)
    end_time = max(times)
    return start_time, end_time


def find_place_in_available_times(grouped_times, start_time, end_time, group_time=2):
    if grouped_times:
        for time in grouped_times:
            if abs(start_time - grouped_times[time]["start_time"]) < timedelta(hours=group_time) and \
                    abs(end_time - grouped_times[time]["end_time"]) < timedelta(hours=group_time):
                start_time = min(
                    start_time, grouped_times[time]["start_time"])
                end_time = max(
                    end_time, grouped_times[time]["end_time"])
                return time, start_time, end_time
    return "", start_time, end_time


def find_time_span(groups):
    """
    time can be -1 for 1h before
    """
    times = {}
    count = 0
    for group in groups:
        time, start_time, end_time = find_place_in_available_times(
            times, group["start_time"], group["end_time"])
        if time:
            times[time]["start_time"] = start_time
            times[time]["end_time"] = end_time
        else:
            count += 1
            times[f"time_{count}"] = {"start_time": start_time,
                                      "end_time": end_time}
    return times.values()


def add_gps_path(pairs):
    new_pairs = []
    for pair in pairs:
        if "gps" not in pair:
            pair["gps"] = get_gps(pair["current"])
        pair["current"] = add_full_scene(pair["scene"], pair["current"])
        if "before" in pair:
            pair["before"] = add_full_scene(pair["scene"], pair["before"])
        if "after" in pair:
            pair["after"] = add_full_scene(pair["scene"], pair["after"])
        # pair["gps_path"] = pair["gps"]
        new_pairs.append(pair)
    return new_pairs
