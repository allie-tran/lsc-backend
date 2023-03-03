import json
import os
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
import geopy.distance
import requests
from ..nlp_utils.common import cache, FILE_DIRECTORY

basic_dict = json.load(open(f"{FILE_DIRECTORY}/info_dict.json"))
all_images = list(basic_dict.keys())

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
    return get_dict(image)["date"]

def get_location(image):
    return get_dict(image)["location"]


def spell_correct(sent):
    corrected = speller(sent)
    if corrected != sent:
        print(f"Spell correction: {sent} ---> {corrected}")
    return corrected


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

    id_images = []
    aggregations = []
    scroll_id = None

    if response.status_code == 200:
        # stt = "Success"
        response_json = response.json()  # Convert to json as dict formatted
        id_images = [[d["_source"], d["_score"]]
                     for d in response_json["hits"]["hits"]]
        if "aggregations" in response_json:
            aggregations = response_json["aggregations"]
        scroll_id = response_json["_scroll_id"] if scroll else None
    else:
        print(f'Response status {response.status_code}')
        print(response.text)

    if not id_images:
        with open("request.log", "a") as f:
            f.write(json_query + '\n')
        # print(json_query)
        print(f'Empty results. Output in request.log')
    return id_images, scroll_id, aggregations


def post_mrequest(json_query, index="lsc2019_combined_text_bow"):
    with open('request.log', 'w') as f:
        f.write(json_query + '\n')
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


    return id_images

def get_min_event(images, event_type="group"):
    return np.argmin([int(image[event_type].split('_')[-1])
                      for image in images])


def get_max_event(images, event_type="group"):
    return np.argmax([int(image[event_type].split('_')[-1])
                      for image in images])


def find_place_in_available_group(regrouped_results, group, begin_time, end_time, group_time=2):
    if regrouped_results:
        for i, regroup in enumerate(regrouped_results):
            # if regroup["group"] == group:
                # earlier
            if end_time < regroup["begin_time"] and regroup["end_time"] - begin_time < timedelta(hours=group_time):
                return i, begin_time, regroup["end_time"]
            if begin_time > regroup["end_time"] and end_time - regroup["begin_time"] < timedelta(hours=group_time):
                return i, regroup["begin_time"], end_time
    return -1, begin_time, end_time


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

        results_with_info = []
        scores = []
        for group, images_with_scores in grouped_results.items():
            score = images_with_scores[0][1] # maximum (first) score of the scenes in the same group
            scores.append(score)
            # images = [grouped_info_dict[img] for scene in images_with_scores for img in scene[0]["current"]]
            begin_time, end_time = get_time_of_group([scene[0] for scene in images_with_scores], "begin_time")
            results_with_info.append({
                "current": [img for scene in images_with_scores for img in scene[0]["current"]][:5],
                "begin_time": begin_time,
                "end_time": end_time,
                "group": group})
        final_results = results_with_info
    else:
        scores = []
        final_results = []
        for scene in results:
            # TODO! choose what to present in scenes
            # scene[0]["current"] = random.choice(scene[0]["current"], k=min(5, len(scene[0]["current"])))
            scene[0]["begin_time"] = datetime.strptime(
                scene[0]["begin_time"], "%Y/%m/%d %H:%M:%S%z")
            scene[0]["end_time"] = datetime.strptime(
                scene[0]["end_time"], "%Y/%m/%d %H:%M:%S%z")
            final_results.append(scene[0])
            scores.append(scene[1])
    return final_results, scores


def format_single_result(results, factor="dummy", group_more_by=0):
    size = len(results)
    if size == 0:
        return [], []
    grouped_results = defaultdict(lambda: [])
    count = 0
    results_with_info = []
    scores = []
    for result, score in results:
        scores.append(score)
        results_with_info.append({
            "current": [result["path"]],
            "begin_time": result["time"],
            "end_time": result["time"],
            "date_identifier": result["date_identifier"],
            "date": result["date"],
            "group": result["group"],
            "scene": result["scene"],
            "person": result["person"],
            "visit": result["visit"],
            "location": f'{result["hour"]:0>2}:{result["minute"]:0>2} on {result["date"]}\n{result["person"]}'})
    return results_with_info, scores


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
        begin_time, end_time = get_time_of_group(images)
        results_with_info.append({
            "current": [image["path"] for image in images][:5],
            "begin_time": begin_time,
            "end_time": end_time,
            "location": images[0]["location"],
            "group": images[0]["group"],
            "scene": images[0]["scene"]})

    if results_with_info and group_more_by:
        final_results = [results_with_info[0]]
        final_scores = [scores[0]]
        for result, score in zip(results_with_info[1:], scores[1:]):
            ind, begin_time, end_time = find_place_in_available_group(final_results, result["group"],
                                                     result["begin_time"], result["end_time"], group_more_by)
            if ind == -1:
                final_results.append(result)
                final_scores.append(score)
            else:
                final_results[ind]["current"].extend(result["current"][:5])
                final_results[ind]["begin_time"] = begin_time
                final_results[ind]["end_time"] = end_time
    else:
        final_results = results_with_info
        final_scores = scores
    return final_results, final_scores

def get_time_of_group(images, field="time"):
    times = [datetime.strptime(
        image[field], "%Y/%m/%d %H:%M:%S%z") for image in images]
    begin_time = min(times)
    end_time = max(times)
    return begin_time, end_time


def find_place_in_available_times(grouped_times, begin_time, end_time, group_time=2):
    if grouped_times:
        for time in grouped_times:
            if abs(begin_time - grouped_times[time]["begin_time"]) < timedelta(hours=group_time) and \
                    abs(end_time - grouped_times[time]["end_time"]) < timedelta(hours=group_time):
                begin_time = min(
                    begin_time, grouped_times[time]["begin_time"])
                end_time = max(
                    end_time, grouped_times[time]["end_time"])
                return time, begin_time, end_time
    return "", begin_time, end_time


def find_time_span(groups):
    """
    time can be -1 for 1h before
    """
    times = {}
    count = 0
    for group in groups:
        time, begin_time, end_time = find_place_in_available_times(
            times, group["begin_time"], group["end_time"])
        if time:
            times[time]["begin_time"] = begin_time
            times[time]["end_time"] = end_time
        else:
            count += 1
            times[f"time_{count}"] = {"begin_time": begin_time,
                                      "end_time": end_time}
    return times.values()


def add_gps_path(pairs):
    return pairs
