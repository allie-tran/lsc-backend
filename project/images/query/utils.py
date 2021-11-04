import json
import os
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
import geopy.distance
import requests
from ..nlp_utils.common import cache

COMMON_PATH = os.getenv("COMMON_PATH")
grouped_info_dict = json.load(open(f"{COMMON_PATH}/basic_dict.json"))


@cache
def get_info(image):
    time = datetime.strptime(grouped_info_dict[image]["time"], "%Y/%m/%d %H:%M:%S+00")
    return time.strftime("%A, %d %B %Y, %I:%M%p")


def get_location(image):
    return grouped_info_dict[image]["location"]


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
            images = [grouped_info_dict[image] for image in images]
        sorted_by_time = [image["gps"] for image in sorted(
            images, key=lambda x: x["id"])]
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
        id_images = []
        scroll_id = None
    
    if not id_images:
        print(f'Empty results. Output in request.log')
        with open('request.log', 'a') as f:
            f.write(json_query + '\n')
    return id_images, scroll_id

def get_min_event(images, event_type="group"):
    return np.argmin([int(image[event_type].split('_')[-1])
                      for image in images])


def get_max_event(images, event_type="group"):
    return np.argmax([int(image[event_type].split('_')[-1])
                      for image in images])


def get_before_after(images):
    min_group = get_min_event(images)
    max_group = get_max_event(images)
    return images[min_group]["before"], images[max_group]["after"]


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
    print(f"Ungrouped: ", size)

    if factor == "group":
        grouped_results = defaultdict(lambda: [])
        count = 0
        for result in results:
            group = result[0]["group"]
            grouped_results[group].append(result)

        results_with_info = []
        scores = []
        concepts = Counter()
        feedback = True
        for images_with_scores in grouped_results.values():
            score = images_with_scores[0][1]
            scores.append(score)
            images = [grouped_info_dict[img] for scene in images_with_scores for img in scene[0]["current"]]
            begin_time, end_time = get_time_of_group(images)
            results_with_info.append({
                "current": [img for scene in images_with_scores for img in scene[0]["current"]][:5],
                "begin_time": begin_time,
                "end_time": end_time,
                "group": images[0]["group"]})
        final_results = results_with_info
    else:
        scores = []
        final_results = []
        for scene in results:
            # TODO! choose what to present in scenes
            # scene[0]["current"] = random.choice(scene[0]["current"], k=min(5, len(scene[0]["current"])))
            scene[0]["begin_time"] = datetime.strptime(
                scene[0]["begin_time"], "%Y/%m/%d %H:%M:%S+00")
            scene[0]["end_time"] = datetime.strptime(
                scene[0]["end_time"], "%Y/%m/%d %H:%M:%S+00")
            final_results.append(scene[0])
            scores.append(scene[1])

    print(f"Grouped in to {len(final_results)} groups.")
    print("Score:", min(scores) if scores else None,
          '-', max(scores) if scores else None)
    return final_results, scores


def group_results(results, factor="group", group_more_by=0):
    size = len(results)
    print(f"Ungrouped: ", size)

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
            "current": [image["image_path"] for image in images][:5],
            "before": images[0]["before"],
            "after": images[0]["after"],
            "begin_time": begin_time,
            "end_time": end_time,
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

    print(f"Grouped in to {len(final_results)} groups.")
    print("Score:", min(scores) if scores else None, '-', max(scores) if scores else None)
    return final_results, scores

def get_time_of_group(images):
    times = [datetime.strptime(
        image["time"], "%Y/%m/%d %H:%M:%S+00") for image in images]
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
    new_pairs = []
    for pair in pairs:
        pair["gps"] = [get_gps(pair["before"]), get_gps(
            pair["current"]), get_gps(pair["after"])]
        pair["gps_path"] = pair["gps"][0] + pair["gps"][1] + pair["gps"][2]
        new_pairs.append(pair)
    return new_pairs




def time_to_filters(begin_time, end_time, dates, scene_group=False):
    # Time
    time_filters = add_time_query(
        set(), "after", begin_time, scene_group=scene_group)
    time_filters = add_time_query(
        time_filters, "before", end_time, scene_group=scene_group)
    if (end_time[0] < begin_time[0]) or (end_time[0] == begin_time[0] and end_time[1] < begin_time[1]):
        time_filters = [
            f' ({"||".join(time_filters)}) '] if time_filters else []
    else:
        time_filters = [
            f' ({"&&".join(time_filters)}) '] if time_filters else []

    # Date
    date_filters = set()
    factor = 'begin_time' if scene_group else 'time'
    for y, m, d in dates:
        this_filter = []
        if y:
            this_filter.append(
                f" (doc['{factor}'].value.getYear() == {y}) ")
        if m:
            this_filter.append(
                f" (doc['{factor}'].value.getMonthValue() == {m}) ")
        if d:
            this_filter.append(
                f" (doc['{factor}'].value.getDayOfMonth() == {d}) ")
        date_filters.add(f' ({"&&".join(this_filter)}) ')
    date_filters = [
        f' ({"||".join(date_filters)}) '] if date_filters else []
    return time_filters, date_filters
