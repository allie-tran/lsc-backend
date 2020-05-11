import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import geopy.distance
import requests

COMMON_PATH = os.getenv("COMMON_PATH")
group_info = json.load(open(f"{COMMON_PATH}/group_info.json"))
grouped_info_dict = json.load(open(f"{COMMON_PATH}/grouped_info_dict.json"))


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


# TODO!: not sorted by image but time
def get_gps(images):
    if images:
        if isinstance(images[0], str):
            all_gps = [grouped_info_dict[image]["gps"] for image in images]

            sorted_by_time = [gps for (gps, image) in sorted(
                zip(all_gps, images), key=lambda x: x[1])]
        elif isinstance(images[0], dict) and "gps" in images[0]:
            all_gps = [image["gps"] for image in images]
            sorted_by_time = [image["gps"] for image in sorted(
                images, key=lambda x: x["image_path"])]
        else:
            raise NotImplementedError
        return sorted_by_time
    return None


def post_request(json_query, index="lsc2019_combined_text_bow"):
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"http://localhost:9200/{index}/_search", headers=headers, data=json_query)
    if response.status_code == 200:
        # stt = "Success"
        response_json = response.json()  # Convert to json as dict formatted
        id_images = [[d["_source"], d["_score"]]
                     for d in response_json["hits"]["hits"]]
    else:
        print('Wrong')
        # print(json_query)
        print(response.status_code)
        id_images = []
    return id_images


def find_place_in_available_group(regrouped_results, new_group, group_time=2):
    begin_time, end_time = get_time_of_group([res[0] for res in new_group])
    if regrouped_results:
        for regroup in regrouped_results:
            if abs(begin_time - regrouped_results[regroup]["begin_time"]) < timedelta(hours=group_time) and \
                    abs(end_time - regrouped_results[regroup]["end_time"]) < timedelta(hours=group_time):
                begin_time = min(
                    begin_time, regrouped_results[regroup]["begin_time"])
                end_time = max(
                    end_time, regrouped_results[regroup]["end_time"])
                return regroup, begin_time, end_time
    return "", begin_time, end_time


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


def group_results(results, factor="group", sort_by_time=False):
    print(f"Ungrouped: ", len(results))
    grouped_results = defaultdict(lambda: [])
    for result in results:
        group = result[0][factor]
        grouped_results[group].append(result)

    # Group again for hours < 2h, same location
    regrouped_results = {}
    count = 0
    for group in grouped_results:
        new_group = grouped_results[group]
        regroup, begin_time, end_time = find_place_in_available_group(
            regrouped_results, new_group)
        if regroup and factor == "group":
            regrouped_results[regroup]["raw_results"].extend(new_group)
            regrouped_results[regroup]["begin_time"] = begin_time
            regrouped_results[regroup]["end_time"] = end_time
        else:
            count += 1
            regrouped_results[f"group_{count}"] = {"raw_results": grouped_results[group],
                                                   "begin_time": begin_time,
                                                   "end_time": end_time}

    sorted_groups = []
    for group in regrouped_results:
        sorted_with_scores = sorted(
            regrouped_results[group]["raw_results"], key=lambda x: (-x[1] if x[1] else 0, x[0]["time"]), reverse=False)
        score = sorted_with_scores[0][1]
        sorted_groups.append((score,
                              [res[0] for res in sorted_with_scores],
                              regrouped_results[group]["begin_time"],
                              regrouped_results[group]["end_time"]))

    sorted_groups = sorted(
        sorted_groups, key=lambda x: (-x[0] if x[0] else 0, x[2]), reverse=False)

    final_results = []

    for score, images, begin_time, end_time in sorted_groups:
        final_results.append({
            "current": [image["image_path"] for image in images],
            "before": images[0]["before"],
            "after": images[0]["after"],
            "begin_time": begin_time,
            "end_time": end_time,
            "gps": [get_gps(images[0]["before"]), get_gps(images), get_gps(images[0]["after"])]})
    print(f"Grouped in to {len(final_results)} groups.")
    return final_results


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


def find_gps_path(pair):
    start_id = min([grouped_info_dict[img]["id"] for img in pair["before"]])
    end_id = max([grouped_info_dict[img]["id"] for img in pair["after"]])
    gps_query = {
        "_source": {
            "includes": ["id", "image_path"]
        },
        "query": {
            "range": {
                "id": {
                    "gte": start_id,
                    "lte": end_id
                }
            }
        },
        "sort":
        {"id": {"order": "asc"}}
    }
    gps_data = [img[0]["gps"]
                for img in post_request(json.dumps(gps_query), "lsc2020")]
    return gps_data
