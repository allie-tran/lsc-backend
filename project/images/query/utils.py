import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import geopy.distance
import requests

COMMON_PATH = os.getenv("COMMON_PATH")
grouped_info_dict = json.load(open(f"{COMMON_PATH}/basic_dict.json"))


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
        print(json_query)
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

    # IMAGECLEF
    if factor == "scene":
        final_results = []
        scores = []
        for images_with_scores in grouped_results.values():
            score = images_with_scores[0][1]
            scores.append(score)
            images = [res[0] for res in images_with_scores]
            final_results.append({
                "current": [image["image_path"] for image in images],
                "before": images[0]["before"],
                "after": images[0]["after"]})
        print(f"Grouped in to {len(final_results)} groups.")
        return final_results, scores
    else:
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
        scores = []
        for score, images, begin_time, end_time in sorted_groups:
            scores.append(score)
            final_results.append({
                "current": [image["image_path"] for image in images],
                "before": images[0]["before"],
                "after": images[0]["after"],
                "begin_time": begin_time,
                "end_time": end_time})
        print(f"Grouped in to {len(final_results)} groups.")
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
        # if pair["before"]:
        #     start_id = min([grouped_info_dict[img]["id"]
        #                     for img in pair["before"]])
        # else:
        #     start_id = min([grouped_info_dict[img]["id"]
        #                     for img in pair["current"]])
        # if pair["after"]:
        #     end_id = max([grouped_info_dict[img]["id"]
        #                   for img in pair["after"]])
        # else:
        #     end_id = max([grouped_info_dict[img]["id"]
        #                   for img in pair["current"]])
        # start_current_id = min([grouped_info_dict[img]["id"]
        #                         for img in pair["current"]])
        # end_current_id = max([grouped_info_dict[img]["id"]
        #                       for img in pair["current"]])
        # gps_query = {
        #     "size": end_id - start_id + 1,
        #     "_source": {
        #         "includes": ["id", "gps", "image_path"]
        #     },
        #     "query": {
        #         "range": {
        #             "id": {
        #                 "gte": start_id,
        #                 "lte": end_id
        #             }
        #         }
        #     },
        #     "sort":
        #     {"id": {"order": "asc"}}
        # }
        # gps_data = post_request(json.dumps(gps_query), "lsc2020")
        # pair["gps_path"] = [img[0]["gps"]
        #                     for img in gps_data]

        # pair["gps"] = [pair["gps_path"][:start_current_id - start_id],
        #                pair["gps_path"][start_current_id -
        #                                 start_id:end_current_id + 1],
                    #    pair["gps_path"][end_current_id + 1:]]
        pair["gps"] = [get_gps(pair["before"]), get_gps(
            pair["current"]), get_gps(pair["after"])]
        pair["gps_path"] = pair["gps"][0] + pair["gps"][1] + pair["gps"][2]
        new_pairs.append(pair)
    return new_pairs


def time_es_query(prep, hour, minute):
    if prep in ["before", "earlier than", "sooner than"]:
        if hour != 24 or minute != 0:
            return f"(doc['time'].value.getHour() < {hour} || (doc['time'].value.getHour() == {hour} && doc['time'].value.getMinute() <= {minute}))"
        else:
            return None
    if prep in ["after", "later than"]:
        if hour != 0 or minute != 0:
            return f"(doc['time'].value.getHour() > {hour} || (doc['time'].value.getHour() == {hour} && doc['time'].value.getMinute() >= {minute}))"
        else:
            return None
    return f"abs(doc['time'].value.getHour() - {hour}) < 1"


def add_time_query(time_filters, prep, time):
    query = time_es_query(prep, time[0], time[1])
    if query:
        time_filters.add(query)
    return time_filters


def time_to_filters(start_time, end_time, dates):
    # Time
    time_filters = add_time_query(set(), "after", start_time)
    time_filters = add_time_query(time_filters, "before", end_time)
    if (end_time[0] < start_time[0]) or (end_time[0] == start_time[0] and end_time[1] < start_time[1]):
        time_filters = [
            f' ({"||".join(time_filters)}) '] if time_filters else []
    else:
        time_filters = [
            f' ({"&&".join(time_filters)}) '] if time_filters else []

    # Date
    date_filters = set()
    for y, m, d in dates:
        this_filter = []
        if y:
            this_filter.append(
                f" (doc['time'].value.getYear() == {y}) ")
        if m:
            this_filter.append(
                f" (doc['time'].value.getMonthValue() == {m}) ")
        if d:
            this_filter.append(
                f" (doc['time'].value.getDayOfMonth() == {d}) ")
        date_filters.add(f' ({"&&".join(this_filter)}) ')
    date_filters = [
        f' ({"||".join(date_filters)}) '] if date_filters else []
    return time_filters, date_filters
