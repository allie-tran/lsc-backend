from .query_types import *
from .timeline import time_info
from .utils import *
from ..nlp_utils.extract_info import Query
from ..nlp_utils.common import countries, map_visualisation, basic_dict, COMMON_DIRECTORY
from datetime import timedelta, datetime
import time as timecounter
from collections import defaultdict
import copy
import pandas as pd
import numpy as np
from clip import clip
import joblib
from torch import torch

full_similar_images = json.load(open(f"{FILE_DIRECTORY}/full_similar_images.json"))
full_similar_images = [f"{image[:6]}/{image[6:8]}/{image}" for image in full_similar_images]
# similar_features = joblib.load("/mnt/data/duyen/sift_and_vgg.feature")

multiple_pairs = {}
INCLUDE_SCENE = ["scene"]
INCLUDE_FULL_SCENE = ["current", "begin_time", "end_time", "gps", "scene", "group", "timestamp", "location"]
INCLUDE_IMAGE = ["image_path", "time", "gps", "scene", "group", "location"]

cached_queries = None
cached_filters =  {"bool": {"filter": [],
                               "should": [],
                               "must": {"match_all": {}},
                               "must_not": []}}

# format_func = format_single_result # ntcir
format_func = group_results

# CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)

def clip_query(main_query):
    with torch.no_grad():
        text_encoded = clip_model.encode_text(
            clip.tokenize(main_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    text_features = text_encoded.cpu().numpy()
    return text_features

def query_all(query, includes, index, group_factor):
    query = {"match_all": {}}
    request = {
        "size": 2000,
        "_source": {
            "includes": includes
        },
        "query": query,
        "sort": [
            {"time": {
                "order": "asc"
            }}
        ]
    }
    return query_text, format_func(post_request(json.dumps(request), index), group_factor)


def es_more(scroll_id, size=200):
    global multiple_pairs
    if scroll_id == 'pairs':
        position = multiple_pairs["position"]
        new_position = min(position + 24, len(multiple_pairs["pairs"]))
        last_results = multiple_pairs["pairs"][position: new_position]
        multiple_pairs["position"] = new_position
        return scroll_id, add_gps_path(last_results), []
    if scroll_id:
        start = timecounter.time()
        response = requests.post(
            f"http://localhost:9200/_search/scroll", headers={"Content-Type": "application/json"},
            data=json.dumps({"scroll": "5m",
                            "scroll_id": scroll_id}))

        assert response.status_code == 200, "Wrong request"
        response_json = response.json()  # Convert to json as dict formatted
        results = [[d["_source"], d["_score"]]
                for d in response_json["hits"]["hits"]]

        scroll_id = response_json["_scroll_id"]
        scenes = [scene[0]["scene"] for scene in results]
        must_queries, should_queries = cached_queries
        filter_queries = {"bool": {"filter": [{"terms": {"scene": scenes}}],
                                "should": [],
                                "must": {"match_all": {}},
                                "must_not": []}}
        # CONSTRUCT JSON
        json_query = get_json_query(must_queries, should_queries, filter_queries, size, includes=INCLUDE_IMAGE)
        results, _ = post_request(json.dumps(json_query), "lsc2022", scroll=False)
        print("Num Results:", len(results))
        results, scores = format_func(results, 'scene', 0)
        print("TOTAL TIMES:", timecounter.time() - start, " seconds.")
        return scroll_id, add_gps_path(results), scores
    return None, [], []


def es(query, gps_bounds, size=200, share_info=False):
    start = timecounter.time()
    query_info = {}
    raw_results = []
    scroll_id = None
    if query["before"] and query["after"]:
        query, before_query, after_query, (results, scores), scroll_id = es_three_events(
            query["current"], query["before"], query["beforewhen"], query["after"], query["afterwhen"], gps_bounds, share_info=share_info)
        query_info = query.get_info()
        cond_query_info = before_query.get_info()
        for key in cond_query_info:
            query_info[key].extend(cond_query_info[key])
        cond_query_info = after_query.get_info()
        for key in cond_query_info:
            query_info[key].extend(cond_query_info[key])
    elif query["before"]:
        query, cond_query, (results, scores), scroll_id = es_two_events(
            query["current"], query["before"], "before", query["beforewhen"], gps_bounds, share_info=share_info)
        query_info = query.get_info()
        cond_query_info = cond_query.get_info()
        for key in cond_query_info:
            query_info[key].extend(cond_query_info[key])
    elif query["after"]:
        query, cond_query, (results, scores), scroll_id = es_two_events(
            query["current"], query["after"], "after", query["afterwhen"], gps_bounds, share_info=share_info)
        query_info = query.get_info()
        cond_query_info = cond_query.get_info()
        for key in cond_query_info:
            query_info[key].extend(cond_query_info[key])
    else:
        query, (results, scores), scroll_id = individual_es(
            query["current"], gps_bounds, group_factor="scene", size=size, scroll=True)
        query_info = query.get_info()
    print("TOTAL TIMES:", timecounter.time() - start, " seconds.")
    return scroll_id, add_gps_path(results), scores, query_info


def query_list(query_list):
    return query_list[0] if len(query_list) == 1 else query_list


def get_json_query(must_queries, should_queries, filter_queries, size, includes, min_score=0):
    # CONSTRUCT JSON
    main_query = {}
    if must_queries:
        main_query["must"] = query_list(must_queries)
    else:
        main_query["must"] = {"match_all": {}}

    if should_queries:
        main_query["should"] = query_list(should_queries)
        main_query["minimum_should_match"] = 1

    if filter_queries["bool"]["filter"] or filter_queries["bool"]["should"] or filter_queries["bool"]["must_not"]:
        if "should" in filter_queries["bool"] and filter_queries["bool"]["should"]:
            filter_queries["bool"]["minimum_should_match"] = 1
        main_query["filter"] = filter_queries
    main_query = {"bool": main_query}

    json_query = {
        "size": size,
        "_source": {
            "includes": includes
        },
        "query": main_query,
        "sort": [
            "_score",
            {"timestamp": {
                "order": "asc"
            }}
        ]
    }
    if min_score:
        json_query["min_score"] = min_score
    return json_query


def get_neighbors(image, lsc, query_info, gps_bounds):
    if lsc:
        global cached_filters
        # print("Using cached filters")
        # print(cached_filters)
        filter_queries = copy.deepcopy(cached_filters)
    else:
        filter_queries = []

    should_queries = {
            "elastiknn_nearest_neighbors": {
                "field": "clip_vector",                # 1
                "vec": {                               # 2
                    "index": "lsc2022",
                    "field": "clip_vector",
                    "id": image
                },
                "model": "permutation_lsh",            # 3
                "similarity": "cosine",                # 4
                "candidates": 1000                   # 5
            }
        }

    json_query = get_json_query([should_queries], [], filter_queries, 40,
                includes=["image_path", "group", "location", "weekday", "time"])

    results, _ = post_request(json.dumps(json_query), "lsc2022")
    new_results = dict([(r[0]["image_path"], r[0]) for r in results])

    grouped_results = defaultdict(lambda: [])
    weekdays = {}
    dates = {}
    locations = {}
    for image in new_results:
        group = new_results[image]["group"]
        weekdays[group] = new_results[image]["weekday"].capitalize()
        dates[group] = new_results[image]["time"]
        locations[group] = new_results[image]["location"]

        grouped_results[group].append(image)
    times = [(grouped_results[group], locations[group] + "\n" + weekdays[group] + " " + dates[group].split(" ")[0] + "\n" + time_info[group])
                for group in grouped_results]
    return times[:100]


def individual_es(query, gps_bounds=None, extra_filter_scripts=None, group_factor="group", size=200, scroll=False):
    if isinstance(query, str):
        query = Query(query)

    if not query.original_text and not gps_bounds:
        return query_all(query, INCLUDE_IMAGE, "lsc2022", group_factor)
    return construct_es(query, gps_bounds, extra_filter_scripts, group_factor, size=size, scroll=scroll)


def construct_es(query, gps_bounds=None, extra_filter_scripts=None, group_factor="group", size=200, scroll=False):
    time_filters, date_filters = query.time_to_filters()
    must_queries = []
    # !TODO
    should_queries = []
    is_empty = True

    if query.ocr:
        should_queries.extend(query.make_ocr_query())
        is_empty = False

    filter_queries = {"bool": {"filter": [],
                               "should": [],
                               "must": {"match_all": {}},
                               "must_not": []}}

    if query.locations:
        should_queries.append(
            {"match": {"location": {"query": " ".join(query.locations), "boost": 0.01}}})
        location_filters = query.make_location_query()
        # if location_query:
            # should_queries.append(location_query)
        filter_queries["bool"]["should"].extend(location_filters)
        filter_queries["bool"]["should"].append({"match": {"location": {"query": " ".join(query.locations), "boost": 0.01}}})
        is_empty = False

    # FILTERS
    if query.regions:
        is_empty = False
        filter_queries["bool"]["filter"].extend([{"term": {"region": region}} for region in query.regions])
    if query.weekdays:
        is_empty = False
        filter_queries["bool"]["filter"].append(
            {"terms": {"weekday": query.weekdays}})

    if time_filters:
        if query.start[0] != 0 and query.end[0] != 24:
            is_empty = False
            filter_queries["bool"]["filter"].append(time_filters)

    if date_filters:
        is_empty = False
        filter_queries["bool"]["filter"].extend(date_filters)

    if gps_bounds:
        is_empty = False
        filter_queries["bool"]["filter"].append(get_gps_filter(gps_bounds))

    clip_script = None
    if query.clip_text:
        embedding = clip_query(query.clip_text)
        # clip_script = {
        #     "source": "((doc['clip_vector'].size() == 0 ? 0 : cosineSimilarity(params.embedding, 'clip_vector')) + 1) * 100 + _score",
        #     # "source": "(cosineSimilarity(params.embedding, doc['clip_vector']) + 1) * 100 + _score",
        #     "params": {"embedding": embedding.tolist()[0]}
        # }
        clip_script = {
            "elastiknn_nearest_neighbors": {
                "field": "clip_vector",                     # 1
                "vec": {                               # 2
                    "values": embedding.tolist()[0]
                },
                "model": "exact",            # 3
                "similarity": "cosine",                # 4
                "candidates": 1000                     # 5
            }
        }
        should_queries.append(clip_script)

    if scroll:
        global cached_filters
        cached_filters = filter_queries
        if filter_queries["bool"]["should"] or filter_queries["bool"]["filter"]:
            is_empty = False

    # CONSTRUCT JSON
    json_query = get_json_query(must_queries, should_queries,
                                filter_queries, size, includes=INCLUDE_IMAGE)
    global cached_queries
    cached_queries = (must_queries, should_queries)
    results, scroll_id = post_request(
        json.dumps(json_query), "lsc2022", scroll=True)
    print("Num Images:", len(results))
    # print([r[1] for r in results])
    return query, format_func(results, group_factor), scroll_id


def msearch(query, gps_bounds=None, extra_filter_scripts=None):
    if isinstance(query, str):
        query = Query(query)
        start = timecounter.time()

    if not query.original_text and not gps_bounds:
        return query_all(query, INCLUDE_IMAGE, "lsc2022", group_factor)

    time_filters, date_filters = query.time_to_filters()
    must_queries = []
    # !TODO
    should_queries = []

    if query.ocr:
        should_queries.extend(query.make_ocr_query())

    filter_queries = {"bool": {"filter": [],
                               "should": [],
                               "must": {"match_all": {}},
                               "must_not": []}}

    if query.locations:
        should_queries.append(
            {"match": {"location": {"query": " ".join(query.locations), "boost": 0.01}}})
        location_filters = query.make_location_query()
        # if location_query:
            # should_queries.append(location_query)
        filter_queries["bool"]["should"].extend(location_filters)
        filter_queries["bool"]["should"].append({"match": {"location": {"query": " ".join(query.locations), "boost": 0.01}}})

    # FILTERS
    if query.regions:
        filter_queries["bool"]["filter"].extend([{"term": {"region": region}} for region in query.regions])

    if query.weekdays:
        filter_queries["bool"]["filter"].append(
            {"terms": {"weekday": query.weekdays}})

    if time_filters:
        filter_queries["bool"]["filter"].append(time_filters)


    if date_filters:
        filter_queries["bool"]["filter"].extend(date_filters)


    if gps_bounds:
        filter_queries["bool"]["filter"].append(get_gps_filter(gps_bounds))


    clip_script = None
    if query.clip_text:
        embedding = clip_query(query.clip_text)
        clip_script = {
            "elastiknn_nearest_neighbors": {
                "field": "clip_vector",                # 1
                "vec": {                               # 2
                    "values": embedding.tolist()[0]
                },
                "model": "exact",            # 3
                "similarity": "cosine",                # 4
                "candidates": 1000                       # 5
            }
        }
        should_queries.append(clip_script)

    mquery = []
    for script in extra_filter_scripts:
        new_filter_queries = copy.deepcopy(filter_queries)
        new_filter_queries["bool"]["filter"].append(script)
        mquery.append(json.dumps({}))
        mquery.append(json.dumps(get_json_query(
            must_queries, should_queries, new_filter_queries, 1, INCLUDE_IMAGE, min_score=2.2 if query.clip_text else 0.2)))

    results = post_mrequest("\n".join(mquery) + "\n", "lsc2022")
    return query, results


def forward_search(query, conditional_query, condition, time_limit, gps_bounds):
    print("-" * 80)
    print("Main")
    start = timecounter.time()
    query_infos = []
    query, (main_events, scores), _ = individual_es(
        query, gps_bounds[0], size=1000, group_factor="scene")

    extra_filter_scripts = []
    time_limit = float(time_limit)
    time_limit = timedelta(hours=time_limit)
    max_score = scores[0] if scores else 1
    for event in main_events:
        if condition == "before":
            start_time = event["begin_time"] - time_limit
            end_time = event["begin_time"] - time_limit // 2
        elif condition == "after":
            start_time = event["end_time"] + time_limit // 2
            end_time = event["end_time"] + time_limit

        extra_filter_scripts.append(create_time_range_query(
            start_time.timestamp(), end_time.timestamp()))

    print("Results:", len(main_events))
    print("Time:", timecounter.time() - start, "seconds.")
    print("-" * 80)
    print(f"Conditional({condition}) with {time_limit}")
    start = timecounter.time()
    conditional_query, conditional_events = msearch(
        conditional_query, gps_bounds=gps_bounds[1], extra_filter_scripts=extra_filter_scripts)

    print("Time:", timecounter.time()-start, "seconds.")
    return query, conditional_query, main_events, conditional_events, scores


def add_pairs(main_events, conditional_events, condition, time_limit, scores, already_done=None, reverse=False):
    if not main_events or not conditional_events:
        return [], None
    pair_events = []
    if not already_done:
        already_done = set()
    print(len(main_events), len(scores), len(conditional_events))
    for i, main_event in enumerate(main_events):
        s1 = scores[i]
        cond_events, cond_scores = format_func(
            conditional_events[i], factor="scene")
        for conditional_event, s2 in zip(cond_events, cond_scores):
            if reverse:
                if conditional_event["scene"] not in already_done:
                    pair_event = conditional_event.copy()
                    pair_event[condition] = main_event["current"]
                    pair_event[f"location_{condition}"] = main_event["location"]
                    pair_events.append((pair_event, s2, s1))
                    already_done.add(
                        conditional_event["scene"])
            else:
                if main_event["scene"] not in already_done:
                    pair_event = main_event.copy()
                    pair_event[condition] = conditional_event["current"]
                    pair_event[f"location_{condition}"] = conditional_event["location"]
                    pair_events.append((pair_event, s1, s2))
                    already_done.add(
                        main_event["scene"])
    return pair_events, already_done


def es_two_events(query, conditional_query, condition, time_limit, gps_bounds, return_extra_filter=False, share_info=False):
    print("Share info:", share_info)
    global multiple_pairs
    if not time_limit:
        time_limit = "1"
    else:
        time_limit = time_limit.strip("h")

    if isinstance(query, str):
        query = Query(query)
    if isinstance(conditional_query, str):
        conditional_query = Query(conditional_query, shared_filters=query)

    # Forward search
    print("Forward Search")
    query, conditional_query, main_events, conditional_events, scores = forward_search(
        query, conditional_query, condition, time_limit, gps_bounds=[gps_bounds, gps_bounds if share_info else None])

    max_score1 = scores[0]
    pair_events, already_done = add_pairs(main_events, conditional_events,
                                          condition, time_limit, scores)

    # print("Backward Search")
    conditional_query, query, conditional_events, main_events, scores = forward_search(
        conditional_query, query, "before" if condition == "after" else "after",
        time_limit, gps_bounds=[gps_bounds if share_info else None, gps_bounds])

    max_score2 = scores[0]
    pair_events2, _ = add_pairs(conditional_events, main_events, condition,
                                time_limit, scores, already_done=already_done, reverse=True)

    pair_events.extend(pair_events2)
    pair_events = sorted(pair_events, key=lambda x: -x[1]/max_score1 - x[2]/max_score2)

    total_scores = [(s1, s2) for (event, s1, s2) in pair_events]
    pair_events = [event for (event, s1, s2) in pair_events]
    print("Max Scores:", max_score1, max_score2)
    print("Pairs:", len(pair_events))
    multiple_pairs = {"position": 24,
                      "pairs": pair_events,
                      "total_scores": total_scores}
    print(pair_events[0])
    return query, conditional_query, (pair_events[:24], total_scores[:24]), "pairs"


def es_three_events(query, before, beforewhen, after, afterwhen, gps_bounds, share_info=False):
    global multiple_pairs
    if not afterwhen:
        afterwhen = "1"
    else:
        afterwhen = afterwhen.strip('h')
    if not beforewhen:
        beforewhen = "1"
    else:
        beforewhen = beforewhen.strip('h')

    query = Query(query)
    before_query = Query(before, shared_filters=query)
    after_query = Query(after, shared_filters=query)

    query, before_query, * \
        _ = es_two_events(query, before, "before", beforewhen,
                          gps_bounds, share_info=share_info)
    print("-" * 80)
    print("Search for after events")
    before_pairs = multiple_pairs["pairs"]
    total_scores = multiple_pairs["total_scores"]

    extra_filter_scripts = []
    time_limit = float(afterwhen)
    time_limit = timedelta(hours=time_limit)
    for time_group, score in zip(before_pairs, total_scores):
        start_time = time_group["end_time"]
        end_time = time_group["end_time"] + time_limit
        extra_filter_scripts.append(create_time_range_query(
            start_time.timestamp(), end_time.timestamp()))

    after_query, after_events = msearch(
        after_query, gps_bounds=gps_bound if share_info else None, extra_filter_scripts=extra_filter_scripts)

    pair_events = []
    max_score1 = max([s1 for s1, s2 in total_scores])
    max_score2 = max([s2 for s1, s2 in total_scores])
    for i, before_pair in enumerate(before_pairs):
        s1, s2 = total_scores[i]
        cond_events, cond_scores = format_func(
            after_events[i], factor="scene")
        for conditional_event, s3 in zip(cond_events, cond_scores):
            pair_event = before_pair.copy()
            pair_event["after"] = conditional_event["current"]
            pair_event["location_after"] = conditional_event["location"]
            pair_events.append((pair_event, s1, s2, s3))

    after_query, (_, scores), _ = individual_es(
        after_query, gps_bounds=gps_bound if share_info else None, size=1, group_factor="scene")
    max_score3 = scores[0]

    pair_events = sorted([(event, s1/max_score1 + s2/max_score2 + s3/max_score3)
                          for (event, s1, s2, s3) in pair_events], key=lambda x: -x[1])
    total_scores = [s for event, s in pair_events]
    pair_events = [event for event, s in pair_events]
    print("Pairs:", len(pair_events))

    multiple_pairs = {"position": 24,
                      "pairs": pair_events,
                      "total_scores": total_scores}
    return query, before_query, after_query, (pair_events[:24], []), "pairs"
