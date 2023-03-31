from .query_types import *
from .timeline import time_info
from .utils import *
from ..nlp_utils.extract_info import Query
from datetime import timedelta, datetime
import time as timecounter
from collections import defaultdict
import copy
import pandas as pd
import numpy as np
from .common_nn import *
from sklearn.cluster import OPTICS
cluster = OPTICS(min_samples=2, max_eps=0.9, metric='cosine')

multiple_pairs = {}
INCLUDE_SCENE = ["scene"]
INCLUDE_FULL_SCENE = ["images", "start_time", "end_time", "gps",
                      "scene", "group", "timestamp", "location", "cluster_images", "weights", "country"]
INCLUDE_IMAGE = ["image_path", "time", "gps", "scene", "group", "location"]

cached_queries = None
cached_filters =  {"bool": {"filter": [],
                               "should": [],
                               "must": {"match_all": {}},
                               "must_not": []}}
cached_embeddings = None

# format_func = format_single_result # ntcir
format_func = group_scene_results
INDEX = "lsc2023"
SCENE_INDEX = "lsc2023_scene_mean"

def format_results(K, scene_results, embedding):
    (scene_results, scores), cluster_scores = remove_duplications(scene_results)
    scene_results, scores = group_scene_results(list(zip(scene_results, scores)), "group")
    final_scenes = []
    if scene_results:
        final_scenes = organise_based_on_similarities(K, embedding, scene_results, cluster_scores)
    return final_scenes, scores

def query_all(query, includes, index, scroll, K, timestamp="timestamp"):
    request = {
        "size": 200,
        "_source": {
            "includes": includes
        },
        "query": {"match_all": {}},
        "sort": [
            {timestamp: {
                "order": "asc"
            }}
        ]
    }
    scene_results, scroll_id = post_request(json.dumps(request), index, scroll=scroll)
    final_scenes, scores = format_results(K, scene_results, None)
    return query, (final_scenes, scores), scroll_id


def es_more(scroll_id, size=200, K=4):
    global multiple_pairs
    if scroll_id == 'pairs':
        position = multiple_pairs["position"]
        new_position = min(position + 24, len(multiple_pairs["pairs"]))
        last_results = multiple_pairs["pairs"][position: new_position]
        multiple_pairs["position"] = new_position
        return scroll_id, last_results, []
    if scroll_id:
        start = timecounter.time()
        response = requests.post(
            f"http://localhost:9200/_search/scroll", headers={"Content-Type": "application/json"},
            data=json.dumps({"scroll": "5m",
                            "scroll_id": scroll_id}))

        assert response.status_code == 200, "Wrong request"
        
        response_json = response.json()  # Convert to json as dict formatted
        scene_results = [[d["_source"], d["_score"]]
                for d in response_json["hits"]["hits"]]
        scroll_id = response_json["_scroll_id"]
        global cached_embeddings
        final_scenes, scores = format_results(K, scene_results, cached_embeddings)
        return scroll_id, final_scenes, scores
    return None, [], []


def es(query, gps_bounds, size, share_info):
    start = timecounter.time()
    query_info = {}
    scroll_id = None
    if query["before"] and query["after"]:
        query, before_query, after_query, (results, scores), scroll_id = es_three_events(
            query["current"], query["before"], query["beforewhen"], query["after"], query["afterwhen"], gps_bounds, share_info, K=2)
        query_info = query.get_info()
        cond_query_info = before_query.get_info()
        for key in cond_query_info:
            query_info[key].extend(cond_query_info[key])
        cond_query_info = after_query.get_info()
        for key in cond_query_info:
            query_info[key].extend(cond_query_info[key])
    elif query["before"]:
        query, cond_query, (results, scores), scroll_id = es_two_events(
            query["current"], query["before"], "before", query["beforewhen"], gps_bounds, share_info, K=2)
        query_info = query.get_info()
        cond_query_info = cond_query.get_info()
        for key in cond_query_info:
            query_info[key].extend(cond_query_info[key])
    elif query["after"]:
        query, cond_query, (results, scores), scroll_id = es_two_events(
            query["current"], query["after"], "after", query["afterwhen"], gps_bounds, share_info, K=2)
        query_info = query.get_info()
        cond_query_info = cond_query.get_info()
        for key in cond_query_info:
            query_info[key].extend(cond_query_info[key])
    else:
        query, (results, scores), scroll_id = individual_es(
            query["current"], gps_bounds, size, scroll=True, K=4)
        query_info = query.get_info()
    print(f"TOTAL TIMES: {(timecounter.time() - start):0.4f} seconds.")
    return scroll_id, results, scores, query_info


def query_list(query_list):
    return query_list[0] if len(query_list) == 1 else query_list


def get_json_query(must_queries, should_queries, filter_queries, size, includes, min_score=0, timestamp="timestamp"):
    # CONSTRUCT JSON
    main_query = {}
    if must_queries:
        main_query["must"] = query_list(must_queries)

    if should_queries:
        main_query["should"] = query_list(should_queries)
        main_query["minimum_should_match"] = 1

    # if filter_queries["bool"]["filter"] or filter_queries["bool"]["should"]:
    #     # if "should" in filter_queries["bool"] and filter_queries["bool"]["should"]:
    #     #     filter_queries["bool"]["minimum_should_match"] = 1
    #     main_query["filter"] = filter_queries
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
            {timestamp: {
                "order": "asc"
            }}
        ]
    }
    if min_score:
        json_query["min_score"] = min_score
    return json_query


def get_neighbors(image, lsc):
    if lsc:
        global cached_filters
        filter_queries = copy.deepcopy(cached_filters)
    else:
        filter_queries = []

    should_queries = {
            "elastiknn_nearest_neighbors": {
                "field": "clip_vector",                # 1
                "vec": {                               # 2
                    "index": INDEX,
                    "field": "clip_vector",
                    "id": image
                },
                "model": "permutation_lsh",            # 3
                "similarity": "cosine",                # 4
                "candidates": 1000                   # 5
            }
        }

    json_query = get_json_query([should_queries], [], filter_queries, 40,
                includes=["image_path", "group", "location", "weekday", "time"], timestamp="timestamp")

    results, _ = post_request(json.dumps(json_query), INDEX)
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


def extra_info(key, scene):
    scene["gps"] = get_gps(scene[key])
    # scene["location"] = scene["location"] + "\n" + scene["country"]
    return scene

def remove_duplications(scene_results):
    cluster_scores = defaultdict(lambda: defaultdict(float))
    new_scenes = []
    new_scores = {}
    for scene_info, score in scene_results:
        if score is None:
            score = 0.0
        scene = scene_info["scene"]
        group = scene_info["group"]
        if scene not in new_scores:
            new_scenes.append(scene_info)
            new_scores[scene] = score
        if "weights" in scene_info and scene_info["weights"] is not None:
            for image, weight in zip(scene_info["cluster_images"], scene_info["weights"]):
                cluster_scores[group][image] = max(cluster_scores[group][image], score * weight)
        else:
            for image in scene_info["cluster_images"]:
                cluster_scores[group][image] = max(cluster_scores[group][image], score)
                
    scene_results = sorted(new_scenes, key=lambda x: -new_scores[x["scene"]])
    scores = sorted(new_scores.values(), reverse=True)
    return (scene_results, scores), cluster_scores

def arrange_scene(K, scene, weights, key):
    best = np.argmax(weights)
                # Left side:
    topk = min(best, K)
    left_ind = np.argpartition(weights[:best], -topk)[-topk:].tolist()
    topk = min(len(weights[best:]), K)
    right_ind = [best + i for i in np.argpartition(weights[best:], -topk)[-topk:].tolist()]
    new_images = []
    for j, (image, weight) in enumerate(zip(scene[key], weights)):
        if j == best:
            new_best = len(new_images)
            new_images.append((image, weight, True))
        elif j in left_ind or j in right_ind:
            new_images.append((image, weight, False))
                
    best = new_best
                # max_side_images = max(best, len(new_images) - best - 1)
    max_side_images = K
    if best < max_side_images:
                    # print(i, best, len(new_images), max_side_images, "added", (max_side_images - best), "to the left")
        new_images = [("", 0, False)] * (max_side_images - best) + new_images
    best = max_side_images
    if (max_side_images - len(new_images) + best + 1) > 0:
                    # print(i, best, len(new_images), max_side_images, "added", max_side_images - len(new_images) + best , "to the right")
        new_images = new_images + [("", 0, False)] * (max_side_images - len(new_images) + best + 1) 
    return new_images


def organise_based_on_similarities(K, embedding, scene_results, cluster_scores):
    key = "current" if "current" in scene_results[0] else "images"
    images = [image for scene in scene_results for image in scene[key]]
    if embedding is None:
        image_scores = defaultdict(lambda: 1)
    else:
        image_scores = {image: score for image, score in zip(images, score_images(images, embedding))}

    final_scenes = []
    for i, scene in enumerate(scene_results):
        # Extra information
        scene = extra_info(key, scene)
        
        # Arrange
        cluster_score = cluster_scores[scene["group"]]
        logit_scale = len(scene[key]) ** 0.5
            
        # Filter
        weights = softmax(np.array([image_scores[image] * logit_scale * cluster_score[image]
                            for image in scene[key]])).round(2)
        scene[key] = arrange_scene(K, scene, weights, key)
        if key != "current":
            scene["current"] = scene[key]
            del scene[key]
        final_scenes.append(scene)
    return final_scenes


def individual_es(query, gps_bounds, size, scroll, K):
    start = timecounter.time()
    if isinstance(query, str):
        query = Query(query)
    print(f"Parse query: {(timecounter.time() - start):0.4f} seconds.")

    if not query.original_text and not gps_bounds:
        return query_all(query, INCLUDE_FULL_SCENE, SCENE_INDEX, scroll, K, timestamp="start_timestamp")
    return construct_es(query, gps_bounds, size, scroll, K)


def construct_es(query, gps_bounds, size, scroll, K):
    start = timecounter.time()
    time_filters, date_filters = query.time_to_filters()
    must_queries = []
    # !TODO
    should_queries = []

    filter_queries = {"bool": {"filter": [],
                               "should": []}}

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
        filter_queries["bool"]["filter"].append({"terms": {"region": query.regions}})
    if query.weekdays:
        filter_queries["bool"]["filter"].append(
            {"terms": {"weekday": query.weekdays}})

    if time_filters:
        if query.start[0] != 0 or query.start[1] != 0 or query.end[0] != 24 or query.end[1] != 0:
            filter_queries["bool"]["filter"].append(time_filters)

    if date_filters:
        filter_queries["bool"]["filter"].append(date_filters)

    if gps_bounds:
        filter_queries["bool"]["filter"].append(get_gps_filter(gps_bounds))
    clip_script = None
    if query.clip_text:
        query.clip_embedding = encode_query(query.clip_text)
        if scroll:
            global cached_embeddings
            cached_embeddings = query.clip_embedding
        clip_script = {
            "elastiknn_nearest_neighbors": {
                "field": "clip_vector",                     # 1
                "vec": {                               # 2
                    "values": query.clip_embedding.tolist()[0]
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

    global cached_queries
    cached_queries = (must_queries, should_queries)
    print(f"Create ElasticSearch query: {(timecounter.time() - start):0.4f} seconds.")
    
    # CONSTRUCT JSON
    scroll_id = None
    if filter_queries["bool"]["filter"] or filter_queries["bool"]["should"]:
        must_queries.append(filter_queries)
    json_query = get_json_query(must_queries, should_queries, [], size, includes=INCLUDE_FULL_SCENE,
                                timestamp="start_timestamp")
    start = timecounter.time()
    scene_results, scroll_id = post_request(json.dumps(json_query), SCENE_INDEX, scroll=scroll)
    if scene_results:
        print("Max score: ", scene_results[0][1])
        print("Min score: ", scene_results[-1][1])
    print(f"Search ElasticSearch: {(timecounter.time() - start):0.4f} seconds.")

    start = timecounter.time()
    final_scenes, scores = format_results(K, scene_results, query.clip_embedding)
    
    print(f"Format results: {(timecounter.time() - start):0.4f} seconds.")
    return query, (final_scenes, scores), scroll_id


def msearch(query, gps_bounds, extra_filter_scripts):
    if isinstance(query, str):
        query = Query(query)

    if not query.original_text and not gps_bounds:
        return query_all(query, INCLUDE_FULL_SCENE, SCENE_INDEX, timestamp="start_timestamp")

    time_filters, date_filters = query.time_to_filters()
    must_queries = []
    # !TODO
    should_queries = []
    filter_queries = {"bool": {"filter": [],
                               "should": []}}
    min_score = 0.0
    if query.locations:
        should_queries.append(
            {"match": {"location": {"query": " ".join(query.locations), "boost": 0.01}}})
        location_filters = query.make_location_query()
        # if location_query:
            # should_queries.append(location_query)
        filter_queries["bool"]["should"].extend(location_filters)
        filter_queries["bool"]["should"].append({"match": {"location": {"query": " ".join(query.locations), "boost": 0.01}}})
        min_score += 0.01

    # FILTERS
    if query.regions:
        filter_queries["bool"]["filter"].append({"terms": {"region": query.regions}})

    if query.weekdays:
        filter_queries["bool"]["filter"].append(
            {"terms": {"weekday": query.weekdays}})

    if time_filters:
        if query.start[0] != 0 or query.start[1] != 0 or query.end[0] != 24 or query.end[1] != 0:
            filter_queries["bool"]["filter"].append(time_filters)

    if date_filters:
        filter_queries["bool"]["filter"].append(date_filters)

    if gps_bounds:
        filter_queries["bool"]["filter"].append(get_gps_filter(gps_bounds))

    clip_script = None
    if query.clip_text:
        query.clip_embedding = encode_query(query.clip_text)
        clip_script = {
            "elastiknn_nearest_neighbors": {
                "field": "clip_vector",                # 1
                "vec": {                               # 2
                    "values": query.clip_embedding.tolist()[0]
                },
                "model": "exact",            # 3
                "similarity": "cosine",                # 4
                "candidates": 1000                       # 5
            }
        }
        should_queries.append(clip_script)
        min_score += 1.2
    
    if filter_queries["bool"]["filter"] or filter_queries["bool"]["should"]:
        must_queries.append(filter_queries)
    mquery = []
    print("Min Score: ", min_score)
    for script in extra_filter_scripts:
        # new_filter_queries = copy.deepcopy(filter_queries)
        new_filter_queries = script
        mquery.append(json.dumps({}))
        mquery.append(json.dumps(get_json_query(
            must_queries, should_queries, new_filter_queries, 1, INCLUDE_FULL_SCENE, min_score=min_score, timestamp="start_timestamp")))

    results = post_mrequest("\n".join(mquery) + "\n", SCENE_INDEX)
    return query, results


def forward_search(query, conditional_query, condition, time_limit, gps_bounds, K, reverse=False):
    print("-" * 80)
    print("Main")
    start = timecounter.time()
    query, (main_events, main_scores), _ = individual_es(
        query, gps_bounds[0], size=200, scroll=False, K=K-1 if reverse else K)

    extra_filter_scripts = []
    time_limit = float(time_limit)
    leeway = timedelta(hours=time_limit - 1)
    time_limit = timedelta(hours=time_limit)
    for event in main_events:
        if condition == "before":
            start_time = event["start_time"] - time_limit 
            end_time = event["start_time"] - leeway
        elif condition == "after":
            start_time = event["end_time"] + leeway
            end_time = event["end_time"] + time_limit

        extra_filter_scripts.append(create_time_range_query(
            start_time.timestamp(), end_time.timestamp(), condition=condition))
    print("Results:", len(main_events))
    print("Time:", timecounter.time() - start, "seconds.")
    print("-" * 80)
    print(f"Conditional({condition}) with {time_limit}")
    start = timecounter.time()
    conditional_query, conditional_events = msearch(
        conditional_query, gps_bounds[1], extra_filter_scripts)

    images = []
    for events in conditional_events:
        for event, score in events:
            images.extend(event["images"])
        
    # Weigh images
    if conditional_query.clip_embedding is None or len(images) == 0:
        image_scores = defaultdict(lambda: 1)
    else:
        image_scores = {image: score for image, score in zip(images, score_images(images, conditional_query.clip_embedding))}
    
    new_events = []
    key = 'current'
    for events in conditional_events:
        cond_events, cond_scores = group_scene_results(events, 'group')
        new_scenes = []
        for scene, cluster_score in zip(cond_events, cond_scores):
            # Extra information
            scene = extra_info(key, scene)
            logit_scale = len(scene[key]) ** 0.5
            # Filter
            weights = softmax(np.array([image_scores[image] * logit_scale * cluster_score
                            for image in scene[key]])).round(2)
            scene[key] = arrange_scene(K if reverse else K-1, scene, weights, key)
            new_scenes.append((scene, cluster_score))
        new_events.append(new_scenes)
    conditional_events = new_events
    print("Time:", timecounter.time()-start, "seconds.")
    return query, conditional_query, main_events, conditional_events, main_scores


def add_pairs(main_events, conditional_events, condition, scores, already_done=None, reverse=False):
    if not main_events or not conditional_events:
        return [], None
    pair_events = []
    if not already_done:
        already_done = set()
    print(len(main_events), len(scores), len(conditional_events))
    factor = "group"
    for i, main_event in enumerate(main_events):
        try:
            s1 = scores[i]
        except Exception as e:
            print("Scores", scores)
            raise(e)
        for conditional_event, s2 in conditional_events[i]:
            if reverse:
                if conditional_event[factor] not in already_done:
                    pair_event = conditional_event.copy()
                    pair_event[condition] = main_event["current"]
                    pair_event[f"location_{condition}"] = main_event["location"]
                    pair_events.append((pair_event, s2, s1))
                    already_done.add(
                        conditional_event[factor])
            else:
                if main_event[factor] not in already_done:
                    pair_event = main_event.copy()
                    pair_event[condition] = conditional_event["current"]
                    pair_event[f"location_{condition}"] = conditional_event["location"]
                    pair_events.append((pair_event, s1, s2))
                    already_done.add(
                        main_event[factor])
    return pair_events, already_done


def es_two_events(query, conditional_query, condition, time_limit, gps_bounds, share_info, K):
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
        query, conditional_query, condition, time_limit, gps_bounds=[gps_bounds, gps_bounds if share_info else None], K=K)
    max_score1 = scores[0]
    pair_events, already_done = add_pairs(main_events, conditional_events,
                                          condition, scores)

    print("Backward Search")
    conditional_query, query, conditional_events, main_events, scores = forward_search(
        conditional_query, query, "before" if condition == "after" else "after",
        time_limit, gps_bounds=[gps_bounds if share_info else None, gps_bounds], K=K, reverse=True)

    max_score2 = scores[0]
    pair_events2, _ = add_pairs(conditional_events, main_events, condition,
                                scores, already_done=already_done, reverse=True)

    pair_events.extend(pair_events2)
    pair_events = sorted(pair_events, key=lambda x: -x[1]/max_score1 - x[2]/max_score2 * 0.5)

    total_scores = [(s1, s2) for (event, s1, s2) in pair_events]
    pair_events = [event for (event, s1, s2) in pair_events]
    print("Max Scores:", max_score1, max_score2)
    print("Pairs:", len(pair_events))
    multiple_pairs = {"position": 24,
                      "pairs": pair_events,
                      "total_scores": total_scores}
    return query, conditional_query, (pair_events[:24], total_scores[:24]), "pairs"


def es_three_events(query, before, beforewhen, after, afterwhen, gps_bounds, share_info, K):
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
                          gps_bounds, share_info=share_info, K=K)
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
            start_time.timestamp(), end_time.timestamp(), condition="after"))

    after_query, after_events = msearch(
        after_query, gps_bounds if share_info else None, extra_filter_scripts)

    images = []
    for events in after_events:
        for event, score in events:
            images.extend(event["images"])
        
    # Weigh images
    if after_query.clip_text is None:
        image_scores = defaultdict(lambda: 1)
    else:
        image_scores = {image: score for image, score in zip(images, score_images(images, after_query.clip_embedding))}

    key = 'current'
    new_events = []
    for events in after_events:
        cond_events, cond_scores = group_scene_results(events, 'group')
        new_scenes = []
        for scene, cluster_score in zip(cond_events, cond_scores):
            scene["gps"] = get_gps(scene[key])
            logit_scale = len(scene[key]) ** 0.5
            # Filter
            weights = softmax(np.array([image_scores[image] * logit_scale * cluster_score
                            for image in scene[key]])).round(2)
            scene[key] = arrange_scene(K-1, scene, weights, key) 
            new_scenes.append((scene, cluster_score))
        new_events.append(new_scenes)
    after_events = new_events
        
    pair_events = []
    max_score1 = max([s1 for s1, s2 in total_scores])
    max_score2 = max([s2 for s1, s2 in total_scores])
    for i, before_pair in enumerate(before_pairs):
        s1, s2 = total_scores[i]
        for conditional_event, s3 in after_events[i]:
            pair_event = before_pair.copy()
            pair_event["after"] = conditional_event["current"]
            pair_event["location_after"] = conditional_event["location"]
            pair_events.append((pair_event, s1, s2, s3))

    after_query, (_, scores), _ = individual_es(
        after_query, gps_bounds=gps_bounds if share_info else None, size=1, scroll=False, K=1)
    max_score3 = scores[0]

    pair_events = sorted([(event, s1/max_score1 + s2/max_score2 * 0.5 + s3/max_score3 * 0.5)
                          for (event, s1, s2, s3) in pair_events], key=lambda x: -x[1])
    total_scores = [s for event, s in pair_events]
    pair_events = [event for event, s in pair_events]
    print("Pairs:", len(pair_events))

    multiple_pairs = {"position": 24,
                      "pairs": pair_events,
                      "total_scores": total_scores}
    return query, before_query, after_query, (pair_events[:24], []), "pairs"
