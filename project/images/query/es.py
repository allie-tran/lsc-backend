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
                      "scene", "group", "timestamp", "location", "cluster_images", "weights", "country",
                      "ocr", "country", "location_info", "duration"]
INCLUDE_IMAGE = ["image_path", "time", "gps", "scene", "group", "location"]

cached_filters = []
cached_query = None

# format_func = format_single_result # ntcir
format_func = group_scene_results
INDEX = "lsc2023"
SCENE_INDEX = "lsc2023_scene_mean"
CLIP_MIN_SCORE = 1.2 # 1.2 is the normal score, 1.0 is for Transf
delete_scroll_id("_all")

def format_results(K, scene_results, embedding, group_factor):
    (scene_results, scores), cluster_scores = remove_duplications(scene_results, group_factor)
    scene_results, scores = group_scene_results(list(zip(scene_results, scores)), group_factor)
    print("Top scores:", scores[:10])
    final_scenes = []
    if scene_results:
        final_scenes = organise_based_on_similarities(K, embedding, scene_results, cluster_scores, group_factor)
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
    scene_results, scroll_id, _ = post_request(json.dumps(request), index, scroll=scroll)
    final_scenes, scores = format_results(K, scene_results, None, group_factor="group")
    return query, (final_scenes, scores), scroll_id

def es_more(scroll_id, size=200, K=4):
    if scroll_id in ['2pairs', '3pairs']:
        global multiple_pairs
        position = multiple_pairs["position"]
        if len(multiple_pairs["pairs"]) == 0:
            return None, [], []
        if position >= len(multiple_pairs["pairs"]):
            print("Geting more results froms scroll_ids")
            if scroll_id == "2pairs":
                query, conditional_query = multiple_pairs["cached_queries"]
                *_, (last_results, last_scores ), _ = es_two_events(query, conditional_query, 
                                                                    multiple_pairs["condition"], 
                                                                    multiple_pairs["time_limit"], 
                                                                    multiple_pairs["gps_bounds"],
                                                                    multiple_pairs["share_info"],
                                                                    multiple_pairs["K"])
            else:
                query, before, after = multiple_pairs["cached_queries"]
                *_, (last_results, last_scores ), _ = es_three_events(query, before, multiple_pairs["beforewhen"],
                                                                      after, multiple_pairs["afterwhen"],
                                                                      multiple_pairs["gps_bounds"],
                                                                      multiple_pairs["share_info"],
                                                                      multiple_pairs["K"])
        else:
            new_position = min(position + 24, len(multiple_pairs["pairs"]))
            last_results = multiple_pairs["pairs"][position: new_position]
            last_scores = multiple_pairs["total_scores"][position: new_position]
            multiple_pairs["position"] = new_position
        return scroll_id, last_results, last_scores
    elif scroll_id:
        global cached_query
        scene_results, scroll_id = get_scroll_request(cached_query.scroll_id)
        if scroll_id != cached_query.scroll_id:
            delete_scroll_id(cached_query.scroll_id)
            cached_query.scroll_id = scroll_id
        final_scenes, scores = format_results(K, scene_results, cached_query.clip_embedding, group_factor="group")
        return scroll_id, final_scenes, scores
    return None, [], []

import re
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# Remove ending words that are stop words from clip_text
def strip_stopwords(sentence):
    if sentence:
        words = sentence.split()
        for i in reversed(range(len(words))):
            if words[i].lower() in stop_words:
                words.pop(i)
            else:
                break
        return " ".join(words)
    return ""
import string

def extract_phrases(text):
    text = text.lstrip(',. ')
    info = {"current": text, "beforewhen": "", "afterwhen": ""}
    # Extract the matched groups
    if "before" in text or "after" in text:
        last_before = text.rfind("before")
        last_after = text.rfind("after")

        if last_before > last_after:
            remain = text[:last_before]
            phrase = text[last_before + 6:]
            actual_phrase = phrase
            rest = ""
            for punct in string.punctuation:
                if punct in phrase:
                    actual_phrase = phrase[:phrase.find(punct)]
                    rest = phrase[phrase.find(punct):]
                    break
                    
            remain += rest
            info["after"] = strip_stopwords(actual_phrase)
            remain = strip_stopwords(remain)
            info["current"] = remain
            if remain:
                info.update(extract_phrases(remain))
        else:
            remain = text[:last_after]
            phrase = text[last_after + 5:]
            actual_phrase = phrase
            rest = ""
            for punct in string.punctuation:
                if punct in phrase:
                    actual_phrase = phrase[:phrase.find(punct)]
                    rest = phrase[phrase.find(punct):]
                    break
            remain += rest
            info["before"] = strip_stopwords(actual_phrase)
            remain = strip_stopwords(remain)
            info["current"] = remain
            
            if remain:
                info.update(extract_phrases(remain))
    return info

import re
def extract_phrase(text):
    # Define the regular expression pattern
    pattern = r'after\s+(.*?)[,\.]'

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # Extract the matched group
    if match:
        phrase = match.group(1)
        remaining_text = re.sub(pattern, '', text)
        return (phrase, remaining_text.strip())
    else:
        return None
    
def es(query, gps_bounds, size, share_info, isQuestion=False):
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
    elif query["before"] or query["after"]:
        time_key = "before" if query["before"] else "after"
        query, cond_query, (results, scores), scroll_id = es_two_events(
            query["current"], query[time_key], time_key, query[f"{time_key}when"], gps_bounds, share_info, K=2)
        query_info = query.get_info()
        cond_query_info = cond_query.get_info()
        for key in cond_query_info:
            query_info[key].extend(cond_query_info[key])
    else:
        query["current"] = query["current"].lower()
        # if "before" in query["current"] or "after" in query["current"]:
        #     new_query = extract_phrases(query["current"].lower())
        #     print("New query:", new_query)
        #     if "after" not in new_query:
        #         new_query["after"] = ""
        #     if "before" not in new_query:
        #         new_query["before"] = ""
        #     return es(new_query, gps_bounds, size, share_info)
        # Normal query
        query, (results, scores), scroll_id = individual_es(
            query["current"], gps_bounds, size, scroll=True, K=4, ignore_limit_score=False, cache=True)
        query_info = query.get_info()
    print(f"TOTAL TIMES: {(timecounter.time() - start):0.4f} seconds.")
    return scroll_id, results, scores, query_info


def query_list(query_list):
    return query_list[0] if len(query_list) == 1 else query_list


def get_json_query(must_queries, 
                   should_queries, 
                   filter_queries, 
                   size, 
                   includes, min_score=0, timestamp="timestamp"):
    # CONSTRUCT JSON
    main_query = {}
    if must_queries:
        main_query["must"] = query_list(must_queries)

    if should_queries:
        main_query["should"] = query_list(should_queries)
        main_query["minimum_should_match"] = 1
    
    if filter_queries:
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
    if size == 1:
        json_query["aggs"] = {
            "score_stats": {
                "extended_stats": {
                    "script" : "_score",
                    "sigma": 2.0
                }
            }
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

    results, *_ = post_request(json.dumps(json_query), INDEX)
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

def remove_duplications(scene_results, group_factor):
    cluster_scores = defaultdict(lambda: defaultdict(float))
    new_scenes = []
    new_scores = {}
    for scene_info, score in scene_results:
        if score is None:
            score = 0.0
        scene = scene_info["scene"]
        group = scene_info[group_factor]
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

def arrange_scene(K, scene, weights, key, uniform=False):
    if len(weights) > 2:
        best = 2 + np.argmax(weights[2:])
    else:
        best = np.argmax(weights)
    # Left side:
    topk = min(best, K)
    left_ind = np.argpartition(weights[:best], -topk)[-topk:].tolist()
    topk = min(len(weights[best + 1:]), K)
    right_ind = [best + 1 + i for i in np.argpartition(weights[best + 1:], -topk)[-topk:].tolist()]
    new_images = []
    
    for j, (image, weight) in enumerate(zip(scene[key], weights.round(6))):
        info = get_image_time_info(image)
        if j == best:
            new_best = len(new_images)
            new_images.append((image, info, True))
        if weight == 0:
            continue
        elif j in left_ind or j in right_ind:
            new_images.append((image, info, uniform))
                
    best = new_best
                # max_side_images = max(best, len(new_images) - best - 1)
    max_side_images = K
    if best < max_side_images:
                    # print(i, best, len(new_images), max_side_images, "added", (max_side_images - best), "to the left")
        new_images = [("", 0, uniform)] * (max_side_images - best) + new_images
    best = max_side_images
    if (max_side_images - len(new_images) + best + 1) > 0:
                    # print(i, best, len(new_images), max_side_images, "added", max_side_images - len(new_images) + best , "to the right")
        new_images = new_images + [("", 0, uniform)] * (max_side_images - len(new_images) + best + 1) 
    return new_images

@cache
def get_image_time_info(image):
    info = datetime.strptime(get_dict(image)["time"], "%Y/%m/%d %H:%M:%S%z")
    info = info.strftime("%I:%M%p")
    return info


def organise_based_on_similarities(K, embedding, scene_results, cluster_scores, group_factor):
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
        cluster_score = cluster_scores[scene[group_factor]]
        logit_scale = len(scene[key]) ** 0.5
        
        if "201904/19/20190419_215250_000.jpg" in scene[key]:
            print(i)
            print(cluster_score["201904/19/20190419_215738_000.jpg"],
                  image_scores["201904/19/20190419_215738_000.jpg"])
            print(cluster_score["201904/19/20190419_215250_000.jpg"],
                  image_scores["201904/19/20190419_215250_000.jpg"])
            print(sorted(scene[key], key=lambda x: image_scores[x], reverse=True).index("201904/19/20190419_215250_000.jpg"))
        # Filter
        image_final_scores = np.array([(image_scores[image] + 1) * logit_scale * cluster_score[image]
                            for image in scene[key]])
        weights = softmax(image_final_scores)
        weights[image_final_scores < 0] = 0
        
        scene[key] = arrange_scene(K, scene, weights, key)
        if key != "current":
            scene["current"] = scene[key]
            del scene[key]
        final_scenes.append(scene)
    return final_scenes


def individual_es(query, gps_bounds, size, scroll, K, ignore_limit_score, cache):
    start = timecounter.time()
    if isinstance(query, str):
        query = Query(query)
    print(f"Parse query: {(timecounter.time() - start):0.4f} seconds.")

    if not query.original_text and not gps_bounds:
        return query_all(query, INCLUDE_FULL_SCENE, SCENE_INDEX, scroll, K, timestamp="start_timestamp")
    return construct_es(query, gps_bounds, size, scroll, K, ignore_limit_score, cache)

def construct_es(query, gps_bounds, size, scroll, K, ignore_limit_score, cache):
    scene_results, scroll_id = [], ""
    # Purpose: get scene_results and (maybe new) scroll_id
    if query.scroll_id:
        scene_results, scroll_id = get_scroll_request(query.scroll_id)
        # Update scroll_id
        if scroll_id != query.scroll_id:
            delete_scroll_id(query.scroll_id)
            query.scroll_id = scroll_id
    else: 
        if query.cached:
            should_queries, filter_queries = query.es_should, query.es_filters
            min_score, max_score = query.min_score, query.max_score
        else:
            should_queries, filter_queries, min_score, max_score = get_es_queries(query, gps_bounds, ignore_limit_score)
            
        # Construct json request and post to ElasticSearch
        scroll_id = None
        json_query = get_json_query([], should_queries, filter_queries, size, includes=INCLUDE_FULL_SCENE, 
                                    min_score=min_score,
                                    timestamp="start_timestamp")
        scene_results, scroll_id, _ = post_request(json.dumps(json_query), SCENE_INDEX, scroll=scroll)
        print("Raw results:", len(scene_results))
        # Cache/update the queries
        query.es_filters = filter_queries
        query.es_should = should_queries
        query.cached = True
        query.scroll_id = scroll_id
        query.min_score = min_score
        query.max_score = max_score
    
    if cache:
        global cached_query
        cached_query = query

    final_scenes, scores = format_results(K, scene_results, query.clip_embedding, group_factor="group")
    return query, (final_scenes, scores), scroll_id

def get_es_queries(query, gps_bounds, ignore_limit_score):
    time_filters, date_filters, duration_filters = query.time_to_filters()
    should_queries = []

    min_score = 0.0
    filter_queries = []
    if query.locations:
        should_queries.append(
                {"match": {"location": {"query": " ".join(query.locations), "boost": 0.01}}})
        location_filters = query.make_location_query()
        filter_queries.append(location_filters)
        min_score += 0.01
        
    if query.location_infos:
        should_queries.append(
                {"match": {"location_info": {"query": " ".join(query.location_infos), "boost": 0.003}}})
        min_score += 0.003

        # FILTERS
    if query.regions:
        filter_queries.append({"terms": {"region": query.regions}})
    if query.weekdays:
        filter_queries.append(
                {"terms": {"weekday": query.weekdays}})

    if time_filters:
        if query.start[0] != 0 or query.start[1] != 0 or query.end[0] != 24 or query.end[1] != 0:
            filter_queries.append(time_filters)

    if date_filters:
        filter_queries.append(date_filters)
            
    if duration_filters:
        should_queries.append(duration_filters)
        min_score += 0.05

    if gps_bounds:
        filter_queries.append(get_gps_filter(gps_bounds))
    clip_script = None
    if query.clip_text:
        if query.clip_embedding is None:
            query.clip_embedding = encode_query(query.clip_text)
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
    if ignore_limit_score:
        if clip_script is not None:
            min_score = CLIP_MIN_SCORE - 0.15
        else:
            min_score = 0.0
        max_score = 1.0
    else:
        # Send a search request for 1 image to get the max score
        # NEW! This request ignores any filters to reserve consistency amongst different filters
        json_query = None
        if clip_script is not None:
            json_query = get_json_query([], [clip_script], [], 1, includes=INCLUDE_FULL_SCENE, min_score=0, timestamp="start_timestamp")
            results, _, aggs  = post_request(
                json.dumps(json_query), SCENE_INDEX, scroll=False)
            clip_max_score = results[0][1]
            # Calculate the min score based on the max score
            clip_min_score = aggs["score_stats"]["std_deviation_bounds"]["upper"]
            max_score = min_score + clip_max_score
            min_score = min_score + clip_min_score
        elif should_queries:
            json_query = get_json_query([], should_queries, [], 1, includes=INCLUDE_FULL_SCENE, min_score=0, timestamp="start_timestamp")
            results, _, aggs  = post_request(
                json.dumps(json_query), SCENE_INDEX, scroll=False)
            max_score = results[0][1]
            # Calculate the min score based on the max score
            min_score = max_score / 2
        else:
            min_score = 0.0
            max_score = 1.0
    print("Min score", min_score)
    return should_queries,filter_queries, min_score, max_score


def msearch(query, gps_bounds, extra_filter_scripts):
    if query.cached:
        should_queries, filter_queries = query.es_should, query.es_filters
        min_score, max_score = query.min_score, query.max_score
    else:
        should_queries, filter_queries, min_score, max_score = get_es_queries(query, gps_bounds, ignore_limit_score=False)
        # Cache/update the queries
        query.es_filters = filter_queries
        query.es_should = should_queries
        query.cached = True
        query.min_score = min_score
        query.max_score = max_score
        
    mquery = []
    for script in extra_filter_scripts:
        new_filter_queries = copy.deepcopy(filter_queries)
        new_filter_queries.append(script)
        mquery.append(json.dumps({}))
        mquery.append(json.dumps(get_json_query(
            [], should_queries, new_filter_queries, 1, INCLUDE_FULL_SCENE, min_score=min_score, timestamp="start_timestamp")))

    results = post_mrequest("\n".join(mquery) + "\n", SCENE_INDEX)
    return query, results


def forward_search(query, conditional_query, condition, time_limit, gps_bounds, K, reverse=False):
    print("-" * 80)
    print("Main")
    start = timecounter.time()
    query, (main_events, main_scores), _ = individual_es(
        query, gps_bounds[0], size=500, scroll=True, 
        K=K-1 if reverse else K, ignore_limit_score=False, cache=False)
    if len(main_events) == 0:
        return query, conditional_query, [], [], [] 
    extra_filter_scripts = []
    time_limit = float(time_limit) 
    leeway = timedelta(hours=min(time_limit - 1, time_limit / 2))
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
        cond_events, cond_scores = group_scene_results(events, group_factor="group")
        new_scenes = []
        for scene, cluster_score in zip(cond_events, cond_scores):
            # Extra information
            scene = extra_info(key, scene)
            logit_scale = len(scene[key]) ** 0.5
            # Filter
            weights = np.array([image_scores[image] * logit_scale * cluster_score
                            for image in scene[key]])
            
            if "weights" in scene and scene["weights"] is not None:
                if len(scene["weights"]) > len(scene[key]):
                    weights = weights * scene["weights"][:len(scene[key])]
            weights = softmax(weights)
            
            scene[key] = arrange_scene(K if reverse else K-1, scene, weights, key, uniform=True)
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
    multiple_pairs = {"condition": condition, 
                      "time_limit": time_limit,
                      "gps_bounds": gps_bounds, 
                      "share_info": share_info, 
                      "K": K}
    
    if isinstance(query, str):
        query = Query(query)
    if isinstance(conditional_query, str):
        conditional_query = Query(conditional_query, shared_filters=query)
        
    if not time_limit:
        if query.duration:
            time_limit = query.duration / 3600
        else:
            time_limit = "1"
    else:
        time_limit = time_limit.strip("h")

    # Forward search
    print("Forward Search")
    query, conditional_query, main_events, conditional_events, scores = forward_search(
        query, conditional_query, condition, time_limit, gps_bounds=[gps_bounds, gps_bounds if share_info else None], K=K)
    max_score1 = query.max_score
    pair_events, already_done = add_pairs(main_events, conditional_events,
                                          condition, scores)
    print("Pairs:", len(pair_events))
    print("-" * 80)
    print("Backward Search")
    conditional_query, query, conditional_events, main_events, scores = forward_search(
        conditional_query, query, "before" if condition == "after" else "after",
        time_limit, gps_bounds=[gps_bounds if share_info else None, gps_bounds], K=K, reverse=True)

    max_score2 = conditional_query.max_score
    pair_events2, _ = add_pairs(conditional_events, main_events, condition,
                                scores, already_done=already_done, reverse=True)

    pair_events.extend(pair_events2)
    if max_score1 == 0:
        max_score1 = 1
    if max_score2 == 0:
        max_score2 = 1
    pair_events = sorted(pair_events, key=lambda x: -x[1]/max_score1 - x[2]/max_score2 * 0.5)

    total_scores = [(s1, s2) for (event, s1, s2) in pair_events]
    pair_events = [event for (event, s1, s2) in pair_events]
    print("Pairs:", len(pair_events))
    multiple_pairs.update({"cached_queries": (query, conditional_query),
                            "position": 24,
                            "pairs": pair_events,
                            "total_scores": total_scores
                            })
    return query, conditional_query, (pair_events[:24], total_scores[:24]), "2pairs"


def es_three_events(query, before, beforewhen, after, afterwhen, gps_bounds, share_info, K):
    global multiple_pairs
    query, before_query, * \
        _ = es_two_events(query, before, "before", beforewhen,
                          gps_bounds, share_info=share_info, K=K)
    before_pairs = multiple_pairs["pairs"]
    total_scores = multiple_pairs["total_scores"]
    
    multiple_pairs = {"beforewhen": beforewhen, 
                      "afterwhen": afterwhen,
                      "gps_bounds": gps_bounds, 
                      "share_info": share_info, 
                      "K": K}

    print("-" * 80)
    print("Search for after events")
    if isinstance(after, str):
        after_query = Query(after, shared_filters=query)
    else:
        after_query = after
    if not afterwhen:
        afterwhen = "1"
        if query.duration:
            afterwhen = query.duration / 3600
        else:
            afterwhen = "1"
    else:
        afterwhen = afterwhen.strip('h')

    extra_filter_scripts = []
    time_limit = float(afterwhen)
    leeway = timedelta(hours=min(time_limit - 1, time_limit / 2))
    time_limit = timedelta(hours=time_limit)
    for time_group, score in zip(before_pairs, total_scores):
        start_time = time_group["end_time"] + leeway
        end_time = time_group["end_time"] + time_limit
        extra_filter_scripts.append(create_time_range_query(
            start_time.timestamp(), end_time.timestamp(), condition="after"))

    # Search for after events
    after_query, after_events = msearch(
        after_query, gps_bounds if share_info else None, extra_filter_scripts)
    max_score3 = after_query.max_score
    # Weigh images
    images = []
    for events in after_events:
        for event, score in events:
            images.extend(event["images"])
    if after_query.clip_embedding is None:
        image_scores = defaultdict(lambda: 1)
    else:
        image_scores = {image: score for image, score in zip(images, score_images(images, after_query.clip_embedding))}

    key = 'current'
    new_events = []
    for events in after_events:
        cond_events, cond_scores = group_scene_results(events, group_factor="group")
        new_scenes = []
        for scene, cluster_score in zip(cond_events, cond_scores):
            scene["gps"] = get_gps(scene[key])
            logit_scale = len(scene[key]) ** 0.5
            # Filter
            weights = np.array([image_scores[image] * logit_scale * cluster_score
                            for image in scene[key]])
            if "weights" in scene and scene["weights"] is not None:
                if len(scene["weights"]) > len(scene[key]):
                    weights = weights * scene["weights"][:len(scene[key])]
            weights = softmax(weights)
            scene[key] = arrange_scene(K-1, scene, weights, key, uniform=True) 
            new_scenes.append((scene, cluster_score))
        new_events.append(new_scenes)
    after_events = new_events
    
    # Add pairs    
    pair_events = []
    for i, before_pair in enumerate(before_pairs):
        s1, s2 = total_scores[i]
        for conditional_event, s3 in after_events[i]:
            pair_event = before_pair.copy()
            pair_event["after"] = conditional_event["current"]
            pair_event["location_after"] = conditional_event["location"]
            pair_events.append((pair_event, s1, s2, s3))
    
    # Sort pairs
    max_score1 = query.max_score
    max_score2 = before_query.max_score
    if max_score1 == 0:
        max_score1 = 1
    if max_score2 == 0:
        max_score2 = 1
    if max_score3 == 0:
        max_score3 = 1
    pair_events = sorted([(event, s1/max_score1 + s2/max_score2 * 0.5 + s3/max_score3 * 0.5)
                          for (event, s1, s2, s3) in pair_events], key=lambda x: -x[1])
    total_scores = [s for event, s in pair_events]
    pair_events = [event for event, s in pair_events]
    print("Pairs:", len(pair_events))

    multiple_pairs.update({"cached_queries": (query, before_query, after_query),
                      "position": 24,
                      "pairs": pair_events,
                      "total_scores": total_scores})
    return query, before_query, after_query, (pair_events[:24], total_scores[:24]), "3pairs"
