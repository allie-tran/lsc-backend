from .query_types import *
from .timeline import time_info
from .utils import *
from ..nlp_utils.extract_info import Query
from ..nlp_utils.synonym import process_string, freq
from ..nlp_utils.common import to_vector, visualisations, countries, stop_words, ocr_keywords, gps_locations
from datetime import timedelta, datetime
import time as timecounter
from collections import defaultdict

COMMON_PATH = os.getenv('COMMON_PATH')
full_similar_images = json.load(
    open(f"{COMMON_PATH}/full_similar_images.json"))
multiple_pairs = {}
INCLUDE_SCENE = ["current", "begin_time", "end_time",
    "gps", "scene", "group", "before", "after"]
INCLUDE_IMAGE = ["image_path", "current", "time",
    "gps", "scene", "group", "before", "after"]

cached_queries = None
cached_filters = []

def query_all(query_text, includes, index, group_factor):
    if query_text:
        query = {"match": {"address": query_text}}
    else:
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
    return query_text, group_results(post_request(json.dumps(request), index), group_factor)

def es_more(scroll_id):
    global multiple_pairs
    if scroll_id == 'pairs':
        position = multiple_pairs["position"]
        new_position = min(position + 21, len(multiple_pairs["pairs"]))
        last_results = multiple_pairs["pairs"][position: new_position]
        multiple_pairs["position"] = new_position
        return scroll_id, add_gps_path(last_results), []

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
    images = [image for scene in results for image in scene[0]["current"]]
    must_queries, should_queries, functions = cached_queries
    filter_queries = [{"terms": {"image_path": images}}]
    # CONSTRUCT JSON
    json_query = get_json_query(must_queries, should_queries, filter_queries, functions, min(len(images), 2500), includes=INCLUDE_IMAGE)
    results, _ = post_request(json.dumps(json_query), "lsc2020", scroll=False)
    print("Num Results:", len(results))
    results, scores = group_results(results, 'scene', 0)
    print("TOTAL TIMES:", timecounter.time() - start, " seconds.")
    return scroll_id, add_gps_path(results), scores


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
            query["current"], gps_bounds, group_factor="scene", size=size)
        query_info = query.get_info()

    print("TOTAL TIMES:", timecounter.time() - start, " seconds.")
    return scroll_id, add_gps_path(results), scores, query_info

def query_list(query_list):
    return query_list[0] if len(query_list) == 1 else query_list

def get_json_query(must_queries, should_queries, filter_queries, functions, size, includes):
    # CONSTRUCT JSON
    main_query = {}
    if must_queries:
        main_query["must"] = query_list(must_queries)
    else:
        main_query["must"] = {"match_all": {}}
    if should_queries:
        main_query["should"] = query_list(should_queries)
    if filter_queries:
        main_query["filter"] = filter_queries
    main_query = {"bool": main_query}

    if functions:
        main_query = {"function_score": {
            "query": main_query,
            "boost": 5,
            "functions": functions,
            "score_mode": "sum",
            "boost_mode": "sum",
        }}

    json_query = {
        "size": size,
        "_source": {
            "includes": includes
        },
        "query": main_query
    }
    return json_query

def get_neighbors(image, query_info, gps_bounds):
    img_index = full_similar_images.index(image)
    if img_index >= 0:
        filter_queries = cached_filters
        request = {
            "size": 1,
            "_source": {
                "includes": ["similars"]
            },
            "query": {
                "term": {"image_index": img_index}
            },
        }
        results = post_request(json.dumps(request), "lsc2020_similar")
        if results:
            images = [full_similar_images[r]
                      for r in results[0][0][0]["similars"]][:2500]
            filter_queries.append(
                {"terms": {"image_path": images}})
            json_query = get_json_query([], [], filter_queries, [], len(images), includes=["image_path", "scene", "weekday"])
            results, _ = post_request(json.dumps(json_query), "lsc2020", scroll=False)
            new_results = dict(
                [(r[0]["image_path"], (r[0]["weekday"], r[0]["scene"])) for r in results])
            images = [image for image in images if image in new_results]
            grouped_results = defaultdict(lambda: [])
            weekdays = {}
            for image in images:
                if image in new_results:
                    scene = new_results[image][1]
                    weekdays[scene] = new_results[image][0]
                    grouped_results[scene].append(image)
            times = [(grouped_results[scene], weekdays[scene] + "\n" + scene.split("_")[0] + "\n" + time_info[scene])
                     for scene in grouped_results]
            return times[:100]
    return []

def individual_es(query, gps_bounds=None, extra_filter_scripts=None, group_factor="group", size=200):
    if isinstance(query, str):
        query = Query(query)
        start = timecounter.time()

    if not query.original_text and not gps_bounds:
        return query_all(query_text, INCLUDE_IMAGE, "lsc2020", group_factor)
    query.expand()
    return construct_scene_es(query, gps_bounds, extra_filter_scripts, group_factor, size=size)

def construct_scene_es(query, gps_bounds=None, extra_filter_scripts=None, group_factor="group", size=200):
    time_filters, date_filters = query.time_to_filters(True)
    must_queries = []
    # !TODO
    # should_queries = [{"match": {"address": {"query": " ".join(query.unigrams)}}},
                    #   {"match": {"location": {"query": " ".join(query.unigrams)}}}]
    should_queries = []
    if query.ocr:
        should_queries.extend(query.make_ocr_query())
    filter_queries = []
    functions = []

    # MUST
    if query.keywords:
        must_queries.append({"terms": {"scene_concepts": list(query.scores.keys())}})

    if query.locations:
        for loc in query.locations:
            for place in gps_locations:
                if set(loc.split()).issubset(set(place.lower().split())):
                    should_queries.append({
                        "distance_feature": {
                            "field": "gps",
                            "pivot": "50m",
                            "origin": gps_locations[place]
                        }
                    })

    # FILTERS
    if query.regions:
        filter_queries.append({"terms_set": {"region": {"terms": query.regions,
                                                        "minimum_should_match_script": {
                                                            "source": "1"}}}})
    if query.weekdays:
        filter_queries.append({"terms": {"weekday": query.weekdays}})

    if date_filters + time_filters:
        script = "&&".join(date_filters + time_filters)
        filter_queries.append({"script": {
            "script": {
                "source": script
            }}})

    if gps_bounds:
        filter_queries.append(get_gps_filter(gps_bounds))

    if extra_filter_scripts:
        if filter_queries:
            filter_queries = {"bool": {
                "filter": query_list(filter_queries),
                "should": extra_filter_scripts,
                "must": {"match_all": {}}
            }}
        else:
            filter_queries = {"bool":
            {"must": {"match_all": {}},
             "should": extra_filter_scripts}}
    else:
        if filter_queries:
            filter_queries = query_list(filter_queries)


    for word in query.scores:
        functions.append({"filter": {"term": {"scene_concepts":
                                                word}}, "weight": query.scores[word] * (freq[word] if word in freq else 1) / 20})

    json_query = get_json_query(
        must_queries, should_queries, filter_queries, functions, size, INCLUDE_SCENE)

    results, scroll_id = post_request(json.dumps(
        json_query), "lsc2020_scene", scroll=True)

    grouped_results = group_scene_results(results, group_factor)

    if group_factor != 'scene':
        return query, grouped_results, scroll_id

    if not extra_filter_scripts:
        global cached_filters
        cached_filters = filter_queries
    images = [image for scene in results for image in scene[0]["current"]]
    functions = []
    # should_queries = [{"match": {"address": {"query": " ".join(query.unigrams)}}},
    #                   {"match": {"location": {"query": " ".join(query.unigrams)}}}]
    should_queries = []
    should_queries.extend(query.make_ocr_query())
    # ATFIDF
    should_queries.extend([{"rank_feature":
                            {"field": f"atfidf.{obj}", "boost": query.scores[obj]}} for obj in query.scores])
    filter_queries = [{"terms": {"image_path": images}}]
    for word in query.scores:
        functions.append({"filter": {"term": {"descriptions":
                                                word}}, "weight": query.scores[word] * (freq[word] if word in freq else 1) / 20})

    # CONSTRUCT JSON
    json_query = get_json_query(must_queries, should_queries,
                                filter_queries, functions, min(len(images), 1000), includes=INCLUDE_IMAGE)
    global cached_queries
    cached_queries = (must_queries, should_queries, functions)
    results, _ = post_request(json.dumps(json_query), "lsc2020", scroll=False)
    print("Num Results:", len(results))
    return query, group_results(results, group_factor), scroll_id

def forward_search(query, conditional_query, condition, time_limit, gps_bounds, group_factor=["scene", "group"]):
    print("-" * 80)
    print("Main")
    start = timecounter.time()
    query_infos = []
    query, (main_events, scores), _ = individual_es(
        query, gps_bounds[0], size=1000, group_factor=group_factor[0])

    extra_filter_scripts = []
    time_limit = float(time_limit)
    time_limit = timedelta(hours=time_limit)

    for time_group in main_events:
        if condition == "before":
            start_time = time_group["begin_time"] - time_limit
            end_time = time_group["begin_time"]
        elif condition == "after":
            start_time = time_group["end_time"]
            end_time = time_group["end_time"] + time_limit
        # extra_filter_scripts.append(
            # f"({start_time.timestamp()} < doc['timestamp'].value &&  doc['timestamp'].value < {end_time.timestamp()})")
        extra_filter_scripts.append(create_time_range_query(start_time.timestamp(), end_time.timestamp()))

    # extra_filter_scripts = {"script":
        # {"script": {"source": "||".join(extra_filter_scripts)}}
    # }
    print("Time:", timecounter.time() - start, "seconds.")
    print("-" * 80)
    print("Conditional")
    start = timecounter.time()
    conditional_query, (conditional_events, scores_cond), _ = individual_es(conditional_query, size=500,
                                                                            gps_bounds=gps_bounds[1], extra_filter_scripts=extra_filter_scripts, group_factor=group_factor[1])

    print("Time:", timecounter.time()-start, "seconds.")
    return query, conditional_query, main_events, conditional_events, extra_filter_scripts, scores, scores_cond


def add_pairs(main_events, conditional_events, condition, time_limit, scores, scores_cond, already_done=None):
    if not main_events or not conditional_events:
        return {}, 0, 0
    pair_events = []
    total_scores = []
    max_score1 = scores[0]
    max_score2 = scores_cond[0]
    if already_done is None:
        already_done = {}
    time_limit = float(time_limit)
    time_limit = timedelta(hours=time_limit)
    for main_event, s1 in zip(main_events, scores):
        scene = main_event["scene"]
        cond_scenes = already_done[scene] if scene in already_done else {}
        for conditional_event, s2 in zip(conditional_events, scores_cond):
            to_take = False
            if condition == "after" and timedelta() < conditional_event["begin_time"] - main_event["end_time"] < time_limit:
                to_take = True
            elif condition == "before" and timedelta() < main_event["begin_time"] - conditional_event["end_time"] < time_limit:
                to_take = True
            if to_take:
                group = conditional_event['group']
                if group in cond_scenes:
                    s1 = max(s1, cond_scenes[group][1])
                    s2 = max(s2, cond_scenes[group][2])
                cond_scenes[group] = [{"current": main_event["current"],
                                       "before": main_event["before"],
                                       "after": conditional_event["current"],
                                       "begin_time": main_event["begin_time"],
                                       "end_time": main_event["end_time"]}, s1, s2]
        already_done[scene] = cond_scenes
    return already_done, max_score1, max_score2


def es_two_events(query, conditional_query, condition, time_limit, gps_bounds, return_extra_filter=False, share_info=False):
    print("Share info:", share_info)
    global multiple_pairs
    if not time_limit:
        time_limit = "1"
    else:
        time_limit = time_limit.strip("h")

    query = Query(query)
    conditional_query = Query(conditional_query)

    # Forward search
    print("Forward Search")
    query, conditional_query, main_events, conditional_events, extra_filter_scripts, scores, scores_cond = forward_search(
            query, conditional_query, condition, time_limit, gps_bounds=[gps_bounds, gps_bounds if share_info else None])

    already_done, max_score1, _ = add_pairs(
        main_events, conditional_events, condition, time_limit, scores, scores_cond)

    print("Backward Search")
    # Backward search
    conditional_query, query, conditional_events, main_events, _, scores, scores_cond = forward_search(
        conditional_query, query, "before" if condition == "after" else "after",
        time_limit, gps_bounds=[gps_bounds if share_info else None, gps_bounds],
        group_factor=["group", "scene"])

    already_done, _, max_score2 = add_pairs(main_events, conditional_events, condition, time_limit, scores, scores_cond, already_done)
    pair_events = []
    total_scores = []
    for main_scene in already_done:
        cond_scenes = sorted(already_done[main_scene].items(
        ), key=lambda x: x[1][1]/max_score1 + 1.0 * x[1][2]/max_score2, reverse=True)[:3]
        for group, (cond_scene, score1, score2) in cond_scenes:
            pair_events.append(cond_scene)
            total_scores.append(score1/max_score1 + 1.0 * score2/max_score2)

    # pair_events += new_pair_events
    # total_scores += new_scores

    pair_events=[pair for (pair, score) in sorted(
        zip(pair_events, total_scores), key=lambda x: -x[1])]

    print("Pairs:", len(pair_events))
    multiple_pairs={"position": 21,
                    "pairs": pair_events}
    if return_extra_filter:
        return query, conditional_query, (pair_events[:21], extra_filter_scripts, total_scores), "pairs"
    else:
        return query, conditional_query, (pair_events[:21], total_scores), "pairs"


def es_three_events(query, before, beforewhen, after, afterwhen, gps_bounds, share_info=False):
    global multiple_pairs
    if not afterwhen:
        afterwhen="1"
    else:
        afterwhen=afterwhen.strip('h')
    if not beforewhen:
        beforewhen="1"
    else:
        beforewhen=afterwhen.strip('h')

    query, before_query, (before_pairs, extra_filter_scripts, total_scores), _  = es_two_events(
        query, before, "before", beforewhen, gps_bounds, return_extra_filter=True)
    print("-" * 80)
    print("Search for after events")
    after_query, (after_events, scores), _ = individual_es(after, size=500, extra_filter_scripts=extra_filter_scripts)
    # print(len(before_pairs), len(after_events))

    pair_events=[]
    pair_scores=[]
    max_score1=total_scores[0]
    max_score2=scores[0]
    for before_pair, s1 in zip(before_pairs, total_scores):
        for after_event, s2 in zip(after_events, scores):
            if timedelta() < after_event["begin_time"] - before_pair["end_time"] < timedelta(hours=float(afterwhen) + 2):
                pair_events.append({"current": before_pair["current"],
                                    "before": before_pair["before"],
                                    "after": after_event["current"],
                                    "begin_time": before_pair["begin_time"],
                                    "end_time": before_pair["end_time"]})
                pair_scores.append(s1/max_score1 + s2/max_score2)
                break
    print("Pairs:", len(pair_events))

    pair_events=[pair for (pair, score) in sorted(
        zip(pair_events, pair_scores), key=lambda x: -x[1])]

    multiple_pairs={"position": 21,
                    "pairs": pair_events}
    return query, before_query, after_query, (pair_events[:21], []), "pairs"


if __name__ == "__main__":
    query="woman in red top"
    info, keywords, region, location, weekdays, start_time, end_time, dates=process_query2(
        query)
    exact_terms, must_terms, expansion, expansion_score=process_string(
        info, keywords, [])
    print(exact_terms, must_terms, expansion, expansion_score)
