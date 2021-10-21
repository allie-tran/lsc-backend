from .query_types import *
from .utils import *
from ..nlp_utils.extract_info import process_query2, process_query3, Query
from ..nlp_utils.synonym import process_string, freq
from ..nlp_utils.common import to_vector, visualisations, countries, stop_words
from datetime import timedelta, datetime
import time as timecounter

multiple_pairs = {}


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
    query_info = {"exact_terms": [],
                  "must_terms": [],
                  "must_not_terms": [],
                  "expansion": [],
                  "expansion_score": {},
                  "weekdays": [],
                  "start_time": (0, 0),
                  "end_time": (24, 0),
                  "dates": [],
                  "region": [],
                  "location": []}
    return group_results(post_request(json.dumps(request), index), group_factor), query_info


def es_date(query, gps_bounds, size, starting_from):
    raw_results, results, size, scores, query_info = es(
        query, gps_bounds, size, starting_from)
    date_dict = defaultdict(lambda: defaultdict(lambda: []))
    count = 0
    for pair, s in zip(results, scores):
        date_dict[count // 8][0].append(pair)
        count += 1

        # date = pair["current"][0].split('/')[0]
        # group = grouped_info_dict[pair["current"][0]]["group"]
        # date_dict[date][group].append(pair)
    print(f"Grouped into {len(date_dict)} days")
    padded_dates = []
    for date in date_dict:
        padded = []
        for group in date_dict[date]:
            padded.extend(date_dict[date][group])
            # padded.append(None)
        padded_dates.append(padded)
        # if len(padded_dates) > 50:
        #     break
    print("Finished")
    return raw_results, padded_dates, size, query_info


def es_more(scroll_id, query_info):
    global multiple_pairs
    if scroll_id == 'pairs':
        position = multiple_pairs["position"]
        new_position = min(position + 21, len(multiple_pairs["pairs"]))
        last_results = multiple_pairs["pairs"][position: new_position]
        multiple_pairs["position"] = new_position
        return None, scroll_id, add_gps_path(last_results), len(last_results), [], {}

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
    query_text = query_info["query_text"]
    scores = query_info["expansion_score"]
    includes = ["id",
                "image_path",
                "time",
                "gps",
                "scene",
                "group",
                "before",
                "after"]

    must_queries = [] # TODO!
    should_queries = [{"match": {"address": {"query": ' '.join(query_text)}}}]
    filter_queries = {"bool": {"must": {"terms": {"image_path": images}}}}

    # TEST
    functions = []
    for word in scores:
        functions.append({"filter": {"term": {"descriptions":
                                              word}}, "weight": scores[word] * (freq[word] if word in freq else 1)})

    # ATFIDF
    should_queries.extend([{"rank_feature":
                            {"field": f"atfidf.{obj}", "boost": scores[obj]}} for obj in scores])

    # CONSTRUCT JSON
    main_query = {}
    test = True
    if must_queries:
        main_query["must"] = must_queries[0] if len(
            must_queries) == 1 else must_queries
    else:
        main_query["must"] = {"match_all": {}}
    if should_queries:
        main_query["should"] = should_queries[0] if len(
            should_queries) == 1 else should_queries
    if filter_queries:
        main_query["filter"] = filter_queries
    main_query = {"bool": main_query}

    main_query = {"function_score": {
        "query": main_query,
        "boost": 5,
        "functions": functions,
        "score_mode": "sum",
        "boost_mode": "sum",
    }}
    # =============
    json_query = {
        "size": min(len(images), 2500),
        "_source": {
            "includes": includes
        },
        "query": main_query
    }
    results, _ = post_request(json.dumps(json_query), "lsc2020", scroll=False)
    print("Num Results:", len(results))
    raw_results, last_results, size, scores, stats = group_results(results, 'scene', 0)
    print("TOTAL TIMES:", timecounter.time() - start, " seconds.")
    return None, scroll_id, add_gps_path(last_results), size, [], {}


def es(query, gps_bounds, size, starting_from, use_simple_process=False, scroll=False):
    start = timecounter.time()
    # print(query, gps_bounds)
    query_info = {}
    raw_results = []
    scroll_id = None
    if query["before"] and query["after"]:
        (raw_results, last_results, size, scores, stats), scroll_id, query_info = es_three_events(
            query["current"], query["before"], query["beforewhen"], query["after"], query["afterwhen"], gps_bounds)
    elif query["before"]:
        (raw_results, last_results, size, scores, stats), scroll_id, query_info = es_two_events(
            query["current"], query["before"], "before", query["beforewhen"], gps_bounds)
    elif query["after"]:
        (raw_results, last_results, size, scores, stats), scroll_id, query_info = es_two_events(
            query["current"], query["after"], "after", query["afterwhen"], gps_bounds)
    else:
        # For LSC, discard info
        # if "info" in query:
        #     (raw_results, last_results, size, scores, stats), query_info = individual_es_from_info(
        #         query["info"], gps_bounds, group_factor="scene", size=size, starting_from=starting_from)
        # else:
        (raw_results, last_results, size, scores, stats), scroll_id, query_info = individual_es(
            query["current"], gps_bounds, group_factor="scene", size=size, starting_from=starting_from, use_simple_process=use_simple_process, scroll=scroll, group_more_by=0.5)
        query_info["stats"] = stats
    print("TOTAL TIMES:", timecounter.time() - start, " seconds.")
    return raw_results, scroll_id, add_gps_path(last_results), size, scores, query_info


def construct_scene_es(query_text, exact_terms, must_terms, must_not_terms, expansion, expansion_score, query_visualisation,
                       weekdays, start_time, end_time, dates,
                       region, location, gps_bounds=None, extra_filter_scripts=None, group_factor="group",
                       use_exact_scores=False, size=2500, starting_from=0, scroll=False, group_more_by=0):
    includes = ["id",
                "current",
                "begin_time",
                "start_time",
                "end_time",
                "gps",
                "scene",
                "group",
                "before",
                "after"]
    time_filters, date_filters = time_to_filters(
        start_time, end_time, dates, True)
    if use_exact_scores:
        expansion = list(expansion_score.keys())
    else:
        expansion.extend(must_terms)
        expansion.extend(exact_terms)
        expansion = list(set(expansion))

    must_queries = []
    should_queries = [{"match": {"address": {"query": ' '.join(query_text)}}}]
    filter_queries = {}
    must_not_queries = []
    place_to_visualise = []
    country_to_visualise = []

    for text in query_text:
        if text not in stop_words:
            for place in visualisations:
                if text in place.lower().split():
                    place_to_visualise.append(place)

    # MUST
    if expansion:
        must_queries.append({"terms": {
            "scene_concepts":  expansion}})

    if region:
        must_queries.append({"terms_set": {"region": {"terms": region,
                                                      "minimum_should_match_script": {
                                                          "source": "1"}}}})
        for reg in region:
            for country in countries:
                if reg == country.lower():
                    country_to_visualise.append(country)

    if location:
        must_queries.append(
            {"match": {"location": {"query": ' '.join(location)}}})

        # should_queries.append({"match": {"location": {"query": ' '.join(location)}}})
        for loc in location:
            for place in visualisations:
                if set(loc.split()).issubset(set(place.lower().split())):
                    place_to_visualise.append(place)

    # FILTERS
    if weekdays:
        filter_queries["bool"] = {}
        filter_queries["bool"]["filter"] = [{"terms": {"weekday": weekdays}}]

    if extra_filter_scripts:
        if "bool" not in filter_queries:
            filter_queries["bool"] = {}
        filter_queries["bool"]["should"] = []
        filter_queries["bool"]["should"].extend(extra_filter_scripts)

    if date_filters + time_filters:
        if "bool" not in filter_queries:
            filter_queries["bool"] = {}
        if "must" not in filter_queries["bool"]:
            filter_queries["bool"]["must"] = []
        script = "&&".join(date_filters + time_filters)
        filter_queries["bool"]["must"].append({"script": {
            "script": {
                "source": script
            }}})

    if gps_bounds:
        if "bool" not in filter_queries:
            filter_queries["bool"] = {"must": []}
        if "must" not in filter_queries["bool"]:
            filter_queries["bool"]["must"] = []
        filter_queries["bool"]["must"].append(get_gps_filter(gps_bounds))

    # TEST
    functions = []
    if use_exact_scores:
        scores = expansion_score
    else:
        scores = defaultdict(lambda: 0)
        for word in expansion:
            scores[word] += expansion_score[word] if word in expansion_score else 1
        for word in must_terms:
            scores[word] += 1.5 * \
                expansion_score[word] if word in expansion_score else 1
        for word in exact_terms:
            scores[word] += 2

    scores = dict([(keyword, score) for (keyword, score) in sorted(
        scores.items(), key=lambda x: -x[1]) if score > 1])
    for word in scores:
        functions.append({"filter": {"term": {"scene_concepts":
                                              word}}, "weight": scores[word] * (freq[word] if word in freq else 1)})

    # CONSTRUCT JSON
    main_query = {}
    test = True
    if must_queries:
        main_query["must"] = must_queries[0] if len(
            must_queries) == 1 else must_queries
    else:
        main_query["must"] = {"match_all": {}}
    if should_queries:
        main_query["should"] = should_queries[0] if len(
            should_queries) == 1 else should_queries
    if filter_queries:
        main_query["filter"] = filter_queries
    main_query = {"bool": main_query}

    main_query = {"function_score": {
        "query": main_query,
        "boost": 5,
        "functions": functions,
        "score_mode": "sum",
        "boost_mode": "sum",
    }}
    # END TEST
    # =============
    json_query = {
        "size": size,
        "from": starting_from,
        "_source": {
            "includes": includes
        },
        "query": main_query
    }

    # query info
    query_info = {"exact_terms": list(exact_terms),
                  "must_terms": list(must_terms),
                  "must_not_terms": list(must_not_terms),
                  "expansion": list(expansion),
                  "expansion_score": scores,
                  "weekdays": list(weekdays),
                  "start_time": start_time,
                  "end_time": end_time,
                  "dates": list(dates),
                  "region": region,
                  "location": location,
                  "place_to_visualise": place_to_visualise,
                  "country_to_visualise": country_to_visualise,
                  "query_text": query_text,
                  "query_visualisation": query_visualisation}
    results, scroll_id = post_request(json.dumps(
        json_query), "lsc2020_scene", scroll=scroll)
    grouped_results = group_scene_results(results, group_factor, group_more_by)
    if group_factor != 'scene':
        return grouped_results, scroll_id, query_info

    images = [image for scene in results for image in scene[0]["current"]]

    includes = ["id",
                "image_path",
                "time",
                "gps",
                "scene",
                "group",
                "before",
                "after"]

    should_queries = [{"match": {"address": {"query": ' '.join(query_text)}}}]
    filter_queries = {"bool": {"must": {"terms": {"image_path": images}}}}

    # TEST
    functions = []
    for word in scores:
        functions.append({"filter": {"term": {"descriptions":
                                              word}}, "weight": scores[word] * (freq[word] if word in freq else 1)})

    # ATFIDF
    should_queries.extend([{"rank_feature":
                            {"field": f"atfidf.{obj}", "boost": scores[obj]}} for obj in scores])

    # CONSTRUCT JSON
    main_query = {}
    test = True
    if must_queries:
        main_query["must"] = must_queries[0] if len(
            must_queries) == 1 else must_queries
    else:
        main_query["must"] = {"match_all": {}}
    if should_queries:
        main_query["should"] = should_queries[0] if len(
            should_queries) == 1 else should_queries
    if filter_queries:
        main_query["filter"] = filter_queries
    main_query = {"bool": main_query}

    main_query = {"function_score": {
        "query": main_query,
        "boost": 5,
        "functions": functions,
        "score_mode": "sum",
        "boost_mode": "sum",
    }}
    # =============
    json_query = {
        "size": min(len(images), 2500),
        "_source": {
            "includes": includes
        },
        "query": main_query
    }
    results, _ = post_request(json.dumps(json_query), "lsc2020", scroll=False)
    print("Num Results:", len(results))
    return group_results(results, group_factor, group_more_by), scroll_id, query_info


def construct_es(query_text, exact_terms, must_terms, must_not_terms, expansion, expansion_score, query_visualisation,
                 weekdays, start_time, end_time, dates,
                 region, location, gps_bounds=None, extra_filter_scripts=None, group_factor="group",
                 use_exact_scores=False, size=2500, starting_from=0, scroll=False, group_more_by=0):
    includes = ["id",
                "image_path",
                "time",
                "gps",
                "scene",
                "group",
                "before",
                "after", "scene_concepts"]

    time_filters, date_filters = time_to_filters(start_time, end_time, dates)
    if use_exact_scores:
        expansion = list(expansion_score.keys())
    else:
        expansion.extend(must_terms)
        expansion.extend(exact_terms)
        expansion = list(set(expansion))

    must_queries = []
    should_queries = []
    filter_queries = {}
    must_not_queries = []
    place_to_visualise = []
    country_to_visualise = []

    for text in query_text:
        if text not in stop_words:
            for place in visualisations:
                if text in place.lower().split():
                    place_to_visualise.append(place)

    # MUST
    if expansion:
        must_queries.append({"terms": {
            "scene_concepts":  expansion}})

    if region:
        must_queries.append({"terms_set": {"region": {"terms": region,
                                                      "minimum_should_match_script": {
                                                          "source": "1"}}}})
        for reg in region:
            for country in countries:
                if reg == country.lower():
                    country_to_visualise.append(country)

    if location:
        must_queries.append(
            {"match": {"location": {"query": ' '.join(location)}}})
        # should_queries.append({"match": {"location": {"query": ' '.join(location)}}})
        for loc in location:
            for place in visualisations:
                if set(loc.split()).issubset(set(place.lower().split())):
                    place_to_visualise.append(place)

    # FILTERS
    if weekdays:
        filter_queries["bool"] = {}
        filter_queries["bool"]["must"] = [{"terms": {"weekday": weekdays}}]

    if extra_filter_scripts:
        if "bool" not in filter_queries:
            filter_queries["bool"] = {}
        filter_queries["bool"]["should"] = []
        filter_queries["bool"]["should"].extend(extra_filter_scripts)

    if date_filters + time_filters:
        if "bool" not in filter_queries:
            filter_queries["bool"] = {}
        if "must" not in filter_queries["bool"]:
            filter_queries["bool"]["must"] = []
        script = "&&".join(date_filters + time_filters)
        filter_queries["bool"]["must"].append({"script": {
            "script": {
                "source": script
            }}})

    if gps_bounds:
        if "bool" not in filter_queries:
            filter_queries["bool"] = {"must": []}
        filter_queries["bool"]["must"].append(get_gps_filter(gps_bounds))

    # TEST
    functions = []
    if use_exact_scores:
        scores = expansion_score
    else:
        scores = defaultdict(lambda: 0)
        for word in expansion:
            scores[word] += expansion_score[word] if word in expansion_score else 1
        for word in must_terms:
            scores[word] += 1.5 * \
                expansion_score[word] if word in expansion_score else 1
        for word in exact_terms:
            scores[word] += 2

    scores = dict([(keyword, score) for (keyword, score) in sorted(
        scores.items(), key=lambda x: -x[1]) if score > 1])
    for word in scores:
        # functions.append({"filter": {"term": {"descriptions":
        # word}}, "weight": scores[word] * (freq[word] if word in freq else 1)})
        functions.append({"filter": {"term": {"scene_concepts":
                                              word}}, "weight": scores[word] * (freq[word] if word in freq else 1)})

    # MUST NOT
    # if must_not_terms:
    #     print("Must not:", must_not_terms)
    #     for word in must_not_terms:
    #         functions.append({"filter": {"term": {"descriptions_not":
    #             word}}, "weight": 150 / len(must_not_terms)})
    #         functions.append({"filter": {"term": {"scene_concepts_not":
    #             word}}, "weight": 150 / len(must_not_terms)})

    # ATFIDF
    should_queries.extend([{"rank_feature":
                            {"field": f"atfidf.{obj}", "boost": scores[obj]}} for obj in scores])

    # CONSTRUCT JSON
    main_query = {}
    test = True
    if must_queries:
        main_query["must"] = must_queries[0] if len(
            must_queries) == 1 else must_queries
    else:
        main_query["must"] = {"match_all": {}}
    if should_queries:
        main_query["should"] = should_queries[0] if len(
            should_queries) == 1 else should_queries
    if filter_queries:
        main_query["filter"] = filter_queries
    main_query = {"bool": main_query}

    main_query = {"function_score": {
        "query": main_query,
        "boost": 5,
        "functions": functions,
        "score_mode": "sum",
        "boost_mode": "sum",
    }}
    # END TEST
    # =============
    json_query = {
        "size": size,
        "from": starting_from,
        "_source": {
            "includes": includes
        },
        "query": main_query
    }

    # query info
    query_info = {"exact_terms": list(exact_terms),
                  "must_terms": list(must_terms),
                  "must_not_terms": list(must_not_terms),
                  "expansion": list(expansion),
                  "expansion_score": scores,
                  "weekdays": list(weekdays),
                  "start_time": start_time,
                  "end_time": end_time,
                  "dates": list(dates),
                  "region": region,
                  "location": location,
                  "place_to_visualise": place_to_visualise,
                  "country_to_visualise": country_to_visualise,
                  "query_text": query_text,
                  "query_visualisation": query_visualisation}
    results, scroll_id = post_request(
        json.dumps(json_query), "lsc2020", scroll=scroll)
    print("Num Results:", len(results))
    return group_results(results, group_factor, group_more_by), scroll_id, query_info


def individual_es_from_info(query_info, gps_bounds=None, extra_filter_scripts=None, group_factor="group", size=2000, starting_from=0, scroll=False, group_more_by=0):
    query_text = query_info["query_text"]
    exact_terms = query_info["exact_terms"]
    must_terms = query_info["must_terms"]
    must_not_terms = query_info["must_not_terms"]
    expansion = query_info["expansion"]
    expansion_score = query_info["expansion_score"],
    weekdays = query_info["weekdays"]
    start_time = query_info["start_time"]
    end_time = query_info["end_time"]
    dates = query_info["dates"]
    region = query_info["region"]
    location = query_info["location"]
    query_visualisation = query_info["query_visualisation"]
    if isinstance(expansion_score, tuple):
        expansion_score = expansion_score[0]
    return construct_scene_es(query_text, exact_terms, must_terms, must_not_terms, expansion, expansion_score, query_visualisation,
                              weekdays, start_time, end_time, dates,
                              region,
                              location, gps_bounds, extra_filter_scripts, group_factor,
                              use_exact_scores=True, size=size, starting_from=starting_from, scroll=scroll, group_more_by=group_more_by)


def individual_es(query_text, gps_bounds=None, extra_filter_scripts=None, group_factor="group", size=2000, starting_from=0, use_simple_process=False, scroll=False, group_more_by=0):
    includes = ["id",
                "image_path",
                "time",
                "gps",
                "scene",
                "group",
                "before",
                "after"]
    if not query_text:
        return query_all(query_text, includes, "lsc2020", group_factor)

    # process = process_query3 if use_simple_process else process_query2
    # info, keywords, region, location, weekdays, start_time, end_time, dates = process(
    #     query)

    query = Query(query_text)

    if not (query.locations or query.keywords or query.exacts or query.weekdays or query.regions or query.dates or query.start_time or query.end_time):
        return query_all(query_text, includes, "lsc2020", group_factor)

    must_not_terms = {}

    exact_terms, must_terms, expansion, expansion_score, query_visualisation = query.expand(
        must_not_terms)

    return construct_scene_es(query.text, exact_terms, must_terms, must_not_terms, expansion, expansion_score, query_visualisation,
                              list(query.weekdays), list(query.start_time), list(
                                  query.end_time), list(query.dates),
                              list(query.regions),
                              list(query.locations), gps_bounds, extra_filter_scripts, group_factor, size=size, starting_from=starting_from, scroll=scroll, group_more_by=group_more_by)


def forward_search(query, conditional_query, condition, time_limit, gps_bounds=None, group_factor=["scene", "group"], from_info=False):
    print("-" * 80)
    print("Main")
    start = timecounter.time()
    query_infos = []
    if from_info:
        (_, main_events, _, scores, _), _, query_info = individual_es_from_info(query,
                                                                                gps_bounds, size=1000, group_factor=group_factor[0])
    else:
        (_, main_events, _, scores, _), _, query_info = individual_es(
            query, gps_bounds, size=1000, group_factor=group_factor[0])

    query_infos.append(query_info)

    extra_filter_scripts = []
    for time_group in find_time_span(main_events):
        if condition == "before":
            start_time = time_group["begin_time"] - \
                timedelta(hours=float(time_limit) + 1)
            end_time = time_group["begin_time"]
        else:
            start_time = time_group["end_time"]
            end_time = start_time + timedelta(hours=float(time_limit) + 1)
        extra_filter_scripts.append(create_time_range_query(
            start_time.timestamp(), end_time.timestamp()))
    print("Time:", timecounter.time()-start, "seconds.")
    print("-" * 80)
    print("Conditional")
    start = timecounter.time()
    if from_info:
        (_, conditional_events, _, scores_cond, _), _, query_info = individual_es_from_info(conditional_query, gps_bounds=gps_bounds, extra_filter_scripts=extra_filter_scripts,
                                                                                            size=500, group_factor=group_factor[1])
    else:
        (_, conditional_events, _, scores_cond, _), _, query_info = individual_es(conditional_query, size=500,
                                                                                  extra_filter_scripts=extra_filter_scripts, group_factor=group_factor[1])

    query_infos.append(query_info)
    print("Time:", timecounter.time()-start, "seconds.")
    return main_events, conditional_events, extra_filter_scripts, scores, scores_cond, query_infos


def add_pairs(main_events, conditional_events, condition, time_limit, scores, scores_cond, already_done=None):
    pair_events = []
    total_scores = []
    max_score1 = scores[0]
    max_score2 = scores_cond[0]
    if already_done is None:
        already_done = set()
    for main_event, s1 in zip(main_events, scores):
        if main_event["scene"] in already_done:
            continue
        for conditional_event, s2 in zip(conditional_events, scores_cond):
            if condition == "after" and timedelta() < conditional_event["begin_time"] - main_event["begin_time"] < timedelta(hours=float(time_limit) + 2):
                already_done.add(main_event["scene"])
                pair_events.append({"current": main_event["current"],
                                    "before": main_event["before"],
                                    "after": conditional_event["current"],
                                    "begin_time": main_event["begin_time"],
                                    "end_time": main_event["end_time"]})
                total_scores.append(s1/max_score1 + s2/max_score2)
                break
            elif condition == "before" and timedelta() < main_event["begin_time"] - conditional_event["begin_time"] < timedelta(hours=float(time_limit) + 2):
                already_done.add(main_event["scene"])
                pair_events.append({"current": main_event["current"],
                                    "before": conditional_event["current"],
                                    "after": main_event["after"],
                                    "begin_time": main_event["begin_time"],
                                    "end_time": main_event["end_time"]})
                total_scores.append(s1/max_score1 + s2/max_score2)
                break

    return pair_events, total_scores, already_done


def es_two_events(query, conditional_query, condition, time_limit, gps_bounds, return_extra_filter=False):
    global multiple_pairs
    if not time_limit:
        time_limit = "1"
    else:
        time_limit = time_limit.strip("h")
    # Forward search
    print("Forward Search")
    main_events, conditional_events, extra_filter_scripts, scores, scores_cond, query_infos = forward_search(query, conditional_query,
                                                                                                             condition, time_limit, gps_bounds)
    pair_events, total_scores, already_done = add_pairs(
        main_events, conditional_events, condition, time_limit, scores, scores_cond)

    print("Backward Search")
    # # Backward search
    conditional_events, main_events, _, scores, scores_cond, _ = forward_search(
        query_infos[1], query_infos[0], "before" if condition == "after" else "after", time_limit, group_factor=["group", "scene"], from_info=True)
    new_pair_events, new_scores, _ = add_pairs(main_events,
                                            conditional_events, condition, time_limit, scores, scores_cond, already_done)

    pair_events += new_pair_events
    total_scores += new_scores
    query_info = query_infos[0]
    query_info["query_visualisation"].extend(
        query_infos[1]["query_visualisation"])
    query_info["place_to_visualise"].extend(
        query_infos[1]["place_to_visualise"])
    query_info["country_to_visualise"].extend(
        query_infos[1]["country_to_visualise"])

    pair_events = [pair for (pair, score) in sorted(
        zip(pair_events, total_scores), key=lambda x: -x[1])]
    print("Pairs:", len(pair_events))
    multiple_pairs = {"position": 21,
                      "pairs": pair_events}
    # (raw_results, last_results, size, scores, stats), query_info
    if return_extra_filter:
        return ({}, pair_events[:21], extra_filter_scripts, 0, total_scores, None), "pairs",  query_info
    else:
        return ({}, pair_events[:21], 0, total_scores, None), "pairs",  query_info


def es_three_events(query, before, beforewhen, after, afterwhen, gps_bounds):
    global multiple_pairs
    if not afterwhen:
        afterwhen = "1"
    else:
        afterwhen = afterwhen.strip('h')
    if not beforewhen:
        beforewhen = "1"
    else:
        beforewhen = afterwhen.strip('h')

    (_, before_pairs, extra_filter_scripts, _, total_scores, _), _, query_info = es_two_events(
        query, before, "before", beforewhen, gps_bounds, return_extra_filter=True)
    print("-" * 80)
    print("Search for after events")
    (_, after_events, _, scores, _), _, query_info_cond = individual_es(after,
                                                                        size=500, extra_filter_scripts=extra_filter_scripts)
    # print(len(before_pairs), len(after_events))

    query_info["query_visualisation"].extend(
        query_info_cond["query_visualisation"])
    query_info["place_to_visualise"].extend(
        query_info_cond["place_to_visualise"])
    query_info["country_to_visualise"].extend(
        query_info_cond["country_to_visualise"])

    pair_events = []
    pair_scores = []
    max_score1 = total_scores[0]
    max_score2 = scores[0]
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
    pair_events = [pair for (pair, score) in sorted(
        zip(pair_events, pair_scores), key=lambda x: -x[1])]
    multiple_pairs = {"position": 21,
                      "pairs": pair_events}
    return ({}, pair_events[:21], 0, [], {}), "pairs", query_info


if __name__ == "__main__":
    query = "woman in red top"
    info, keywords, region, location, weekdays, start_time, end_time, dates = process_query2(
        query)
    exact_terms, must_terms, expansion, expansion_score = process_string(
        info, keywords, [])
    print(exact_terms, must_terms, expansion, expansion_score)
