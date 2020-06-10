from .query_types import *
from .utils import *
from ..nlp_utils.extract_info import process_query, process_query2
from ..nlp_utils.synonym import process_string, freq
from datetime import timedelta, datetime


def query_all(includes, index, group_factor):
    request = {
        "size": 2000,
        "_source": {
            "includes": includes
        },
        "query": {
            "match_all": {}
        },
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
    results, size, scores, query_info = es(query, gps_bounds, starting_from)
    date_dict = defaultdict(lambda: defaultdict( lambda: []))
    for pair, s in zip(results, scores):
        date = pair["current"][0].split('/')[0]
        group = grouped_info_dict[pair["current"][0]]["group"]
        date_dict[date][group].append(pair)
    print(f"Grouped into {len(date_dict)} days")
    padded_dates = []
    for date in date_dict:
        padded = []
        for group in date_dict[date]:
            padded.extend(date_dict[date][group])
            padded.append(None)
        padded_dates.append(padded)
        # if len(padded_dates) > 50:
        #     break
    print("Finished")
    return padded_dates, size, query_info


def es(query, gps_bounds, size, starting_from):
    # print(query, gps_bounds)
    query_info = {}
    if query["before"] and query["after"]:
        last_results, scores = es_three_events(
            query["current"], query["before"], query["beforewhen"], query["after"], query["afterwhen"], gps_bounds)
    elif query["before"]:
        last_results, scores = es_two_events(
            query["current"], query["before"], "before", query["beforewhen"], gps_bounds)
    elif query["after"]:
        last_results, scores = es_two_events(
            query["current"], query["after"], "after", query["afterwhen"], gps_bounds)
    else:
        if "info" in query:
            (last_results, size, scores, stats), query_info = individual_es_from_info(
                query["info"], gps_bounds, group_factor="scene", size=size, starting_from=starting_from)
        else:
            (last_results, size, scores, stats), query_info = individual_es(
                query["current"], gps_bounds, group_factor="scene", size=size, starting_from=starting_from)
        query_info["stats"] = stats
    return add_gps_path(last_results), size, scores, query_info


def construct_es(exact_terms, must_terms, must_not_terms, expansion, expansion_score,
                 weekdays, start_time, end_time, dates,
                 region, location, gps_bounds=None, extra_filter_scripts=None, group_factor="group",
                 use_exact_scores=False, size=1000, starting_from=0):
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
    filter_queries = []
    must_not_queries = []

    # MUST
    if expansion:
        must_queries.append({"terms": {
            "scene_concepts":  expansion}})

    if region:
        must_queries.append({"terms_set": {"region": {"terms": region,
                                                      "minimum_should_match_script": {
                                                          "source": "1"}}}})

    if location:
        should_queries.append({"match": {"location": {"query": ' '.join(location)}}})

    # FILTERS
    if weekdays:
        filter_queries.append({"terms": {"weekday": weekdays}})

    script = extra_filter_scripts if extra_filter_scripts else []
    script += date_filters + time_filters

    if script:
        script = "&&".join(script)
        filter_queries.append({"script": {
            "script": {
                "source": script
            }}})

    if gps_bounds:
        filter_queries.append(get_gps_filter(gps_bounds))

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
        main_query["filter"] = filter_queries[0] if len(
            filter_queries) == 1 else filter_queries
    main_query = {"bool": main_query}

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

    scores = dict(sorted(scores.items(), key=lambda x: -x[1]))

    for word in scores:
        functions.append({"filter": {"term": {"descriptions_and_mc":
            word}}, "weight": scores[word] * (freq[word] if word in freq else 1)})
        functions.append({"filter": {"term": {"scene_concepts":
            word}}, "weight": scores[word] * (freq[word] if word in freq else 1)})

    # MUST NOT
    if must_not_terms:
        print("Must not:", must_not_terms)
        for word in must_not_terms:
            functions.append({"filter": {"term": {"descriptions_and_mc_not":
                word}}, "weight": 150 / len(must_not_terms)})
            functions.append({"filter": {"term": {"scene_concepts_not":
                word}}, "weight": 150 / len(must_not_terms)})

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
    print(starting_from)

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
                  "location": location}

    return group_results(post_request(json.dumps(json_query), "lsc2020"), group_factor), query_info


def individual_es_from_info(query_info, gps_bounds=None, extra_filter_scripts=None, group_factor="group", size=2000, starting_from=2000):
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
    if isinstance(expansion_score, tuple):
        expansion_score = expansion_score[0]
    return construct_es(exact_terms, must_terms, must_not_terms, expansion, expansion_score,
                        weekdays, start_time, end_time, dates,
                        region,
                        location, gps_bounds, extra_filter_scripts, group_factor,
                        use_exact_scores=True, size=size, starting_from=starting_from)


def individual_es(query, gps_bounds=None, extra_filter_scripts=None, group_factor="group", size=2000, starting_from=0):
    includes = ["id",
                "image_path",
                "time",
                "gps",
                "scene",
                "group",
                "before",
                "after"]
    if not query:
        return query_all(includes, "lsc2020", group_factor)
    info, keywords, region, location, weekdays, start_time, end_time, dates = process_query2(
        query)

    if not (location or keywords or info or weekdays or region or must_not_terms):
        return query_all(includes, "lsc2020", group_factor)

    must_not_terms = {}

    exact_terms, must_terms, expansion, expansion_score = process_string(
        info, keywords, must_not_terms)

    return construct_es(exact_terms, must_terms, must_not_terms, expansion, expansion_score,
                        weekdays, start_time, end_time, dates,
                        region,
                        location, gps_bounds, extra_filter_scripts, group_factor, size=size, starting_from=starting_from)


def forward_search(query, conditional_query, condition, time_limit, gps_bounds=None):
    (main_events, scores, _) =  individual_es(
        query, gps_bounds, size=1000, group_factor="scene")
    extra_filter_scripts = []

    for time_group in find_time_span(main_events):
        if condition == "before":
            time = datetime.strftime(
                time_group["begin_time"], "%Y, %m, %d, %H, %M, %S")
            time = ', '.join([str(int(i)) for i in time.split(', ')])
            time = f"ZonedDateTime.of({time}, 0, ZoneId.of('Z'))"
            script = f" 0 < ChronoUnit.HOURS.between(doc['time'].value, {time}) &&  ChronoUnit.HOURS.between(doc['time'].value, {time}) < {float(time_limit) + 2} "
        else:
            time = datetime.strftime(
                time_group["end_time"], "%Y, %m, %d, %H, %M, %S")
            time = ', '.join([str(int(i)) for i in time.split(', ')])
            time = f"ZonedDateTime.of({time}, 0, ZoneId.of('Z'))"
            script = f" 0 < ChronoUnit.HOURS.between({time}, doc['time'].value) &&  ChronoUnit.HOURS.between({time}, doc['time'].value) < {float(time_limit)+ 2} "
        extra_filter_scripts.append(f"({script})")
    extra_filter_scripts = [f''"||".join(extra_filter_scripts)]
    (conditional_events, scores_cond, _), _ = individual_es(conditional_query, size = 10000,
                                          extra_filter_scripts=None)

    return main_events, conditional_events, extra_filter_scripts, scores, scores_cond


def add_pairs(main_events, conditional_events, condition, time_limit, scores, scores_cond):
    pair_events = []
    total_scores = []
    for main_event, s1 in zip(main_events, scores):
        for conditional_event, s2 in zip(conditional_events, scores_cond):
            if condition == "after" and timedelta() < conditional_event["begin_time"] - main_event["begin_time"] < timedelta(hours=float(time_limit) + 2):
                pair_events.append({"current": main_event["current"],
                                    "before": main_event["before"],
                                    "after": conditional_event["current"],
                                    "begin_time": main_event["begin_time"],
                                    "end_time": main_event["end_time"]})
            elif condition == "before" and timedelta() < main_event["begin_time"] - conditional_event["begin_time"] < timedelta(hours=float(time_limit) + 2):
                pair_events.append({"current": main_event["current"],
                                    "before": conditional_event["current"],
                                    "after": main_event["after"],
                                    "begin_time": main_event["begin_time"],
                                    "end_time": main_event["end_time"]})
                total_scores.append(s1 + 0.5 * s2)

    return pair_events, total_scores


def es_two_events(query, conditional_query, condition, time_limit, gps_bounds, return_extra_filter=False):
    if not time_limit:
        time_limit = "1"
    else:
        time_limit = time_limit.strip("h")
    # Forward search
    main_events, conditional_events, extra_filter_scripts, scores, scores_cond = forward_search(query, conditional_query,
                                                                           condition, time_limit, gps_bounds)
    pair_events, total_scores = add_pairs(
        main_events, conditional_events, condition, time_limit, scores, scores_cond)

    # Backward search
    conditional_events, main_events, _, scores, scores_cond = forward_search(
        conditional_query, query, "before" if condition == "after" else "after", time_limit)
    new_pair_events, new_scores = add_pairs(main_events,
                             conditional_events, condition, time_limit, scores, scores_cond)
    pair_events += new_pair_events
    total_scores += new_scores
    # print(len(pair_events))
    if return_extra_filter:
        return pair_events, extra_filter_scripts, total_scores
    else:
        return pair_events, total_scores


def es_three_events(query, before, beforewhen, after, afterwhen, gps_bounds):
    if not afterwhen:
        afterwhen = "1"
    else:
        afterwhen = afterwhen.strip('h')
    if not beforewhen:
        beforewhen = "1"
    else:
        beforewhen = afterwhen.strip('h')
    before_pairs, extra_filter_scripts, total_scores = es_two_events(
        query, before, "before", beforewhen, gps_bounds, return_extra_filter=True)
    (after_events, scores, _) =  individual_es(after,
                                    size=5000, extra_filter_scripts=extra_filter_scripts)
    # print(len(before_pairs), len(after_events))

    pair_events = []
    pair_scores = []
    for before_pair, s1 in zip(before_pairs, total_scores):
        for after_event, s2 in zip(after_events, scores):
            if timedelta() < after_event["begin_time"] - before_pair["end_time"] < timedelta(hours=float(afterwhen) + 2):
                pair_events.append({"current": before_pair["current"],
                                    "before": before_pair["before"],
                                    "after": after_event["current"],
                                    "begin_time": before_pair["begin_time"],
                                    "end_time": before_pair["end_time"]})
                pair_scores.append(s1 + 0.5 * s2)
    return pair_events, pair_scores


if __name__ == "__main__":
    query = "woman in red top"
    info, keywords, region, location, weekdays, start_time, end_time, dates = process_query2(
        query)
    exact_terms, must_terms, expansion, expansion_score = process_string(
        info, keywords, [])
    print(exact_terms, must_terms, expansion, expansion_score)
