def create_base_query(num_of_result, includes, bool_query):
    return {
        "size": num_of_result,
        "_source": {
            "includes": includes
        },
        "query": bool_query
    }


def create_query_string(query, field, boost=1):
    result = {
        "query_string": {
            "query": query,
            "default_field": field,
            "boost": boost
        }
    }
    return result


def create_script_score(source, params):
    query = {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": source
            }
        }
    }
    if params:
        query["script_score"]["script"]["params"] = params

    return query


def create_script_query(source, params):
    query = {
        "script": {
            "source": source
        },
        "boost": 100
    }
    if params:
        query["script"]["params"] = params

    return query


def create_term_query(terms, field, boost):
    return {
        "terms": {
            field: terms,
            "boost": boost
        }
    }


def create_dismax_query(queries, tie_breaker):
    return {
        "dis_max": {
            "queries": queries,
            "tie_breaker": tie_breaker
        }
    }


def create_geo_distance(distance, lat, lon):
    return {
        "distance": "0km",
        "location": {
            "lat": 53.3932218,
            "lon": -6.2632873
        },
        "_name": "gps"
    }


def create_script_filter(source, params):
    query = {
        "script": {
            "script": {
                "source": source
            }
        }
    }
    if params:
        query["script"]["script"]["params"] = params

    return query


def create_bool_query(must_query, filter_query):
    if filter_query:
        return {"bool": {"must": must_query, "filter": filter_query}}
    else:
        return {"bool": {"must": must_query}}


def get_gps_filter(bounds):
    return {"geo_bounding_box": {
        "gps": {
            "top_left": {"lon": float(bounds[0]), "lat": float(bounds[3])},
            "bottom_right": {"lon": float(bounds[2]), "lat": float(bounds[1])}
        }
    }}

def create_time_range_query(start, end):
    return {
            "range":
            {
                "timestamp":
                {
                    "gte": start,
                    "lte": end,
                }
                }
            }
