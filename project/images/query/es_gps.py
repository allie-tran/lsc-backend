import json
import os
from ..nlp_utils.common import basic_dict

def get_query_request(bounds):
    return {
        "_source": {
            "includes": ["image_path",
                         "descriptions",
                         "location",
                         "weekday",
                         "time",
                         "gps"]
        },
        "query": {
            "bool": {
                "must": {
                    "match_all": {}
                },
                "filter": {
                    "geo_bounding_box": {
                        "location": {
                            "top_left": {"lon": float(bounds[0]), "lat": float(bounds[3])},
                            "bottom_right": {"lon": float(bounds[2]), "lat": float(bounds[1])}
                        }
                    }
                }
            }
        }
    }


def get_gps(image):
    return basic_dict[image]["gps"]


def gps_search(es, bounds, images, display_type):
    if display_type == 'normal':
        query_request = get_query_request(bounds)
        if images:
            image_paths = []
            for period in images:
                image_paths.extend([res["image_path"] for res in period[1]])
            image_filter = {
                "terms": {
                    "image_path": image_paths
                }
            }
            query_request["query"]["bool"]["filter"] = [
                query_request["query"]["bool"]["filter"], image_filter]
        query_request_json = json.dumps(query_request)
        # Show all result (9999 results if possible)
        res = es.search(index="lsc2020", body=query_request_json, size=50)
        id_result = [[r["_source"], r["_score"]] for r in res['hits']['hits']]

        return group_time(id_result)
    else:
        new_periods = []
        for period in images:
            query_request = get_query_request(bounds)
            image_filter = {
                "terms": {
                    "id": [res["id"] for res in period[1][0]]
                }
            }
            query_request["query"]["bool"]["filter"] = [
                query_request["query"]["bool"]["filter"], image_filter]
            query_request_json = json.dumps(query_request)
            res = es.search(index="lsc2019_combined_text_bow", body=query_request_json,
                            size=50)  # Show all result (9999 results if possible)
            id_result = [r["_source"] for r in res['hits']['hits']]
            if id_result:
                new_periods.append((period[0], (id_result, period[1][0])))

        return new_periods
