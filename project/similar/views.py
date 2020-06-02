import os
import json
from collections import defaultdict
import csv
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
import random
from sklearn.metrics.pairwise import cosine_distances
import requests

COMMON_PATH = os.getenv('COMMON_PATH')
images = json.load(open(
    '/home/tlduyen/LSC2020/common_full/full_similar_images.json'))
grouped_info_dict = json.load(open(f"{COMMON_PATH}/grouped_info_dict.json"))


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


def get_neighbors(image):
    img_index = images.index(image)
    if img_index >= 0:
        request = {
            "size": 1,
            "_source": {
                "includes": ["similars"]
            },
            "query": {
                "term": {"image_index": img_index}
            }
        }

        results = post_request(json.dumps(request), "lsc2020_similar")
        if results:
            return [images[r] for r in results[0][0]["similars"]][:500]
    return []


def jsonize(response):
    # JSONize
    response = JsonResponse(response)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Credentials"] = "true"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response


@csrf_exempt
def index(request):
    image = request.GET.get('image_id')
    if image not in images:
        image = random.choice(images)
    similar_images = [image for image in get_neighbors(image) if image in grouped_info_dict]
    response = {"image": image, "images": similar_images}
    return jsonize(response)


@csrf_exempt
def group(request):
    image = request.GET.get('image_id')
    if image not in images:
        image = random.choice(images)
    similar_images = get_neighbors(image)[:100]
    scenes = defaultdict(lambda: [])
    for image in similar_images:
        if image in grouped_info_dict:
            scenes[grouped_info_dict[image]["scene"]].append(image)
    response = {"scenes": list(scenes.values())[:25]}
    return jsonize(response)
