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
from tinydb import TinyDB, Query
from datetime import datetime


COMMON_PATH = os.getenv('COMMON_PATH')
group_info = json.load(open(f"{COMMON_PATH}/group_info.json"))
scene_info = json.load(open(f"{COMMON_PATH}/scene_info.json"))
grouped_info_dict = json.load(open(f"{COMMON_PATH}/basic_dict.json"))
db = TinyDB('/mnt/DATA/tlduyen/LQA/anno.json')
Desc = Query()

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
    date = request.GET.get('date')
    groups = [[image for scene in group for image in scene_info[date][scene]] for group in group_info[date].values()]
    names = list(group_info[date].keys())
    descs = [db.get(Desc.scene==name) for name in names]
    descs = [desc["desc"] if desc else None for desc in descs]
    times = [[datetime.strptime(grouped_info_dict[image]["time"], "%Y/%m/%d %H:%M:%S+00").strftime("%H:%M") for image in group] for group in groups]
    scenes = [{"images" : images, "name": name, "time": time} for (images, name, time) in zip(groups, names, times)]
    gps = [[grouped_info_dict[image]["gps"] for image in group] for group in groups]
    response = {"scenes": scenes, "gps": gps, "descriptions": descs}
    return jsonize(response)

@csrf_exempt
def update(request):
    message = json.loads(request.body.decode('utf-8'))
    desc, scene = message['desc'], message['scene']
    if desc == "Descriptions...":
        return jsonize({"status": "IGNORE"})
    # Find existing
    doc = db.get(Desc.scene == scene)
    if doc is None:
        db.insert({'scene': scene, 'desc': desc})
    else:
        db.upsert({'scene': scene, 'desc': desc}, Desc.scene==scene)
    return jsonize({"status": "SUCCESS"})
