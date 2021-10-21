import os
import json
from collections import defaultdict, Counter
import csv
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
import random
from sklearn.metrics.pairwise import cosine_distances
import requests
from tinydb import TinyDB, Query
from pymongo import MongoClient
from datetime import datetime
import random

COMMON_PATH = os.getenv('COMMON_PATH')
MNT = os.getenv('MNT')
db = TinyDB('/home/tlduyen/LQA/mnt/anno.json')
Desc = Query()

group_info = json.load(open(f"{COMMON_PATH}/group_info.json"))
scene_info = json.load(open(f"{COMMON_PATH}/scene_info.json"))
grouped_info_dict = json.load(open(f"{COMMON_PATH}/basic_dict.json"))
dates = list(group_info.keys())


if os.path.isfile('all_qa.json'):
    all_qas = json.load(open('all_qa.json'))
else:
    all_qas = defaultdict(lambda : [])
    for type_file in ["train", "valid", "test"]:
        filename = f"{MNT}/tvqa/lifelog/release/lifelog_{type_file}_multiple.jsonl"
        with open(filename) as f:
            for line in f.readlines():
                qa = json.loads(line)
                all_qas[qa["vid_name"]].append(qa)
    for date in all_qas:
        all_qas[date] = sorted(all_qas[date], key=lambda qa: qa["qid"])
    with open('all_qa.json', 'w') as f:
        json.dump(all_qas, f)

info = json.load(open(f"{COMMON_PATH}/grouped_info_dict.json"))
frames = defaultdict(lambda: [])
for image in info:
    frames[image.split('/')[0]].append(image)

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
    global names
    date = request.GET.get('date')

    date = request.GET.get('date')
    groups = [[image for scene in group for image in scene_info[date][scene]]
              for group in group_info[date].values()]
    names = list(group_info[date].keys())

    descs = [db.get(Desc.scene == name) for name in names]
    descs = [desc["desc"] if desc else None for desc in descs]

    qas = []

    for scene in names:

        scene_keys = sorted(group_info[date][scene].keys())
        start_scene = [image for image in group_info[date]
                         [scene][scene_keys[0]] if image.split('/')[0] == date]
        end_scene = [image for image in group_info[date][scene]
                        [scene_keys[-1]] if image.split('/')[0] == date]
        try:
            position_start = start_scene[0]
            position_start = frames[date].index(position_start)
        except Exception as e:
            position_start = 0
        try:
            position_end = end_scene[-1]
            position_end = frames[date].index(position_end)
        except Exception as e:
            position_end = len(frames[date])

        relevant_qas = []
        for i, qa in enumerate(all_qas[date]):
            images = []
            true_start = qa["true_ts"].split('-')[0]
            if position_start<=int(true_start)<=position_end:
                relevant_qas.append((qa["qid"], qa))
        qas.append(relevant_qas)
    response = {"descriptions": descs, 'qas': qas}
    return jsonize(response)

@csrf_exempt
def update(request):
    global names
    message = json.loads(request.body.decode('utf-8'))
    newdesc, date, qid, key = message['newdesc'], message['date'], message['qid'], message['key']
    qas = []
    print(date, qid, key, newdesc)
    # Find existing
    for i, qa in enumerate(all_qas[date]):
        if qa["qid"] == qid:
            print(qa)
            print(qa[key], newdesc)
            qa["edited"] = True
            # qa["qid"]["key"] = newdesc
            # with open('all_qa.json', 'w') as f:
            #     json.dump(all_qas, f)
        qas.append([(qa["qid"], qa)])
    return jsonize({"status": "SUCCESS", 'qas': qas})
