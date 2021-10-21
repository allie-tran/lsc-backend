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

group_info = json.load(open(f"{COMMON_PATH}/group_info.json"))
scene_info = json.load(open(f"{COMMON_PATH}/scene_info.json"))
grouped_info_dict = json.load(open(f"{COMMON_PATH}/basic_dict.json"))

db = TinyDB('/home/tlduyen/LQA/mnt/anno.json')
Desc = Query()
dates = list(group_info.keys())
length = {date: len(group_info[date]) for date in dates}

client = MongoClient()
db2 = client.pymongo_test.posts
test_file = f"{MNT}/tvqa/full_lifelog/lifelog_test_binary.jsonl"
all_test_qas = []
with open(test_file) as f:
    for line in f.readlines():
        all_test_qas.append(json.loads(line))
test_multiple_file = f"{MNT}/tvqa/full_lifelog/lifelog_test_multiple.jsonl"
all_test_multiple_qas = []
with open(test_multiple_file) as f:
    for line in f.readlines():
        all_test_multiple_qas.append(json.loads(line))
# all_qas = []
# for type_file in ["train", "valid", "test"]:
#     filename = f"{MNT}/tvqa/lifelog/release/lifelog_{type_file}_multiple.jsonl"
#     with open(filename) as f:
#         for line in f.readlines():
#             all_qas.append(json.loads(line))
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
    date = request.GET.get('date')
    groups = [[image for scene in group for image in scene_info[date][scene]] for group in group_info[date].values()]
    names = list(group_info[date].keys())
    descs = [db.get(Desc.scene==name) for name in names]
    descs = [desc["desc"] if desc else None for desc in descs]
    times = [[datetime.strptime(grouped_info_dict[image]["time"], "%Y/%m/%d %H:%M:%S+00").strftime("%H:%M") for image in group] for group in groups]
    scenes = [{"images" : images, "name": name, "time": time} for (images, name, time) in zip(groups, names, times)]
    gps = [[grouped_info_dict[image]["gps"] for image in group] for group in groups]
    # Question and answers
    qas = []
    binary, yes, multiple, no = 0, 0, 0, 0
    questions = Counter()
    # with TinyDB('/home/tlduyen/LQA/anno_qa.json') as db2:
        # QA = Query()
    for name in names:
        scene_qas = sorted(db2.find({"scene": name}), key=lambda qa: qa["i"])
        scene_qas = [(qa["i"], qa["qa"]) for qa in scene_qas]
        only_binary = []
        for i, qa in scene_qas:
            if qa["q"] and "[DEL]" not in qa["q"]:
                if qa[f"a{qa['answer_idx']}"].strip('. ') in ["Yes", "No", ""]:
                    binary += 1
                    if qa[f"a{qa['answer_idx']}"] == "Yes .":
                        yes += 1
                    else:
                        no +=1
                    # only_binary.append((i, qa))
                else:
                    if '[' not in qa["q"]:
                        multiple += 1
                        only_binary.append((i, qa))
                    questions[qa["q"].split()[0].split("'")[0].lower()] += 1
        # only_binary = sorted(only_binary, key=lambda x: x[1]["q"].split(
        #     '[')[-1].split('-')[0].strip(' ]').rjust(5, '0'))
        qas.append(only_binary)

    stats = {qu_word: questions[qu_word] for qu_word in ['what', 'where', 'who', 'how', 'when', 'which', 'why']}
    stats.update({'binary': binary, 'multiple': multiple, 'yes': yes, 'no': no})
    response = {"scenes": scenes, "gps": gps, "descriptions": descs, 'qas': qas, 'stats': stats}
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

@csrf_exempt
def process(request):
    processes = [round(db.count(User.scene.matches(f'*{date}*')) / length[date], 2)  for date in dates]
    return jsonize({"processes": processes})


@csrf_exempt
def update_qa(request):
    message = json.loads(request.body.decode('utf-8'))
    newdesc, scene, i, key = message['newdesc'], message['scene'], message['i'], message['key']
    # Find existing

    doc = db2.find_one({"scene": scene, "i": i})
    if doc is None:
        qa = {"q": "", "a0": "", "a1": "", "a2": "", "a3": "", "a4": "", "answer_idx": 0, "scene": scene, "vid_name": scene.split('_')[0]}
        qa[key] = " ".join(newdesc.split())
        print(i, qa)
        db2.insert_one({'scene': scene, 'qa': qa, 'i': i})
    else:
        qa = doc["qa"]
        qa[key] = " ".join(newdesc.split())
        print(qa)
        print("Updated")
        db2.find_one_and_update({"_id" : doc["_id"]}, {"$set":{'scene': scene, 'qa': qa, 'i': i}}, upsert=True)

    qas = []
    binary, yes, multiple, no = 0, 0, 0, 0
    questions = Counter()
    names = list(group_info[scene.split('_')[0]].keys())
    for name in names:
        scene_qas = sorted(db2.find({"scene": name}), key=lambda qa: qa["i"])

        scene_qas = [(qa["i"], qa["qa"]) for qa in scene_qas]
        only_binary = []

        for i, qa in scene_qas:
            if qa["q"] and "[DEL]" not in qa["q"]:
                if qa[f"a{qa['answer_idx']}"].strip('. ') in ["Yes", "No", "", "Np"]:
                    binary += 1
                    if qa[f"a{qa['answer_idx']}"] == "Yes .":
                        yes += 1
                    else:
                        no +=1
                    # only_binary.append((i, qa))
                else:
                    if '[' not in qa["q"]:
                        multiple += 1
                        only_binary.append((i, qa))
                    questions[qa["q"].split()[0].split("'")[0].lower()] += 1
        # only_binary = sorted(
            # only_binary, key=lambda x: x[1]["q"].split('[')[-1].split('-')[0].strip(' ]').rjust(5, '0'))
        qas.append(only_binary)

    stats = {qu_word: questions[qu_word] for qu_word in ['what', 'where', 'who', 'how', 'when', 'which', 'why']}
    stats.update({'binary': binary, 'multiple': multiple, 'yes': yes, 'no': no})
    return jsonize({"status": "SUCCESS", 'stats': stats, 'qas': qas})

FINISHED_DATE = ["2015-02-23", "2015-02-24", "2015-02-25",
                 "2015-03-01", "2015-03-02", "2015-03-03",
                 "2015-03-20"]

UNCHECKED = ["2015-02-26", "2015-02-27", "2015-02-28",
             "2015-03-04", "2015-03-05", "2015-03-06",
             "2015-03-07", "2015-03-08", "2015-03-09",
             "2015-03-10", "2015-03-11", "2015-03-12",
             "2015-03-13", "2015-03-14", "2015-03-15",
             "2015-03-16", "2015-03-17", "2015-03-18",
             "2015-03-19"]

@csrf_exempt
def get_qa_list(request):
    images = []
    qas = []

    # chosen_qas = random.choices(list(db2.find({})), k=20)
    # chosen_qas = [(qa["i"], qa["scene"], qa["qa"]) for qa in chosen_qas]

    chosen_qas = random.choices(all_test_qas, k=20)
    for i, qa in enumerate(chosen_qas):
        start, end = qa["ts"].split('-')
        start, end = int(start), int(end)
        true_start, true_end = qa["true_ts"].split('-')
        true_start, true_end = int(true_start), int(true_end)
        relevant_images = []
        range_after, range_before = 3, 3
        if "after" in qa["q"] or "before" in qa["q"]:
            range_after = 20
            range_before = 20
        for j, frame in enumerate(frames[qa["vid_name"]]):
            if j > true_start - range_before and j < true_end + range_after:
                relevant_images.append(
                    (frame, "-highlight" if true_start <= j <= true_end else ""))
        images.append(relevant_images)
        # images.append(frames[qa["vid_name"]][start:end+1])
        qas.append(('scene', i, qa))
        # date = random.choice(FINISHED_DATE + UNCHECKED)
        # group = random.choice(list(group_info[date].keys()))
        # try:
        #     qa = random.choice(list(db2.find({"scene": group})))
        #     if f'{group}_{qa["i"]}' not in ids and qa["qa"]["q"] and "[DEL]" not in qa["qa"]["q"]:
        #         ids.append(f'{group}_{qa["i"]}')
        #         qas.append((group, qa["i"], qa["qa"]))
        #         images.append([image for scene in group_info[date][group] for image in scene_info[date][scene]])
        # except IndexError as e:
        #     continue
    
    chosen_qas = random.choices(all_test_multiple_qas, k=20)
    for i, qa in enumerate(chosen_qas):
        start, end = qa["ts"].split('-')
        start, end = int(start), int(end)
        true_start, true_end = qa["true_ts"].split('-')
        true_start, true_end = int(true_start), int(true_end)
        relevant_images = []
        range_after, range_before = 3, 3
        if "after" in qa["q"]:
            range_after = 20
        if "before" in qa["q"]:
            range_before = 20
        for j, frame in enumerate(frames[qa["vid_name"]]):
            if j > true_start - range_before and j < true_end + range_after:
                relevant_images.append(
                    (frame, "-highlight" if true_start <= j <= true_end else ""))
        images.append(relevant_images)
        # images.append(frames[qa["vid_name"]][start:end+1])
        qas.append(('scene', i, qa))

    return jsonize({"status": "SUCCESS", 'qas': qas, 'images': images})
