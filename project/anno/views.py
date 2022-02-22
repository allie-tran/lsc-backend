import os
import json
from collections import defaultdict, Counter
import csv
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
import random
from unicodedata import normalize
from sklearn.metrics.pairwise import cosine_distances
import requests
from tinydb import TinyDB, Query
from pymongo import MongoClient
from datetime import datetime
import random
from lifelog_utils import generate_question, get_no_question_from_distractor, load_all

COMMON_PATH = os.getenv('COMMON_PATH')
ORIGINAL_LSC = os.getenv('ORIGINAL_LSC')

MNT = os.getenv('MNT')
load_all()

grouped_info_dict = json.load(open(f"{COMMON_PATH}/basic_dict.json"))
time_info = json.load(open(f"{COMMON_PATH}/time_info.json"))
gps_dict = {image: (info["gps"], info["location"]) for image, info in json.load(
    open(f"{COMMON_PATH}/grouped_info_dict.json")).items()}

captions = json.load(
    open("/home/tlduyen/LSC2020/original/microsoft_captions.json"))
group_segments = json.load(open(f"{COMMON_PATH}/group_segments.json"))
scene_segments = {}
for date, groups in group_segments.items():
    scene_segments[date] = {}
    for group_name, scenes in groups.items():
        for scene_name, images in scenes.items():
            assert "_S" in scene_name, f"{scene_name} is not a valid scene id"
            scene_segments[date][scene_name] = images

dates = list(group_segments.keys())

client = MongoClient()
qa_db = client.qa_2018.posts
caption_db = client.caption_2018.posts

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

frames = defaultdict(lambda: [])
for image in grouped_info_dict:
    frames[image.split('/')[0]].append(image)

def jsonize(response):
    # JSONize
    response = JsonResponse(response)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Credentials"] = "true"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response

def format_questions(questions, no_questions):
    text = []
    for qa in questions:
        text.append(
            f"S: {qa['source']}" + "\n" + \
            f"Q: {qa['question']}" + "\n" + \
                qa['answer'])
    text.append("NO QUESTIONS")
    for qa in no_questions:
        text.append(
            f"S: {qa['source']}" + "\n" +
            f"Q: {qa['question']}" + "\n" +
                qa['answer'])
    return "\n\n".join(text)

def text_to_questions(text):
    qas = []
    text = text.split("\nNO QUESTIONS")[0].strip()
    for qa in text.split("\n\n"):
        lines = qa.strip().split("\n")
        qas.append({"source": lines[0].replace("S: ", "").strip(),
                    "question": lines[1].replace("Q: ", "").strip(),
                    "answer": "\n".join(lines[2:])})
    return qas

@csrf_exempt
def update(request):
    message = json.loads(request.body.decode('utf-8'))
    desc, scene = message['desc'], message['scene']
    desc = desc.strip()
    if desc == "Descriptions...":
        return jsonize({"status": "IGNORE"})
    # Find existing
    doc = caption_db.find_one({"scene": scene})
    if doc is None:
        caption_db.insert_one({'scene': scene, 'desc': desc})
    else:
        caption_db.update_one({'scene': scene}, {"$set": {'desc': desc}}, upsert=False)
    if desc:
        date = scene.split("_")[0]
        images = scene_segments[date][scene]
        questions = generate_question(desc, images)
        no_questions = get_no_question_from_distractor(questions)
        qas = qa_db.find_one({"scene": scene})
        if qas is None:
            qa_db.insert_one(
                {'scene': scene, 'questions': questions, 'no_questions': no_questions})
        else:
            qa_db.update_one({"scene": scene}, {
                             "$set": {'questions': questions, 'no_questions': no_questions}}, upsert=False)
    else:
        qa_db.find_one_and_delete({"scene": scene})
    return jsonize({"status": "SUCCESS"})

@csrf_exempt
def process(request):
    # processes = [round(db.count(User.scene.matches(f'*{date}*')) / length[date], 2)  for date in dates]
    return jsonize({"processes": []})


@csrf_exempt
def update_qa(request):
    message = json.loads(request.body.decode('utf-8'))
    desc, scene = message['desc'], message['scene']
    qas = text_to_questions(desc)
    print(qas)
    no_questions = get_no_question_from_distractor(qas)
    qa_db.update_one(
        {'scene': scene}, {"$set": {'questions': qas, 'no_questions': no_questions}}, upsert=False)
    return jsonize({"status": "SUCCESS"})

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


# =========================================================================== #
# =========================================================================== #
# NEW API FUNCTION #
# =========================================================================== #
# =========================================================================== #
@csrf_exempt
def get_group(request):
    date = request.GET.get('date')
    groups = []
    notdone_position = 0
    for i, group in enumerate(group_segments[date]):
        group_done = True
        for scene_name in group_segments[date][group]:
            if not caption_db.find_one({'scene': scene_name}):
                group_done = False
                if notdone_position == 0:
                    notdone_position = i
                break
        groups.append(
            (group, list(group_segments[date][group].values())[0][0], "Group" + group.split("_")[-1], group_done))
    datetime_value = datetime.strptime(date, "%Y-%m-%d")
    detail = datetime_value.strftime("%A, %d %B, %Y")
    return jsonize({"timeline": groups, "notdone": notdone_position, "detail": detail})


@csrf_exempt
def get_scene(request):
    group = request.GET.get('group')
    no_empty = request.GET.get('noEmpty') == "true"
    date = group.split("_")[0]
    scene_names = list(group_segments[date][group].keys())
    scenes = []
    notdone_position = 0
    for i, scene_name in enumerate(scene_names):
        done = True
        desc = caption_db.find_one({'scene': scene_name})
        if not desc:
            done = False
            if notdone_position == 0:
                notdone_position = i
        included = True
        if no_empty:
            if not desc or not desc["desc"] or desc["desc"] == "Descriptions...":
                included = False
        if included:
            scenes.append(
                (scene_name, scene_segments[date][scene_name][0], time_info[scene_name], done))
    return jsonize({"timeline": scenes, "notdone": notdone_position})


@csrf_exempt
def get_desc(request):
    scene = request.GET.get('scene')
    date = scene.split("_")[0]
    images = scene_segments[date][scene]
    desc = caption_db.find_one({'scene': scene})
    if not desc:
        desc = ""
        # caps = []
        # for image in images:
        #     if captions[image] not in caps:
        #         caps.append(captions[image])
        # desc = "\n".join(caps)
    else:
        desc = desc["desc"]
    return jsonize({"images": images, "desc": desc})

@csrf_exempt
def get_gps(request):
    scene = request.GET.get('scene')
    date = scene.split("_")[0]
    images = scene_segments[date][scene]
    gps_points = [gps_dict[image][0] for image in images]
    location = gps_dict[images[0]][1]
    return jsonize({"gps": gps_points, "location": location})

@csrf_exempt
def get_question(request):
    scene = request.GET.get('scene')
    qas = qa_db.find_one({"scene": scene})
    if qas is None:
        desc = caption_db.find_one({'scene': scene})
        if not desc or not desc["desc"] or desc["desc"] == "Descriptions...":
            return jsonize({"status": "No descriptions!"})
        else:
            desc = desc["desc"]
            date = scene.split("_")[0]
            images = scene_segments[date][scene]
            questions = generate_question(desc, images)
            no_questions = get_no_question_from_distractor(questions)
            qa_db.insert_one(
                {'scene': scene, 'questions': questions, 'no_questions': no_questions})
            return jsonize({"questions": format_questions(questions, no_questions)})
    else:
        questions = qas["questions"]
        no_questions = qas["no_questions"]
        return jsonize({"questions": format_questions(questions, no_questions)})
