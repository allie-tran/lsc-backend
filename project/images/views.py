import json
import os
import time
from collections import defaultdict

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
from images.query import *
import logging
import requests

sessionId = None
last_scroll_id = None
last_results = None
# server = "https://vbs.videobrowsing.org/api/v1"
server = "http://localhost:8003/"

# Get logger
logger = logging.getLogger('lsc23')
logger.propagate = False
logger.setLevel(logging.DEBUG)
# Create a handler
f_handler = logging.FileHandler('lsc23.log')
logger.addHandler(f_handler) # <-- THIS!
# test
logger.debug("Started logging")

def jsonize(response):
    # JSONize
    response = JsonResponse(response)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Credentials"] = "true"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response


@csrf_exempt
def login(request):
    global sessionId
    global server

    session_id = request.GET.get('session_id')
    if session_id and session_id != "vbs":
        server = "http://localhost:8003/"
        sessionId = session_id
        logger.info(f"{int(time.time())}, LSCBLANK, login, {sessionId}")
        return jsonize({"description": f"Login successful: {sessionId}!"})
    else:
        server = "https://vbs.videobrowsing.org/api/v1"
        url = f"{server}/login/"
        res = requests.post(url, json={"username": "myeachtra",
                                    "password": "mQFbxFf9Cx3q"})
        print(res)
        if res.status_code == 200:
            sessionId = res.json()["sessionId"]
            logger.info(f"{int(time.time())}, LSCBLANK, login, {sessionId}")
            return jsonize({"description": f"Login successful: {sessionId}!"})
        else:
            logger.info(f"{int(time.time())}, LSCBLANK, login, Login error!")
            return jsonize({"description": "Login error!"})

@csrf_exempt
def export(request):
    global sessionId
    image = request.GET.get('image_id')
    scene = str(request.GET.get('scene')) == "true"
    qa_answer = str(request.GET.get('qa_answer')) == "true"
    results = []
    if qa_answer:
        url = f"{server}/submit?text={image}&session={sessionId}"
        res = requests.get(url)
        res = res.json()["description"]
        results.append(f'{image}: {res}')
        logger.info(f"{int(time.time())}, LSCBLANK, result-submit, {image}, {res}")
    else:
        for i, item in enumerate(get_submission(image, scene)):
            url = f"{server}/submit?item={item}&session={sessionId}"
            res = requests.get(url)
            res = res.json()["description"]
            results.append(f'{i + 1}. {item}: {res}')
            logger.info(f"{int(time.time())}, LSCBLANK, result-submit, {item}, {res}")
    return jsonize({"description": "\n".join(results)})

@csrf_exempt
def submit_all(request):
    # submitting saved section
    global sessionId
    message = json.loads(request.body.decode('utf-8'))
    saved_scenes = message["saved"]
    results = []
    submissions = []
    for first in range(0, len(saved_scenes), 2):
        last = first + 1
        submissions.extend(get_image_list(saved_scenes[first][0], saved_scenes[last][-1]))
    for i, item in enumerate(submissions):
        url = f"{server}/submit?item={item}&session={sessionId}"
        res = requests.get(url)
        res = res.json()["description"]
        results.append(f'{i + 1}. {item}: {res}')
        logger.info(f"{int(time.time())}, LSCBLANK, result-submit, {item}, {res}")
    return jsonize({"description": "\n".join(results)})

def form_question(current, before, after):
    current = current.strip(" ?\n")
    if before and after:
        return f"After {before} and before {after}, {current}?"
    elif before:
        return f"After {before}, {current}?"
    elif after:
        return f"Before {after}, {current}?"
    else:
        return f"{current}?"
        
@csrf_exempt
def images(request):
    global last_scroll_id
    global last_results
    global messages
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    print("=" * 80)
    message_to_log = {"query": {key: value for key, value in message["query"].items() if key != "info"},
                      "gps_bounds": message["gps_bounds"]}
    question = message["query"]["current"]
    question = form_question(message["query"]["current"], message["query"]["before"], message["query"]["after"])
    
    # Calculations
    scroll_id, queryset, scores, info = es(message['query'], 
                                           message["gps_bounds"], 
                                           message["size"] if "size" in message else 100,
                                           message['share_info'],
                                           message["query"]["isQuestion"])
    message["query"]["info"] = info
    if last_scroll_id:
        try:
            response = requests.delete(
                f"http://localhost:9200/_search/scroll", headers={"Content-Type": "application/json"}, data=json.dumps({"scroll_id": last_scroll_id}))
            assert response.status_code == 200, f"Wrong request ({response.status_code})"
        except:
            pass
    last_scroll_id = scroll_id
    last_results = (queryset, scores)
    
    #LSC23!
    texts = []
    print("Answering question:", question)
    start = time.time()
    if message["query"]["isQuestion"] and "?" in message["query"]["current"]:
        texts = answer_topk_scenes(question, queryset, scores, k=10)
    print("Answering time:", time.time() - start)
        
    response = {'results': queryset, 
                'size': len(queryset), 
                'info': info, 
                'more': False, 
                'scores': scores, 
                "texts": texts}
    
    def get_images(scene):
        images = []
        for key in ["before", "current", "after"]:
            if key in scene:
                images.extend([image[0] for image in scene[key]])
        return images
    result_to_log = {"images": [get_images(x) for x in queryset], "texts": texts}
    logger.debug(f"{int(time.time())}, LSCBLANK, query-string, [{json.dumps(message_to_log)}], [{json.dumps(result_to_log)}]")
    return jsonize(response)

@csrf_exempt
def more(request):
    global last_scroll_id
    global last_results
    if last_scroll_id:
        scroll_id, queryset, scores = es_more(last_scroll_id)
        last_scroll_id = scroll_id
        last_queryset, last_scores = last_results
        last_queryset.extend(queryset)
        last_scores.extend(scores)
        last_results = (last_queryset, last_scores)
        response = {'results': queryset, 'size': len(queryset), 'more': True, 'scores': scores}
    else:
        response = {'results': [], 'size': 0, 'more': True, 'scores': []}
    return jsonize(response)

@csrf_exempt
def gps(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    # Calculations
    gps = get_gps([message['image']])[0]
    location = get_location(message['image'])
    response = {'gps': gps, "location": location}
    return jsonize(response)

@csrf_exempt
def timeline(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    timeline, line, space, scene_id = get_all_scenes(message['images'])
    response = {'timeline': timeline, 'line': line, 'space': space, 'scene_id': scene_id, 'image': message['images'][0]}
    return jsonize(response)

@csrf_exempt
def more_scenes(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    timeline, line, space = get_more_scenes(message['group'], message['direction'])
    response = {'timeline': timeline, 'direction': message['direction'], 'line': line, 'space': space}
    return jsonize(response)

@csrf_exempt
def detailed_info(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    if 'image' in message:
        info = get_date_info(message['image'])
        response = {'info': info}
    else:
        response = {'info': ""}
    return jsonize(response)


@csrf_exempt
def similar(request):
    message = json.loads(request.body.decode('utf-8'))
    image = message['image_id']
    if 'lsc' in message:
        lsc = message['lsc']
    else:
        lsc = True
    # info = message['info']
    # gps_bounds = message['gps_bounds']
    similar_images = get_neighbors(image, lsc)[:500]
    response = {"scenes": similar_images}
    return jsonize(response)

@csrf_exempt
def answer_scene(request):
    message = json.loads(request.body.decode('utf-8'))
    images = message["images"]
    images = [image[0] for image in images]
    question = message["question"]
    response = {"texts": get_answers_from_images(images, question)}
    print(response)
    return jsonize(response)

@csrf_exempt
def sort_by(request):
    global last_results
    queryset, scores = last_results

    sortby = str(request.GET.get('by'))
    maximum = 100
    if sortby == "time":
        queryset, scores = zip(*sorted(zip(queryset[:maximum], scores[:maximum]), key=lambda x: int(x[0]["scene"].split("_")[1])))
    elif sortby == "time-reverse":
        queryset, scores = zip(*sorted(zip(queryset[:maximum], scores[:maximum]), key=lambda x: int(x[0]["scene"].split("_")[1]), reverse=True))
    return jsonize({"results": queryset, 
                    "scores": scores, 
                    "size": len(queryset),
                    "more": False,
                    "sorted": True})