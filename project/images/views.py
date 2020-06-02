import json
import os
import time
from collections import defaultdict

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from images.query import es, es_date, get_gps, get_timeline, get_timeline_group

saved = defaultdict(lambda : [])

def jsonize(response):
    # JSONize
    response = JsonResponse(response)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Credentials"] = "true"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response


@csrf_exempt
def save(request):
    global saved
    image = request.GET.get('image_id')
    query_id = request.GET.get('query_id')
    if image not in saved[query_id]:
        saved[query_id].append(image)
    return jsonize({"success": True})


@csrf_exempt
def restart(request):
    global saved
    saved=defaultdict(lambda: [])
    return jsonize({"success": True})

@csrf_exempt
def remove(request):
    global saved
    query_id = request.GET.get('query_id')
    image = request.GET.get('image_id')
    saved[query_id] = [img for img in saved[query_id] if img != image]
    return jsonize({"success": True})

@csrf_exempt
def export(request):
    global saved
    json.dump(saved, open(f'results_{time.time()}.json', 'w'))
    return jsonize({"success": True})

@csrf_exempt
def images(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    # Calculations
    queryset, scores, info = es(message['query'], message["gps_bounds"])
    response = {'results': queryset[:100], 'info': info}
    return jsonize(response)

@csrf_exempt
def get_saved(request):
    global saved
    query_id = request.GET.get('query_id')
    return jsonize({"saved": saved[query_id]})

@csrf_exempt
def gps(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    # Calculations
    gps = get_gps([message['image']])[0]
    response = {'gps': gps}
    return jsonize(response)


@csrf_exempt
def date(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    # Calculations
    queryset, info = es_date(message['query'], message["gps_bounds"])
    response = {'results': queryset[:100], 'info': info}
    return jsonize(response)

@csrf_exempt
def timeline_group(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    timeline = get_timeline_group(message['date'])
    response = {'timeline': timeline}
    return jsonize(response)


@csrf_exempt
def timeline(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    timeline, position, group = get_timeline(
        message['images'], message["timeline_type"], message["direction"])
    response = {'timeline': timeline, 'position': position, 'group': group}
    return jsonize(response)


@csrf_exempt
def gpssearch(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    # Calculations
    images = message["scenes"]
    display_type = message["display_type"]
    queryset = es_gps(es, message['query'], images, display_type)
    response = {'results': queryset,
                'error': None}
    return jsonize(response)
