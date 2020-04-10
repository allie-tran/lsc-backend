import os
import json
from collections import defaultdict
import csv
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
import random
from sklearn.metrics.pairwise import cosine_distances

COMMON_PATH = os.getenv('COMMON_PATH')
feature_dict = joblib.load(f'{COMMON_PATH}/feature_dict.joblib')
images = list(feature_dict.keys())
model = joblib.load(f'{COMMON_PATH}/image_model.joblib')
distance_matrix = joblib.load(
    f'{COMMON_PATH}/distance_matrix.joblib')


def get_neighbors(image):
    img_index = images.index(image)
    dist = distance_matrix[img_index]
    k_nearest = model.kneighbors([dist])
    distance = k_nearest[0][0]
    k_nearest_index = k_nearest[1][0]
    rank_list = [images[index] for index in k_nearest_index][:100]
    return rank_list


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
    image = json.loads(request.body.decode('utf-8'))["image"]
    image = random.choice(images)
    similar_images = get_neighbors(image)
    response = {"image": image, "images": similar_images}
    return jsonize(response)
