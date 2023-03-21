from django.shortcuts import render
from django.http import JsonResponse
from .forms import UploadFileForm, UploadProgressCachedHandler
from django.views.decorators.csrf import csrf_exempt
import subprocess, os
from utils.uploaded_data_process import *
from utils.connection_helper import *
import json, threading
from multiprocessing import Pool
from django.http import HttpResponse, HttpResponseServerError
from django.core.cache import cache
from rest_framework.decorators import api_view
from utils.mongodb_helper import MongoDBHelper
import shutil
import platform
from .settings import WINDOWS_IMAGES_FOLDER_PATH, WINDOWS_TEMPORARY_UPLOADING_FOLDER_PATH, \
     LINUX_IMAGES_FOLDER_PATH, LINUX_TEMPORARY_UPLOADING_FOLDER_PATH
from django.contrib.auth import authenticate
# Create your views here.

# Initialize view global variables

# Establish connection to MongoDB
db_helper = MongoDBHelper()
db_helper.connect_db('Deakin')


@csrf_exempt
@api_view(['POST'])
def upload_file(request):
    """
    API for file uploading, the uploaded file must be in zip format
    """
    def handle_uploaded_file(f, source_path):
        with open(source_path, 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)

    if request.method == 'POST':
        request.upload_handlers.insert(0, UploadProgressCachedHandler(request))
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            subject_id = request.POST['subject_id']
            cache_key = get_cache_key(request)
            if cache_key:
                file_name = "{}-{}".format(cache_key, subject_id)
                f = request.FILES['file']
                source_path = os.path.join(LINUX_TEMPORARY_UPLOADING_FOLDER_PATH, file_name + '.zip') # Fix the save path
                
                # Check current operating system to check for path
                if platform.system() == 'Windows':
                    source_path = os.path.join(WINDOWS_TEMPORARY_UPLOADING_FOLDER_PATH, file_name + '.zip')

                handle_uploaded_file(f, source_path)
                if not check_zip_file(source_path):
                    remove_uploaded_zip_file(source_path)
                    return HttpResponseServerError('The uploaded file is not zip file!')
                return HttpResponse(file_name)
            else:
                return HttpResponseServerError(
                   'Server Error: You must provide X-Progress-ID header or query param.')
        return HttpResponseServerError("Error")
    return HttpResponseServerError("Error")


# @csrf_exempt
# @api_view(['GET'])
# def upload_progress(request):
#     """
#     A view to report back on upload progress.
#     Return JSON object with information about the progress of an upload.

#     Copied from:
#     http://djangosnippets.org/snippets/678/

#     See upload.py for file upload handler.
#     """
#     cache_key = get_cache_key(request)
#     if cache_key:
#         data = cache.get(cache_key)
#         return HttpResponse(json.dumps(data))
#     else:
#         return HttpResponseServerError(
#             'Server Error: You must provide X-Progress-ID header or query param.')


@csrf_exempt
@api_view(['POST'])
def extract_uploaded_file(request):
    # cache_key = get_cache_key(request)
    uploaded_file_name = request.data['uploaded_file_name']
    if uploaded_file_name:
        uploaded_file_path = os.path.join(LINUX_TEMPORARY_UPLOADING_FOLDER_PATH, uploaded_file_name + '.zip')
        destination_file_path = f'{LINUX_TEMPORARY_UPLOADING_FOLDER_PATH}/{uploaded_file_name}' # Public file path

        # Check current operating system to check for path
        if platform.system() == "Windows":
            uploaded_file_path = os.path.join(WINDOWS_TEMPORARY_UPLOADING_FOLDER_PATH, uploaded_file_name + '.zip')
            destination_file_path = os.path.join(WINDOWS_TEMPORARY_UPLOADING_FOLDER_PATH, uploaded_file_name)

        if not unzip_uploaded_file(uploaded_file_path, destination_file_path):
            return HttpResponseServerError('Get error when extracing uploaded file')
        else:
            remove_uploaded_zip_file(uploaded_file_path)
    else:
        return HttpResponseServerError(
                   'Server Error: You must provide X-Progress-ID header or query param.')
    return HttpResponse(uploaded_file_name)


@csrf_exempt
@api_view(['POST'])
def insert_data_to_db(request):
    # cache_key = get_cache_key(request)
    uploaded_file_name = request.data['uploaded_file_name']
    if uploaded_file_name:
        data_path = f'{LINUX_TEMPORARY_UPLOADING_FOLDER_PATH}/{uploaded_file_name}'
        #data_path = '/Users/tuninh/DCU/Deakin/Lifelog-Annotation-Platform/public/data'
        if platform.system() == 'Windows':
            data_path = os.path.join(WINDOWS_TEMPORARY_UPLOADING_FOLDER_PATH, uploaded_file_name)

        user_schema, image_annotation_schema = dump_data_structure(data_path)
        try:
            db_helper.insert_to_collection(image_annotation_schema, collection_name='ImageAnnotation', ordered=True)
        except:
            shutil.rmtree(data_path)
            return HttpResponseServerError('The data contains duplicate image to the existing object in the database.')
        try:
            db_helper.insert_to_collection(user_schema, collection_name='User')
        except:
            object_id = user_schema[0]['user_id']
            data = user_schema[0]['data']
            db_helper.update_one(object_id, data, 'data', collection_name='User')
        # Copy data to public image source path after inserting necessary information to db
        subject_id = [d for d in os.listdir(data_path) if not d == '.DS_Store'][0]
        subject_source_path = os.path.join(data_path, subject_id)
        images_folder_path = WINDOWS_IMAGES_FOLDER_PATH if platform.system() == 'Windows' else LINUX_IMAGES_FOLDER_PATH 
        for root, dirs, files in os.walk(subject_source_path):
            for d in dirs:
                dest_folder = os.path.join(root.replace(data_path, images_folder_path), d)
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)
            for f in files:
                if f == '.DS_Store': continue  # Ignore .DS_Store file
                source_file_path = os.path.join(root, f)
                dest_file_path = source_file_path.replace(data_path, images_folder_path)
                if not os.path.exists(dest_file_path):
                    dest_folder = root.replace(data_path, images_folder_path)
                    shutil.move(source_file_path, dest_folder)
        shutil.rmtree(data_path)
    else:
        return HttpResponseServerError(
                   'Server Error: You must provide X-Progress-ID header or query param.') 
    return HttpResponse('Append to database successfully')


@csrf_exempt
@api_view(['POST'])
def update_annotation(request):
    data = request.data
    annotation = data['annotation']
    object_id = data['object_id']

    # Check current operating system to check for path
    if platform.system() == 'Windows':
        object_id = object_id.replace('/', '\\') # Use for Windows path only
        print(object_id)

    db_helper.update_one(object_id, annotation, 'annotation', collection_name='ImageAnnotation')
    db_helper.update_one(object_id, True, 'annotated', collection_name='ImageAnnotation')
    return HttpResponse('Update annotation successfully')


@csrf_exempt
@api_view(['POST'])
def get_next_unlabelled_data(request):
    """
    Get unlabelled image data.
    Use case: When user wants to jump directly to unlabelled images
    """
    data = request.data
    subject_id = data['subject_id']
    date = data['date']
    query_condition = { 'user_id': subject_id, 'date': date, 'annotated': False }
    field_filter = { '_id': 1 }
    results = db_helper.query(query_condition, field_filter, max_return=1, collection_name="ImageAnnotation")
    results = json.loads(results)
    if len(results) == 0:
        query_condition = { 'user_id': subject_id, 'date': date }
        results = db_helper.query(query_condition, field_filter, max_return=1, collection_name='ImageAnnotation')
        results = json.loads(results)
    results = results[0]
    return JsonResponse(results, safe=False)


@csrf_exempt
@api_view(['GET'])
def get_subjects(request):
    """ 
    Return the list of subjects (lifelogger)
    """
    query_condition = {}
    field_filter = {'_id': 1}
    result = db_helper.query(query_condition, field_filter, distinct_by='_id', collection_name='User')
    result = json.loads(result)
    return JsonResponse(result, safe=False)


@csrf_exempt
@api_view(['POST'])
def get_dates_by_subject(request):
    """
    """
    data = request.data
    subject_id = data['subject_id']
    query_condition = {'_id' : subject_id}
    field_filter = {'data.date': 1}
    result = db_helper.query(query_condition, field_filter, distinct_by='data.date', collection_name='User')
    result = json.loads(result)
    return JsonResponse(result, safe=False)


@csrf_exempt
@api_view(['POST'])
def get_image_list_by_date(request):
    data = request.data
    subject_id = data['subject_id']
    date = data['date']
    query_condition = {'_id' : subject_id, 'data.date' : date}
    # field_filter = {'_id': 0, 'data.image_id': 1, 'data.$': 1}
    field_filter = {'_id': 0, 'data.$': 1}
    result = db_helper.query(query_condition, field_filter, collection_name='User')
    result = json.loads(result)
    if len(result) > 0:
        result = result[0]['data'][0]['image_id']
    return JsonResponse(result, safe=False)


@csrf_exempt
@api_view(['POST'])
def get_image_annotation(request):
    data = request.data
    object_id = data['object_id']
    
    # Check current operating system to check for path
    if platform.system() == 'Windows':
        object_id = object_id.replace('/', '\\') # Use for Windows only

    # subject_id = data['subject_id']
    # date = data['date']
    query_condition = {'_id': object_id }
    field_filter = { 'annotation': 1, 'annotated': 1 }
    result = db_helper.query(query_condition, field_filter, collection_name='ImageAnnotation')
    result = json.loads(result)
    result = result[0] if len(result) > 0 else {}
    return JsonResponse(result, safe=False)


@csrf_exempt
@api_view(['POST'])
def count_total_annotated_images(request):
    data = request.data
    subject_id = data['subject_id']
    date = data['date']
    # Get number of annotated images
    query_condition = { 'user_id': subject_id, 'date': date }
    field_filter = {'_id': 0, 'annotated': 1 }
    result = db_helper.query(query_condition, field_filter, collection_name='ImageAnnotation')
    result = json.loads(result)
    annotated = [item['annotated'] for item in result]
    # cnt_annotated_images = sum([1 if item['annotated'] == True else 0 for item in result])
    cnt_annotated_images = sum(annotated)
    cnt_images = len(annotated)

    # Get total number of images
    result = { "total" : f"{cnt_annotated_images}/{cnt_images}" }
    return JsonResponse(result, safe=False)


@csrf_exempt
def cross_server_auth(request):
    if request.method == 'POST':
        data = request.POST
        username = data['username']
        password = data['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            return HttpResponse(status=200)
        else:
            return HttpResponse(status=403)
    else:
        return HttpResponse(status=405)