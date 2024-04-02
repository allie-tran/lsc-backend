from datetime import datetime

from django.contrib.auth import authenticate
from django.http import HttpResponse, HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from database.utils import user_collection


@csrf_exempt
@api_view(["POST"])
def cross_server_auth(request):
    data = request.POST
    username = data["username"]
    password = data["password"]
    user = authenticate(username=username, password=password)
    if user is not None:
        return HttpResponse(status=200)
    else:
        return HttpResponse(status=403)


@csrf_exempt
@api_view(["GET"])
def login_from_frontend(request):
    session_id = request.GET.get("sessionID")
    if not session_id:
        return HttpResponseServerError(reason="No session ID provided")
    # Save session ID to the database to match with the user later
    # Check if the session ID is valid
    user = user_collection.find_one({"session_id": session_id})
    # Check if the last login time is within the last 24 hours
    # If not, delete the user and create a new one
    now = datetime.now()
    if user:
        if (now - user["last_login"]).days > 1:
            user_collection.delete_one({"session_id": session_id})
        else:
            user_collection.update_one(
                {"session_id": session_id}, {"$set": {"last_login": now}}
            )
            return HttpResponse(status=200)

    # Create a new user
    user_collection.insert_one(
        {"session_id": session_id, "last_login": datetime.now(), "queries": []}
    )
    return HttpResponse(status=200)
