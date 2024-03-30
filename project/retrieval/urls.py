from django.urls import path

from . import routers

urlpatterns = [
    path("search/", routers.search),
]
