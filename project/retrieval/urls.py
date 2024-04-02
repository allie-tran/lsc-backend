from django.urls import path

from . import routers

urlpatterns = [
    path("search/", routers.search),
    path("get-stream-results/", routers.get_stream_search),
]
