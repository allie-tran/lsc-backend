from django.urls import path

from . import views

urlpatterns = [
    path('api/images/', views.images),
    path('api/timeline/', views.timeline),
    path('api/timeline/group/', views.timeline_group),
    path('api/timeline/info/', views.detailed_info),
    path('api/more/', views.more),
    path('api/gps/', views.gps),
    path('api/save', views.save),
    path('api/similar', views.similar),
    path('api/remove', views.remove),
    path('api/submit', views.export),
    path('api/restart', views.restart),
    path('api/getsaved', views.get_saved),
    path('aaron/', views.aaron),
    path('aaron_timeline/', views.aaron_timeline)]
