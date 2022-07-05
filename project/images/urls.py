from django.urls import path

from . import views

urlpatterns = [
    path('api/images/', views.images),
    path('api/timeline/', views.timeline),
    path('api/timeline/more_scene/', views.more_scenes),
    path('api/timeline/group/', views.timeline_group),
    path('api/timeline/info/', views.detailed_info),
    path('api/more/', views.more),
    path('api/gps/', views.gps),
    path('api/similar', views.similar),
    path('api/login', views.login),
    path('api/submit', views.export),
    path('api/submit_saved/', views.submit_all),
    path('api/restart', views.restart),
    path('aaron/', views.aaron),
    path('aaron_timeline/', views.aaron_timeline)]
