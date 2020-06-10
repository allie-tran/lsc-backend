from django.urls import path

from . import views

urlpatterns = [
    path('api/image/', views.images),
    path('api/timeline/', views.timeline),
    path('api/timeline/group/', views.timeline_group),
    path('api/date/', views.date),
    path('api/gps/', views.gps),
    path('api/save', views.save),
    path('api/remove', views.remove),
    path('api/submit', views.export),
    path('api/restart', views.restart),
    path('api/getsaved', views.get_saved),
    path('api/aaron/', views.aaron)]
