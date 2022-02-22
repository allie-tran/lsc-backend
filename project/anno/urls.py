from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('update/', views.update, name='index'),
    path('update_qa/', views.update_qa, name='index'),
    path('get_qa_list/', views.get_qa_list, name='index'),
    path('get_group/', views.get_group, name='get_group'),
    path('get_scene/', views.get_scene, name='get_scene'),
    path('get_desc/', views.get_desc, name='get_desc'),
]
