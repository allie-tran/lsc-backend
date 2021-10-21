from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('update/', views.update, name='index'),
    path('update_qa/', views.update_qa, name='index'),
    path('get_qa_list/', views.get_qa_list, name='index'),
]
