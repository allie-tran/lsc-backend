from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('update_qa/', views.update, name='index'),
]
