from django.urls import path
from backend import views

urlpatterns = [
    path('', views.new_session, name='new_session'),
    path('get_query', views.get_query, name='get_query'),
    path('submit', views.submit, name='submit'),
    path('next_clue', views.next_clue, name='next_clue'),
    path('get_score', views.get_score, name='get_score'),
    path('end_query_round', views.end_query_round, name='end_query_round')
]