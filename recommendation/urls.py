# recommendation/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('recommend/', views.get_recommendations, name='get_recommendations'),
    path('feedback/', views.submit_feedback, name='submit_feedback'),
    path('videos/', views.get_video_list, name='get_video_list'),
]