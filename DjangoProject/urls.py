from django.contrib import admin
from django.urls import path, include
from app1 import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('save_test/', views.save_test, name='save_test'),
    path('save_adjustment/', views.save_adjustment, name='save_adjustment'),
    path('save_hobby_survey/', views.save_hobby_survey, name='save_hobby_survey'),

    # 认证系统URL
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('register/', views.register, name='register'),
    path('edit_profile/', views.edit_profile, name='edit_profile'),


    path('upload_mat/', views.upload_mat, name='upload_mat'),

    path('get_recommendations/', views.get_recommendations, name='get_recommendations'),

    path('get_test_records/', views.get_test_records, name='get_test_records'),
    
    # 测试接口
    path('test_recommendation/', views.test_recommendation_system, name='test_recommendation'),
    path('simulate_test/', views.simulate_pressure_test, name='simulate_test'),
    path('submit_feedback/', views.submit_feedback, name='submit_feedback'),
    path('test_page/', views.test_page, name='test_page'),
]
