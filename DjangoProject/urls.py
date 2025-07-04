"""
URL configuration for DjangoProject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from app1 import views


urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('save_test/', views.save_test, name='save_test'),
    path('save_adjustment/', views.save_adjustment, name='save_adjustment'),



    # 添加认证系统的URL配置
    path('accounts/', include('django.contrib.auth.urls')),

    path('register/', views.register, name='register'),

]
