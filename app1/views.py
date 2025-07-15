# views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from .models import PressureTest, PressureAdjustment, UserProfile, UserHobby
from django.contrib.auth import login, authenticate
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import logging
from django.views.decorators.csrf import csrf_exempt
import json
from .forms import UserProfileForm


logger = logging.getLogger(__name__)

@login_required
def dashboard(request):
    """主控制面板"""

    # 检查是否是首次登录
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    show_hobby_survey = profile.first_login


    # 获取用户最近的压力测试记录
    latest_test = PressureTest.objects.filter(user=request.user).order_by('-test_time').first()
    latest_level = latest_test.pressure_level if latest_test else 5

    return render(request, 'dashboard.html', {
        'username': request.user.username,
        'latest_level': latest_level,
        'show_hobby_survey': show_hobby_survey
    })


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()  # 这行会触发信号自动创建 UserProfile

            # 不需要手动创建 UserProfile，因为信号已经处理了
            # UserProfile.objects.create(user=user, first_login=True)

            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'账号 {username} 注册成功！')
                return redirect('dashboard')
            else:
                messages.error(request, '认证失败，请重试')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

@csrf_exempt
@login_required
def save_hobby_survey(request):
    """保存爱好调查结果"""
    if request.method == 'POST':
        try:
            #调试日志
            logger.debug(f"Received POST data: {request.POST}")
            selected_hobbies = request.POST.getlist('hobbies[]')
            logger.debug(f"Selected hobbies: {selected_hobbies}")

            # 正确获取选中的爱好列表
            selected_hobbies = request.POST.getlist('hobbies[]')  # 使用 getlist 获取数组

            # 删除现有爱好记录（可选）
            UserHobby.objects.filter(user=request.user).delete()

            # 保存新选择的爱好
            for hobby in selected_hobbies:
                UserHobby.objects.create(user=request.user, hobby=hobby)

            # 更新用户首次登录状态
            profile = UserProfile.objects.get(user=request.user)
            profile.first_login = False
            profile.save()

            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


@login_required
def save_test(request):
    """保存压力测试结果"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            PressureTest.objects.create(
                user=request.user,
                pressure_level=data['pressure_level'],
                duration=data['duration']
            )
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
@login_required
def save_adjustment(request):
    """保存压力调节记录"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            PressureAdjustment.objects.create(
                user=request.user,
                before_level=data['before_level'],
                after_level=data['after_level'],
                method='video',  # 固定为视频疗法
                video_url=data.get('video_url', ''),
                video_title=data.get('video_title', '')
            )
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


@login_required
def edit_profile(request):
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        # 如果资料不存在则创建
        profile = UserProfile.objects.create(user=request.user)

    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, '个人资料已更新！')
            return redirect('dashboard')
        else:
            # 添加表单错误处理
            messages.error(request, '请修正以下错误')
    else:
        form = UserProfileForm(instance=profile)

    return render(request, 'profile_edit.html', {
        'form': form,
        'user': request.user  # 确保 user 对象传递给模板
    })


import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
import scipy.io
import numpy as np
from scipy.signal import welch
from .models import PressureTest


# 压力分析函数 (示例实现)
def analyze_pressure(mat_data):
    """
    从.mat文件中提取生理信号数据并分析压力水平
    这里是一个简化的示例实现
    """
    try:
        # 假设.mat文件包含名为'ecg'的ECG信号和名为'gsr'的皮电反应数据
        ecg_signal = mat_data['ecg'].flatten()
        gsr_signal = mat_data['gsr'].flatten()

        # 计算ECG信号的HRV (心率变异性)
        # 在实际应用中，这里会有更复杂的处理
        rr_intervals = np.diff(np.where(ecg_signal > 0.5)[0])
        hrv = np.std(rr_intervals) if len(rr_intervals) > 0 else 50

        # 计算皮电反应的特征
        gsr_mean = np.mean(gsr_signal)

        # 简化的压力水平计算 (实际应用会有更复杂的模型)
        pressure_level = min(10, max(1, int(10 - (hrv / 20) + (gsr_mean / 2000))))

        return pressure_level
    except Exception as e:
        raise ValueError(f"数据分析错误: {str(e)}")


import requests


# 修改后的 upload_mat 函数
@csrf_exempt
@login_required
def upload_mat(request):
    """处理.mat文件上传和分析"""
    if request.method == 'POST' and request.FILES.get('mat_file'):
        try:
            # 获取上传的文件
            mat_file = request.FILES['mat_file']

            # 设置API接口地址
            API_URL = "http://localhost:8000/predict"

            # 准备文件数据
            files = {'file': (mat_file.name, mat_file, 'application/octet-stream')}

            try:
                # 发送请求到API接口
                response = requests.post(API_URL, files=files)

                # 检查响应状态
                if response.status_code != 200:
                    return JsonResponse({
                        'success': False,
                        'error': f'API服务错误: 状态码 {response.status_code}'
                    })

                # 解析API返回的JSON数据
                api_data = response.json()

                # 提取所需数据
                prediction = api_data.get('prediction', '中压力')
                confidence = api_data.get('confidence', 0.0)
                probabilities = api_data.get('probabilities', [])
                class_names = api_data.get('class_names', [])

                # 创建概率字典
                probabilities_dict = {}
                if probabilities and class_names and len(probabilities) == len(class_names):
                    probabilities_dict = {name: prob for name, prob in zip(class_names, probabilities)}

                # 映射压力等级数值（1-10）
                pressure_level_map = {
                    "低压力": 3,
                    "中压力": 6,
                    "高压力": 8,
                    "极高压力": 10
                }
                pressure_level = pressure_level_map.get(prediction, 5)

                # 返回分析结果
                return JsonResponse({
                    'success': True,
                    'file_name': mat_file.name,
                    'pressure_level': pressure_level,
                    'confidence': confidence,
                    'probabilities': probabilities_dict,
                    'prediction_label': prediction,
                    'message': '分析成功'
                })

            except requests.exceptions.RequestException as e:
                return JsonResponse({
                    'success': False,
                    'error': f'API连接失败: {str(e)}'
                })
            except ValueError as e:
                return JsonResponse({
                    'success': False,
                    'error': f'API返回数据解析错误: {str(e)}'
                })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({
        'success': False,
        'error': '无效的请求或未上传文件'
    })