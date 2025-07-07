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
                method=data['method']
            )
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})