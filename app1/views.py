# views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from .models import PressureTest, PressureAdjustment
import json



@login_required
def dashboard(request):
    """主控制面板"""
    # 获取用户最近的压力测试记录
    latest_test = PressureTest.objects.filter(user=request.user).order_by('-test_time').first()
    latest_level = latest_test.pressure_level if latest_test else 5

    return render(request, 'dashboard.html', {
        'username': request.user.username,
        'latest_level': latest_level
    })



def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})


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