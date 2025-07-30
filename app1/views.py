# views.py
import json
import logging
import random

import numpy as np
import requests
from django.contrib import messages
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt

from app1.EEGStressMAB import EEGStressMAB
from .forms import UserProfileForm
from .models import PressureAdjustment, PressureTest, UserProfile, UserHobby, VideoRecommendation


logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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
                method=data['method'],
                # 添加视频信息
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


@csrf_exempt
@login_required
def upload_mat(request):
    """处理.mat文件上传和分析"""
    if request.method == 'POST' and request.FILES.get('mat_file'):
        try:
            # 获取上传的文件
            mat_file = request.FILES['mat_file']
            API_URL = "http://localhost:8000/predict"
            files = {'file': (mat_file.name, mat_file, 'application/octet-stream')}

            logger.info(f"开始处理文件上传: {mat_file.name}")

            try:
                # 发送请求到API接口
                response = requests.post(API_URL, files=files)
                logger.info(f"API响应状态码: {response.status_code}")

                # 检查响应状态
                if response.status_code != 200:
                    logger.error(f"API服务错误: 状态码 {response.status_code}")
                    return JsonResponse({
                        'success': False,
                        'error': f'API服务错误: 状态码 {response.status_code}'
                    })

                # 解析API返回的JSON数据
                api_data = response.json()
                logger.info(f"API返回原始数据: {json.dumps(api_data, indent=2, ensure_ascii=False)}")

                # 提取所需数据
                prediction = api_data.get('prediction', '')
                confidence = api_data.get('confidence', 0.0)
                probabilities = api_data.get('probabilities', [])
                class_names = api_data.get('class_names', [])

                logger.info(f"预测结果: {prediction}, 置信度: {confidence}")


                # 回退到字符串映射
                logger.warning("使用回退的压力等级映射")
                pressure_level_map = {
                    "低压力": 1,
                    "中压力": 2,
                    "高压力": 3,
                    "极高压力": 4
                }
                pressure_level = pressure_level_map.get(prediction, 3)  # 默认中等压力

                logger.info(f"映射后的压力等级: {pressure_level}")


                # 验证压力等级范围 (1-4级)
                if pressure_level < 1:
                    pressure_level = 1
                elif pressure_level > 4:
                    pressure_level = 4

                logger.info(f"调整后压力等级: {pressure_level}")

                # 获取压力等级后调用推荐系统
                recommender = EEGStressMAB(request.user)
                recommended_videos = recommender.recommend_videos(pressure_level)

                logger.info(f"推荐视频原始数据: {json.dumps(recommended_videos, indent=2, ensure_ascii=False)}")

                logger.info(f"推荐视频数据: {json.dumps(recommended_videos, indent=2, ensure_ascii=False)}")

                # 保存推荐记录 - 直接存储Python对象
                try:
                    VideoRecommendation.objects.create(
                        user=request.user,
                        stress_level=pressure_level,
                        recommended_videos=recommended_videos
                    )
                    logger.info("推荐记录保存成功")
                except Exception as e:
                    logger.error(f"保存推荐记录失败: {str(e)}")
                    return JsonResponse({
                        'success': False,
                        'error': f'保存推荐记录失败: {str(e)}'
                    })

                # 返回分析结果
                return JsonResponse({
                    'success': True,
                    'file_name': mat_file.name,
                    'pressure_level': pressure_level,
                    'confidence': confidence,
                    'probabilities': {name: float(prob) for name, prob in
                                      zip(class_names, probabilities)} if probabilities else {},
                    'prediction_label': prediction,
                    'message': '分析成功',
                    'recommended_videos': recommended_videos,
                })

            except requests.exceptions.RequestException as e:
                logger.error(f"API连接失败: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'error': f'API连接失败: {str(e)}'
                })
            except (ValueError, json.JSONDecodeError, KeyError) as e:
                logger.error(f"API返回数据解析错误: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'error': f'API返回数据解析错误: {str(e)}'
                })

        except Exception as e:
            logger.exception("处理上传时发生异常")
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({
        'success': False,
        'error': '无效的请求或未上传文件'
    })


@login_required
def get_recommendations(request):
    """获取最新推荐视频"""
    latest_rec = VideoRecommendation.objects.filter(
        user=request.user
    ).order_by('-recommend_time').first()

    if latest_rec:
        # 直接返回JSONField值，不需要json.loads()
        return JsonResponse({
            'success': True,
            'recommended_videos': latest_rec.recommended_videos
        })
    return JsonResponse({'success': False, 'message': 'No recommendations'})


@login_required
def get_test_records(request):
    """获取用户测试记录"""
    records = PressureTest.objects.filter(
        user=request.user
    ).order_by('-test_time')[:5]  # 获取最近5条记录

    records_data = []
    for record in records:
        records_data.append({
            'date': record.test_time.strftime("%Y-%m-%d %H:%M"),
            'level': record.pressure_level,
            'duration': f"{record.duration}秒"
        })

    return JsonResponse({
        'success': True,
        'records': records_data
    })


# 添加反馈处理视图
@csrf_exempt
@login_required
def submit_feedback(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            pre_stress = data['pre_stress']
            post_stress = data['post_stress']
            selected_video_id = data['selected_video_id']

            # 更新推荐模型
            recommender = EEGStressMAB(request.user)  # 修复：传递user对象而不是username
            recommender.update_model(pre_stress, post_stress, selected_video_id)

            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': '无效请求方法'})


@login_required
def test_recommendation_system(request):
    """测试推荐系统功能 - 不需要真实EEG数据"""
    try:
        # 模拟不同压力等级的测试
        test_results = []
        
        for stress_level in range(1, 5):
            try:
                # 初始化推荐系统
                recommender = EEGStressMAB(request.user)
                
                # 获取推荐视频
                videos = recommender.recommend_videos(stress_level)
                
                test_results.append({
                    'stress_level': stress_level,
                    'success': True,
                    'video_count': len(videos),
                    'videos': videos
                })
                
                logger.info(f"压力等级 {stress_level}: 成功推荐 {len(videos)} 个视频")
                
            except Exception as e:
                test_results.append({
                    'stress_level': stress_level,
                    'success': False,
                    'error': str(e)
                })
                logger.error(f"压力等级 {stress_level} 测试失败: {e}")
        
        # 测试反馈机制
        feedback_test = {'success': False}
        try:
            recommender = EEGStressMAB(request.user)
            recommender.update_model(pre_stress=4, post_stress=2, selected_video_id=0)
            feedback_test = {'success': True, 'message': '反馈机制正常'}
            logger.info("反馈机制测试成功")
        except Exception as e:
            feedback_test = {'success': False, 'error': str(e)}
            logger.error(f"反馈机制测试失败: {e}")
        
        return JsonResponse({
            'success': True,
            'test_results': test_results,
            'feedback_test': feedback_test,
            'message': '推荐系统测试完成'
        }, encoder=NumpyEncoder)
        
    except Exception as e:
        logger.exception("推荐系统测试过程中发生异常")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@login_required 
def simulate_pressure_test(request):
    """模拟压力测试 - 直接指定压力等级进行推荐"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            pressure_level = data.get('pressure_level', 3)
            
            # 验证压力等级
            if pressure_level < 1 or pressure_level > 4:
                pressure_level = 3
            
            logger.info(f"模拟压力测试 - 压力等级: {pressure_level}")
            
            # 调用推荐系统
            recommender = EEGStressMAB(request.user)
            recommended_videos = recommender.recommend_videos(pressure_level)
            
            # 保存模拟测试记录
            PressureTest.objects.create(
                user=request.user,
                pressure_level=pressure_level,
                duration=0  # 模拟测试无持续时间
            )
            
            # 保存推荐记录
            VideoRecommendation.objects.create(
                user=request.user,
                stress_level=pressure_level,
                recommended_videos=recommended_videos
            )
            
            return JsonResponse({
                'success': True,
                'pressure_level': pressure_level,
                'recommended_videos': recommended_videos,
                'message': f'模拟测试成功 - 压力等级{pressure_level}'
            }, encoder=NumpyEncoder)
            
        except Exception as e:
            logger.exception("模拟压力测试失败")
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': '请使用POST方法'})


def test_page(request):
    """测试页面"""
    return render(request, 'test_recommendation.html')