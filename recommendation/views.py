# recommendation/views.py
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from .service import EEGStressMAB
from .models import Video
from .service import EEGStressMABWrapper
import json

@csrf_exempt
@login_required
def get_recommendations(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            stress_level = data.get('stress_level')

            if stress_level < 1 or stress_level > 5:
                return JsonResponse({'error': '压力等级必须在1-5之间'}, status=400)

            recommender = EEGStressMABWrapper(request.user)
            recommended_videos = recommender.recommend_videos(stress_level)

            # 转换为前端需要的格式
            video_list = []
            for video in recommended_videos:
                video_list.append({
                    'id': video['id'],
                    'name': video['name'],
                    'description': video['description'],
                    'duration_minutes': video['duration_minutes'],
                    'video_url': video['video_url']
                })

            return JsonResponse({
                'success': True,
                'recommended_videos': video_list
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
@login_required
def submit_feedback(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            pre_stress = data.get('pre_stress')
            post_stress = data.get('post_stress')
            selected_video_id = data.get('selected_video_id')
            recommended_video_ids = data.get('recommended_video_ids', [])
            rating = data.get('rating')

            recommender = EEGStressMABWrapper(request.user)
            recommender.update_model(
                pre_stress, post_stress,
                selected_video_id, recommended_video_ids, rating
            )

            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@login_required
def get_video_list(request):
    """获取视频列表"""
    videos = Video.objects.all().values('id', 'name', 'description', 'duration_minutes', 'video_url')
    return JsonResponse({'videos': list(videos)})