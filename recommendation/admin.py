from django.contrib import admin

# Register your models here.
# 在Django shell中执行
from recommendation.models import Video
Video.objects.create(
    name="自然风光放松视频",
    description="5分钟自然风光视频",
    duration_minutes=5,
    video_url="https://www.bilibili.com/video/BV1ST411E7wb/?spm_id_from=333.337.search-card.all.click&vd_source=ff7c8aac48af9ebfb1338fae9b2f96ff"
)
Video.objects.create(
    name="轻音乐精选",
    description="舒缓轻音乐合集",
    duration_minutes=15,
    video_url="https://www.bilibili.com/video/BV1G1MTz5EAB/?spm_id_from=333.337.search-card.all.click"
)