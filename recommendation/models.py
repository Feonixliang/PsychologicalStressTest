# recommendation/models.py
import numpy as np
from django.db import models
from django.contrib.auth.models import User
from django.db.models import JSONField
from sklearn.cluster import KMeans  # 添加这行导入


class Video(models.Model):
    """降压视频模型"""
    name = models.CharField(max_length=255, verbose_name="视频名称")
    description = models.TextField(verbose_name="视频描述")
    duration_minutes = models.IntegerField(verbose_name="时长(分钟)")
    video_url = models.URLField(verbose_name="视频链接")
    thumbnail = models.ImageField(upload_to='thumbnails/', verbose_name="缩略图", blank=True, null=True)
    category = models.CharField(max_length=100, verbose_name="类别", blank=True, null=True)

    def __str__(self):
        return self.name


class UserRecommendationProfile(models.Model):
    """用户推荐系统档案"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='recommendation_profile')
    total_selections = models.IntegerField(default=0, verbose_name="总选择次数")

    # 用于存储偏好向量（序列化为JSON）
    preference_vector = models.JSONField(default=dict, verbose_name="偏好向量")
    cluster_id = models.CharField(max_length=50, blank=True, null=True, verbose_name="聚类ID")

    def __str__(self):
        return f"{self.user.username}'s recommendation profile"


class UserVideoInteraction(models.Model):
    """用户与视频的交互记录"""
    user_profile = models.ForeignKey(UserRecommendationProfile, on_delete=models.CASCADE, related_name='interactions')
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    stress_level = models.IntegerField(verbose_name="压力等级")  # 1-5
    cumulative_reward = models.FloatField(default=0.0, verbose_name="累积奖励")
    selection_count = models.IntegerField(default=0, verbose_name="选择次数")
    ucb_score = models.FloatField(default=0.0, verbose_name="UCB分数")

    class Meta:
        unique_together = ('user_profile', 'video', 'stress_level')

    def __str__(self):
        return f"{self.user_profile.user.username} - {self.video.name} (Stress: {self.stress_level})"



class Cluster(models.Model):
    """用户聚类模型"""
    cluster_id = models.IntegerField(unique=True, verbose_name="聚类ID")
    preference_vector = JSONField(default=dict, verbose_name="偏好向量")

    def __str__(self):
        return f"Cluster {self.cluster_id}"

    def recluster(self):
        """实现K-means聚类算法"""


        # 获取所有用户偏好向量
        profiles = UserRecommendationProfile.objects.all()
        vectors = [np.array(profile.preference_vector) for profile in profiles]

        if len(vectors) < 5:  # 最少5个用户才聚类
            return

        # 执行K-means聚类
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(vectors)

        # 更新聚类中心
        for i, center in enumerate(kmeans.cluster_centers_):
            cluster, _ = Cluster.objects.update_or_create(
                cluster_id=i,
                defaults={'preference_vector': center.tolist()}
            )

        # 更新用户聚类分配
        for profile, label in zip(profiles, kmeans.labels_):
            profile.cluster_id = f"cluster_{label}"
            profile.save()