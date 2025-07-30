import json

# Create your models here.
# models.py

from django.contrib.auth.models import User
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver



class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    first_login = models.BooleanField(default=True)
    birth_date = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=10, choices=[
        ('M', '男'),
        ('F', '女'),
        ('O', '其他')
    ], blank=True)
    phone = models.CharField(max_length=20, blank=True)
    bio = models.TextField('个人简介', max_length=500, blank=True)

    # 新增MAB算法字段
    total_selections = models.IntegerField(default=0)
    cluster_id = models.IntegerField(null=True, blank=True)
    reward_matrix = models.TextField(default='[]')  # 12x4 奖励矩阵
    selection_counts = models.TextField(default='[]')  # 12x4 选择计数矩阵
    preference_vector = models.TextField(default='[]')  # 12元素偏好向量

    def get_reward_matrix(self):
        return json.loads(self.reward_matrix) if self.reward_matrix else []

    def set_reward_matrix(self, matrix):
        self.reward_matrix = json.dumps(matrix)

    def get_selection_counts(self):
        return json.loads(self.selection_counts) if self.selection_counts else []

    def set_selection_counts(self, counts):
        self.selection_counts = json.dumps(counts)

    def get_preference_vector(self):
        return json.loads(self.preference_vector) if self.preference_vector else []

    def set_preference_vector(self, vector):
        self.preference_vector = json.dumps(vector)



    def __str__(self):
        return f"{self.user.username}'s profile"


class UserHobby(models.Model):
    """用户爱好模型"""
    HOBBY_CHOICES = [
        ('reading', '阅读'),
        ('sports', '运动'),
        ('music', '音乐'),
        ('travel', '旅行'),
        ('gaming', '游戏'),
        ('art', '艺术'),
        ('cooking', '烹饪'),
        ('photography', '摄影'),
        ('dancing', '舞蹈'),
        ('gardening', '园艺'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='hobbies')
    hobby = models.CharField(max_length=50, choices=HOBBY_CHOICES)
    interest_level = models.IntegerField(choices=[(i, str(i)) for i in range(1, 6)], default=3)

    def __str__(self):
        return f"{self.user.username} - {self.get_hobby_display()}"


# 信号：当创建新用户时自动创建用户资料
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """当创建新用户时自动创建用户资料"""
    if created:
        UserProfile.objects.get_or_create(user=instance)

class PressureTest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    test_time = models.DateTimeField(auto_now_add=True)
    pressure_level = models.IntegerField(choices=[(i, f"Level {i}") for i in range(1, 6)])
    duration = models.IntegerField(help_text="测试持续时间(秒)")  # 测试持续时间

    def __str__(self):
        return f"{self.user.username} - {self.get_pressure_level_display()} at {self.test_time}"


class PressureAdjustment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    adjust_time = models.DateTimeField(auto_now_add=True)
    before_level = models.IntegerField()
    after_level = models.IntegerField()
    method = models.CharField(max_length=100, choices=[
        ('breathing', '呼吸法'),
        ('music', '音乐疗法'),
        ('meditation', '冥想'),
        ('exercise', '运动'),
    ])
    video_url = models.URLField('视频链接', blank=True, null=True)
    video_title = models.CharField('视频标题', max_length=255, blank=True, null=True)

    def __str__(self):
        return f"{self.user.username}: {self.before_level} → {self.after_level}"


class VideoRecommendation(models.Model):
    """视频推荐记录"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    recommend_time = models.DateTimeField(auto_now_add=True)
    stress_level = models.IntegerField()
    recommended_videos = models.JSONField()  # 存储推荐视频列表

    def __str__(self):
        return f"{self.user.username} - {self.stress_level} at {self.recommend_time}"

    # 添加压力等级标签方法
    def get_stress_level_display(self):
        levels = {
            1: "低压力",
            2: "中低压力",
            3: "中等压力",
            4: "中高压力",
            5: "高压力"
        }
        return levels.get(self.stress_level, f"未知({self.stress_level})")







# 添加新模型
class UserCluster(models.Model):
    """用户聚类数据"""
    cluster_id = models.IntegerField(unique=True, db_index=True)
    platform = models.CharField(max_length=50, default='default')
    user_count = models.IntegerField(default=0)
    total_selections = models.IntegerField(default=0)
    cluster_reward_matrix = models.TextField(default='[]')
    cluster_selection_counts = models.TextField(default='[]')
    cluster_preference_vector = models.TextField(default='[]')

    # ... 类似getter/setter方法 ...


class SystemStatistics(models.Model):
    """系统统计数据"""
    platform = models.CharField(max_length=50, default='default')
    total_users = models.IntegerField(default=0)
    total_recommendations = models.IntegerField(default=0)
    total_feedbacks = models.IntegerField(default=0)
    last_clustering_update = models.DateTimeField(null=True, blank=True)

    # ... 其他字段 ...
