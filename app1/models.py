
# Create your models here.
# models.py
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

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """确保用户资料存在并保存"""
    # 使用正确的反向关系名称 'profile'
    try:
        instance.profile.save()
    except UserProfile.DoesNotExist:
        # 如果资料不存在则创建
        UserProfile.objects.create(user=instance)

class PressureTest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    test_time = models.DateTimeField(auto_now_add=True)
    pressure_level = models.IntegerField(choices=[(i, f"Level {i}") for i in range(1, 11)])  # 1-10级
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

    def __str__(self):
        return f"{self.user.username}: {self.before_level} → {self.after_level}"