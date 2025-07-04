
# Create your models here.
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    birth_date = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=10, choices=[
        ('M', '男'),
        ('F', '女'),
        ('O', '其他')
    ], blank=True)
    phone = models.CharField(max_length=20, blank=True)

    def __str__(self):
        return self.user.username


# 信号：当创建新用户时自动创建用户资料
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


# 信号：当保存用户时自动保存用户资料
@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.userprofile.save()

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