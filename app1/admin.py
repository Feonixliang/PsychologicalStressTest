

# Register your models here.
# admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User

from .models import UserProfile, PressureTest, PressureAdjustment

# 扩展用户管理界面
class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False

class CustomUserAdmin(UserAdmin):
    inlines = (UserProfileInline,)

admin.site.unregister(User)  # 取消默认注册
admin.site.register(User, CustomUserAdmin)  # 重新注册自定义用户管理

# 注册你的模型
admin.site.register(PressureTest)
admin.site.register(PressureAdjustment)