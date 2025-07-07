from django import forms
from .models import UserProfile
from django.contrib.auth.models import User


class UserProfileForm(forms.ModelForm):
    # 添加邮箱字段（来自User模型）
    email = forms.EmailField(label='电子邮箱')

    class Meta:
        model = UserProfile
        fields = ['birth_date', 'gender', 'phone', 'bio']
        widgets = {
            'birth_date': forms.DateInput(attrs={'type': 'date'}),
            'gender': forms.RadioSelect(),
            'bio': forms.Textarea(attrs={'rows': 3}),
        }

    def __init__(self, *args, **kwargs):
        super(UserProfileForm, self).__init__(*args, **kwargs)
        # 初始化邮箱字段的值
        if self.instance and self.instance.user:
            self.fields['email'].initial = self.instance.user.email
            self.fields['bio'].initial = self.instance.bio  # 确保个人简介字段初始化

    def save(self, commit=True):
        profile = super().save(commit=False)
        # 保存邮箱到User模型
        if profile.user:
            profile.user.email = self.cleaned_data['email']
            profile.user.save()
        # 保存个人简介
        profile.bio = self.cleaned_data['bio']

        if commit:
            profile.save()
        return profile