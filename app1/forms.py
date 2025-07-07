    # forms.py
    # from django import forms
    # from .models import UserHobby


    # class HobbySurveyForm(forms.Form):
    #     """爱好调查表单"""
    #     HOBBY_CHOICES = [
    #         ('reading', '阅读'),
    #         ('sports', '运动'),
    #         ('music', '音乐'),
    #         ('travel', '旅行'),
    #         ('gaming', '游戏'),
    #         ('art', '艺术'),
    #         ('cooking', '烹饪'),
    #         ('photography', '摄影'),
    #         ('dancing', '舞蹈'),
    #         ('gardening', '园艺'),
    #     ]
    #
    #     hobbies = forms.MultipleChoiceField(
    #         choices=HOBBY_CHOICES,
    #         widget=forms.CheckboxSelectMultiple(attrs={'class': 'hobby-checkbox'}),
    #         label="请选择您感兴趣的爱好"
    #     )
    #
    #     def save(self, user):
    #         """保存用户选择的爱好"""
    #         selected_hobbies = self.cleaned_data['hobbies']
    #         for hobby in selected_hobbies:
    #             UserHobby.objects.create(user=user, hobby=hobby)