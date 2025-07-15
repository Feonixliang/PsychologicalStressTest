# recommendation/service.py (完整修正版)
import numpy as np
from django.conf import settings
from .models import UserRecommendationProfile, UserVideoInteraction, Cluster, Video
from .models import Video
from .services.EEGStressMAB import EEGStressMAB

class ContextualMAB:
    """情境多臂老虎机算法实现"""

    def __init__(self, user):
        self.user = user
        self.num_videos = Video.objects.count()
        self.num_stress_levels = 5  # 5个压力等级
        self.exploration_param = 2.0  # 探索参数

        # 获取或创建用户推荐档案
        self.user_profile, _ = UserRecommendationProfile.objects.get_or_create(user=user)

        # 加载配置
        self.config = {
            "threshold_num": 10,
            "non_selected_rewards": -0.1,
            "enable_cluster": True,
            "cluster_reco_ratio": 0.3,
            "mab_reco_ratio": 0.7,
            "cluster_refresh_threshold": 50,
            "assign_cluster_num": 5,
            "recluster_threshold": 100,
            "avoid_indices": {
                0: [7, 10],  # 压力等级1避免的视频索引
                1: [7],
                2: [],
                3: [9],
                4: [1, 7, 10]
            }
        }

    def recommend(self, rec_size, stress_level_index, avoid_indices=None):
        """推荐视频"""
        # 获取当前压力等级下的所有交互记录
        interactions = UserVideoInteraction.objects.filter(
            user_profile=self.user_profile,
            stress_level=stress_level_index + 1  # 调整为1-5
        )

        # 获取UCB分数
        ucb_scores = np.full(self.num_videos, float('inf'))
        for interaction in interactions:
            ucb_scores[interaction.video.id - 1] = interaction.ucb_score

        # 获取所有视频
        all_videos = list(Video.objects.all())

        # 应用避免索引
        if avoid_indices is None:
            avoid_indices = []
        valid_indices = [i for i in range(len(all_videos)) if i not in avoid_indices]

        # 选择分数最高的视频
        sorted_indices = np.argsort(ucb_scores)[::-1]
        recommended_indices = [idx for idx in sorted_indices if idx in valid_indices][:rec_size]

        return [all_videos[idx] for idx in recommended_indices]

    def update(self, video_id, stress_level_index, reward, selected=True):
        """更新模型"""
        stress_level = stress_level_index + 1  # 调整为1-5
        video = Video.objects.get(id=video_id)

        # 获取或创建交互记录
        interaction, created = UserVideoInteraction.objects.get_or_create(
            user_profile=self.user_profile,
            video=video,
            stress_level=stress_level,
            defaults={
                'cumulative_reward': 0,
                'selection_count': 0,
                'ucb_score': float('inf')
            }
        )

        if selected:
            # 更新被选择的视频
            interaction.selection_count += 1
            interaction.cumulative_reward += reward
            self.user_profile.total_selections += 1
            self.user_profile.save()
        else:
            # 更新未被选择的视频
            interaction.cumulative_reward += self.config['non_selected_rewards']

        # 重新计算UCB分数
        if interaction.selection_count > 0:
            exploitation_score = interaction.cumulative_reward / interaction.selection_count
            # 修正括号问题
            exploration_score = np.sqrt(
                (self.exploration_param * np.log(self.user_profile.total_selections + 1)
                 ) / interaction.selection_count)
            interaction.ucb_score = exploitation_score + exploration_score

            interaction.save()

            # 更新聚类数据
            self.update_cluster_data(video_id, stress_level_index)


class EEGStressMABWrapper:
    """Django适配的推荐系统封装"""

    def __init__(self, user):
        self.recommender = EEGStressMAB(user)

    def recommend_videos(self, stress_level):
        """获取视频推荐"""
        # 转换压力等级 (1-10 → 1-5)
        stress_level_5 = convert_pressure_level(stress_level)
        return self.recommender.recommend_videos(stress_level_5)

    def update_model(self, pre_stress, post_stress, selected_video_id, video_ids_list, rating=None):
        """更新模型"""
        pre_stress_5 = convert_pressure_level(pre_stress)
        post_stress_5 = convert_pressure_level(post_stress)
        self.recommender.update_model(
            pre_stress_5, post_stress_5,
            selected_video_id, video_ids_list, rating
        )

        # 定期重新聚类
        if self.recommender.mab.total_selections % 50 == 0:
            from .models import Cluster
            Cluster.objects.first().recluster()