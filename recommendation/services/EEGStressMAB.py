# EEG压力降低视频推荐系统
# 基于脑电神经预测的压力评级，推荐降压视频内容
# 使用观看前后压力差值作为客观反馈，优化推荐策略
import numpy as np
from django.conf import settings
from .ContextualMAB import ContextualMAB
from recommendation.models import Video
import os
import yaml

# 从Django设置获取配置
with open(os.path.join(settings.BASE_DIR, "recommendation/config.yaml"), "r", encoding='utf-8') as yaml_file:
    data = yaml.safe_load(yaml_file)


reco_size = data["recommendation_size"]
stress_levels = data["stress_levels"]  # 5个压力等级
videos = data["videos"]  # 降压视频
video_dict = data["video_dict"]
avoid_indices = data["avoid_indices"]

# 压力相关配置
stress_reduction_threshold = data["stress_reduction_threshold"]
max_stress_increase = data["max_stress_increase"]

# 负反馈学习配置
modify_non_selected = data["modify_non_selected"]
threshold_num = data["threshold_num"]
auxiliary_rewards = data["auxiliary_rewards"]

# 聚类功能配置
enable_cluster = data["enable_cluster"]
cluster_refresh_threshold = data["cluster_refresh_threshold"]
cluster_reco_ratio = data["cluster_reco_ratio"]
mab_reco_ratio = data["mab_reco_ratio"]
assign_cluster_num = data["assign_cluster_num"]
recluster_threshold = data["recluster_threshold"]

# 文件路径配置
video_folder_path = os.path.join(os.getcwd(), data["video_folder_name"])
thumbnail_folder_path = os.path.join(os.getcwd(), data["thumbnail_folder_name"])
general_data_platform = data["general_data_platform"]
general_data_user_hash = data["general_data_user_hash"]

# 创建文件夹
for folder_path in [video_folder_path, thumbnail_folder_path]:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

class EEGStressMAB:
    """
    EEG压力降低视频推荐系统
    
    核心功能：
    1. 接收脑电预测的压力评级(1-5)作为情境输入
    2. 基于UCB算法推荐降压视频
    3. 使用观看前后压力差值作为客观奖励信号
    4. 持续学习优化个性化推荐策略
    """

    def __init__(self, user):
        self.user = user
        # 从数据库获取视频列表
        self.videos = list(Video.objects.all())
        self.video_dict = {v.id: {
            'name': v.name,
            'description': v.description,
            'duration_minutes': v.duration_minutes,
            'video_url': v.video_url
        } for v in self.videos}

        # 初始化MAB实例
        self.mab = ContextualMAB(
            user=user,
            num_videos=len(self.videos),
            num_stress_levels=len(data["stress_levels"]),
            exploration_param=2.0,
            # ... 其他配置参数 ...
        )

    def get_stress_levels(self):
        """返回压力等级列表"""
        return stress_levels

    def get_video_list(self):
        """返回视频列表"""
        return videos

    def stress_level_to_index(self, stress_level):
        """
        将压力评级(1-5)转换为索引(0-4)
        
        参数:
        - stress_level: 脑电预测模型输出的压力评级(1-5)
        
        返回:
        - stress_index: 对应的索引(0-4)
        """
        if not (1 <= stress_level <= 5):
            raise ValueError(f"压力评级必须在1-5之间，当前值: {stress_level}")
        return stress_level - 1

    def recommend_videos(self, stress_level, prev_video_indices=None):
        """
        基于当前压力等级推荐降压视频
        
        参数:
        - stress_level: 当前压力评级(1-5)
        - prev_video_indices: 之前推荐过的视频索引列表(避免重复推荐)
        
        返回:
        - recommended_videos: 推荐的视频信息列表
        """
        stress_idx = self.stress_level_to_index(stress_level)
        ctx_avoid_indices = avoid_indices[stress_idx].copy()
        
        # 合并需要避免的视频索引
        if prev_video_indices is not None:
            ctx_avoid_indices = np.unique(np.concatenate((ctx_avoid_indices, prev_video_indices)).astype(int))

        # 获取推荐的视频索引
        video_indices = self.user_mab_instance.recommend(
            reco_size, stress_idx, 
            self.user_mab_instance.is_first_time(stress_idx, reco_size), 
            ctx_avoid_indices
        )
        
        # 构建推荐视频信息
        recommended_videos = []
        for idx in video_indices:
            video_name = videos[idx]
            video_info = {
                'index': idx,
                'name': video_name,
                'description': video_dict[video_name]['description'],
                'duration_minutes': video_dict[video_name]['duration_minutes'],
                'video_url': video_dict[video_name]['video_url'],
                'thumbnail_path': os.path.join(self.thumbnail_folder_path, video_dict[video_name]['thumbnail'])
            }
            recommended_videos.append(video_info)
        
        return recommended_videos

    def get_video_index(self, video_name):
        """获取视频名称对应的索引"""
        return videos.index(video_name)

    def get_video_info(self, video_idx):
        """获取完整视频信息"""
        video_name = videos[video_idx]
        return video_dict[video_name]

    def calculate_stress_reduction_reward(self, pre_stress, post_stress, subjective_rating=None):
        """
        计算压力降低奖励（融合客观和主观反馈）
        
        将压力降低量和主观评分转换为推荐系统的奖励信号
        奖励设计原则：
        - 主要基于客观压力降低效果
        - 辅助考虑用户主观满意度
        - 两者权重可调整
        
        参数:
        - pre_stress: 观看视频前的压力评级
        - post_stress: 观看视频后的压力评级
        - subjective_rating: 用户主观评分(1-5)，可选
        
        返回:
        - reward: 计算得出的奖励值(可能为负)
        """
        # 客观奖励：基于压力降低量
        stress_reduction = pre_stress - post_stress
        
        if stress_reduction >= stress_reduction_threshold:
            # 有效降压：给予正奖励，降压越多奖励越高
            objective_reward = min(stress_reduction * 2, 5.0)  # 最高5分奖励
        elif stress_reduction > 0:
            # 轻微降压：给予较小正奖励
            objective_reward = stress_reduction * 1.5
        elif stress_reduction >= -max_stress_increase:
            # 压力略微增加：小幅负奖励
            objective_reward = stress_reduction * 0.5
        else:
            # 压力大幅增加：较大负奖励
            objective_reward = -2.0
        
        # 如果没有主观评分，只返回客观奖励
        if subjective_rating is None:
            return max(-3.0, min(5.0, objective_reward))
        
        # 主观奖励：将1-5评分转换为-2到+2的奖励
        subjective_reward = (subjective_rating - 3) * 0.8  # 3分为中性，±0.8为调整幅度
        
        # 融合客观和主观奖励 (客观权重70%，主观权重30%)
        objective_weight = 0.7
        subjective_weight = 0.3
        
        final_reward = (objective_weight * objective_reward + 
                       subjective_weight * subjective_reward)
        
        # 确保奖励在合理范围内
        return max(-3.0, min(5.0, final_reward))

    def update_model(self, pre_stress, post_stress, selected_video_idx, video_indices_list, subjective_rating=None):
        """
        更新推荐模型
        
        基于观看视频前后的压力变化和用户主观评分，更新推荐算法参数
        
        参数:
        - pre_stress: 观看前压力评级
        - post_stress: 观看后压力评级  
        - selected_video_idx: 用户选择的视频索引
        - video_indices_list: 本次推荐的所有视频索引列表
        - subjective_rating: 用户主观评分(1-5)，可选
        """
        stress_idx = self.stress_level_to_index(pre_stress)
        reward = self.calculate_stress_reduction_reward(pre_stress, post_stress, subjective_rating)
        
        # 关闭现有数据库连接，避免并发问题
        self.general_mab_instance.close_db()
        self.user_mab_instance.close_db()
        
        # 重新创建MAB实例
        self._reinitialize_mab_instances()
        
        # 更新推荐模型
        if modify_non_selected:
            # 更新所有推荐视频的反馈
            for video_idx in video_indices_list:
                if video_idx == selected_video_idx:
                    # 被选择的视频：使用实际奖励
                    self.general_mab_instance.update(reward, selected_video_idx, stress_idx, selected=True)
                    self.user_mab_instance.update(reward, selected_video_idx, stress_idx, selected=True)
                else:
                    # 未被选择的视频：给予轻微负面反馈
                    self.general_mab_instance.update(reward, video_idx, stress_idx, selected=False)
                    self.user_mab_instance.update(reward, video_idx, stress_idx, selected=False)
        else:
            # 仅更新被选择的视频
            if selected_video_idx != -1:
                self.general_mab_instance.update(reward, selected_video_idx, stress_idx, selected=True)
                self.user_mab_instance.update(reward, selected_video_idx, stress_idx, selected=True)

    def log_activity(self, pre_stress, post_stress, selected_video_idx, subjective_rating=None):
        """
        记录活动日志
        
        参数:
        - pre_stress: 观看前压力评级
        - post_stress: 观看后压力评级
        - selected_video_idx: 选择的视频索引
        - subjective_rating: 用户主观评分(1-5)，可选
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stress_reduction = pre_stress - post_stress
        reward = self.calculate_stress_reduction_reward(pre_stress, post_stress, subjective_rating)
        
        # 构建活动记录
        general_activity_entry = {
            "user_id": self.user_id,
            "platform": self.platform,
            "timestamp": current_time,
            "pre_stress_level": pre_stress,
            "post_stress_level": post_stress,
            "stress_reduction": stress_reduction,
            "recommended_video": videos[selected_video_idx],
            "calculated_reward": reward
        }
        
        user_activity_entry = {
            "timestamp": current_time,
            "pre_stress_level": pre_stress,
            "post_stress_level": post_stress,
            "stress_reduction": stress_reduction,
            "recommended_video": videos[selected_video_idx],
            "calculated_reward": reward
        }
        
        # 保存到数据库
        self.general_mab_instance.update_activity(general_activity_entry)
        self.user_mab_instance.update_activity(user_activity_entry)

    def _reinitialize_mab_instances(self):
        """
        重新初始化MAB实例（用于避免并发问题）
        后续可以考虑使用连接池来处理并发问题，目前的处理方法过于简单，
        存在性能低效，资源浪费等问题，并且竞争条件仍然存在。
        """
        if modify_non_selected:
            self.general_mab_instance = ContextualMAB(
                len(videos), len(stress_levels), 2, 
                general_data_platform, general_data_user_hash, 
                threshold_num, auxiliary_rewards,
                enable_cluster, cluster_reco_ratio, mab_reco_ratio, 
                cluster_refresh_threshold, assign_cluster_num, recluster_threshold
            )
            self.user_mab_instance = ContextualMAB(
                len(videos), len(stress_levels), 2, 
                self.platform, self.user_id, 
                threshold_num, auxiliary_rewards,
                enable_cluster, cluster_reco_ratio, mab_reco_ratio, 
                cluster_refresh_threshold, assign_cluster_num, recluster_threshold
            )
        else:
            self.general_mab_instance = ContextualMAB(
                len(videos), len(stress_levels), 2, 
                general_data_platform, general_data_user_hash,
                enable_cluster=enable_cluster,
                cluster_reco_ratio=cluster_reco_ratio, 
                mab_reco_ratio=mab_reco_ratio, 
                cluster_refresh_threshold=cluster_refresh_threshold, 
                assign_cluster_num=assign_cluster_num,
                recluster_threshold=recluster_threshold
            )
            self.user_mab_instance = ContextualMAB(
                len(videos), len(stress_levels), 2, 
                self.platform, self.user_id,
                enable_cluster=enable_cluster,
                cluster_reco_ratio=cluster_reco_ratio, 
                mab_reco_ratio=mab_reco_ratio, 
                cluster_refresh_threshold=cluster_refresh_threshold, 
                assign_cluster_num=assign_cluster_num,
                recluster_threshold=recluster_threshold
            )

    def close_connections(self):
        """关闭数据库连接"""
        self.general_mab_instance.close_db()
        self.user_mab_instance.close_db()

    def get_user_statistics(self):
        """
        获取用户使用统计信息
        
        返回:
        - stats: 包含用户使用情况的统计字典
        """
        stats = {
            'total_sessions': self.user_mab_instance.total_selections,
            'stress_levels_used': {},
            'favorite_videos': {},
            'average_stress_reduction': 0.0
        }
        
        # 计算各压力等级的使用次数
        for i, level in enumerate(stress_levels):
            level_selections = np.sum(self.user_mab_instance.selections[:, i])
            stats['stress_levels_used'][level] = int(level_selections)
        
        # 计算最受欢迎的视频
        total_video_selections = np.sum(self.user_mab_instance.selections, axis=1)
        for i, video in enumerate(videos):
            if total_video_selections[i] > 0:
                avg_reward = np.sum(self.user_mab_instance.rewards[i, :]) / total_video_selections[i]
                stats['favorite_videos'][video] = {
                    'selection_count': int(total_video_selections[i]),
                    'average_reward': float(avg_reward)
                }
        
        return stats 