# 这是使用上置信界算法(Upper Confidence Bound Algorithm)实现的多臂老虎机类对象
# 
# 多臂老虎机(Multi-Armed Bandit, MAB)是一个经典的强化学习问题：
# - 想象一个赌场有多台老虎机(多个"臂")，每台机器的奖励概率不同但未知
# - 目标是在有限的尝试次数内，最大化总奖励
# - 核心挑战：平衡"探索"(尝试新选择)和"利用"(选择已知好的选择)
# 
# 情境多臂老虎机(Contextual MAB)扩展了这个概念：
# - 考虑不同的"情境"(如不同压力等级)
# - 在每个情境下，同一个视频可能有不同的效果
# - 算法需要学习每个情境下每个视频的最优选择策略

import numpy as np
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# 定义MongoDB连接字符串和配置参数
load_dotenv()
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")  # 数据库连接字符串
DB_NAME = os.getenv("DB_NAME")                           # 数据库名称
CLUSTER_SIZE = 5                                         # 用户聚类数量
RATIO_DIFF = 0.1  # 聚类权重调整参数：数值越大表示在推荐时给予聚类数据更多权重
                  # 注意：cluster_reco_ratio/mab_reco_ratio +/- RATIO_DIFF 应在[0.0, 1.0]范围内

class ContextualMAB:
    """
    情境多臂老虎机算法实现类
    
    该类实现了基于UCB算法的情境多臂老虎机，用于在不同压力等级下为用户推荐最优的降压视频。
    
    核心算法原理：
    1. UCB算法公式: UCB_i = 平均奖励_i + c * sqrt(ln(总选择次数) / 选择次数_i)
       - 平均奖励_i: 第i个选择的历史平均奖励(利用项)
       - 探索项: c * sqrt(ln(总选择次数) / 选择次数_i)，鼓励探索较少选择的选项
    
    2. 情境感知: 为每个压力等级维护独立的奖励矩阵，学习压力特定的偏好
    
    3. 用户聚类: 将相似偏好的用户聚类，利用群体智慧改善推荐效果
    """
    
    def __init__(self, num_videos, num_stress_levels, exploration_param, platform, user_hash, 
                threshold_num=None, non_selected_rewards=None,
                enable_cluster=False, cluster_reco_ratio=None, mab_reco_ratio=None, 
                cluster_refresh_threshold=None, assign_cluster_num=None, recluster_threshold=None):
        """
        初始化情境多臂老虎机实例
        
        参数说明:
        - num_videos: 降压视频数量(对应MAB中的"臂"数量)
        - num_stress_levels: 压力等级数量(5个等级：1-5)
        - exploration_param: 探索参数c，控制探索与利用的平衡
        - platform: 平台标识(eeg_stress)
        - user_hash: 用户唯一标识符
        - threshold_num: 开始修改未选择视频奖励的阈值选择次数
        - non_selected_rewards: 给予未选择视频的奖励值
        - enable_cluster: 是否启用用户聚类功能
        - cluster_reco_ratio: 聚类推荐权重
        - mab_reco_ratio: MAB推荐权重
        - cluster_refresh_threshold: 聚类刷新阈值
        - assign_cluster_num: 分配用户到聚类所需的最小选择次数
        - recluster_threshold: 重新聚类的阈值
        """
        # 基本参数设置
        self.num_videos = num_videos               # 视频数量
        self.num_stress_levels = num_stress_levels # 压力等级数量(5个)
        self.exploration_param = exploration_param  # 探索参数，用于UCB公式
        self.platform = platform                   # 平台类型
        self.user_hash = user_hash                 # 用户哈希ID
        
        # 可选功能参数
        self.threshold_num = threshold_num                     # 达到此阈值后开始修改未选择项的奖励
        self.non_selected_rewards = non_selected_rewards      # 未选择项的奖励值
        
        # 聚类功能参数
        self.enable_cluster = enable_cluster                   # 是否启用聚类
        self.cluster_reco_ratio = cluster_reco_ratio          # 聚类推荐权重
        self.mab_reco_ratio = mab_reco_ratio                  # MAB推荐权重
        self.cluster_refresh_threshold = cluster_refresh_threshold  # 聚类刷新频率
        self.assign_cluster_num = assign_cluster_num          # 分配到聚类的最小选择数
        self.recluster_threshold = recluster_threshold        # 重新聚类的阈值
        
        # 初始化数据库连接
        self.db_client = MongoClient(DB_CONNECTION_STRING)
        self.db = self.db_client[DB_NAME]
        self.collection = self.db[self.platform]
        
        # 读取并初始化情境多臂老虎机参数
        self.user_documents = self.collection.find_one({'user_id': self.user_hash})
        
        if self.user_documents is None:
            # 新用户：初始化所有数据结构
            # rewards矩阵: num_videos(行) x num_stress_levels(列)，记录累积奖励
            self.rewards = np.zeros((num_videos, num_stress_levels))
            # selections矩阵: 记录每个(视频,压力等级)组合被选择的次数
            self.selections = np.zeros((num_videos, num_stress_levels))
            # mab_scores矩阵: 记录UCB分数，初始设为无穷大鼓励探索
            self.mab_scores = np.full((num_videos, num_stress_levels), float('inf'))
            # 总选择次数计数器
            self.total_selections = 0
            
            # 如果启用聚类且不是总数据平台，初始化偏好向量
            if self.platform != "total_data" and self.enable_cluster:
                self.pref_vector = np.zeros((num_videos, num_stress_levels))
        else:
            # 已存在用户：从数据库加载历史数据
            self.mab_data = self.user_documents.get('mab_data', {})
            self.rewards = np.array(self.mab_data.get('rewards', np.zeros((num_videos, num_stress_levels))))
            self.selections = np.array(self.mab_data.get('selections', np.zeros((num_videos, num_stress_levels))))
            self.mab_scores = np.array(self.mab_data.get('mab_scores', np.full((num_videos, num_stress_levels), float('inf'))), dtype=object)
            self.total_selections = self.mab_data.get('total_selections', 0)
            
            # 加载聚类相关数据
            if self.platform != "total_data" and self.enable_cluster:
                self.user_cluster_data = self.user_documents.get('cluster_data', {})
                # 如果用户已有足够选择次数，分配到聚类
                if self.total_selections >= self.assign_cluster_num:
                    self.user_cluster_id = self.user_cluster_data.get('cluster_id', f'cluster_{np.random.randint(0, CLUSTER_SIZE)}')
                self.pref_vector = np.array(self.user_cluster_data.get('pref_vector', np.zeros((num_videos, num_stress_levels))))
                # 检查用户偏好向量维度是否正确
                self.check_user_pref_vector()
        
        # 获取聚类信息(用于所有用户的群体偏好数据)
        if self.enable_cluster:
            self.cluster_collection = self.db['total_data']
            self.cluster_documents = self.cluster_collection.find_one({'user_id': 'total'})
            if self.cluster_documents is not None:
                self.cluster_data = self.cluster_documents.get('cluster_data', {})
            else:
                self.cluster_data = {}
            # 检查聚类偏好向量维度是否正确
            self.check_cluster_data()

    def check_user_pref_vector(self):
        """
        检查并修正用户偏好向量的维度
        
        当系统配置发生变化(如增加新视频或新压力等级)时，需要调整已有用户的偏好向量维度
        """
        curr_row, curr_column = self.pref_vector.shape
        if curr_row == self.num_videos and curr_column == self.num_stress_levels:
            return  # 维度正确，无需调整
        else:
            # 调整行维度(视频数量)
            if curr_row < self.num_videos:
                # 增加行：新视频初始偏好为0
                extra_rows = np.zeros((self.num_videos - curr_row, curr_column))
                self.pref_vector = np.vstack((self.pref_vector, extra_rows))
            elif curr_row > self.num_videos:
                # 减少行：删除多余视频
                self.pref_vector = self.pref_vector[:self.num_videos, :]

            # 调整列维度(压力等级数量)
            if curr_column < self.num_stress_levels:
                # 增加列：新压力等级初始偏好为0
                extra_cols = np.zeros((self.num_videos, self.num_stress_levels - curr_column))
                self.pref_vector = np.hstack((self.pref_vector, extra_cols))
            elif curr_column > self.num_stress_levels:
                # 减少列：删除多余压力等级
                self.pref_vector = self.pref_vector[:, :self.num_stress_levels]
   
    def check_cluster_data(self):
        """
        检查并修正聚类数据的维度
        
        确保所有聚类的偏好向量维度与当前系统配置一致
        """
        changed = False
        for i in range(CLUSTER_SIZE):
            curr_cluster_data = np.array(self.cluster_data.get(f'cluster_{i}', np.zeros((self.num_videos, self.num_stress_levels))))
            curr_row, curr_column = curr_cluster_data.shape
            
            if curr_row == self.num_videos and curr_column == self.num_stress_levels:
                continue  # 该聚类维度正确
            else:
                changed = True
                # 调整行维度
                if curr_row < self.num_videos:
                    # 新视频使用随机值初始化(范围0-5，模拟压力降低效果)
                    extra_rows = np.random.rand(self.num_videos - curr_row, curr_column) * 3
                    curr_cluster_data = np.vstack((curr_cluster_data, extra_rows))
                elif curr_row > self.num_videos:
                    curr_cluster_data = curr_cluster_data[:self.num_videos, :]

                # 调整列维度
                if curr_column < self.num_stress_levels:
                    # 新压力等级使用随机值初始化
                    extra_cols = np.random.rand(self.num_videos, self.num_stress_levels - curr_column) * 3
                    curr_cluster_data = np.hstack((curr_cluster_data, extra_cols))
                elif curr_column > self.num_stress_levels:
                    curr_cluster_data = curr_cluster_data[:, :self.num_stress_levels]

                # 更新数据库中的聚类数据
                curr_cluster_data_list = curr_cluster_data.tolist()
                self.cluster_collection.find_one_and_update(
                    {'user_id': 'total'},
                    {'$set': {
                        f'cluster_data.cluster_{i}': curr_cluster_data_list,
                    }},
                    return_document=True,  # 返回更新后的文档
                    upsert=True           # 如果文档不存在则创建
                )
        
        # 如果有变化，重新加载聚类数据
        if changed:
            self.cluster_data = self.cluster_documents.get('cluster_data', {})

    def recommend(self, rec_size, stress_level_index, first_time, avoid_indices=None):
        """
        基于给定压力等级推荐top rec_size个视频
        
        这是整个算法的核心推荐函数
        
        输入参数:
        - rec_size: 需要推荐的视频数量
        - stress_level_index: 当前压力等级索引(0-4对应压力等级1-5)
        - first_time: 是否为首次推荐(首次推荐会随机选择以收集初始数据)
        - avoid_indices: 需要避免推荐的视频索引列表
        
        输出:
        - 推荐的视频索引列表，按UCB分数从高到低排序
        
        算法流程:
        1. 提取当前压力等级下所有视频的UCB分数
        2. 如果启用聚类，结合聚类偏好调整分数
        3. 根据是否首次推荐选择不同策略：
           - 首次：随机选择(探索)
           - 非首次：按UCB分数选择(利用+探索平衡)
        """
        # 获取当前压力等级下所有视频的MAB分数
        stress_mab_scores = self.mab_scores[:, stress_level_index]
        
        # 默认只使用MAB分数
        reco_scores = stress_mab_scores
        
        # 如果启用聚类功能，结合聚类偏好
        if self.platform != "total_data" and self.enable_cluster:
            stress_pref_vector = self.pref_vector[:, stress_level_index]
            
            # 用户已被分配到聚类：使用聚类群体偏好
            if self.total_selections >= self.assign_cluster_num:
                user_cluster_data = np.array(self.cluster_data.get(self.user_cluster_id, np.zeros((self.num_videos, self.num_stress_levels))))
                stress_user_cluster_data = user_cluster_data[:, stress_level_index]
                # 组合MAB分数和聚类偏好，动态调整权重
                reco_scores = (self.mab_reco_ratio - RATIO_DIFF) * stress_mab_scores + (self.cluster_reco_ratio + RATIO_DIFF) * stress_user_cluster_data
            else:
                # 用户尚未分配到聚类：使用个人偏好向量
                reco_scores = self.mab_reco_ratio * stress_mab_scores + self.cluster_reco_ratio * stress_pref_vector

        # 根据分数进行推荐
        if first_time:
            # 首次推荐：随机选择以收集初始数据(纯探索策略)
            if avoid_indices is None or len(avoid_indices) == 0:
                if len(reco_scores) < rec_size:
                    indices = np.arange(len(reco_scores))
                    return np.random.permutation(indices)
                else:
                    return np.random.choice(len(reco_scores), size=rec_size, replace=False)
            else:
                # 考虑避免索引的随机选择
                valid_indices = [i for i in range(len(reco_scores)) if i not in avoid_indices]
                if len(valid_indices) < rec_size:
                    return np.random.permutation(valid_indices)
                else:
                    return np.random.choice(valid_indices, size=rec_size, replace=False)
        else:
            # 非首次推荐：基于UCB分数选择(利用+探索策略)
            if avoid_indices is None or len(avoid_indices) == 0:
                # 返回分数最高的rec_size个视频，降序排列
                return np.argsort(reco_scores)[-rec_size:][::-1]
            else:
                # 排除避免索引后选择分数最高的视频
                sorted_indices = np.argsort(reco_scores)
                reco_indices = [idx for idx in sorted_indices if idx not in avoid_indices][-rec_size:][::-1]
                return reco_indices

    def update(self, stress_reduction_reward, video_index, stress_level_index, selected=True):
        """
        基于压力降低效果更新MAB分数和奖励
        
        这是算法学习的核心函数，根据视频降压效果调整推荐策略
        
        输入参数:
        - stress_reduction_reward: 压力降低奖励(基于观看前后压力差值计算)
        - video_index: 被选择(或未被选择)的视频索引
        - stress_level_index: 当前压力等级索引
        - selected: 该视频是否被用户选择
        
        算法更新逻辑:
        1. 如果视频被选择：
           - 累加奖励值
           - 增加选择计数
           - 重新计算UCB分数
           - 更新数据库
           - 如果启用聚类，更新聚类数据
        2. 如果视频未被选择且满足条件：
           - 给予少量负面奖励(降低未选择视频的吸引力)
        """
        if selected:
            # 视频被用户选择：正面学习
            # 累加压力降低奖励到奖励矩阵
            self.rewards[video_index][stress_level_index] += stress_reduction_reward
            self.total_selections += 1  # 增加总选择计数
            self.selections[video_index][stress_level_index] += 1  # 增加特定选择计数
            
            # 重新计算UCB分数
            self.update_mab_score(video_index, stress_level_index)
            
            # 更新数据库中的MAB数据
            self.update_mab_data()
            
            # 如果启用聚类，更新聚类相关数据
            if self.enable_cluster:
                self.update_cluster_data(video_index, stress_level_index)
        else:
            # 视频未被用户选择：负面学习(可选功能)
            if (self.threshold_num is not None and 
                self.non_selected_rewards is not None and 
                self.total_selections > self.threshold_num):
                
                # 给予未选择视频少量负面奖励
                self.rewards[video_index][stress_level_index] += self.non_selected_rewards
                
                # 只有当该视频之前被选择过时才更新MAB分数
                if self.selections[video_index][stress_level_index] > 0:
                    self.update_mab_score(video_index, stress_level_index)
                
                # 更新数据库
                self.update_mab_data()

    def update_mab_score(self, video_index, stress_level_index):
        """
        更新MAB分数 (利用 + 探索)
        
        实现UCB(Upper Confidence Bound)算法的核心公式：
        UCB_i = μ_i + c * sqrt(ln(t) / n_i)
        
        其中：
        - μ_i: 视频i的平均奖励(利用项)
        - c: 探索参数
        - t: 总选择次数
        - n_i: 视频i被选择的次数
        - sqrt(ln(t) / n_i): 探索项，鼓励选择较少被尝试的视频
        
        参数:
        - video_index: 要更新的视频索引
        - stress_level_index: 当前压力等级索引
        """
        # 计算利用分数：平均奖励
        exploitation_score = self.rewards[video_index][stress_level_index] / self.selections[video_index][stress_level_index]
        
        # 计算探索分数：置信上界
        exploration_score = np.sqrt((self.exploration_param * np.log(self.total_selections)) / self.selections[video_index][stress_level_index])
        
        # 更新UCB分数 = 利用分数 + 探索分数
        self.mab_scores[video_index][stress_level_index] = exploitation_score + exploration_score

    def update_mab_data(self):
        """
        更新数据库中的MAB数据
        
        将内存中的奖励矩阵、选择计数矩阵、UCB分数矩阵和总选择次数
        同步保存到MongoDB数据库中，确保数据持久化
        """
        # 将numpy数组转换为Python列表以便JSON序列化
        rewards_list = self.rewards.tolist()
        mab_scores_list = self.mab_scores.tolist()
        selections_list = self.selections.tolist()
        
        # 使用MongoDB的find_one_and_update方法原子性地更新用户数据
        self.collection.find_one_and_update(
            {'user_id': self.user_hash},  # 查找条件：匹配用户ID
            {'$set': {                    # 更新操作：设置MAB相关数据
                'mab_data.rewards': rewards_list,
                'mab_data.selections': selections_list,
                'mab_data.mab_scores': mab_scores_list,
                'mab_data.total_selections': self.total_selections
            }},
            return_document=True,  # 返回更新后的文档
            upsert=True           # 如果用户不存在则创建新文档
        )
   
    def update_cluster_data(self, video_index, stress_level_index):
        """
        更新聚类信息
        
        当用户做出选择后，更新其偏好向量并重新评估聚类分配
        这是协同过滤推荐的核心：利用相似用户的偏好改善推荐效果
        
        参数:
        - video_index: 被选择的视频索引
        - stress_level_index: 当前压力等级索引
        
        更新流程:
        1. 更新用户偏好向量(基于平均奖励)
        2. 如果用户有足够选择历史，重新分配聚类
        3. 将更新后的偏好向量保存到数据库
        4. 如果是总数据平台且满足条件，触发重新聚类
        """
        if self.platform != "total_data":
            # 第一步：更新用户偏好向量信息
            # 偏好向量存储每个(视频,压力等级)组合的平均奖励，反映用户真实偏好
            self.pref_vector[video_index][stress_level_index] = self.rewards[video_index][stress_level_index] / self.selections[video_index][stress_level_index]
            
            # 第二步：如果用户需要被分配到聚类或更换聚类
            if self.total_selections >= self.assign_cluster_num:
                self.update_user_cluster()
            
            # 第三步：更新用户聚类信息和偏好向量到数据库
            pref_vector_list = self.pref_vector.tolist()
            self.collection.find_one_and_update(
                {'user_id': self.user_hash},
                {'$set': {
                    'cluster_data.pref_vector': pref_vector_list
                    # 注意：cluster_id在update_user_cluster()中单独更新
                }},
                return_document=True,  # 返回更新后的文档
                upsert=True           # 如果文档不存在则创建
            )
        
        # 最后：检查是否需要重新安排聚类
        if (self.platform == "total_data" and 
            self.total_selections >= self.recluster_threshold and 
            self.total_selections % self.cluster_refresh_threshold == 0):
            self.refresh_cluster()

    def update_user_cluster(self):
        """
        更新用户的聚类分配
        
        根据用户当前的偏好向量，计算其与各个聚类中心的距离，
        将用户分配到距离最近的聚类中
        
        算法流程:
        1. 计算用户偏好向量与每个聚类中心的欧几里得距离
        2. 选择距离最小的聚类
        3. 更新数据库中的用户聚类ID
        """
        distances = []
        
        # 计算与每个聚类中心的距离
        for i in range(CLUSTER_SIZE):
            distance = self.get_euclidean_dist(
                self.pref_vector, 
                self.cluster_data.get(f'cluster_{i}', 
                                      np.zeros((self.num_videos, self.num_stress_levels)))
            )
            distances.append(distance)
        
        # 获取距离最短的聚类索引并更新用户聚类分配
        target_cluster_index = np.argmin(distances)
        self.user_cluster_id = f'cluster_{target_cluster_index}'  # 更新实例属性
        self.collection.find_one_and_update(
            {'user_id': self.user_hash},
            {'$set': {
                'cluster_data.cluster_id': self.user_cluster_id
            }},
            return_document=True,  # 返回更新后的文档
            upsert=True           # 如果文档不存在则创建
        )

    def refresh_cluster(self):
        """
        重新聚类 (K-means聚类算法)
        
        定期重新计算聚类中心，确保聚类能够反映用户偏好的变化
        使用经典的K-means算法优化聚类质量
        """
        max_iteration_count = 1000  # 最大迭代次数，防止无限循环
        tolerance = 1e-4           # 收敛阈值
        
        # 获取当前聚类中心点
        pivots = []  # 存储聚类中心，列表格式便于操作
        for i in range(CLUSTER_SIZE):
            pivots.append(self.cluster_data.get(f'cluster_{i}', 
                                                np.zeros((self.num_videos, self.num_stress_levels))))
        
        # 收集所有用户的聚类数据
        user_cluster_data = []
        platforms = ["eeg_stress"]  # EEG压力推荐平台
        
        for platform in platforms:
            curr_plat = self.db[platform]
            # 查询该平台所有用户的偏好向量
            plat_users = list(curr_plat.find({}, {"cluster_data.pref_vector": 1, "_id": 0}))
            for user in plat_users:
                user_pref_vector = user.get("cluster_data", {}).get("pref_vector")
                if user_pref_vector is not None:
                    user_cluster_data.append(user_pref_vector)
        
        # K-means迭代过程
        for iteration in range(max_iteration_count):
            # 步骤1：将每个用户数据点分配到最近的聚类中心
            labels = []
            for user_data in user_cluster_data:
                min_distance = float('inf')
                min_index = 0
                for i, pivot in enumerate(pivots):
                    distance = self.get_euclidean_dist(np.array(user_data), np.array(pivot))
                    if distance < min_distance:
                        min_distance = distance
                        min_index = i
                labels.append(min_index)

            # 步骤2：重新计算聚类中心
            new_pivots = []
            for i in range(CLUSTER_SIZE):
                # 获取分配到第i个聚类的所有用户数据
                cluster_points = [user_cluster_data[j] for j, label in enumerate(labels) if label == i]
                if cluster_points:
                    # 计算该聚类所有用户的平均偏好向量作为新的聚类中心
                    cluster_arrays = [np.array(point) for point in cluster_points]
                    stacked_array = np.stack(cluster_arrays)
                    pivot_mean = np.mean(stacked_array, axis=0)
                    new_pivots.append(pivot_mean.tolist())
                else:
                    # 如果某个聚类没有分配到用户，保持原有中心
                    new_pivots.append(pivots[i])

            # 步骤3：检查收敛性
            pivot_diff = self.get_euclidean_dist(np.array(pivots), np.array(new_pivots))
            if pivot_diff < tolerance:
                break  # 收敛，停止迭代

            # 步骤4：更新聚类中心
            pivots = new_pivots

        # 将新的聚类中心更新到数据库
        for i in range(CLUSTER_SIZE):
            self.cluster_data[f'cluster_{i}'] = pivots[i]
            self.cluster_collection.find_one_and_update(
                {'user_id': 'total'},
                {'$set': {
                    f'cluster_data.cluster_{i}': self.cluster_data[f'cluster_{i}'],
                }},
                return_document=True,  # 返回更新后的文档
                upsert=True           # 如果文档不存在则创建
            )

    def get_euclidean_dist(self, arr1, arr2):
        """
        计算两个2D数组之间的欧几里得距离
        
        用于衡量用户偏好向量之间的相似性，距离越小表示偏好越相似
        
        参数:
        - arr1, arr2: 要比较的两个numpy数组
        
        返回:
        - 欧几里得距离(标量值)
        """
        return np.linalg.norm(arr1 - arr2)
   
    def update_activity(self, activity_entry):
        """
        更新数据库中的活动数据
        
        记录用户的活动历史，用于系统分析和改进推荐算法
        
        参数:
        - activity_entry: 包含时间戳、压力等级、推荐视频、压力降低效果等信息的字典
        """
        self.collection.find_one_and_update(
            {'user_id': self.user_hash},
            {'$push': {  # 使用$push将新记录添加到数组末尾
                'activity_history': activity_entry
            }},
            return_document=True,  # 返回更新后的文档
            upsert=True           # 如果文档不存在则创建
        )
   
    def is_first_time(self, stress_level_index, reco_size):
        """
        检查当前数据状态，判断是否应该进行首次推荐
        
        首次推荐采用纯探索策略(随机选择)，以收集足够的初始数据
        非首次推荐则采用UCB策略平衡探索与利用
        
        参数:
        - stress_level_index: 当前压力等级索引
        - reco_size: 推荐数量
        
        返回:
        - True: 应该进行随机推荐(探索)
        - False: 应该基于UCB分数推荐(利用+探索)
        
        判断逻辑:
        - 如果启用聚类：总是使用UCB策略，因为可以利用群体智慧
        - 如果未启用聚类：当总选择次数为0或当前压力等级选择次数不足时，进行随机推荐
        """
        # 如果启用聚类功能，不给出随机建议，因为可以利用聚类信息
        if self.enable_cluster:
            return False
        
        # 如果未启用聚类，检查数据充分性
        stress_selections = self.selections[:, stress_level_index]  # 当前压力等级下的所有选择计数
        if self.total_selections == 0 or np.sum(stress_selections) < reco_size:
            return True  # 数据不足，进行随机探索
        else:
            return False  # 数据充分，使用UCB策略

    def close_db(self):
        """
        关闭数据库客户端连接
        
        在完成所有数据库操作后调用，释放资源防止连接泄漏
        """
        self.db_client.close() 