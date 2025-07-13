"""
数据分析模块

该模块负责从原始交互数据中挖掘更深层次的洞察，
包括但不限于：
- 题目画像分析 (难度、区分度、参与度等)
- 学生学习行为分析 (学习速度、稳健性等)
- “学生-题目”交互匹配度分析

这些分析结果将为上层应用（如LLM决策）提供数据支持。
"""

import pandas as pd
import numpy as np
import torch
import json
import datetime

from .data_loader import DataLoader
from .orcdf.model import ORCDF
from .train_orcdf import to_sparse_tensor

class QuestionAnalytics:
    """
    负责计算和管理所有与题目（问题）相关的分析指标。
    """
    def __init__(self, interaction_data: pd.DataFrame, problem_map: dict):
        """
        使用已经加载过的数据进行初始化。

        Args:
            interaction_data (pd.DataFrame): 包含交互记录的DataFrame。
            problem_map (dict): 问题ID到索引的映射。
        """
        self.interactions = interaction_data
        self.problem_map_rev = {v: k for k, v in problem_map.items()}
        
        # 为不同的计算准备不同的DataFrame
        self.interactions_fordev = self.interactions[['student_id_idx', 'problem_id_idx', 'correct']]
        self.interactions_foreng = self.interactions[['problem_id_idx','ms_first_response','hint_count']]

        # 用于缓存计算结果
        self._difficulty = None

    def calculate_difficulty(self) -> pd.DataFrame:
        """
        计算每道题的难度（P-value），即题目通过率。
        难度值越接近1，题目越简单；越接近0，题目越难。

        Returns:
            pd.DataFrame: 包含 'problem_id' 和 'difficulty' 两列。
        """
        if self._difficulty is not None:
            return self._difficulty

        print("Calculating question difficulty (p-value)...")
        
        # 按题目ID分组，计算每次交互的平均正确率
        difficulty_df = self.interactions_fordev.groupby('problem_id_idx')['correct'].mean().reset_index()
        difficulty_df.rename(columns={'correct': 'difficulty'}, inplace=True)
        
        # 添加原始 problem_id
        difficulty_df['problem_id'] = difficulty_df['problem_id_idx'].map(self.problem_map_rev)
        
        # 重新排列并筛选列
        self._difficulty = difficulty_df[['problem_id', 'difficulty']]
        
        print(f"Difficulty calculated for {len(self._difficulty)} problems.")
        return self._difficulty

    def calculate_discrimination(self, upper_group_ratio=0.27, lower_group_ratio=0.27) -> pd.DataFrame:
        """
        计算每道题的区分度指数。

        区分度衡量了题目在区分高能力和低能力学生方面的有效性。
        计算方法：(高分组正确率) - (低分组正确率)。
        值越高，区分能力越强。负值可能表示题目有问题。

        Args:
            upper_group_ratio (float): 定义为高分组的学生比例。
            lower_group_ratio (float): 定义为低分组的学生比例。

        Returns:
            pd.DataFrame: 包含 'problem_id' 和 'discrimination' 两列。
        """
        print("Calculating question discrimination index...")
        
        # 1. 计算每个学生的总体表现（总正确率）
        student_performance = self.interactions_fordev.groupby('student_id_idx')['correct'].mean().reset_index()
        student_performance.rename(columns={'correct': 'overall_correct_rate'}, inplace=True)
        
        # 2. 根据表现划分高分组和低分组
        num_students = len(student_performance)
        upper_quantile = 1.0 - upper_group_ratio
        lower_quantile = lower_group_ratio
        
        high_performers_threshold = student_performance['overall_correct_rate'].quantile(upper_quantile)
        low_performers_threshold = student_performance['overall_correct_rate'].quantile(lower_quantile)
        
        high_performer_ids = student_performance[student_performance['overall_correct_rate'] >= high_performers_threshold]['student_id_idx']
        low_performer_ids = student_performance[student_performance['overall_correct_rate'] <= low_performers_threshold]['student_id_idx']

        # 3. 分别计算高、低分组在每道题上的正确率
        high_group_interactions = self.interactions_fordev[self.interactions_fordev['student_id_idx'].isin(high_performer_ids)]
        low_group_interactions = self.interactions_fordev[self.interactions_fordev['student_id_idx'].isin(low_performer_ids)]
        
        high_group_difficulty = high_group_interactions.groupby('problem_id_idx')['correct'].mean().rename('high_group_p')
        low_group_difficulty = low_group_interactions.groupby('problem_id_idx')['correct'].mean().rename('low_group_p')
        
        # 4. 合并数据并计算区分度
        discrimination_df = pd.concat([high_group_difficulty, low_group_difficulty], axis=1).fillna(0)
        discrimination_df['discrimination'] = discrimination_df['high_group_p'] - discrimination_df['low_group_p']
        
        discrimination_df.reset_index(inplace=True)
        discrimination_df['problem_id'] = discrimination_df['problem_id_idx'].map(self.problem_map_rev)
        
        print(f"Discrimination calculated for {len(discrimination_df)} problems.")
        return discrimination_df[['problem_id', 'discrimination']]

    def calculate_engagement_metrics(self) -> pd.DataFrame:
        """
        计算每道题的参与度指标。

        这包括：
        - 平均首次作答时长 (ms_first_response)
        - 平均提示使用次数 (hint_count)

        这些指标可以反映题目的认知负荷和迷惑程度。

        Returns:
            pd.DataFrame: 包含 'problem_id' 和各种参与度指标的列。
        """
        print("Calculating question engagement metrics...")

        # 确保 'interactions' DataFrame 中包含所需列
        required_cols = ['problem_id_idx', 'ms_first_response', 'hint_count']
        if not all(col in self.interactions_foreng.columns for col in required_cols):
            # 如果原始数据中没有这些列，我们需要重新加载
            print("Engagement columns not found. This feature requires the original interaction columns.")
            # 这个实现假设了 __init__ 时传入的 interactions 已经包含了这些列。
            # 如果没有，需要修改 QuestionAnalytics 的初始化逻辑。
            # 这里暂时返回一个空 DataFrame
            return pd.DataFrame(columns=['problem_id'])

        engagement_df = self.interactions_foreng.groupby('problem_id_idx').agg(
            avg_ms_first_response=('ms_first_response', 'mean'),
            avg_hint_count=('hint_count', 'mean')
        ).reset_index()

        engagement_df['problem_id'] = engagement_df['problem_id_idx'].map(self.problem_map_rev)

        print(f"Engagement metrics calculated for {len(engagement_df)} problems.")
        return engagement_df[['problem_id', 'avg_ms_first_response', 'avg_hint_count']]

class StudentAnalytics:
    """
    负责计算和管理所有与学生相关的分析指标。
    """
    def __init__(self, raw_data: dict, model: ORCDF, device: torch.device):
        """
        使用已经加载过的数据和模型进行初始化。

        Args:
            raw_data (dict): 从 DataLoader.load_orcdf_data() 返回的完整数据字典。
            model (ORCDF): 预加载的ORCDF模型实例。
            device (torch.device): 'cpu' or 'cuda'.
        """
        # 传入的是 orcdf_data，其中的 interactions 是一个 list of tuples
        # 我们需要将它转换为包含所有所需列的 DataFrame
        # 为此，我们还是需要最原始的 skill_builder_data
        self.interactions = raw_data['skill_builder_interactions']
        self.student_map_rev = {v: k for k, v in raw_data['student_map'].items()}
        self.student_map = raw_data['student_map']
        self.skill_map = raw_data['skill_map']
        # 'q_matrix' is a numpy array. For indexing like in the original code, we convert it to a DataFrame.
        self.problem_skill_matrix = pd.DataFrame(raw_data['q_matrix'])

        self.model = model
        self.device = device
        
        # 预先将图矩阵转换为稀疏张量以提高效率
        self.q_matrix_tensor = to_sparse_tensor(raw_data['q_matrix'], self.device)
        # 注意：这里的a_matrix和ia_matrix需要从orcdf_data中获取，我们在main中加载
        self.a_matrix_tensor = None
        self.ia_matrix_tensor = None


    def _run_prediction(self, student_ids: list, problem_ids: list) -> np.ndarray:
        """内部辅助函数，用于运行模型预测并返回numpy数组。"""
        if self.a_matrix_tensor is None or self.ia_matrix_tensor is None:
            raise ValueError("A/IA matrices not set. Call set_graph_matrices first.")

        student_tensor = torch.LongTensor(student_ids).to(self.device)
        problem_tensor = torch.LongTensor(problem_ids).to(self.device)
        
        with torch.no_grad():
            preds = self.model(
                student_tensor,
                problem_tensor,
                self.a_matrix_tensor,
                self.ia_matrix_tensor,
                self.q_matrix_tensor
            )
        return preds.cpu().numpy()

    def set_graph_matrices(self, a_matrix: np.ndarray, ia_matrix: np.ndarray):
        """设置图的邻接矩阵。"""
        self.a_matrix_tensor = to_sparse_tensor(a_matrix, self.device)
        self.ia_matrix_tensor = to_sparse_tensor(ia_matrix, self.device)

    def calculate_behavioral_patterns(self) -> pd.DataFrame:
        """
        计算每个学生的总体行为模式。

        包括：
        - 总体正确率 (overall_accuracy)
        - 平均作答时长 (avg_response_time)
        - 平均提示使用次数 (avg_hint_usage)

        Returns:
            pd.DataFrame: 包含学生ID和其行为模式指标。
        """
        print("Calculating student behavioral patterns...")

        student_behaviors = self.interactions.groupby('student_id_idx').agg(
            overall_accuracy=('correct', 'mean'),
            avg_response_time=('ms_first_response', 'mean'),
            avg_hint_usage=('hint_count', 'mean')
        ).reset_index()
        
        student_behaviors['student_id'] = student_behaviors['student_id_idx'].map(self.student_map_rev)

        print(f"Behavioral patterns calculated for {len(student_behaviors)} students.")
        return student_behaviors[['student_id', 'overall_accuracy', 'avg_response_time', 'avg_hint_usage']]

    def calculate_learning_velocity(self, student_id: int, skill_id: int, window_size: int = 10) -> dict:
        """
        计算指定学生在特定技能上的学习速度。

        Args:
            student_id (int): 学生的原始ID。
            skill_id (int): 技能的原始ID。
            window_size (int): 计算掌握度的滚动窗口大小。

        Returns:
            dict: 包含掌握度变化序列和计算出的学习速度。
        """
        print(f"\nCalculating learning velocity for student {student_id} on skill {skill_id}...")
        
        # 预先定义一个符合API响应模型的字典结构
        response = {
            "student_id": student_id,
            "skill_id": skill_id,
            "mastery_trend": None,
            "learning_velocity": None,
            "error": None
        }

        if student_id not in self.student_map:
            response["error"] = "Student ID not found."
            return response
        if skill_id not in self.skill_map:
            response["error"] = "Skill ID not found."
            return response
            
        student_idx = self.student_map[student_id]
        skill_idx = self.skill_map[skill_id]

        # 1. 筛选出与该技能相关的所有问题
        problem_indices_for_skill = self.problem_skill_matrix.index[self.problem_skill_matrix[skill_idx] == 1].tolist()
        
        # 2. 筛选出该学生与这些问题相关的所有交互，并按时间排序
        student_interactions = self.interactions[
            (self.interactions['student_id_idx'] == student_idx) &
            (self.interactions['problem_id_idx'].isin(problem_indices_for_skill))
        ].sort_values(by='order_id')
        
        if len(student_interactions) < window_size:
            response["error"] = f"Not enough interactions ({len(student_interactions)}) for this skill to calculate velocity."
            return response
            
        # 3. 在交互序列上滚动计算掌握度
        mastery_over_time = []
        problem_ids_for_pred = student_interactions['problem_id_idx'].tolist()
        
        # 预测所有相关交互的表现
        preds = self._run_prediction([student_idx] * len(problem_ids_for_pred), problem_ids_for_pred)
        
        # 使用滚动窗口计算平均掌握度
        rolling_mastery = pd.Series(preds).rolling(window=window_size).mean().dropna().tolist()
        
        if len(rolling_mastery) < 2:
            response["error"] = "Not enough data points to calculate a trend."
            return response
            
        # 4. 计算学习速度（线性回归的斜率）
        x = np.arange(len(rolling_mastery))
        y = np.array(rolling_mastery)
        slope, intercept = np.polyfit(x, y, 1) # y = slope * x + intercept
        
        response["mastery_trend"] = rolling_mastery
        response["learning_velocity"] = slope
        
        return response


# --- 用于独立测试该模块的示例代码 ---
def main():
    """
    运行完整的分析流程并将结果保存到JSON文件中。
    """
    print("--- Running Full Analytics Pipeline ---")
    
    # 为了让这个脚本能被直接运行，需要处理路径问题
    import os
    import sys
    # 将项目根目录添加到 sys.path
    if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # 1. 初始化 DataLoader 并加载数据
    try:
        from dkg_mvp.data_loader import DataLoader
        print("Running from a script context.")
    except ImportError:
        from .data_loader import DataLoader
        print("Running from a module context.")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_path = os.path.join(project_root, 'dataset')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found at '{dataset_path}'")
        return

    print("Loading data once for all analytics...")
    loader = DataLoader(dataset_path)
    raw_data = loader.load_skill_builder_data()
    # 为学生分析器额外加载orcdf格式数据以获取图矩阵
    orcdf_data = loader.load_orcdf_data()

    if not raw_data or not orcdf_data:
        print("Failed to load necessary data. Exiting.")
        return
        
    # 将原始交互数据添加到 orcdf_data 字典中，以便传递给 StudentAnalytics
    orcdf_data['skill_builder_interactions'] = raw_data['interactions']
    interaction_df = raw_data['interactions']
    
    # --- 题目分析 ---
    print("\n--- Starting Question Analytics ---")
    question_analytics = QuestionAnalytics(interaction_df, raw_data['problem_map'])
    
    difficulty_df = question_analytics.calculate_difficulty()
    discrimination_df = question_analytics.calculate_discrimination()
    engagement_df = question_analytics.calculate_engagement_metrics()
    
    # 合并所有题目分析结果
    question_analytics_df = pd.merge(difficulty_df, discrimination_df, on='problem_id', how='outer')
    question_analytics_df = pd.merge(question_analytics_df, engagement_df, on='problem_id', how='outer')
    question_analytics_df.set_index('problem_id', inplace=True)
    
    print(f"Combined analytics for {len(question_analytics_df)} questions.")

    # --- 学生分析 ---
    print("\n--- Starting Student Analytics ---")
    # 为学生分析器加载模型
    print("Loading pre-trained ORCDF model for student analytics...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(project_root, 'models', 'orcdf_best_model_seed42.pt')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Cannot perform student analysis.")
        return

    model = ORCDF(
        num_students=raw_data['num_students'],
        num_problems=raw_data['num_problems'],
        num_skills=raw_data['num_skills'],
        embed_dim=64,
        num_layers=2
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    student_analytics = StudentAnalytics(orcdf_data, model, device)
    student_analytics.set_graph_matrices(orcdf_data['a_matrix'], orcdf_data['ia_matrix'])
    
    behavior_df = student_analytics.calculate_behavioral_patterns()
    behavior_df.set_index('student_id', inplace=True)
    print(f"Behavioral patterns calculated for {len(behavior_df)} students.")

    # --- 结果整合与保存 ---
    print("\n--- Consolidating and Saving Analytics Data ---")
    
    # 转换为字典格式
    # fillna(None)是为了让NaN在json中变成null，而不是保留NaN
    question_analytics_dict = question_analytics_df.where(pd.notnull(question_analytics_df), None).to_dict(orient='index')
    student_analytics_dict = behavior_df.where(pd.notnull(behavior_df), None).to_dict(orient='index')

    final_data = {
        "questions": question_analytics_dict,
        "students": student_analytics_dict,
        "metadata": {
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "source_file": loader.data_file_path,
            "total_questions": len(question_analytics_dict),
            "total_students": len(student_analytics_dict),
        }
    }
    
    output_path = os.path.join(os.path.dirname(__file__), 'analytics_data.json')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved analytics data to '{output_path}'")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")

    print("\n--- Analytics Pipeline Finished ---")

if __name__ == '__main__':
    main() 