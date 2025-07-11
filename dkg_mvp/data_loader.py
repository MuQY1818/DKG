"""
数据加载模块 - 处理Skill Builder数据集
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import warnings

class DataLoader:
    """教育数据集加载器"""
    
    def __init__(self, base_path: str):
        """
        初始化数据加载器
        
        Args:
            base_path: 数据集所在的根目录
        """
        self.base_dir = base_path
        if not os.path.isdir(self.base_dir):
            raise FileNotFoundError(f"The specified base directory does not exist: {self.base_dir}")
        
        self.clear_dataset_path = os.path.join(self.base_dir, 'clear_dataset')
        self.processed_data_dir = os.path.join(self.base_dir, 'processed')
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def load_skill_builder_data(self, interaction_file: str = 'skill_builder_data_filter15.csv') -> Optional[Dict]:
        """
        从筛选后的Skill Builder日志数据构建DKG所需的数据结构。

        Args:
            interaction_file: 包含学生交互记录的文件名。
                              默认为 'skill_builder_data_filter15.csv'。
                              也可以使用 'log79021.csv' 或 'builder_top100.csv'。

        Returns:
            处理后的数据集字典，包含 'interactions', 'problems', 'skills', 
            和 'problem_skill_matrix'。
        """
        print(f"Loading filtered Skill Builder dataset from '{interaction_file}'...")
        
        # 定义文件路径
        interactions_path = os.path.join(self.clear_dataset_path, interaction_file)
        problems_path = os.path.join(self.clear_dataset_path, 'builder_problem.csv')
        skills_path = os.path.join(self.clear_dataset_path, 'builder_skill.csv')

        # 检查文件是否存在
        for path in [interactions_path, problems_path, skills_path]:
            if not os.path.exists(path):
                print(f"Error: Dataset file not found at {path}")
                return None

        try:
            # 加载交互、问题和技能数据
            interactions_df = pd.read_csv(interactions_path, encoding='latin1', low_memory=False)
            problems_df = pd.read_csv(problems_path, encoding='latin1')
            skills_df = pd.read_csv(skills_path, encoding='latin1')
        except Exception as e:
            print(f"Error reading dataset files: {e}")
            return None

        # --- 数据预处理 ---
        
        # 1. 清洗和验证
        interactions_df.dropna(subset=['user_id', 'problem_id', 'correct', 'skill_id'], inplace=True)
        interactions_df = interactions_df[interactions_df['skill_id'] != ''].copy()

        # 确保ID为整数类型
        for col in ['user_id', 'problem_id', 'correct', 'skill_id']:
            interactions_df[col] = pd.to_numeric(interactions_df[col], errors='coerce')
        interactions_df.dropna(subset=['user_id', 'problem_id', 'correct', 'skill_id'], inplace=True)
        interactions_df['skill_id'] = interactions_df['skill_id'].astype(int)

        # 2. 按时间排序
        if 'order_id' in interactions_df.columns:
            interactions_df.sort_values(by='order_id', inplace=True)
        
        # 3. 创建ID映射
        student_ids = sorted(interactions_df['user_id'].unique())
        problem_ids = sorted(problems_df['problem_id'].unique())
        skill_ids = sorted(skills_df['skill_id'].unique())
        
        student_map = {sid: i for i, sid in enumerate(student_ids)}
        problem_map = {pid: i for i, pid in enumerate(problem_ids)}
        skill_map = {skid: i for i, skid in enumerate(skill_ids)}

        # 应用映射
        interactions_df['student_id'] = interactions_df['user_id'].map(student_map)
        interactions_df['problem_idx'] = interactions_df['problem_id'].map(problem_map)
        interactions_df['skill_idx'] = interactions_df['skill_id'].map(skill_map)
        
        # 筛选出在映射中存在的交互
        interactions_df.dropna(subset=['student_id', 'problem_idx', 'skill_idx'], inplace=True)

        # 4. 构建 problem_skill_matrix
        problem_skill_relations = problems_df[['problem_id', 'skill_id']].copy()
        problem_skill_relations.dropna(inplace=True)
        
        num_problems = len(problem_ids)
        num_skills = len(skill_ids)
        
        problem_skill_matrix = pd.DataFrame(0, index=range(num_problems), columns=range(num_skills))

        for _, row in problem_skill_relations.iterrows():
            p_id = row['problem_id']
            s_id = row['skill_id']
            if p_id in problem_map and s_id in skill_map:
                p_idx = problem_map[p_id]
                s_idx = skill_map[s_id]
                problem_skill_matrix.iloc[p_idx, s_idx] = 1
        
        # 5. 创建技能字典 (skill_idx -> skill_name)
        skills = {
            skill_map[row['skill_id']]: row['skill_name']
            for _, row in skills_df.iterrows() if row['skill_id'] in skill_map
        }

        # 6. 准备最终的 interactions DataFrame
        final_interactions = interactions_df[[
            'student_id', 'problem_idx', 'correct', 'order_id', 
            'ms_first_response', 'hint_count', 'user_id', 'problem_id'
        ]].copy()

        # 重命名索引列以便区分
        final_interactions.rename(columns={
            'student_id': 'student_id_idx', 
            'problem_idx': 'problem_id_idx'
        }, inplace=True)
        
        # 保留原始ID列，因为DKGBuilder需要它们
        final_interactions['student_id'] = final_interactions['user_id']
        # 'problem_id' 列已经存在并且是正确的原始ID

        # 清理不再需要的user_id列
        final_interactions.drop(columns=['user_id'], inplace=True)

        print(f"Finished loading and processing. Found:")
        print(f"  - {len(student_ids)} students")
        print(f"  - {len(problem_ids)} problems")
        print(f"  - {len(skill_ids)} skills")
        print(f"  - {len(final_interactions)} interactions")
        
        return {
            'dataset_name': 'SkillBuilder_Filtered',
            'interactions': final_interactions,
            'problem_skill_matrix': problem_skill_matrix,
            'skills': skills,
            'num_students': len(student_ids),
            'num_problems': len(problem_ids),
            'num_skills': len(skill_ids),
            'problem_descriptions': problems_df.set_index('problem_id'),
            'skill_descriptions': skills_df.set_index('skill_id')
        }

def main():
    """用于测试数据加载器的示例"""
    print("Testing DataLoader...")
    
    # 假设项目根目录下有 "dataset" 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    dataset_path = os.path.join(project_root, 'dataset')

    loader = DataLoader(dataset_path)
    
    # 测试新的加载器
    skill_builder_data = loader.load_skill_builder_data()
    
    if skill_builder_data:
        print("\n--- Skill Builder Filtered Data Summary ---")
        print(f"Dataset Name: {skill_builder_data['dataset_name']}")
        print(f"Number of Students: {skill_builder_data['num_students']}")
        print(f"Number of Problems: {skill_builder_data['num_problems']}")
        print(f"Number of Skills: {skill_builder_data['num_skills']}")
        print("\nInteractions (first 5):")
        print(skill_builder_data['interactions'].head())
        print("\nProblem-Skill Matrix (sample):")
        print(skill_builder_data['problem_skill_matrix'].head())
        print("\nSkills (first 5):")
        print(list(skill_builder_data['skills'].items())[:5])
    else:
        print("\nFailed to load Skill Builder filtered data.")

if __name__ == '__main__':
    main()