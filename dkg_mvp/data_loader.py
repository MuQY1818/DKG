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
        self.data_file_path = None # 将在这里存储被加载的数据文件名
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def load_skill_builder_data(self, interaction_file: str = 'skill_builder_data_filter15.csv', nrows: Optional[int] = None) -> Optional[Dict]:
        """
        从筛选后的Skill Builder日志数据构建DKG所需的数据结构。

        Args:
            interaction_file: 包含学生交互记录的文件名。
                              默认为 'skill_builder_data_filter15.csv'。
                              也可以使用 'log79021.csv' 或 'builder_top100.csv'。
            nrows: 要读取的交互记录行数，用于快速测试。

        Returns:
            处理后的数据集字典，包含 'interactions', 'problems', 'skills', 
            和 'problem_skill_matrix'。
        """
        print(f"Loading filtered Skill Builder dataset from '{interaction_file}'...")
        
        # 定义文件路径
        interactions_path = os.path.join(self.clear_dataset_path, interaction_file)
        self.data_file_path = interactions_path # 存储文件路径
        problems_path = os.path.join(self.clear_dataset_path, 'builder_problem.csv')
        skills_path = os.path.join(self.clear_dataset_path, 'builder_skill.csv')

        # 检查文件是否存在
        for path in [interactions_path, problems_path, skills_path]:
            if not os.path.exists(path):
                print(f"Error: Dataset file not found at {path}")
                return None

        try:
            # 加载交互、问题和技能数据
            interactions_df = pd.read_csv(interactions_path, encoding='latin1', low_memory=False, nrows=nrows)
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

        # 终极修复：确保 problem_skill_matrix 的列名是整数类型，以匹配后续操作
        problem_skill_matrix.columns = problem_skill_matrix.columns.astype(int)

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
            'skill_descriptions': skills_df.set_index('skill_id'),
            'student_map': student_map,
            'problem_map': problem_map,
            'skill_map': skill_map
        }

    def load_orcdf_data(self, **kwargs) -> Optional[Dict]:
        """
        加载并格式化数据以适应ORCDF模型的需求。

        该方法会复用 load_skill_builder_data 的结果，并将其处理成
        适合构建图和进行GNN训练的格式。

        Args:
            **kwargs: 传递给 load_skill_builder_data 的参数,
                      例如 nrows 用于调试。

        Returns:
            一个为ORCDF准备的数据字典，包含:
            - num_students, num_problems, num_skills: 实体数量
            - interactions: [(student_idx, problem_idx, correct), ...]
            - problem_skill_relations: [(problem_idx, skill_idx), ...]
            - a_matrix: 邻接矩阵 (correct subgraph)
            - ia_matrix: 邻接矩阵 (incorrect subgraph)
            - q_matrix: 练习-技能邻接矩阵
            - entity maps
        """
        # 1. 使用现有方法加载基础数据
        base_data = self.load_skill_builder_data(**kwargs)
        if not base_data:
            return None
        
        print("Formatting data for ORCDF model...")

        # 2. 提取所需数据
        interactions_df = base_data['interactions']
        ps_matrix = base_data['problem_skill_matrix']
        num_students = base_data['num_students']
        num_problems = base_data['num_problems']
        num_skills = base_data['num_skills']

        # 3. 格式化交互数据
        # 确保列名正确
        interactions_list = list(interactions_df[[
            'student_id_idx', 'problem_id_idx', 'correct'
        ]].itertuples(index=False, name=None))

        # 4. 格式化练习-技能关系
        ps_relations = []
        for p_idx in ps_matrix.index:
            for s_idx in ps_matrix.columns:
                if ps_matrix.loc[p_idx, s_idx] == 1:
                    ps_relations.append((p_idx, s_idx))
        
        # 5. 构建邻接矩阵 (Adjacency Matrices)
        # ResG (Response Graph) 的两个组成部分
        # a_matrix: 正确作答子图 (Correct Subgraph)
        # ia_matrix: 错误作答子图 (Incorrect Subgraph)
        a_matrix = np.zeros((num_students, num_problems))
        ia_matrix = np.zeros((num_students, num_problems))
        
        for s_idx, p_idx, correct in interactions_list:
            if correct == 1:
                a_matrix[s_idx, p_idx] = 1
            else:
                ia_matrix[s_idx, p_idx] = 1

        # q_matrix: 练习-技能关系矩阵 (numpy array)
        q_matrix = ps_matrix.to_numpy()

        print("Finished formatting.")
        
        return {
            'num_students': num_students,
            'num_problems': num_problems,
            'num_skills': num_skills,
            'interactions': interactions_list,
            'problem_skill_relations': ps_relations,
            'a_matrix': a_matrix,
            'ia_matrix': ia_matrix,
            'q_matrix': q_matrix,
            'student_map': base_data['student_map'],
            'problem_map': base_data['problem_map'],
            'skill_map': base_data['skill_map'],
            'skills': base_data['skills'],
            'problem_descriptions': base_data['problem_descriptions']
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

    # 测试为ORCDF准备的数据
    print("\n" + "="*50)
    print("Testing ORCDF data loader...")
    orcdf_data = loader.load_orcdf_data(nrows=1000) # 使用少量数据进行测试
    if orcdf_data:
        print("\n--- ORCDF Data Summary ---")
        print(f"Num Students: {orcdf_data['num_students']}")
        print(f"Num Problems: {orcdf_data['num_problems']}")
        print(f"Num Skills: {orcdf_data['num_skills']}")
        print(f"Num Interactions: {len(orcdf_data['interactions'])}")
        print(f"Num Problem-Skill Relations: {len(orcdf_data['problem_skill_relations'])}")
        print(f"Shape of A matrix (correct): {orcdf_data['a_matrix'].shape}")
        print(f"Shape of IA matrix (incorrect): {orcdf_data['ia_matrix'].shape}")
        print(f"Shape of Q matrix (problem-skill): {orcdf_data['q_matrix'].shape}")
        print("\nSample interaction:", orcdf_data['interactions'][0])
        print("Sample relation:", orcdf_data['problem_skill_relations'][0])
    else:
        print("\nFailed to load ORCDF data.")


if __name__ == '__main__':
    main()