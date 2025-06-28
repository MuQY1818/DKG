"""
数据加载模块 - 处理Math2015和Assistments数据集
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
        
        # 定义子数据集路径 (如果需要)
        self.frcsub_path = os.path.join(self.base_dir, 'math2015', 'FrcSub')
        self.math1_path = os.path.join(self.base_dir, 'math2015', 'Math1')
        self.math2_path = os.path.join(self.base_dir, 'math2015', 'Math2')
        self.assistments_path = os.path.join(self.base_dir, 'assistments-p(j)-data')
        self.processed_data_dir = os.path.join(self.base_dir, 'processed')
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def load_frcsub_dataset(self) -> Dict:
        """
        加载FrcSub数据集(分数减法)
        
        Returns:
            包含学生答题数据、题目-技能关系、技能名称等的字典
        """
        # 加载学生答题数据 (536学生 × 20题目)
        data_file = os.path.join(self.frcsub_path, 'data.txt')
        student_responses = pd.read_csv(data_file, sep='\t', header=None)
        
        # 加载题目-技能关系矩阵 (20题目 × 8技能)
        q_file = os.path.join(self.frcsub_path, 'q.txt') 
        problem_skill_matrix = pd.read_csv(q_file, sep='\t', header=None)
        
        # 加载技能名称
        qnames_file = os.path.join(self.frcsub_path, 'qnames.txt')
        with open(qnames_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        skills = {}
        for line in lines[1:]:  # 跳过表头
            if line.strip() and '\t' in line:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    skill_id = int(parts[0])
                    skill_name = parts[1].rstrip(',')
                    skills[skill_id] = skill_name
        
        # 加载题目描述
        problemdesc_file = os.path.join(self.frcsub_path, 'problemdesc.txt')
        problem_desc = pd.read_csv(problemdesc_file, sep='\t')
        
        return {
            'dataset_name': 'FrcSub',
            'student_responses': student_responses,  # 536×20矩阵
            'problem_skill_matrix': problem_skill_matrix,  # 20×8矩阵
            'skills': skills,  # 技能ID到名称的映射
            'problem_descriptions': problem_desc,
            'num_students': student_responses.shape[0],
            'num_problems': student_responses.shape[1], 
            'num_skills': len(skills)
        }
    
    def load_math_dataset(self, subset: str = 'Math1') -> Dict:
        """
        加载Math1或Math2数据集(高中数学)
        
        Args:
            subset: 'Math1' 或 'Math2'
            
        Returns:
            包含数学数据集信息的字典
        """
        math_path = self.math1_path if subset == 'Math1' else self.math2_path
        
        # 加载标准化分数数据
        data_file = os.path.join(math_path, 'data.txt')
        normalized_scores = pd.read_csv(data_file, sep='\t', header=None)
        
        # 加载原始分数数据
        raw_file = os.path.join(math_path, 'rawdata.txt')
        raw_scores = pd.read_csv(raw_file, sep='\t', header=None)
        
        # 加载题目-技能关系矩阵
        q_file = os.path.join(math_path, 'q.txt')
        problem_skill_matrix = pd.read_csv(q_file, sep='\t', header=None)
        
        # 加载技能名称
        qnames_file = os.path.join(math_path, 'qnames.txt')
        with open(qnames_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        skills = {}
        for line in lines[1:]:  # 跳过表头
            if line.strip() and '\t' in line:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    skill_id = int(parts[0])
                    skill_name = parts[1].rstrip(',')
                    skills[skill_id] = skill_name
        
        # 加载题目描述
        problemdesc_file = os.path.join(math_path, 'problemdesc.txt')
        problem_desc = pd.read_csv(problemdesc_file, sep='\t')
        
        return {
            'dataset_name': subset,
            'normalized_scores': normalized_scores,
            'raw_scores': raw_scores,
            'problem_skill_matrix': problem_skill_matrix,
            'skills': skills,
            'problem_descriptions': problem_desc,
            'num_students': normalized_scores.shape[0],
            'num_problems': normalized_scores.shape[1],
            'num_skills': len(skills)
        }
    
    def load_assistments_dataset(self) -> Dict:
        """
        加载Assistments P(J)数据集(学习行为时序数据)
        
        Returns:
            包含学习行为数据的字典
        """
        # 加载主要的行为数据
        actions_file = os.path.join(self.assistments_path, 'PJ Data (Actions).csv')
        actions_df = pd.read_csv(actions_file)
        
        # 加载特征和标签数据
        features_file = os.path.join(self.assistments_path, 'PJ Data (Features and Labels).csv')
        if os.path.exists(features_file):
            features_df = pd.read_csv(features_file)
        else:
            features_df = None
        
        return {
            'dataset_name': 'Assistments_PJ',
            'actions_data': actions_df,
            'features_data': features_df,
            'num_interactions': len(actions_df),
            'unique_students': actions_df['userId'].nunique() if 'userId' in actions_df.columns else 0,
            'unique_problems': actions_df['problemId'].nunique() if 'problemId' in actions_df.columns else 0,
            'unique_skills': actions_df['skill'].nunique() if 'skill' in actions_df.columns else 0
        }
    
    def load_assistments_log_data(self, dataset_name: str = 'skill_builder', nrows: Optional[int] = None) -> Optional[Dict]:
        """
        从ASSISTments 2009-2010日志数据构建DKG所需的数据结构
        
        Args:
            dataset_name: 数据集名称
            nrows: 限制读取的行数
            
        Returns:
            处理后的数据集字典
        """
        if dataset_name == 'skill_builder':
            file_path = os.path.join(self.base_dir, "skill_builder_data09-10.csv")
            print("Loading Skill Builder 2009-2010 dataset...")
        elif dataset_name == 'assistments':
            file_path = os.path.join(self.base_dir, "assistments_2009_2010.csv")
            print("Loading ASSISTments 2009-2010 dataset...")
        else:
            print(f"Unknown dataset name: {dataset_name}")
            return None

        if not os.path.exists(file_path):
            print(f"Error: Dataset file not found at {file_path}")
            return None

        try:
            # 使用nrows参数限制读取的行数，并跳过可能存在的注释行
            df = pd.read_csv(file_path, encoding='latin1', low_memory=False, nrows=nrows, skiprows=[1])
        except Exception as e:
            print(f"Error reading dataset file: {e}")
            return None
        
        # 2. 数据清洗和预处理
        df.dropna(subset=['skill_id', 'skill_name'], inplace=True)
        df = df[df['skill_id'] != '']
        
        # 将可能的多技能ID转换为单个（取第一个）
        df['skill_id'] = df['skill_id'].apply(
            lambda x: int(str(x).split(',')[0]) if isinstance(x, str) and ',' in x else int(x)
        )
        
        # 3. 按时间戳排序，并为每个(学生, 题目)组合保留最后一次尝试
        df.sort_values(by='order_id', inplace=True)
        
        # 4. 创建ID到索引的映射
        student_ids = sorted(df['user_id'].unique())
        problem_ids = sorted(df['problem_id'].unique())
        
        # 筛选出在问题和学生ID都在列表中的交互
        df = df[df['user_id'].isin(student_ids) & df['problem_id'].isin(problem_ids)]

        student_map = {sid: i for i, sid in enumerate(student_ids)}
        problem_map = {pid: i for i, pid in enumerate(problem_ids)}

        # 转换原始ID为索引ID
        df['student_id'] = df['user_id'].map(student_map)
        df['problem_id'] = df['problem_id'].map(problem_map)

        # 5. 保留每个(学生, 题目)的最后一次有效作答
        # 确保correct列与subjet/problem对齐
        interactions = df.sort_values(by='order_id').drop_duplicates(
            subset=['student_id', 'problem_id'], 
            keep='last'
        )

        # 6. 构建 problem_skill_matrix 矩阵
        skill_ids = sorted(df['skill_id'].unique())
        skill_map = {skid: i + 1 for i, skid in enumerate(skill_ids)} # 技能ID从1开始
        num_skills = len(skill_ids)

        # 转换skill_id
        df['skill_idx'] = df['skill_id'].map(skill_map)

        problem_skill_matrix = pd.DataFrame(0, index=range(len(problem_ids)), columns=range(1, num_skills + 1))
        
        problem_skill_relations = df[['problem_id', 'skill_idx']].dropna().drop_duplicates()
        
        for _, row in problem_skill_relations.iterrows():
            p_idx = int(row['problem_id'])
            s_idx = int(row['skill_idx'])
            if p_idx < len(problem_ids) and s_idx <= num_skills:
                problem_skill_matrix.iloc[p_idx, s_idx - 1] = 1 # 列索引是 s_idx - 1
                
        # 7. 创建技能字典 (skill_idx -> skill_name)
        skills_df = df[['skill_id', 'skill_name', 'skill_idx']].drop_duplicates('skill_idx')
        skills = {
            row['skill_idx']: row['skill_name']
            for _, row in skills_df.iterrows() if pd.notna(row['skill_idx'])
        }
        
        print(f"Finished loading and processing. Found:")
        print(f"  - {len(student_ids)} students")
        print(f"  - {len(problem_ids)} problems")
        print(f"  - {num_skills} skills")

        return {
            'dataset_name': dataset_name,
            'num_students': len(student_ids),
            'num_problems': len(problem_ids),
            'num_skills': num_skills,
            'interactions': interactions, # 直接传递交互日志
            'problem_skill_matrix': problem_skill_matrix,
            'skills': skills,
        }
    
    def get_student_subset(self, data: Dict, student_ids: List[int]) -> Dict:
        """
        从完整数据集中提取一部分学生的数据子集。
        这个方法只筛选与学生直接相关的交互数据，保持知识空间（题目、技能）的完整性。

        Args:
            data: 完整的数据集字典
            student_ids: 需要提取的学生ID（索引）列表

        Returns:
            提取后的数据子集
        """
        subset_data = data.copy()
        
        # 1. 筛选核心交互数据
        if 'interactions' in data:
            interactions = data['interactions']
            subset_interactions = interactions[interactions['student_id'].isin(student_ids)].copy()
            subset_data['interactions'] = subset_interactions
            
            # 2. 更新学生数量
            subset_data['num_students'] = len(student_ids)
            
            # 3. 其他所有元数据 (num_problems, num_skills, problem_skill_matrix, skills) 保持不变
            #    因为它们定义了整个知识空间，不应随学生子集的变化而变化。
            
        elif 'student_responses' in data:
            # 对于旧的矩阵格式，同样只筛选学生相关的行
            subset_data['student_responses'] = data['student_responses'].iloc[student_ids]
            subset_data['num_students'] = len(student_ids)
        
        return subset_data
    
    def split_student_data_for_simulation(self, data: Dict, student_id: int, train_ratio: float = 0.7) -> Tuple[Dict, Dict]:
        """
        为单个学生的模拟实验分割数据。

        Args:
            data: 完整的数据集字典。
            student_id: 要进行模拟的学生ID（索引）。
            train_ratio: 用于构建初始状态的训练数据比例。

        Returns:
            一个元组 (initial_state_data, oracle_map):
            - initial_state_data: 用于构建初始DKG的数据集。
            - oracle_map: 该学生未来的交互记录字典 {problem_id: correct}。
        """
        if 'interactions' not in data:
            raise ValueError("此功能仅支持基于'interactions'的日志数据集。")

        all_interactions = data['interactions']
        
        # 1. 分离出目标学生和其他学生的数据
        student_interactions = all_interactions[all_interactions['student_id'] == student_id].copy()
        other_students_interactions = all_interactions[all_interactions['student_id'] != student_id].copy()
        
        # 2. 按时间排序并分割目标学生的数据
        student_interactions.sort_values(by='order_id', inplace=True)
        split_point = int(len(student_interactions) * train_ratio)
        train_interactions = student_interactions.iloc[:split_point]
        future_interactions_oracle = student_interactions.iloc[split_point:]
        
        # 3. 构建用于初始化的数据集
        initial_interactions = pd.concat([other_students_interactions, train_interactions])
        
        initial_state_data = data.copy()
        initial_state_data['interactions'] = initial_interactions
        
        # 将oracle中的problem_id和correct值做成一个字典，方便查询
        oracle_map = pd.Series(future_interactions_oracle.correct.values, index=future_interactions_oracle.problem_id).to_dict()

        print(f"为学生 {student_id} 分割数据:")
        print(f"  - 初始状态交互数: {len(train_interactions)}")
        print(f"  - 未来行为(Oracle)交互数: {len(future_interactions_oracle)}")
        
        return initial_state_data, oracle_map
    
    def compute_skill_mastery(self, data: Dict, threshold: float = 0.7) -> pd.DataFrame:
        """
        计算数据集中每个学生在所有技能上的掌握程度（正确率）。
        
        Args:
            data: 数据集字典
            threshold: 掌握阈值
            
        Returns:
            学生×技能的掌握程度矩阵
        """
        if 'student_responses' in data:
            # 二值化数据 (FrcSub)
            responses = data['student_responses'].values
            problem_skill = data['problem_skill_matrix'].values
            
            # 计算每个学生在每个技能上的表现
            skill_mastery = []
            for student_idx in range(responses.shape[0]):
                student_skills = []
                for skill_idx in range(problem_skill.shape[1]):
                    # 找到需要该技能的题目
                    skill_problems = np.where(problem_skill[:, skill_idx] == 1)[0]
                    if len(skill_problems) > 0:
                        # 计算该技能相关题目的正确率
                        correct_rate = responses[student_idx, skill_problems].mean()
                        student_skills.append(correct_rate)
                    else:
                        student_skills.append(0.0)
                skill_mastery.append(student_skills)
            
            return pd.DataFrame(skill_mastery, 
                              columns=[f'Skill_{i+1}' for i in range(len(student_skills))])
        
        elif 'normalized_scores' in data:
            # 连续分数数据 (Math1/Math2)
            scores = data['normalized_scores'].values
            problem_skill = data['problem_skill_matrix'].values
            
            skill_mastery = []
            for student_idx in range(scores.shape[0]):
                student_skills = []
                for skill_idx in range(problem_skill.shape[1]):
                    skill_problems = np.where(problem_skill[:, skill_idx] == 1)[0]
                    if len(skill_problems) > 0:
                        # 计算该技能相关题目的平均得分
                        avg_score = scores[student_idx, skill_problems].mean()
                        student_skills.append(avg_score)
                    else:
                        student_skills.append(0.0)
                skill_mastery.append(student_skills)
            
            return pd.DataFrame(skill_mastery,
                              columns=[f'Skill_{i+1}' for i in range(len(student_skills))])
        
        else:
            raise ValueError("不支持的数据格式")

def main():
    """测试数据加载功能"""
    # 初始化数据加载器
    loader = DataLoader('../dataset')
    
    print("=== 加载FrcSub数据集 ===")
    frcsub_data = loader.load_frcsub_dataset()
    print(f"数据集: {frcsub_data['dataset_name']}")
    print(f"学生数: {frcsub_data['num_students']}")
    print(f"题目数: {frcsub_data['num_problems']}")
    print(f"技能数: {frcsub_data['num_skills']}")
    print(f"技能列表: {list(frcsub_data['skills'].values())[:3]}...")
    
    # 计算技能掌握度
    skill_mastery = loader.compute_skill_mastery(frcsub_data)
    print(f"技能掌握度矩阵形状: {skill_mastery.shape}")
    print(f"前3个学生的技能掌握度:\n{skill_mastery.head(3)}")
    
    print("\n=== 加载Math1数据集 ===")
    math1_data = loader.load_math_dataset('Math1')
    print(f"数据集: {math1_data['dataset_name']}")
    print(f"学生数: {math1_data['num_students']}")
    print(f"题目数: {math1_data['num_problems']}")
    print(f"技能数: {math1_data['num_skills']}")
    
    print("\n=== 加载Assistments数据集 ===")
    try:
        assistments_data = loader.load_assistments_dataset()
        print(f"数据集: {assistments_data['dataset_name']}")
        print(f"交互记录数: {assistments_data['num_interactions']}")
        print(f"学生数: {assistments_data['unique_students']}")
        print(f"题目数: {assistments_data['unique_problems']}")
        print(f"技能数: {assistments_data['unique_skills']}")
    except Exception as e:
        print(f"加载Assistments数据集失败: {e}")

if __name__ == "__main__":
    main()