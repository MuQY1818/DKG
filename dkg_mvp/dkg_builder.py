"""
DKG构建模块 - 构建动态知识图谱
"""
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import os
import pickle # 新增导入

class DKGBuilder:
    """动态知识图谱构建器"""
    
    def __init__(self):
        """初始化DKG构建器"""
        self.graph = nx.MultiDiGraph()
        self.node_types = {'student', 'problem', 'skill', 'session'}
        self.relation_types = {'solve', 'require', 'master', 'prerequisite', 'similar'}
    
    def build_from_data(self, data: Dict) -> nx.MultiDiGraph:
        """
        从教育数据构建动态知识图谱
        
        Args:
            data: 数据集字典(来自DataLoader)
            
        Returns:
            构建完成的知识图谱
        """
        self.graph.clear()
        
        # 1. 创建基础节点
        self._create_nodes(data)
        
        # 2. 创建基础关系
        self._create_basic_relations(data)
        
        # 3. 推导隐含关系
        self._infer_skill_prerequisites(data)
        self._compute_skill_similarities(data)
        
        # 4. 添加图谱元信息 (从图中直接统计)
        self.graph.graph['dataset_name'] = data.get('dataset_name', 'Unknown')
        self.graph.graph['num_students'] = len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'student'])
        self.graph.graph['num_problems'] = len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'problem'])
        self.graph.graph['num_skills'] = len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'skill'])
        
        return self.graph
    
    def save_graph(self, file_path: str):
        """
        将图谱保存到文件
        
        Args:
            file_path: 文件路径 (推荐使用.graphml)
        """
        # GraphML不支持None值, 所以在保存前将其转换为空字符串
        graph_copy = self.graph.copy()
        for node, data in graph_copy.nodes(data=True):
            for key, value in data.items():
                if value is None:
                    data[key] = ''
        
        for u, v, key, data in graph_copy.edges(keys=True, data=True):
            for attr_key, value in data.items():
                if value is None:
                    data[attr_key] = ''

        try:
            nx.write_graphml(graph_copy, file_path)
            print(f"DKG successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving DKG to {file_path}: {e}")

    def save_with_pickle(self, file_path: str):
        """
        使用 pickle 将整个 DKGBuilder 实例保存到文件。
        这比 GraphML 更快且更可靠。

        Args:
            file_path: 文件路径 (推荐使用.pkl)
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"DKG instance successfully pickled to {file_path}")
        except Exception as e:
            print(f"Error pickling DKG instance to {file_path}: {e}")

    @classmethod
    def load_graph(cls, file_path: str) -> 'DKGBuilder':
        """
        从文件加载图谱
        
        Args:
            file_path: 文件路径
            
        Returns:
            一个包含已加载图谱的DKGBuilder实例
        """
        builder = cls()
        try:
            builder.graph = nx.read_graphml(file_path)
            # Since node attributes like skill_id are stored as strings after saving,
            # we need to convert them back to their original types for consistency.
            for node, data in builder.graph.nodes(data=True):
                for key, value in data.items():
                    # Attempt to convert numeric-like strings back to numbers
                    if isinstance(value, str) and value.isdigit():
                        data[key] = int(value)
                    elif isinstance(value, str):
                        try:
                            # Attempt to convert float strings
                            data[key] = float(value)
                        except (ValueError, TypeError):
                            pass # Keep as string if conversion fails
            print(f"DKG successfully loaded from {file_path}")
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            builder.graph = nx.MultiDiGraph() # Return with an empty graph
        except Exception as e:
            print(f"Error loading DKG from {file_path}: {e}")
            builder.graph = nx.MultiDiGraph() # Return with an empty graph
        return builder

    @classmethod
    def load_with_pickle(cls, file_path: str) -> Optional['DKGBuilder']:
        """
        使用 pickle 从文件加载 DKGBuilder 实例。

        Args:
            file_path: 文件路径
            
        Returns:
            一个包含已加载图谱的DKGBuilder实例，如果失败则返回None。
        """
        try:
            with open(file_path, 'rb') as f:
                builder = pickle.load(f)
            print(f"DKG instance successfully unpickled from {file_path}")
            return builder
        except FileNotFoundError:
            print(f"Error: The pickle file {file_path} was not found.")
        except Exception as e:
            print(f"Error unpickling DKG from {file_path}: {e}")
        return None

    def _create_nodes(self, data: Dict):
        """创建图谱节点"""
        interactions = data.get('interactions')
        problems_df = data.get('problem_descriptions')
        skills_df = data.get('skill_descriptions')

        if interactions is None or problems_df is None or skills_df is None:
            print("Warning: Missing interactions, problems, or skills data. Cannot create all nodes.")
            return

        # 创建学生节点
        student_ids = interactions['student_id'].unique()
        for student_id in student_ids:
            self.graph.add_node(
                f"student_{student_id}",
                type='student',
                student_id=int(student_id),
                learning_rate=np.random.normal(1.0, 0.1),
                perseverance=np.random.randint(3, 7),
                curiosity=np.random.uniform(0.05, 0.2),
                ability_vector=None,
                learning_style=None,
                progress_rate=0.0
            )
        
        # 创建题目节点 (从 problems_df 创建所有题目)
        for problem_id in problems_df.index:
            self.graph.add_node(
                f"problem_{problem_id}",
                type='problem',
                problem_id=int(problem_id),
                difficulty=problems_df.loc[problem_id].get('initDifficulty', 0.5)
            )
        
        # 创建技能节点 (从 skills_df 创建所有技能)
        for skill_id, skill_row in skills_df.iterrows():
            self.graph.add_node(
                f"skill_{skill_id}",
                type='skill',
                skill_id=int(skill_id),
                skill_name=skill_row.get('skill_name', ''),
                difficulty_level=skill_row.get('initDifficulty', 0.5)
            )
    
    def _create_basic_relations(self, data: Dict):
        """创建基础关系"""
        # 创建SOLVE关系(学生-题目)
        self._create_solve_relations(data)
        
        # 创建REQUIRE关系(题目-技能)
        self._create_require_relations(data)
        
        # 创建MASTER关系(学生-技能)
        self._create_master_relations(data)
    
    def _create_solve_relations(self, data: Dict):
        """创建学生解题关系"""
        if 'interactions' in data:
            # 日志型数据
            interactions = data['interactions']
            for _, row in interactions.iterrows():
                # 使用原始ID来创建图中的节点关系
                student_id = row['student_id']
                problem_id = row['problem_id']
                correct = int(row['correct'])

                self.graph.add_edge(
                    f"student_{student_id}",
                    f"problem_{problem_id}",
                    type='solve',
                    correct=correct,
                    score=correct,  # 在这个数据集中，分数等于正确性
                    attempts=1,
                    time_taken=row.get('ms_first_response'),
                    hints_used=row.get('hint_count', 0)
                )

        elif 'student_responses' in data:
            # 二值化数据 (FrcSub)
            responses = data['student_responses'].values
            for student_idx in range(responses.shape[0]):
                for problem_idx in range(responses.shape[1]):
                    correct = int(responses[student_idx, problem_idx])
                    
                    self.graph.add_edge(
                        f"student_{student_idx}",
                        f"problem_{problem_idx}",
                        type='solve',
                        correct=correct,
                        score=correct,  # 二值情况下分数等于正确性
                        attempts=1,
                        time_taken=None,
                        hints_used=0
                    )
        
        elif 'normalized_scores' in data:
            # 连续分数数据 (Math1/Math2)
            scores = data['normalized_scores'].values
            for student_idx in range(scores.shape[0]):
                for problem_idx in range(scores.shape[1]):
                    score = float(scores[student_idx, problem_idx])
                    correct = 1 if score > 0.6 else 0  # 阈值判断
                    
                    self.graph.add_edge(
                        f"student_{student_idx}",
                        f"problem_{problem_idx}",
                        type='solve',
                        correct=correct,
                        score=score,
                        attempts=1,
                        time_taken=None,
                        hints_used=0
                    )
    
    def _create_require_relations(self, data: Dict):
        """创建题目-技能需求关系"""
        problems_df = data.get('problem_descriptions')
        if problems_df is None or 'skill_id' not in problems_df.columns:
            # 如果使用旧的数据格式，则回退到矩阵方法
            problem_skill_matrix = data.get('problem_skill_matrix')
            if problem_skill_matrix is not None:
                matrix = problem_skill_matrix.values
                skill_ids = sorted(data.get('skills', {}).keys()) # 获取排序后的技能索引

                for p_idx in range(matrix.shape[0]):
                    problem_id = problem_skill_matrix.index[p_idx] # 假设索引是problem_id
                    for s_idx_in_matrix in range(matrix.shape[1]):
                        if matrix[p_idx, s_idx_in_matrix] == 1:
                            skill_id = skill_ids[s_idx_in_matrix]
                            self.graph.add_edge(
                                f"problem_{problem_id}",
                                f"skill_{skill_id}",
                                type='require',
                                necessity_level='required'
                            )
            return

        # 首选方法：直接从DataFrame创建关系
        # 重置索引，以便我们可以迭代行
        if not isinstance(problems_df.index, pd.RangeIndex):
            problems_df = problems_df.reset_index()

        for _, row in problems_df.iterrows():
            if pd.notna(row['problem_id']) and pd.notna(row['skill_id']):
                problem_id = int(row['problem_id'])
                skill_id = int(row['skill_id'])
                
                # 确保节点存在于图中
                problem_node = f"problem_{problem_id}"
                skill_node = f"skill_{skill_id}"
                
                if self.graph.has_node(problem_node) and self.graph.has_node(skill_node):
                    self.graph.add_edge(
                        problem_node,
                        skill_node,
                        type='require',
                        necessity_level='required'
                    )

    def _create_master_relations(self, data: Dict):
        """创建学生-技能掌握关系"""
        skill_mastery = self._compute_skill_mastery_direct(data)
        
        # 修正：不再需要 student_map 或 inv_student_map 来查找 student_id，
        # 因为 skill_mastery 的索引已经是原始的 student_id。
        skill_map = data.get('skill_map', {})
        if not skill_map:
            print("Warning: skill_map not found in data. Cannot create 'master' relations.")
            return
            
        inv_skill_map = {v: k for k, v in skill_map.items()}

        # 修正：iterrows() 的第一个元素 student_id 本身就是原始ID，不再是索引
        for student_id, row in skill_mastery.iterrows():
            # 这里的 student_id 就是我们需要的原始ID，例如 14
            for skill_idx, mastery_level in row.items():
                # 终极修正：从DataFrame列名中获取的 skill_idx 是字符串，必须转为整数才能查询map
                skill_id = inv_skill_map.get(int(skill_idx))
                if skill_id is None:
                    continue
                
                # mastery_level 可能是 NaN，如果是则跳过
                if pd.isna(mastery_level):
                    continue

                confidence = min(mastery_level * 1.2, 1.0)

                self.graph.add_edge(
                    f"student_{student_id}",
                    f"skill_{skill_id}",
                    type='master',
                    mastery_level=float(mastery_level),
                    confidence=confidence,
                    last_updated=None,
                    attempts_count=0 
                )
    
    def _infer_skill_prerequisites(self, data: Dict):
        """推断技能先修关系"""
        skills = data.get('skills', {})
        problem_skill_matrix = data.get('problem_skill_matrix')
        
        if problem_skill_matrix is None or len(skills) == 0:
            return
        
        matrix = problem_skill_matrix.values
        
        # 基于题目共现模式推断先修关系
        skill_cooccurrence = np.dot(matrix.T, matrix)  # 技能×技能共现矩阵
        
        for i in range(len(skills)):
            for j in range(len(skills)):
                if i != j:
                    # 推断逻辑：如果技能j的出现很大程度上意味着技能i也会出现，
                    # 但技能i的出现不一定意味着j会出现，则i可能是j的先修技能。
                    # 例如: i=一元一次方程, j=二元一次方程组
                    # 几乎所有二元一次方程组的题目都包含一元一次方程的知识, P(i|j)很高。
                    # 但反过来，P(j|i)则较低。
                    cooc_ij = skill_cooccurrence[i, j]
                    cooc_ii = skill_cooccurrence[i, i]  # 技能i出现的总次数
                    cooc_jj = skill_cooccurrence[j, j]  # 技能j出现的总次数
                    
                    if cooc_ii > 0 and cooc_jj > 0:
                        # 计算条件概率 P(j|i) 和 P(i|j)
                        prob_j_given_i = cooc_ij / cooc_ii
                        prob_i_given_j = cooc_ij / cooc_jj
                        
                        # 核心判断：如果P(i|j)远大于P(j|i)且P(i|j)足够大，则i是j的先修
                        if prob_i_given_j > prob_j_given_i and prob_i_given_j > 0.7:
                            dependency_strength = prob_i_given_j - prob_j_given_i
                            
                            self.graph.add_edge(
                                f"skill_{i + 1}",
                                f"skill_{j + 1}",
                                type='prerequisite',
                                dependency_strength=dependency_strength,
                                learning_order=1,
                                confidence=prob_i_given_j
                            )
    
    def _compute_skill_similarities(self, data: Dict):
        """计算技能相似性关系"""
        problem_skill_matrix = data.get('problem_skill_matrix')
        
        if problem_skill_matrix is None:
            return
        
        matrix = problem_skill_matrix.values.T  # 技能×题目矩阵
        
        # 计算技能间的余弦相似度
        similarities = cosine_similarity(matrix)
        
        num_skills = similarities.shape[0]
        for i in range(num_skills):
            for j in range(i + 1, num_skills):
                similarity = similarities[i, j]
                
                # 只保留相似度较高的关系
                if similarity > 0.3:
                    # 计算共现频率
                    cooccurrence = np.sum(matrix[i] * matrix[j])
                    
                    self.graph.add_edge(
                        f"skill_{i + 1}",
                        f"skill_{j + 1}",
                        type='similar',
                        similarity_score=similarity,
                        cooccurrence_frequency=int(cooccurrence),
                        relation_type='bidirectional'
                    )
    
    def record_interaction(self, interaction: Dict):
        """
        记录一次学习交互，并更新图谱状态
        这是更新DKG最核心的入口
        
        Args:
            interaction: 包含交互详情的字典
                         e.g., {'student_id': int, 'problem_id': int, 'correct': int, 
                                 'score': float, 'timestamp': str}
        """
        # This method replaces the old update_student_state
        student_id = interaction.get('student_id')
        student_node = f"student_{student_id}"

        if student_node not in self.graph:
            # Handle case where student does not exist. 
            # For now, we'll just print a warning.
            print(f"Warning: Student node {student_node} not found. Skipping interaction.")
            return
        
        # 1. 更新或创建SOLVE关系
        problem_id = interaction.get('problem_id')
        if problem_id is not None:
            problem_node = f"problem_{problem_id}"
            
            # 更新或创建solve关系
            edge_key_to_update = None
            if self.graph.has_edge(student_node, problem_node):
                for key, edge_attr in self.graph[student_node][problem_node].items():
                    if edge_attr.get('type') == 'solve':
                        edge_key_to_update = key
                        break
            
            if edge_key_to_update is not None:
                # 更新现有关系
                edge_attr = self.graph[student_node][problem_node][edge_key_to_update]
                edge_attr.update({
                    'correct': interaction.get('correct', edge_attr.get('correct')),
                    'score': interaction.get('score', edge_attr.get('score')),
                    'attempts': edge_attr.get('attempts', 0) + 1,
                    'time_taken': interaction.get('time_taken'),
                    'hints_used': interaction.get('hints_used', 0)
                })
            else:
                # 创建新关系
                self.graph.add_edge(
                    student_node,
                    problem_node,
                    type='solve',
                    correct=interaction.get('correct', 0),
                    score=interaction.get('score', 0.0),
                    attempts=1,
                    time_taken=interaction.get('time_taken'),
                    hints_used=interaction.get('hints_used', 0),
                    timestamp=interaction.get('timestamp')
                )
        
        # 2. 更新相关的MASTER关系 (技能掌握度)
        self._update_skill_mastery(student_id, interaction)

    def record_interactions_batch(self, interactions: List[Dict]):
        """批量处理学习交互记录"""
        for interaction in interactions:
            self.record_interaction(interaction)

    def update_skill_mastery_manual(self, student_id: int, skill_id: int, mastery_level: float):
        """手动干预，直接修改某个学生对某个技能的掌握度"""
        student_node = f"student_{student_id}"
        skill_node = f"skill_{skill_id}"
        if self.graph.has_edge(student_node, skill_node):
            for key, attr in self.graph[student_node][skill_node].items():
                if attr['type'] == 'master':
                    attr['mastery_level'] = mastery_level
                    print(f"Manually updated mastery for {student_node} on {skill_node} to {mastery_level}")
                    return
        print(f"Warning: No 'master' edge found between {student_node} and {skill_node} to update.")

    def get_student_profile(self, student_id: int, num_recent: int = 5) -> Dict:
        """获取一个学生的完整画像"""
        student_node = f"student_{student_id}"
        if student_node not in self.graph:
            return {"error": "Student not found"}

        # 1. Inferred traits
        traits = {k: v for k, v in self.graph.nodes[student_node].items() if k not in ['type', 'student_id']}
        
        # 2. Knowledge Summary
        knowledge_state = self.get_student_knowledge_state(student_id)
        skills_by_mastery = sorted(knowledge_state.values(), key=lambda x: x['mastery_level'])
        
        strongest = [{'skill_name': s['skill_name'], 'mastery': s['mastery_level']} for s in skills_by_mastery[-5:]]
        weakest = [{'skill_name': s['skill_name'], 'mastery': s['mastery_level']} for s in skills_by_mastery[:5]]
        
        # 3. Recent Activity
        solve_edges = []
        for u, v, attr in self.graph.out_edges(student_node, data=True):
            if attr.get('type') == 'solve' and 'timestamp' in attr:
                solve_edges.append(attr)
        
        # Sort by timestamp if available
        solve_edges.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        recent_activity = solve_edges[:num_recent]

        return {
            "student_id": student_id,
            "inferred_traits": traits,
            "knowledge_summary": {
                "strongest_skills": strongest[::-1],
                "weakest_skills": weakest,
            },
            "recent_activity": recent_activity
        }

    def get_skill_details(self, skill_id: int) -> Dict:
        """获取特定技能的详细信息"""
        skill_node = f"skill_{skill_id}"
        if skill_node not in self.graph:
            return {"error": "Skill not found"}

        details = dict(self.graph.nodes[skill_node])
        
        # Find relationships
        details['prerequisites'] = [
            self.graph.nodes[u]['skill_name'] for u, v, attr in self.graph.in_edges(skill_node, data=True) 
            if attr['type'] == 'prerequisite'
        ]
        details['postrequisites'] = [
            self.graph.nodes[v]['skill_name'] for u, v, attr in self.graph.out_edges(skill_node, data=True) 
            if attr['type'] == 'prerequisite'
        ]
        details['related_problems'] = [
            self.graph.nodes[u]['problem_id'] for u, v, attr in self.graph.in_edges(skill_node, data=True) 
            if attr['type'] == 'require'
        ]
        
        return details

    def get_problem_details(self, problem_id: int) -> Dict:
        """获取特定题目的详细信息"""
        problem_node = f"problem_{problem_id}"
        if problem_node not in self.graph:
            return {"error": "Problem not found"}
            
        details = dict(self.graph.nodes[problem_node])
        details['required_skills'] = [
            self.graph.nodes[v]['skill_name'] for u, v, attr in self.graph.out_edges(problem_node, data=True)
            if attr['type'] == 'require'
        ]
        return details
    
    def update_student_state(self, student_id: int, new_interaction: Dict):
        """
        基于新的学习交互更新学生状态
        
        DEPRECATED: This method is now replaced by the more specific `record_interaction`.
        It is kept for backward compatibility for now but will be removed.
        Please use record_interaction(interaction) instead.
        """
        print("Warning: update_student_state is deprecated. Use record_interaction instead.")
        self.record_interaction(new_interaction)
        
    def _update_skill_mastery(self, student_id: int, interaction: Dict):
        """更新学生技能掌握关系，并向下传播强化效果。"""
        student_node = f"student_{student_id}"
        problem_id = interaction.get('problem_id')
        if problem_id is None:
            return
        
        problem_node = f"problem_{problem_id}"
        related_skills = [edge[1] for edge in self.graph.out_edges(problem_node, data=True) if edge[2].get('type') == 'require']
        
        for skill_node in related_skills:
            if self.graph.has_edge(student_node, skill_node):
                edge_data = self.graph[student_node][skill_node]
                for key, edge_attr in edge_data.items():
                    if edge_attr.get('type') == 'master':
                        correct = interaction.get('correct', 0)
                        is_epiphany = interaction.get('is_epiphany', False)
                        learning_rate = self.graph.nodes[student_node].get('learning_rate', 1.0)
                        
                        # 计算本次交互带来的基础变化量 (受学习效率影响)
                        base_change = (0.1 * learning_rate) if correct else -0.02
                        epiphany_bonus = 0.25 if is_epiphany else 0
                        total_change = base_change + epiphany_bonus

                        # 更新主技能
                        current_mastery = edge_attr.get('mastery_level', 0.5)
                        new_mastery = max(0.0, min(current_mastery + total_change, 1.0))
                        edge_attr['mastery_level'] = new_mastery
                        edge_attr['confidence'] = min(new_mastery * 1.1, 1.0)
                        edge_attr['attempts_count'] = edge_attr.get('attempts_count', 0) + 1
                        
                        if is_epiphany:
                            print(f"✨ 触发顿悟! 技能 '{self.graph.nodes[skill_node].get('skill_name')}' 掌握度大幅提升!")

                        # 递归地传播强化效果到先决条件
                        self._propagate_reinforcement(student_node, skill_node, total_change)

    def _propagate_reinforcement(self, student_node: str, skill_node: str, change_amount: float, decay_factor: float = 0.4):
        """递归地将掌握度变化传播给先决条件技能。"""
        # 找到所有直接的先决条件
        prerequisites = [u for u, v, attr in self.graph.in_edges(skill_node, data=True) if attr.get('type') == 'prerequisite']
        
        for prereq_node in prerequisites:
            # 强化效果按衰减因子递减
            reinforcement_change = change_amount * decay_factor
            
            if self.graph.has_edge(student_node, prereq_node):
                edge_data = self.graph[student_node][prereq_node]
                for key, edge_attr in edge_data.items():
                    if edge_attr.get('type') == 'master':
                        current_prereq_mastery = edge_attr.get('mastery_level', 0.5)
                        new_prereq_mastery = max(0.0, min(current_prereq_mastery + reinforcement_change, 1.0))
                        edge_attr['mastery_level'] = new_prereq_mastery
                        
                        skill_name = self.graph.nodes[prereq_node].get('skill_name', 'Unknown')
                        print(f"🔗 知识强化: '{skill_name}' (作为先决条件) 掌握度因上层技能学习而提升 {reinforcement_change:+.3f}")
                        
                        # 继续向下递归传播
                        self._propagate_reinforcement(student_node, prereq_node, reinforcement_change, decay_factor)
    
    def get_student_knowledge_state(self, student_id: int) -> Dict:
        """
        获取学生当前知识状态
        
        Args:
            student_id: 学生ID
            
        Returns:
            学生知识状态字典
        """
        student_node = f"student_{student_id}"
        
        if student_node not in self.graph:
            return {}
        
        knowledge_state = {}
        
        # 获取技能掌握情况
        for edge in self.graph.out_edges(student_node, data=True):
            if edge[2].get('type') == 'master':
                skill_node = edge[1]
                skill_id = self.graph.nodes[skill_node].get('skill_id', 0)
                skill_name = self.graph.nodes[skill_node].get('skill_name', 'Unknown')
                
                knowledge_state[f"skill_{skill_id}"] = {
                    'skill_name': skill_name,
                    'mastery_level': edge[2].get('mastery_level', 0.0),
                    'confidence': edge[2].get('confidence', 0.0),
                    'attempts': edge[2].get('attempts_count', 0)
                }
        
        return knowledge_state
    
    def recommend_next_problems(self, student_id: int, num_recommendations: int = 5) -> List[Dict]:
        """
        为学生推荐下一步学习的题目
        
        Args:
            student_id: 学生ID
            num_recommendations: 推荐题目数量
            
        Returns:
            推荐题目列表
        """
        student_node = f"student_{student_id}"
        
        if student_node not in self.graph:
            return []
        
        # 获取学生当前知识状态
        knowledge_state = self.get_student_knowledge_state(student_id)
        
        # 找到薄弱技能
        weak_skills = []
        for skill_key, skill_info in knowledge_state.items():
            if skill_info['mastery_level'] < 0.7:  # 掌握度阈值
                weak_skills.append((skill_key, skill_info['mastery_level']))
        
        # 按掌握度排序，优先推荐最薄弱的技能相关题目
        weak_skills.sort(key=lambda x: x[1])
        
        recommendations = []
        
        for skill_key, mastery_level in weak_skills[:3]:  # 最多考虑3个薄弱技能
            skill_id = int(skill_key.split('_')[1])
            skill_node = f"skill_{skill_id}"
            
            # 找到需要该技能的题目
            skill_problems = []
            for edge in self.graph.in_edges(skill_node, data=True):
                if edge[2].get('type') == 'require':
                    problem_node = edge[0]
                    problem_id = self.graph.nodes[problem_node].get('problem_id', 0)
                    difficulty = self.graph.nodes[problem_node].get('difficulty', 0.5)
                    
                    # 计算适合度分数
                    suitability = self._calculate_problem_suitability(
                        student_id, problem_id, mastery_level, difficulty
                    )
                    
                    skill_problems.append({
                        'problem_id': problem_id,
                        'skill_id': skill_id,
                        'skill_name': knowledge_state[skill_key]['skill_name'],
                        'current_mastery': mastery_level,
                        'problem_difficulty': difficulty,
                        'suitability_score': suitability,
                        'recommendation_reason': f"提升{knowledge_state[skill_key]['skill_name']}技能"
                    })
            
            # 按适合度排序
            skill_problems.sort(key=lambda x: x['suitability_score'], reverse=True)
            recommendations.extend(skill_problems[:2])  # 每个技能推荐最多2道题
        
        # 按适合度总排序，返回top N
        recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)
        return recommendations[:num_recommendations]
    
    def _calculate_problem_suitability(self, student_id: int, problem_id: int, 
                                     mastery_level: float, problem_difficulty: float) -> float:
        """
        计算题目对学生的适合度
        
        Args:
            student_id: 学生ID
            problem_id: 题目ID
            mastery_level: 当前技能掌握度
            problem_difficulty: 题目难度
            
        Returns:
            适合度分数[0,1]
        """
        # 检查学生是否已经做过这道题
        student_node = f"student_{student_id}"
        problem_node = f"problem_{problem_id}"
        
        has_solved = False
        if self.graph.has_edge(student_node, problem_node):
            for edge_data in self.graph[student_node][problem_node].values():
                if edge_data.get('type') == 'solve':
                    has_solved = True
                    break
        
        # 如果已经做过且做对了，适合度降低
        if has_solved:
            return 0.3
        
        # 基于"最近发展区"理论计算适合度
        # 理想难度应该略高于当前掌握度
        ideal_difficulty = mastery_level + 0.1
        difficulty_diff = abs(problem_difficulty - ideal_difficulty)
        
        # 难度适合度：差异越小越好
        difficulty_suitability = max(0, 1 - difficulty_diff * 2)
        
        # 挑战性：不能太简单也不能太难
        if problem_difficulty < mastery_level - 0.2:
            challenge_factor = 0.5  # 太简单
        elif problem_difficulty > mastery_level + 0.3:
            challenge_factor = 0.3  # 太难
        else:
            challenge_factor = 1.0  # 适中
        
        return difficulty_suitability * challenge_factor
    
    def _compute_skill_mastery_direct(self, data: Dict) -> pd.DataFrame:
        """
        直接计算学生在各技能上的掌握程度
        
        Args:
            data: 数据集字典
            
        Returns:
            学生×技能的掌握程度矩阵
        """
        if 'interactions' not in data or 'problem_skill_matrix' not in data:
            return pd.DataFrame()

        interactions = data['interactions'].copy()
        problem_skill = data['problem_skill_matrix']

        # 创建一个 problem_idx -> list of skill_idx 的映射
        problem_to_skills = {
            p_idx: problem_skill.columns[problem_skill.iloc[p_idx] == 1].tolist()
            for p_idx in range(problem_skill.shape[0])
        }

        # 使用 problem_id_idx (题目索引) 来映射技能
        interactions['skill_indices'] = interactions['problem_id_idx'].map(problem_to_skills)
        
        # 展开数据，使得每行是一个 (学生, 题目, 技能索引) 的交互
        exploded_df = interactions.explode('skill_indices').dropna(subset=['skill_indices'])
        
        if exploded_df.empty:
            return pd.DataFrame()

        # 按 (学生索引, 技能索引) 分组计算正确率
        # 确保列是整数类型以进行正确分组
        exploded_df['skill_indices'] = exploded_df['skill_indices'].astype(int)
        
        # 修正：分组时使用原始的 student_id，因为后续步骤需要它作为索引
        # 我们需要确保 'student_id' 列在 exploded_df 中存在
        if 'student_id' not in exploded_df.columns:
            # 如果不存在，需要从 data['interactions'] 重新合并进来
            # 但更简单的方法是确保它在 data_loader 中一直被传递下来
            print("FATAL: 'student_id' column is missing for mastery computation.")
            return pd.DataFrame()

        skill_mastery = exploded_df.groupby(['student_id', 'skill_indices'])['correct'].mean().unstack(fill_value=0.0)
        
        return skill_mastery

    def generate_llm_prompt(self, student_id: int, target_skill_ids: List[int], num_weakest: int = 5, num_strongest: int = 5) -> str:
        """
        为大语言模型生成一个用于学习路径规划的详细Prompt

        Args:
            student_id: 学生ID
            target_skill_ids: 学生希望学习的目标技能ID列表
            num_weakest: 在Prompt中包含的最薄弱技能的数量
            num_strongest: 在Prompt中包含的最强技能的数量

        Returns:
            一个结构化的、信息丰富的Prompt字符串
        """
        # 1. 获取学生知识状态
        knowledge_state = self.get_student_knowledge_state(student_id)
        if not knowledge_state:
            return "无法为该学生生成Prompt，知识状态为空。"

        skills_by_mastery = sorted(knowledge_state.items(), key=lambda item: item[1]['mastery_level'])
        
        weakest_skills = skills_by_mastery[:num_weakest]
        strongest_skills = skills_by_mastery[-num_strongest:][::-1]
        
        # 2. 提取相关的先修关系
        prerequisite_rules = []
        relevant_skills = set(target_skill_ids)
        
        # 找到目标技能的所有前置依赖
        for target_id in target_skill_ids:
            try:
                ancestors = nx.ancestors(self.graph, f"skill_{target_id}")
                for prereq_node in ancestors:
                    if self.graph.nodes[prereq_node].get('type') == 'skill':
                        relevant_skills.add(self.graph.nodes[prereq_node]['skill_id'])
            except nx.NetworkXError:
                # 节点可能不在图中
                pass
        
        # 提取这些相关技能之间的所有先修关系
        for u, v, attr in self.graph.edges(data=True):
            if attr.get('type') == 'prerequisite':
                u_id = self.graph.nodes[u].get('skill_id')
                v_id = self.graph.nodes[v].get('skill_id')
                if u_id in relevant_skills or v_id in relevant_skills:
                     u_name = self.graph.nodes[u].get('skill_name', f'技能{u_id}')
                     v_name = self.graph.nodes[v].get('skill_name', f'技能{v_id}')
                     prerequisite_rules.append(f"- **'{u_name}'** 是 **'{v_name}'** 的先修技能。")


        # 3. 组装Prompt
        prompt = f"""
# **角色：**
你是一位顶级的AI教育规划专家，精通认知科学和教学设计。

# **任务：**
基于以下提供的学生知识状态和知识图谱规则，为该学生设计一条个性化的、循序渐进的学习路径，以帮助他掌握目标技能。

---

# **学生知识画像 (Student Profile):**

## **1. 基本信息:**
- **学生ID:** {student_id}

## **2. 知识强项 (已掌握的技能):**
"""
        for skill_key, info in strongest_skills:
            prompt += f"- **{info['skill_name']}**: 掌握度 {info['mastery_level']:.2f}\n"

        prompt += """
## **3. 知识弱项 (最需要提升的技能):**
"""
        for skill_key, info in weakest_skills:
            prompt += f"- **{info['skill_name']}**: 掌握度 {info['mastery_level']:.2f}\n"

        prompt += f"""
---

# **学习目标与规则 (Goal and Constraints):**

## **1. 学习目标:**
学生希望系统地学习并掌握以下技能：
"""
        for target_id in target_skill_ids:
            skill_name = self.graph.nodes.get(f"skill_{target_id}", {}).get('skill_name', f'技能{target_id}')
            prompt += f"- **{skill_name}**\n"

        prompt += """
## **2. 必须遵守的规则 (知识结构):**
学习路径的规划必须严格遵守以下知识点之间的先修关系。一个技能必须在其所有的先修技能都被掌握后才能开始学习。
"""
        if prerequisite_rules:
            prompt += "\n".join(prerequisite_rules)
        else:
            prompt += "该目标技能无特定的先修关系，但仍需从学生的薄弱知识点开始补足。"

        prompt += """

---

# **输出要求 (Output Format):**

请提供一个清晰的、分步骤的学习计划。每个步骤应包含：
1.  **学习的技能名称**。
2.  **推荐学习该技能的理由** (例如：因为它是掌握目标技能XXX的必要前提，或者是学生当前的知识薄弱点)。
3.  **学习顺序**：请确保整个计划的顺序严格遵循先修关系，并从学生最需要弥补的知识点开始。

请开始生成学习计划：
"""
        return prompt.strip()

def main():
    """主函数，用于演示和测试"""
    # run_autonomous_simulation()

    # --- 新增：直接构建和保存DKG ---
    print("Starting DKG build process from real data...")
    # 延迟导入以避免循环依赖
    from dkg_mvp.data_loader import DataLoader

    # 1. 加载数据
    # 指定数据集的根目录 (修正：只到 "dataset" 这一级)
    dataset_root = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    loader = DataLoader(dataset_root)
    # 加载我们正在使用的数据集
    data_dict = loader.load_skill_builder_data('skill_builder_data_filter15.csv')

    # 修正：检查 data_dict 是否为 None
    if data_dict and not data_dict.get('interactions', pd.DataFrame()).empty:
        # 2. 构建DKG
        builder = DKGBuilder()
        dkg = builder.build_from_data(data_dict)
        print(f"DKG built successfully with {dkg.number_of_nodes()} nodes and {dkg.number_of_edges()} edges.")
        
        # 3. 保存图谱
        # 将图谱保存在项目根目录
        save_path = os.path.join(os.path.dirname(__file__), '..', 'dkg.pkl')
        builder.save_with_pickle(save_path)
    else:
        print("Failed to load data. DKG build process aborted.")
    # --- 结束新增代码 ---

def run_autonomous_simulation():
    """运行一个完全自主的智能体学习模拟，并可视化其学习过程。"""
    from .simulation import SimulationEngine
    from .interactive_visualization import InteractiveDKGVisualizer
    import os

    print("\n\n" + "="*15 + " 自主学习智能体模拟 (终极版) " + "="*15)
    print("目标: 观察一个拥有个性、会遗忘、能自主决策的智能体的学习过程。")

    # 1. 创建知识空间和具有个性的学生
    builder = DKGBuilder()
    student_id = 0
    # 在DKGBuilder的_create_nodes中已通过np.random实现个性化
    builder.graph.add_node(f"student_{student_id}", type='student', student_id=student_id, 
                           learning_rate=np.random.normal(1.0, 0.1),
                           perseverance=np.random.randint(3, 7),
                           curiosity=np.random.uniform(0.05, 0.2))
    
    skills = {1: "代数基础", 2: "一元一次方程", 3: "一元二次方程"}
    problems = {}
    problem_id_counter = 101
    for skill_id in skills.keys():
        for _ in range(30): # 每个技能30题
            problems[problem_id_counter] = [skill_id]
            problem_id_counter += 1

    for skill_id, skill_name in skills.items():
        builder.graph.add_node(f"skill_{skill_id}", type='skill', skill_id=skill_id, skill_name=skill_name)
    
    for problem_id, req_skills in problems.items():
        builder.graph.add_node(f"problem_{problem_id}", type='problem', problem_id=problem_id)
        for skill_id in req_skills:
            builder.graph.add_edge(f"problem_{problem_id}", f"skill_{skill_id}", type='require')

    builder.graph.add_edge("skill_1", "skill_2", type='prerequisite')
    builder.graph.add_edge("skill_2", "skill_3", type='prerequisite')

    for skill_id in skills:
        builder.graph.add_edge(f"student_{student_id}", f"skill_{skill_id}", type='master', mastery_level=0.0)

    # 2. 初始化自主学习引擎 (不再需要路径)
    engine = SimulationEngine(builder, problems)
    
    # 3. 运行完全自主的模拟
    history = engine.run_simulation(student_id=student_id, num_steps=150) # 步数可以更长

    # 4. 可视化结果
    if not history.empty:
        output_dir = os.path.join('dkg_mvp', 'visualizations')
        visualizer = InteractiveDKGVisualizer(output_dir=output_dir)
        save_path = os.path.join(output_dir, "autonomous_agent_simulation.html")
        visualizer.plot_simulation_history(history, student_id, save_path=save_path)
        print(f"\n📈 终极学习曲线已保存到: {save_path}")

    print("\n" + "="*40)
    print("自主学习智能体模拟已完成！")

def run_real_data_simulation():
    """使用真实的ASSISTments数据集运行模拟。"""
    from .data_loader import DataLoader
    from .simulation import SimulationEngine
    from .interactive_visualization import InteractiveDKGVisualizer
    import os
    
    # --- 数据加载 ---
    loader = DataLoader('dataset')
    full_data = loader.load_assistments_log_data(dataset_name='skill_builder')
    if not full_data:
        print("数据加载失败，测试中止。")
        return
        
    # --- 模拟准备 ---
    print("\n\n" + "="*15 + " 学习路径模拟 (真实数据) " + "="*15)
    sim_student_id = 3
    
    # 1. 分割数据
    initial_data, oracle = loader.split_student_data_for_simulation(full_data, student_id=sim_student_id, train_ratio=0.7)
    
    # 2. 用初始数据构建DKG
    print("\n构建初始DKG...")
    builder = DKGBuilder()
    builder.build_from_data(initial_data)
    
    # --- 运行模拟 ---
    # 3. 初始化并运行模拟器
    engine = SimulationEngine(builder, oracle)
    history = engine.run_simulation(student_id=sim_student_id, num_steps=15)
    
    # --- 结果可视化 ---
    # 4. 可视化模拟结果
    if not history.empty:
        print("\n--- 可视化模拟结果 ---")
        output_dir = os.path.join('dkg_mvp', 'visualizations')
        visualizer = InteractiveDKGVisualizer(output_dir=output_dir)
        visualizer.plot_simulation_history(history, sim_student_id)
    
    print("\n" + "="*40)
    print("模拟与可视化已完成！")
    print(f"请在 'dkg_mvp/visualizations' 目录下查看 simulation_history_student_{sim_student_id}.html 文件。")


if __name__ == "__main__":
    main()