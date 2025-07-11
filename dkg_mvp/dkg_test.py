import unittest
import os
import sys
import pandas as pd
import networkx as nx

# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dkg_mvp.data_loader import DataLoader
from dkg_mvp.dkg_builder import DKGBuilder

class TestDKGBuilder(unittest.TestCase):
    """
    测试 DKGBuilder 类的逻辑是否正确。
    """
    
    @classmethod
    def setUpClass(cls):
        """在所有测试开始前，加载数据并构建DKG。"""
        print("Setting up test class: loading data and building DKG...")
        
        # 使用相对路径定位数据集
        cls.loader = DataLoader('dataset')
        cls.data_dict = cls.loader.load_skill_builder_data()
        
        if cls.data_dict is None:
            raise ValueError("Failed to load data, cannot run tests.")
            
        cls.dkg_builder = DKGBuilder()
        cls.graph = cls.dkg_builder.build_from_data(cls.data_dict)
        
        # 为方便测试，预先计算一些统计数据
        cls.interactions = cls.data_dict['interactions']
        cls.problems_df = pd.read_csv(os.path.join('dataset', 'clear_dataset', 'builder_problem.csv'))
        
        # 统计 problem-skill 关系数量
        cls.expected_require_edges = cls.problems_df['skill_id'].notna().sum()

    def test_data_loaded(self):
        """测试数据是否成功加载。"""
        self.assertIsNotNone(self.data_dict, "Data dictionary should not be None.")
        self.assertIn('interactions', self.data_dict)
        self.assertIn('problem_skill_matrix', self.data_dict)
        self.assertIn('skills', self.data_dict)

    def test_node_counts(self):
        """测试图中节点数量是否与数据源一致。"""
        expected_students = self.data_dict['num_students']
        expected_problems = self.data_dict['num_problems']
        expected_skills = self.data_dict['num_skills']
        
        students_in_graph = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'student']
        problems_in_graph = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'problem']
        skills_in_graph = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'skill']
        
        self.assertEqual(len(students_in_graph), expected_students, "Student node count mismatch.")
        self.assertEqual(len(problems_in_graph), expected_problems, "Problem node count mismatch.")
        self.assertEqual(len(skills_in_graph), expected_skills, "Skill node count mismatch.")

    def test_edge_counts(self):
        """测试图中边的数量是否正确。"""
        # 'solve' 边的数量应等于交互记录的数量
        expected_solve_edges = len(self.interactions)
        solve_edges = [e for e in self.graph.edges(data=True) if e[2].get('type') == 'solve']
        self.assertEqual(len(solve_edges), expected_solve_edges, "Solve edge count mismatch.")
        
        # 'require' 边的数量
        require_edges = [e for e in self.graph.edges(data=True) if e[2].get('type') == 'require']
        self.assertEqual(len(require_edges), self.expected_require_edges, "Require edge count mismatch.")

    def test_random_student_node(self):
        """随机抽查一个学生节点，验证其属性。"""
        random_interaction = self.interactions.sample(1).iloc[0]
        student_id = random_interaction['student_id']
        
        student_node_name = f"student_{student_id}"
        self.assertIn(student_node_name, self.graph.nodes, f"Student node {student_node_name} not in graph.")
        
        node_data = self.graph.nodes[student_node_name]
        self.assertEqual(node_data['type'], 'student')
        self.assertEqual(node_data['student_id'], student_id)

    def test_random_problem_node(self):
        """随机抽查一个问题节点，验证其属性。"""
        random_interaction = self.interactions.sample(1).iloc[0]
        problem_id = random_interaction['problem_id']
        
        problem_node_name = f"problem_{problem_id}"
        self.assertIn(problem_node_name, self.graph.nodes, f"Problem node {problem_node_name} not in graph.")
        
        node_data = self.graph.nodes[problem_node_name]
        self.assertEqual(node_data['type'], 'problem')
        self.assertEqual(node_data['problem_id'], problem_id)

    def test_random_solve_edge(self):
        """随机抽查一条'solve'边，验证其属性和连接。"""
        random_interaction = self.interactions.sample(1).iloc[0]
        student_id = random_interaction['student_id']
        problem_id = random_interaction['problem_id']
        expected_correct = int(random_interaction['correct'])
        
        student_node = f"student_{student_id}"
        problem_node = f"problem_{problem_id}"
        
        self.assertTrue(self.graph.has_edge(student_node, problem_node), f"Edge between {student_node} and {problem_node} not found.")
        
        # 在MultiDiGraph中，两个节点间可以有多条边，用key区分
        edge_data = self.graph.get_edge_data(student_node, problem_node)
        
        # 查找 'solve' 类型的边
        found_solve_edge = False
        for key, data in edge_data.items():
            if data.get('type') == 'solve':
                self.assertEqual(data['correct'], expected_correct, "Solve edge 'correct' attribute mismatch.")
                found_solve_edge = True
                break
        
        self.assertTrue(found_solve_edge, "No 'solve' type edge found for the interaction.")

    def test_random_require_edge(self):
        """随机抽查一条'require'边，验证其连接。"""
        # 找到一个存在技能ID的问题
        problem_with_skill = self.problems_df.dropna(subset=['skill_id']).sample(1).iloc[0]
        problem_id = int(problem_with_skill['problem_id'])
        skill_id = int(problem_with_skill['skill_id'])

        problem_node = f"problem_{problem_id}"
        skill_node = f"skill_{skill_id}"
        
        self.assertIn(problem_node, self.graph.nodes, f"Problem node for require edge test ({problem_node}) not in graph.")
        self.assertIn(skill_node, self.graph.nodes, f"Skill node for require edge test ({skill_node}) not in graph.")

        self.assertTrue(self.graph.has_edge(problem_node, skill_node), f"Require edge between {problem_node} and {skill_node} not found.")
        edge_data = self.graph.get_edge_data(problem_node, skill_node)
        
        found_require_edge = any(data.get('type') == 'require' for data in edge_data.values())
        self.assertTrue(found_require_edge, "No 'require' type edge found.")


if __name__ == '__main__':
    unittest.main() 