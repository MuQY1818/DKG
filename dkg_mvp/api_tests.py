import os
import sys
import pandas as pd

# 将项目根目录添加到sys.path，以便导入dkg_mvp中的模块
# This assumes the script is run from the project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dkg_mvp.dkg_builder import DKGBuilder
from dkg_mvp.data_loader import DataLoader

def setup_test_dkg() -> DKGBuilder:
    """创建一个用于测试的、小型的、已构建的DKG实例"""
    print("--- Setting up a test DKG instance ---")
    loader = DataLoader('dataset')
    # 使用一小部分数据来加速测试
    full_data = loader.load_assistments_log_data(dataset_name='skill_builder', nrows=1000)
    
    if not full_data:
        print("Failed to load data for testing. Aborting.")
        sys.exit(1)
        
    builder = DKGBuilder()
    builder.build_from_data(full_data)
    print("--- Test DKG setup complete ---")
    return builder

def test_save_and_load_graph(builder: DKGBuilder):
    """测试图的保存和加载功能"""
    print("\n--- Test: Save and Load Graph ---")
    test_file = "test_dkg.graphml"
    
    # 1. 保存图
    builder.save_graph(test_file)
    assert os.path.exists(test_file), "Graph file was not created."
    print(f"Graph saved to {test_file}")
    
    # 2. 加载图
    loaded_builder = DKGBuilder.load_graph(test_file)
    
    # 3. 验证
    original_nodes = builder.graph.number_of_nodes()
    loaded_nodes = loaded_builder.graph.number_of_nodes()
    original_edges = builder.graph.number_of_edges()
    loaded_edges = loaded_builder.graph.number_of_edges()
    
    print(f"Original graph: {original_nodes} nodes, {original_edges} edges.")
    print(f"Loaded graph: {loaded_nodes} nodes, {loaded_edges} edges.")
    
    assert original_nodes == loaded_nodes, "Node count mismatch after loading."
    assert original_edges == loaded_edges, "Edge count mismatch after loading."
    
    # 随机抽查一个节点的属性
    random_node = list(builder.graph.nodes)[0]
    original_attrs = builder.graph.nodes[random_node]
    loaded_attrs = loaded_builder.graph.nodes[random_node]
    assert original_attrs['type'] == loaded_attrs['type'], "Node attribute mismatch after loading."

    print("✅ Test_save_and_load_graph: PASSED")
    
    # 清理测试文件
    os.remove(test_file)
    print(f"Cleaned up {test_file}")

def test_record_interaction(builder: DKGBuilder):
    """测试记录单次学习交互的接口"""
    print("\n--- Test: Record Interaction ---")
    student_id = 10
    problem_id = 3
    skill_id = 1 # 假设problem 3考察skill 1
    
    # 1. 获取更新前的状态
    mastery_before = builder.get_student_knowledge_state(student_id).get(f"skill_{skill_id}", {}).get('mastery_level', 0)
    print(f"Mastery of skill {skill_id} for student {student_id} before interaction: {mastery_before:.3f}")
    
    # 2. 构造并记录一次新的交互（做对了）
    interaction = {
        'student_id': student_id,
        'problem_id': problem_id,
        'correct': 1,
        'score': 1.0,
        'timestamp': pd.Timestamp.now()
    }
    builder.record_interaction(interaction)
    print("Recorded a new 'correct' interaction.")
    
    # 3. 验证状态是否更新
    mastery_after = builder.get_student_knowledge_state(student_id).get(f"skill_{skill_id}", {}).get('mastery_level', 0)
    print(f"Mastery of skill {skill_id} for student {student_id} after interaction: {mastery_after:.3f}")
    
    assert mastery_after > mastery_before, "Mastery level should increase after a correct interaction."
    
    # 验证solve关系是否被添加或更新
    solve_edge_found = False
    if builder.graph.has_edge(f"student_{student_id}", f"problem_{problem_id}"):
        for key, attr in builder.graph[f"student_{student_id}"][f"problem_{problem_id}"].items():
            if attr.get('type') == 'solve':
                solve_edge_found = True
                assert attr['correct'] == 1
                break
    assert solve_edge_found, "The 'solve' edge was not created or updated."

    print("✅ test_record_interaction: PASSED")

def test_query_apis(builder: DKGBuilder):
    """测试查询接口的功能"""
    print("\n--- Test: Query APIs ---")
    
    # 从图中动态选择一个存在的学生、技能和题目ID进行测试
    student_node = next(n for n, d in builder.graph.nodes(data=True) if d['type'] == 'student')
    skill_node = next(n for n, d in builder.graph.nodes(data=True) if d['type'] == 'skill')
    problem_node = next(n for n, d in builder.graph.nodes(data=True) if d['type'] == 'problem')
    
    student_id = builder.graph.nodes[student_node]['student_id']
    skill_id = builder.graph.nodes[skill_node]['skill_id']
    problem_id = builder.graph.nodes[problem_node]['problem_id']


    # 1. 测试 get_student_profile
    profile = builder.get_student_profile(student_id)
    print(f"Student profile for {student_id}: (output suppressed for brevity)")
    assert "student_id" in profile and profile["student_id"] == student_id, "get_student_profile failed: ID mismatch."
    assert "knowledge_summary" in profile, "get_student_profile failed: missing knowledge_summary."
    assert "strongest_skills" in profile["knowledge_summary"], "get_student_profile failed: missing strongest_skills."
    print("✅ get_student_profile seems to work.")

    # 2. 测试 get_skill_details
    skill_details = builder.get_skill_details(skill_id)
    print(f"Details for skill {skill_id}: (output suppressed for brevity)")
    assert "skill_name" in skill_details, "get_skill_details failed: missing skill_name."
    assert "prerequisites" in skill_details, "get_skill_details failed: missing prerequisites."
    print("✅ get_skill_details seems to work.")
    
    # 3. 测试 get_problem_details
    problem_details = builder.get_problem_details(problem_id)
    print(f"Details for problem {problem_id}: (output suppressed for brevity)")
    assert "problem_id" in problem_details and problem_details["problem_id"] == problem_id, "get_problem_details failed: ID mismatch."
    assert "required_skills" in problem_details, "get_problem_details failed: missing required_skills."
    print("✅ get_problem_details seems to work.")

def main():
    """主函数，运行所有测试"""
    # 1. 构建一个基础的DKG用于测试
    test_builder = setup_test_dkg()
    
    # 2. 运行所有测试用例
    test_save_and_load_graph(test_builder)
    test_record_interaction(test_builder)
    test_query_apis(test_builder)
    
    print("\n🎉 All API tests completed successfully!")

if __name__ == "__main__":
    main() 