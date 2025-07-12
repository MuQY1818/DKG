import os
import sys
import pandas as pd
import json

# 确保能找到dkg_mvp模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dkg_mvp.dkg_builder import DKGBuilder


MODELS_DIR = "models"
DKG_SAVE_PATH = os.path.join(MODELS_DIR, "dkg_skill_builder.graphml")

# --- 第一部分: 系统初始化 ---
# 检查是否存在已保存的DKG模型，如果否则构建并保存一个
if not os.path.exists(DKG_SAVE_PATH):
    print("未找到已保存的DKG，正在从头构建...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    from dkg_mvp.data_loader import DataLoader
    
    loader = DataLoader('dataset')
    log_data = loader.load_skill_builder_data()
    
    if log_data:
        builder = DKGBuilder()
        builder.build_from_data(log_data)
        builder.save_graph(DKG_SAVE_PATH)
        print("新的DKG已构建并保存。")
else:
    print("从文件加载已存在的DKG...")
    builder = DKGBuilder.load_graph(DKG_SAVE_PATH)
    print("DKG加载完毕。")

# --- 第二部分: 核心API功能展示 ---
if 'builder' in locals() and builder.graph.number_of_nodes() > 0:
    
    # --- 准备工作: 动态选择用于测试的实体 ---
    student_id_to_test = 20
    if f"student_{student_id_to_test}" not in builder.graph:
        print(f"警告：学生ID {student_id_to_test} 不在图中。")
        student_node = next((n for n, d in builder.graph.nodes(data=True) if d['type'] == 'student'), None)
        if student_node:
            student_id_to_test = builder.graph.nodes[student_node]['student_id']
            print(f"已自动选择一个存在的学生进行测试: ID {student_id_to_test}")
        else:
            print("错误：图中找不到任何学生。")
            sys.exit(1)

    # 动态选择一个问题和相关技能
    problem_node = next((n for n, d in builder.graph.nodes(data=True) if d['type'] == 'problem'), "problem_0")
    problem_id_to_test = builder.graph.nodes[problem_node]['problem_id']
    
    required_skills = [v for u, v, attr in builder.graph.out_edges(problem_node, data=True) if attr['type'] == 'require']
    if not required_skills:
        print(f"错误: 题目 {problem_id_to_test} 未关联任何技能，无法继续测试。")
        sys.exit(1)
    skill_node_to_test = required_skills[0]
    skill_id_to_test = builder.graph.nodes[skill_node_to_test]['skill_id']

    print(f"\n--- 将使用以下实体进行API功能演示 ---")
    print(f"  - 学生ID: {student_id_to_test}")
    print(f"  - 题目ID: {problem_id_to_test}")
    print(f"  - 技能ID: {skill_id_to_test} (技能名: '{builder.graph.nodes[skill_node_to_test]['skill_name']}')")
    print("-" * 40)

    # 1. 演示: 查询实体详情
    print("\n【API演示 1: 查询实体详情 (get_problem_details & get_skill_details)】")
    problem_details = builder.get_problem_details(problem_id_to_test)
    print(f"\n--- 题目 {problem_id_to_test} 的详情 ---")
    print(json.dumps(problem_details, indent=2, ensure_ascii=False))

    skill_details = builder.get_skill_details(skill_id_to_test)
    print(f"\n--- 技能 {skill_id_to_test} 的详情 ---")
    print(json.dumps(skill_details, indent=2, ensure_ascii=False))

    # 2. 演示: 获取学生初始画像
    print(f"\n【API演示 2: 获取学生 {student_id_to_test} 交互前的知识画像 (get_student_profile)】")
    profile_before = builder.get_student_profile(student_id_to_test)
    print(json.dumps(profile_before['knowledge_summary'], indent=2, ensure_ascii=False))

    # 3. 演示: 单次交互与手动干预
    print(f"\n【API演示 3: 记录单次交互与手动更新 (record_interaction & update_skill_mastery_manual)】")
    interaction = {'student_id': student_id_to_test, 'problem_id': problem_id_to_test, 'correct': 1, 'timestamp': pd.Timestamp.now()}
    builder.record_interaction(interaction)
    print(f"\n--- 模拟交互：学生 {student_id_to_test} 做对了题目 {problem_id_to_test}。交互已记录。")

    builder.update_skill_mastery_manual(student_id_to_test, skill_id_to_test, 0.95)
    print(f"--- 模拟线下辅导：已手动将学生在技能 {skill_id_to_test} 上的掌握度更新为 0.95。")

    # 4. 演示: 批量交互
    print(f"\n【API演示 4: 批量记录交互 (record_interactions_batch)】")
    # 动态选择另外两个问题
    next_problems = [n for n,d in builder.graph.nodes(data=True) if d['type'] == 'problem' and d['problem_id'] != problem_id_to_test]
    p1 = builder.graph.nodes[next_problems[0]]['problem_id']
    p2 = builder.graph.nodes[next_problems[1]]['problem_id']
    
    batch_interactions = [
        {'student_id': student_id_to_test, 'problem_id': p1, 'correct': 1, 'timestamp': pd.Timestamp.now()},
        {'student_id': student_id_to_test, 'problem_id': p2, 'correct': 0, 'timestamp': pd.Timestamp.now()},
        {'student_id': student_id_to_test, 'problem_id': p2, 'correct': 1, 'timestamp': pd.Timestamp.now()}
    ]
    builder.record_interactions_batch(batch_interactions)
    print(f"--- 批量交互：学生 {student_id_to_test} 又完成了 {len(batch_interactions)} 次练习。批量记录已更新。")

    # 5. 演示: 获取最终状态与推荐
    print(f"\n【API演示 5: 获取学生最终状态与推荐 (get_student_knowledge_state & recommend_next_problems)】")
    state_after = builder.get_student_knowledge_state(student_id_to_test)
    print(f"\n--- 学生 {student_id_to_test} 的最终原始知识状态 (与演示技能相关) ---")
    # 为了简洁，只展示与测试技能相关的部分
    filtered_state = {k:v for k,v in state_after.items() if str(skill_id_to_test) in k}
    print(json.dumps(filtered_state, indent=2, ensure_ascii=False))

    recommendations = builder.recommend_next_problems(student_id_to_test)
    print(f"\n--- 基于最终状态，为学生 {student_id_to_test} 推荐下一步练习 ---")
    print(json.dumps(recommendations, indent=2, ensure_ascii=False))

    # 6. 演示: 生成LLM Prompt
    print(f"\n【API演示 6: 为LLM生成学习规划Prompt (generate_llm_prompt)】")
    # 动态选择一个学生还没掌握好的技能作为学习目标
    profile_after = builder.get_student_profile(student_id_to_test)
    weak_skills = profile_after['knowledge_summary'].get('weakest_skills', [])
    if weak_skills:
        target_skill_name = weak_skills[0]['skill_name']
        target_skill_node = next((n for n,d in builder.graph.nodes(data=True) if d.get('skill_name') == target_skill_name), None)
        if target_skill_node:
            target_skill_id = builder.graph.nodes[target_skill_node]['skill_id']
            print(f"\n--- 目标: 帮助学生 {student_id_to_test} 学习掌握技能 '{target_skill_name}' (ID: {target_skill_id}) ---")
            llm_prompt = builder.generate_llm_prompt(student_id_to_test, [target_skill_id])
            print(llm_prompt)
        else:
            print("未能动态找到目标技能。")
    else:
        print("该学生没有薄弱技能，无法生成学习目标Prompt。")

else:
    print("DKG未能成功加载或构建，测试中止。") 