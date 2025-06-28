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
    log_data = loader.load_assistments_log_data('skill_builder')
    
    if log_data:
        builder = DKGBuilder()
        builder.build_from_data(log_data)
        builder.save_graph(DKG_SAVE_PATH)
        print("新的DKG已构建并保存。")
else:
    print("从文件加载已存在的DKG...")
    builder = DKGBuilder.load_graph(DKG_SAVE_PATH)
    print("DKG加载完毕。")

# --- 第二部分: 模拟一次新的学习交互并观察变化 ---
if 'builder' in locals() and builder.graph.number_of_nodes() > 0:
    student_id_to_test = 20
    
    # 确保要测试的学生存在于图中
    if f"student_{student_id_to_test}" not in builder.graph:
        print(f"错误：学生ID {student_id_to_test} 不在测试数据集中。请尝试其他ID。")
        # 动态选择一个存在的学生ID
        student_node = next((n for n, d in builder.graph.nodes(data=True) if d['type'] == 'student'), None)
        if student_node:
            student_id_to_test = builder.graph.nodes[student_node]['student_id']
            print(f"已自动选择一个存在的学生进行测试: ID {student_id_to_test}")
        else:
            print("错误：图中找不到任何学生。")
            sys.exit(1)


    # 1. 查看交互前的学生画像
    print(f"\n--- 交互前，学生 {student_id_to_test} 的知识画像 ---")
    profile_before = builder.get_student_profile(student_id_to_test)
    print(json.dumps(profile_before['knowledge_summary'], indent=2, ensure_ascii=False))

    # 2. 模拟一次交互：该学生正确解答了问题15
    print(f"\n--- 模拟交互：学生 {student_id_to_test} 做对了题目 15 ---")
    interaction = {
        'student_id': student_id_to_test,
        'problem_id': 15,
        'correct': 1,
        'timestamp': pd.Timestamp.now()
    }
    builder.record_interaction(interaction)
    
    # 3. 查看交互后的学生画像，对比变化
    print(f"\n--- 交互后，学生 {student_id_to_test} 的新知识画像 ---")
    profile_after = builder.get_student_profile(student_id_to_test)
    print(json.dumps(profile_after['knowledge_summary'], indent=2, ensure_ascii=False))
    
    # 4. 基于更新后的状态，为学生推荐下一步练习
    print(f"\n--- 为学生 {student_id_to_test} 推荐下一步练习 ---")
    recommendations = builder.recommend_next_problems(student_id_to_test)
    print(json.dumps(recommendations, indent=2, ensure_ascii=False))

else:
    print("DKG未能成功加载或构建，测试中止。") 