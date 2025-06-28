"""
应用示例模块 - DKG在教育场景中的应用演示
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from data_loader import DataLoader
from dkg_builder import DKGBuilder
from visualization import DKGVisualizer
from interactive_visualization import InteractiveDKGVisualizer

def example_1_basic_dkg_construction():
    """
    示例1: 基础DKG构建和分析
    展示如何从FrcSub数据集构建知识图谱
    """
    print("=" * 60)
    print("示例1: 基础DKG构建和分析")
    print("=" * 60)
    
    # 1. 加载数据
    print("1. 加载数据集...")
    loader = DataLoader('../dataset')
    
    # --- 加载FrcSub数据集并获取子集 ---
    full_data = loader.load_frcsub_dataset() 
    data = loader.get_student_subset(full_data, list(range(20)))
    
    # 获取数据集统计信息
    num_students = data.get('num_students', 0)
    num_problems = data.get('num_problems', 0)
    num_skills = data.get('num_skills', 0)
    
    print(f"   数据集规模: {num_students}学生 × {num_problems}题目 × {num_skills}技能")
    print(f"   技能列表:")
    for skill_id, skill_name in list(data['skills'].items())[:5]:
        print(f"     {skill_id}. {skill_name}")
    if len(data['skills']) > 5:
        print(f"     ... 共{len(data['skills'])}个技能")
    
    # 为了演示，使用前20个学生的数据
    subset_data = loader.get_student_subset(data, list(range(20)))
    print(f"   演示使用: {subset_data['num_students']}个学生的数据子集")
    
    # 2. 构建DKG
    print("\\n2. 构建动态知识图谱...")
    builder = DKGBuilder()
    dkg = builder.build_from_data(subset_data)
    
    # 分析图谱结构
    node_types = {}
    edge_types = {}
    
    for node, attr in dkg.nodes(data=True):
        node_type = attr.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    for u, v, attr in dkg.edges(data=True):
        edge_type = attr.get('type', 'unknown') 
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print(f"   图谱节点数: {dkg.number_of_nodes()}")
    print(f"   图谱边数: {dkg.number_of_edges()}")
    print(f"   节点类型分布: {node_types}")
    print(f"   关系类型分布: {edge_types}")
    
    # 3. 可视化展示
    print("\\n3. 生成交互式可视化图表...")
    # 使用传统matplotlib可视化
    visualizer = DKGVisualizer()
    visualizer.plot_knowledge_graph(dkg, layout='spring', show_labels=True)
    
    # 使用交互式Plotly可视化
    interactive_viz = InteractiveDKGVisualizer()
    interactive_viz.plot_interactive_knowledge_graph(dkg, layout='spring')
    interactive_viz.plot_interactive_skill_heatmap(dkg, student_ids=list(range(10)))
    
    return dkg, subset_data

def example_2_student_diagnosis():
    """
    示例2: 学生能力诊断
    基于少量题目推断学生技能掌握状态
    """
    print("\\n" + "=" * 60)
    print("示例2: 学生能力诊断")
    print("=" * 60)
    
    # 加载数据并构建DKG
    loader = DataLoader('../dataset')
    frcsub_data = loader.load_frcsub_dataset()
    subset_data = loader.get_student_subset(frcsub_data, list(range(10)))
    
    builder = DKGBuilder()
    dkg = builder.build_from_data(subset_data)
    
    # 选择一个学生进行诊断
    target_student = 3
    print(f"1. 对学生{target_student}进行能力诊断...")
    
    # 获取学生知识状态
    knowledge_state = builder.get_student_knowledge_state(target_student)
    
    print(f"\\n   学生{target_student}的技能掌握状态:")
    print(f"   {'技能名称':<30} {'掌握度':<10} {'置信度':<10}")
    print(f"   {'-'*50}")
    
    for skill_key, skill_info in knowledge_state.items():
        skill_name = skill_info['skill_name']
        mastery = skill_info['mastery_level']
        confidence = skill_info['confidence']
        
        # 评估等级
        if mastery >= 0.8:
            level = "优秀"
        elif mastery >= 0.6:
            level = "良好"
        elif mastery >= 0.4:
            level = "一般"
        else:
            level = "薄弱"
        
        print(f"   {skill_name:<30} {mastery:<10.3f} {confidence:<10.3f} [{level}]")
    
    # 识别薄弱技能
    weak_skills = []
    strong_skills = []
    
    for skill_key, skill_info in knowledge_state.items():
        if skill_info['mastery_level'] < 0.5:
            weak_skills.append((skill_info['skill_name'], skill_info['mastery_level']))
        elif skill_info['mastery_level'] > 0.8:
            strong_skills.append((skill_info['skill_name'], skill_info['mastery_level']))
    
    print(f"\\n2. 诊断结果分析:")
    print(f"   优势技能 ({len(strong_skills)}个):")
    for skill_name, mastery in strong_skills:
        print(f"     • {skill_name} (掌握度: {mastery:.3f})")
    
    print(f"\\n   薄弱技能 ({len(weak_skills)}个):")
    for skill_name, mastery in weak_skills:
        print(f"     • {skill_name} (掌握度: {mastery:.3f})")
    
    # 生成诊断报告
    print(f"\\n3. 学习建议:")
    if weak_skills:
        print(f"   建议优先学习以下技能:")
        # 按掌握度排序，最薄弱的优先
        weak_skills.sort(key=lambda x: x[1])
        for i, (skill_name, mastery) in enumerate(weak_skills[:3]):
            print(f"     {i+1}. {skill_name} - 当前掌握度仅为{mastery:.1%}")
    else:
        print(f"   该学生在所有基础技能上表现良好，可以进行进阶学习。")
    
    # 可视化学生学习轨迹
    print("\\n4. 生成学习轨迹可视化...")
    interactive_viz = InteractiveDKGVisualizer()
    interactive_viz.plot_interactive_learning_trajectory(dkg, target_student)
    
    return knowledge_state

def example_3_personalized_recommendation():
    """
    示例3: 个性化题目推荐
    为学生推荐合适的下一步学习题目
    """
    print("\\n" + "=" * 60)
    print("示例3: 个性化题目推荐")
    print("=" * 60)
    
    # 构建DKG
    loader = DataLoader('../dataset')
    frcsub_data = loader.load_frcsub_dataset()
    subset_data = loader.get_student_subset(frcsub_data, list(range(15)))
    
    builder = DKGBuilder()
    dkg = builder.build_from_data(subset_data)
    
    # 为多个学生生成推荐
    target_students = [0, 5, 10]
    
    for student_id in target_students:
        print(f"\\n1. 为学生{student_id}生成个性化推荐...")
        
        # 获取推荐
        recommendations = builder.recommend_next_problems(student_id, num_recommendations=5)
        
        if not recommendations:
            print(f"   未能为学生{student_id}生成推荐")
            continue
        
        print(f"   推荐题目列表:")
        print(f"   {'题目ID':<8} {'目标技能':<25} {'当前掌握度':<12} {'题目难度':<12} {'适合度':<10}")
        print(f"   {'-'*75}")
        
        for i, rec in enumerate(recommendations):
            print(f"   {rec['problem_id']:<8} {rec['skill_name'][:24]:<25} "
                  f"{rec['current_mastery']:<12.3f} {rec['problem_difficulty']:<12.3f} "
                  f"{rec['suitability_score']:<10.3f}")
        
        # 分析推荐策略
        skills_in_rec = [rec['skill_name'] for rec in recommendations]
        unique_skills = list(set(skills_in_rec))
        
        print(f"\\n   推荐策略分析:")
        print(f"     • 涉及{len(unique_skills)}个不同技能")
        print(f"     • 平均适合度: {np.mean([rec['suitability_score'] for rec in recommendations]):.3f}")
        print(f"     • 针对薄弱技能: {', '.join(unique_skills[:2])}")
        
        # 可视化推荐分析
        if student_id == target_students[0]:  # 只为第一个学生生成详细可视化
            print(f"\\n   生成交互式推荐分析图...")
            interactive_viz = InteractiveDKGVisualizer()
            interactive_viz.plot_interactive_recommendations(recommendations, student_id)

def example_4_learning_path_planning():
    """
    示例4: 学习路径规划
    基于技能依赖关系生成最优学习顺序
    """
    print("\\n" + "=" * 60)
    print("示例4: 学习路径规划")
    print("=" * 60)
    
    # 构建DKG
    loader = DataLoader('../dataset')
    frcsub_data = loader.load_frcsub_dataset()
    subset_data = loader.get_student_subset(frcsub_data, list(range(10)))
    
    builder = DKGBuilder()
    dkg = builder.build_from_data(subset_data)
    
    print("1. 分析技能依赖关系...")
    
    # 收集技能依赖信息
    skill_dependencies = {}
    skill_names = {}
    
    for node, attr in dkg.nodes(data=True):
        if attr.get('type') == 'skill':
            skill_id = attr.get('skill_id', 0)
            skill_name = attr.get('skill_name', f'Skill_{skill_id}')
            skill_names[node] = skill_name
            skill_dependencies[node] = []
    
    # 收集先修关系
    for u, v, attr in dkg.edges(data=True):
        if attr.get('type') == 'prerequisite':
            dependency_strength = attr.get('dependency_strength', 0.5)
            skill_dependencies[v].append((u, dependency_strength))
    
    print(f"   发现的技能依赖关系:")
    for skill_node, dependencies in skill_dependencies.items():
        if dependencies:
            skill_name = skill_names.get(skill_node, 'Unknown')
            print(f"     {skill_name}:")
            for dep_skill, strength in dependencies:
                dep_name = skill_names.get(dep_skill, 'Unknown')
                print(f"       ← {dep_name} (强度: {strength:.3f})")
    
    # 生成学习路径
    print(f"\\n2. 生成推荐学习路径...")
    
    # 简单的拓扑排序算法
    def topological_sort_skills():
        in_degree = {skill: 0 for skill in skill_names.keys()}
        
        # 计算入度
        for skill_node, dependencies in skill_dependencies.items():
            in_degree[skill_node] = len(dependencies)
        
        # 拓扑排序
        queue = [skill for skill, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # 更新后续技能的入度
            for skill_node, dependencies in skill_dependencies.items():
                for dep_skill, _ in dependencies:
                    if dep_skill == current:
                        in_degree[skill_node] -= 1
                        if in_degree[skill_node] == 0:
                            queue.append(skill_node)
        
        return result
    
    learning_path = topological_sort_skills()
    
    print(f"   推荐学习顺序:")
    for i, skill_node in enumerate(learning_path):
        skill_name = skill_names.get(skill_node, 'Unknown')
        print(f"     {i+1}. {skill_name}")
        
        # 找到该技能对应的题目
        skill_problems = []
        for u, v, attr in dkg.edges(data=True):
            if attr.get('type') == 'require' and v == skill_node:
                problem_node = u
                problem_id = dkg.nodes[problem_node].get('problem_id', 0)
                skill_problems.append(problem_id)
        
        if skill_problems:
            print(f"        推荐练习题目: {', '.join(map(str, sorted(skill_problems)))}")
    
    # 可视化技能依赖图
    print(f"\\n3. 生成技能依赖关系可视化...")
    # 注意：技能依赖图暂时使用传统matplotlib，因为需要特殊的层次布局
    visualizer = DKGVisualizer()
    visualizer.plot_skill_dependency_graph(dkg)
    
    return learning_path

def example_5_dynamic_update_simulation():
    """
    示例5: 动态更新模拟
    模拟学生学习过程中的知识状态更新
    """
    print("\\n" + "=" * 60)
    print("示例5: 动态更新模拟")
    print("=" * 60)
    
    # 构建DKG
    loader = DataLoader('../dataset')
    frcsub_data = loader.load_frcsub_dataset()
    subset_data = loader.get_student_subset(frcsub_data, list(range(5)))
    
    builder = DKGBuilder()
    dkg = builder.build_from_data(subset_data)
    
    # 选择一个学生进行模拟
    student_id = 2
    print(f"1. 模拟学生{student_id}的学习过程...")
    
    # 获取初始状态
    initial_state = builder.get_student_knowledge_state(student_id)
    print(f"\\n   初始技能状态:")
    for skill_key, skill_info in list(initial_state.items())[:3]:
        print(f"     {skill_info['skill_name']}: {skill_info['mastery_level']:.3f}")
    
    # 模拟学习交互
    simulated_interactions = [
        {'problem_id': 1, 'correct': 1, 'score': 1.0, 'time_taken': 120, 'hints_used': 0},
        {'problem_id': 3, 'correct': 0, 'score': 0.0, 'time_taken': 180, 'hints_used': 2},
        {'problem_id': 5, 'correct': 1, 'score': 1.0, 'time_taken': 90, 'hints_used': 0},
        {'problem_id': 7, 'correct': 1, 'score': 1.0, 'time_taken': 110, 'hints_used': 1},
    ]
    
    print(f"\\n2. 应用学习交互更新...")
    for i, interaction in enumerate(simulated_interactions):
        print(f"   交互{i+1}: 题目{interaction['problem_id']}, "
              f"正确性: {'✓' if interaction['correct'] else '✗'}, "
              f"用时: {interaction['time_taken']}秒")
        
        # 更新学生状态
        builder.update_student_state(student_id, interaction)
    
    # 获取更新后的状态
    updated_state = builder.get_student_knowledge_state(student_id)
    
    print(f"\\n3. 对比学习前后的变化:")
    print(f"   {'技能名称':<25} {'学习前':<10} {'学习后':<10} {'变化':<10}")
    print(f"   {'-'*55}")
    
    for skill_key in initial_state.keys():
        skill_name = initial_state[skill_key]['skill_name'][:24]
        before = initial_state[skill_key]['mastery_level']
        after = updated_state[skill_key]['mastery_level']
        change = after - before
        
        change_str = f"+{change:.3f}" if change > 0 else f"{change:.3f}"
        print(f"   {skill_name:<25} {before:<10.3f} {after:<10.3f} {change_str:<10}")
    
    # 重新生成推荐
    print(f"\\n4. 基于更新状态的新推荐:")
    new_recommendations = builder.recommend_next_problems(student_id, num_recommendations=3)
    
    for i, rec in enumerate(new_recommendations):
        print(f"   {i+1}. 题目{rec['problem_id']} - {rec['skill_name'][:20]} "
              f"(适合度: {rec['suitability_score']:.3f})")
    
    return initial_state, updated_state

def run_all_examples():
    """运行所有示例"""
    print("🚀 DKG MVP 应用示例演示")
    print("本演示将展示动态知识图谱在教育场景中的应用")
    
    try:
        # 示例1: 基础DKG构建
        dkg, data = example_1_basic_dkg_construction()
        
        # 示例2: 学生能力诊断
        knowledge_state = example_2_student_diagnosis()
        
        # 示例3: 个性化推荐
        example_3_personalized_recommendation()
        
        # 示例4: 学习路径规划
        learning_path = example_4_learning_path_planning()
        
        # 示例5: 动态更新模拟
        initial_state, updated_state = example_5_dynamic_update_simulation()
        
        # 创建综合仪表板
        print("\\n" + "=" * 60)
        print("🎨 创建综合交互式仪表板")
        print("=" * 60)
        
        # 重新构建DKG用于仪表板（使用更多学生数据）
        loader = DataLoader('../dataset')
        frcsub_data = loader.load_frcsub_dataset()
        dashboard_data = loader.get_student_subset(frcsub_data, list(range(20)))
        
        builder = DKGBuilder()
        dashboard_dkg = builder.build_from_data(dashboard_data)
        
        # 创建综合仪表板
        interactive_viz = InteractiveDKGVisualizer()
        dashboard_path = interactive_viz.create_dashboard(dashboard_dkg, target_student=5)
        
        print("\\n" + "=" * 60)
        print("🎉 所有示例演示完成！")
        print("=" * 60)
        print("总结:")
        print("✅ 成功构建了动态知识图谱")
        print("✅ 实现了学生能力诊断")
        print("✅ 生成了个性化题目推荐")
        print("✅ 规划了最优学习路径")
        print("✅ 模拟了动态状态更新")
        print("✅ 创建了交互式综合仪表板")
        print("\\n🌟 主要特色:")
        print("📊 交互式图表 - 支持缩放、悬停、点击等操作")
        print("💾 自动保存 - 所有图表自动保存为HTML文件")
        print("🎨 美观设计 - 现代化UI设计，专业视觉效果")
        print("📱 响应式 - 支持不同屏幕尺寸")
        print("🔍 详细信息 - 悬停显示详细数据")
        print("\\n💻 查看方式:")
        print(f"🌐 在浏览器中打开: {dashboard_path}")
        print("📁 或查看visualizations目录下的单独图表文件")
        print("\\nDKG MVP展示了将静态教育数据转换为动态知识图谱的完整流程，")
        print("为个性化学习和智能教育提供了强有力的技术支撑。")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        print("请检查数据集路径和依赖包是否正确安装。")

if __name__ == "__main__":
    # 设置matplotlib中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行所有示例
    run_all_examples()