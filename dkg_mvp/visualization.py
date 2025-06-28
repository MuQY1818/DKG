"""
可视化模块 - DKG图谱和学习分析可视化
"""
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

class DKGVisualizer:
    """DKG可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化器
        
        Args:
            figsize: 图片尺寸
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_knowledge_graph(self, graph: nx.MultiDiGraph, layout: str = 'spring', 
                           show_labels: bool = True, save_path: Optional[str] = None):
        """
        可视化知识图谱结构
        
        Args:
            graph: DKG图谱
            layout: 布局算法('spring', 'circular', 'kamada_kawai')
            show_labels: 是否显示节点标签
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 根据节点类型分类
        node_colors = {}
        node_sizes = {}
        color_map = {
            'student': '#FF6B6B',    # 红色
            'problem': '#4ECDC4',    # 青色
            'skill': '#45B7D1',      # 蓝色
            'session': '#96CEB4'     # 绿色
        }
        
        for node, attr in graph.nodes(data=True):
            node_type = attr.get('type', 'unknown')
            node_colors[node] = color_map.get(node_type, '#CCCCCC')
            
            # 节点大小基于度数
            degree = graph.degree(node)
            node_sizes[node] = min(50 + degree * 10, 300)
        
        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # 绘制不同类型的边
        edge_colors = {
            'solve': '#FFB6C1',      # 浅粉色
            'require': '#87CEEB',    # 天蓝色
            'master': '#DDA0DD',     # 梅花色
            'prerequisite': '#F0E68C', # 卡其色
            'similar': '#98FB98'     # 浅绿色
        }
        
        for edge_type, color in edge_colors.items():
            edges_of_type = [(u, v) for u, v, d in graph.edges(data=True) 
                           if d.get('type') == edge_type]
            if edges_of_type:
                nx.draw_networkx_edges(graph, pos, edgelist=edges_of_type,
                                     edge_color=color, alpha=0.6, width=1.5,
                                     arrows=True, arrowsize=10, ax=ax)
        
        # 绘制节点
        nx.draw_networkx_nodes(graph, pos, 
                             node_color=[node_colors[node] for node in graph.nodes()],
                             node_size=[node_sizes[node] for node in graph.nodes()],
                             alpha=0.8, ax=ax)
        
        # 绘制标签(仅显示技能节点的标签)
        if show_labels:
            skill_labels = {}
            for node, attr in graph.nodes(data=True):
                if attr.get('type') == 'skill':
                    skill_name = attr.get('skill_name', '')
                    if len(skill_name) > 15:  # 截断过长的标签
                        skill_name = skill_name[:15] + '...'
                    skill_labels[node] = skill_name
            
            nx.draw_networkx_labels(graph, pos, labels=skill_labels,
                                  font_size=8, ax=ax)
        
        # 添加图例
        legend_elements = []
        for node_type, color in color_map.items():
            if any(attr.get('type') == node_type for _, attr in graph.nodes(data=True)):
                legend_elements.append(mpatches.Patch(color=color, label=f'{node_type.title()} 节点'))
        
        for edge_type, color in edge_colors.items():
            if any(d.get('type') == edge_type for _, _, d in graph.edges(data=True)):
                legend_elements.append(mpatches.Patch(color=color, label=f'{edge_type.title()} 关系'))
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        ax.set_title('动态知识图谱结构', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_skill_mastery_heatmap(self, graph: nx.MultiDiGraph, student_ids: Optional[List[int]] = None,
                                  save_path: Optional[str] = None):
        """
        绘制学生技能掌握度热力图
        
        Args:
            graph: DKG图谱
            student_ids: 要显示的学生ID列表，None表示显示所有
            save_path: 保存路径
        """
        # 收集技能掌握数据
        mastery_data = []
        skill_names = {}
        
        # 获取所有技能信息
        for node, attr in graph.nodes(data=True):
            if attr.get('type') == 'skill':
                skill_id = attr.get('skill_id', 0)
                skill_name = attr.get('skill_name', f'Skill_{skill_id}')
                skill_names[node] = skill_name
        
        # 收集掌握度数据
        for node, attr in graph.nodes(data=True):
            if attr.get('type') == 'student':
                student_id = attr.get('student_id', 0)
                
                if student_ids is None or student_id in student_ids:
                    student_mastery = {'student_id': student_id}
                    
                    # 获取该学生的技能掌握情况
                    for edge in graph.out_edges(node, data=True):
                        if edge[2].get('type') == 'master':
                            skill_node = edge[1]
                            if skill_node in skill_names:
                                mastery_level = edge[2].get('mastery_level', 0.0)
                                student_mastery[skill_names[skill_node]] = mastery_level
                    
                    mastery_data.append(student_mastery)
        
        if not mastery_data:
            print("没有找到技能掌握数据")
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(mastery_data)
        df = df.set_index('student_id')
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(max(10, len(df.columns)), max(6, len(df) * 0.3)))
        
        # 使用自定义颜色映射
        colors = ['#ff4d4d', '#ff9999', '#ffcc99', '#99ff99', '#66ff66']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('mastery', colors, N=n_bins)
        
        sns.heatmap(df, annot=True, fmt='.2f', cmap=cmap, vmin=0, vmax=1,
                   cbar_kws={'label': '技能掌握度'}, ax=ax)
        
        ax.set_title('学生技能掌握度热力图', fontsize=16, fontweight='bold')
        ax.set_xlabel('技能', fontsize=12)
        ax.set_ylabel('学生ID', fontsize=12)
        
        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_learning_trajectory(self, graph: nx.MultiDiGraph, student_id: int,
                               save_path: Optional[str] = None):
        """
        绘制学生学习轨迹
        
        Args:
            graph: DKG图谱
            student_id: 学生ID
            save_path: 保存路径
        """
        student_node = f"student_{student_id}"
        
        if student_node not in graph:
            print(f"未找到学生{student_id}的数据")
            return
        
        # 收集学生的解题记录
        solve_records = []
        for edge in graph.out_edges(student_node, data=True):
            if edge[2].get('type') == 'solve':
                problem_node = edge[1]
                problem_id = graph.nodes[problem_node].get('problem_id', 0)
                correct = edge[2].get('correct', 0)
                score = edge[2].get('score', 0.0)
                
                solve_records.append({
                    'problem_id': problem_id,
                    'correct': correct,
                    'score': score
                })
        
        if not solve_records:
            print(f"学生{student_id}没有解题记录")
            return
        
        # 按题目ID排序(模拟时间顺序)
        solve_records.sort(key=lambda x: x['problem_id'])
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制正确率轨迹
        problems = [r['problem_id'] for r in solve_records]
        correct_rates = [r['correct'] for r in solve_records]
        scores = [r['score'] for r in solve_records]
        
        ax1.plot(problems, correct_rates, 'o-', color='#FF6B6B', linewidth=2, markersize=6)
        ax1.fill_between(problems, correct_rates, alpha=0.3, color='#FF6B6B')
        ax1.set_ylabel('正确率', fontsize=12)
        ax1.set_title(f'学生{student_id}的学习轨迹', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)
        
        # 绘制分数轨迹
        ax2.plot(problems, scores, 's-', color='#4ECDC4', linewidth=2, markersize=6)
        ax2.fill_between(problems, scores, alpha=0.3, color='#4ECDC4')
        ax2.set_xlabel('题目ID', fontsize=12)
        ax2.set_ylabel('得分', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.1, max(scores) * 1.1 if scores else 1.1)
        
        # 添加平均线
        if correct_rates:
            avg_correct = np.mean(correct_rates)
            ax1.axhline(y=avg_correct, color='red', linestyle='--', alpha=0.7,
                       label=f'平均正确率: {avg_correct:.2f}')
            ax1.legend()
        
        if scores:
            avg_score = np.mean(scores)
            ax2.axhline(y=avg_score, color='red', linestyle='--', alpha=0.7,
                       label=f'平均得分: {avg_score:.2f}')
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_skill_dependency_graph(self, graph: nx.MultiDiGraph, save_path: Optional[str] = None):
        """
        绘制技能依赖关系图
        
        Args:
            graph: DKG图谱
            save_path: 保存路径
        """
        # 创建技能子图
        skill_graph = nx.DiGraph()
        
        # 添加技能节点
        skill_nodes = {}
        for node, attr in graph.nodes(data=True):
            if attr.get('type') == 'skill':
                skill_id = attr.get('skill_id', 0)
                skill_name = attr.get('skill_name', f'Skill_{skill_id}')
                skill_graph.add_node(node, skill_name=skill_name, skill_id=skill_id)
                skill_nodes[node] = skill_name
        
        # 添加先修关系边
        for u, v, attr in graph.edges(data=True):
            if attr.get('type') == 'prerequisite':
                if u in skill_nodes and v in skill_nodes:
                    strength = attr.get('dependency_strength', 0.5)
                    skill_graph.add_edge(u, v, weight=strength)
        
        if skill_graph.number_of_nodes() == 0:
            print("没有找到技能节点")
            return
        
        # 创建图表
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 使用层次布局
        try:
            pos = nx.nx_agraph.graphviz_layout(skill_graph, prog='dot')
        except:
            # 如果graphviz不可用，使用其他布局
            pos = nx.spring_layout(skill_graph, k=3, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(skill_graph, pos, node_color='#45B7D1',
                             node_size=1000, alpha=0.8, ax=ax)
        
        # 绘制边，粗细代表依赖强度
        edges = skill_graph.edges(data=True)
        if edges:
            weights = [d.get('weight', 0.5) for u, v, d in edges]
            nx.draw_networkx_edges(skill_graph, pos, width=[w*3 for w in weights],
                                 edge_color='#FFB6C1', alpha=0.6, arrows=True,
                                 arrowsize=20, ax=ax)
        
        # 绘制标签
        labels = {}
        for node in skill_graph.nodes():
            skill_name = skill_nodes.get(node, '')
            if len(skill_name) > 10:
                skill_name = skill_name[:10] + '...'
            labels[node] = skill_name
        
        nx.draw_networkx_labels(skill_graph, pos, labels, font_size=8, ax=ax)
        
        ax.set_title('技能依赖关系图', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # 添加说明
        ax.text(0.02, 0.98, '箭头方向: 先修技能 → 后续技能\n边的粗细: 依赖强度',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_recommendation_analysis(self, recommendations: List[Dict], student_id: int,
                                   save_path: Optional[str] = None):
        """
        绘制推荐分析图
        
        Args:
            recommendations: 推荐结果列表
            student_id: 学生ID
            save_path: 保存路径
        """
        if not recommendations:
            print("没有推荐数据")
            return
        
        # 准备数据
        problems = [f"题目{r['problem_id']}" for r in recommendations]
        suitability = [r['suitability_score'] for r in recommendations]
        mastery = [r['current_mastery'] for r in recommendations]
        difficulty = [r['problem_difficulty'] for r in recommendations]
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 适合度得分
        bars1 = ax1.bar(problems, suitability, color='#FF6B6B', alpha=0.7)
        ax1.set_title('题目适合度得分', fontsize=14, fontweight='bold')
        ax1.set_ylabel('适合度得分', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, score in zip(bars1, suitability):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 当前掌握度vs题目难度
        x_pos = np.arange(len(problems))
        width = 0.35
        
        bars2 = ax2.bar(x_pos - width/2, mastery, width, label='当前掌握度', 
                       color='#4ECDC4', alpha=0.7)
        bars3 = ax2.bar(x_pos + width/2, difficulty, width, label='题目难度',
                       color='#45B7D1', alpha=0.7)
        
        ax2.set_title('掌握度 vs 题目难度', fontsize=14, fontweight='bold')
        ax2.set_ylabel('水平', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(problems)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. 技能分布饼图
        skills = [r['skill_name'] for r in recommendations]
        skill_counts = pd.Series(skills).value_counts()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(skill_counts)))
        ax3.pie(skill_counts.values, labels=skill_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax3.set_title('推荐题目的技能分布', fontsize=14, fontweight='bold')
        
        # 4. 推荐原因文本
        ax4.axis('off')
        ax4.set_title('推荐原因', fontsize=14, fontweight='bold')
        
        reasons_text = f"为学生{student_id}的个性化推荐:\n\n"
        for i, rec in enumerate(recommendations):
            reasons_text += f"{i+1}. {rec['recommendation_reason']}\n"
            reasons_text += f"   适合度: {rec['suitability_score']:.2f}\n\n"
        
        ax4.text(0.05, 0.95, reasons_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle(f'学生{student_id}的题目推荐分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_student_knowledge_radar(self, dkg_builder, student_id: int, top_n: int = 6, save_path: Optional[str] = None):
        """
        为单个学生绘制知识状态雷达图

        Args:
            dkg_builder: DKGBuilder 实例
            student_id: 学生ID
            top_n: 显示的技能数量（默认6个最薄弱的）
            save_path: 保存路径
        """
        student_state = dkg_builder.get_student_knowledge_state(student_id)
        if not student_state:
            print(f"未找到学生 {student_id} 的知识状态。")
            return

        # 筛选掌握度并排序，找到最薄弱的N个技能
        skills = sorted(student_state.items(), key=lambda item: item[1]['mastery_level'])
        
        # 如果技能总数少于top_n，则全部显示
        top_skills = skills[:top_n]
        
        if not top_skills:
            print(f"学生 {student_id} 没有可显示的技能。")
            return

        labels = [item[1]['skill_name'] for item in top_skills]
        values = [item[1]['mastery_level'] for item in top_skills]

        # 雷达图需要闭合，所以将第一个点的数据追加到末尾
        labels_radar = labels + [labels[0]]
        values_radar = values + [values[0]]
        
        # 设置雷达图的角度
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles_radar = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # 绘制数据
        ax.plot(angles_radar, values_radar, color='#4ECDC4', linewidth=2, linestyle='solid')
        ax.fill(angles_radar, values_radar, color='#4ECDC4', alpha=0.25)
        
        # 设置坐标轴标签
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=12)

        # 设置雷达图范围和标题
        ax.set_ylim(0, 1.0)
        ax.set_rlabel_position(22.5) # 标签位置
        ax.grid(True)
        
        ax.set_title(f'学生 {student_id} 知识状态雷达图 (Top {len(labels)} 薄弱技能)', size=16, color='black', y=1.1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()

def main():
    """测试可视化功能"""
    # 这个main函数需要一个已经构建好的DKG实例
    # 在实际使用中，这个函数会从其他模块接收dkg对象
    # 此处仅为演示目的，创建一个简单的示例图谱
    
    print("这是一个演示可视化功能的脚本。")
    print("请从其他模块（如dkg_builder.py）调用DKGVisualizer中的方法。")
    
    # 示例:
    # from data_loader import DataLoader
    # from dkg_builder import DKGBuilder
    # loader = DataLoader('../dataset')
    # data = loader.load_frcsub_dataset()
    # builder = DKGBuilder()
    # dkg = builder.build_from_data(data)
    # visualizer = DKGVisualizer()
    # visualizer.plot_skill_mastery_heatmap(dkg, student_ids=[0, 1, 2])

if __name__ == "__main__":
    main()