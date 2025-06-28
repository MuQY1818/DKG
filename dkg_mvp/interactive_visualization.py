import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
from pyvis.network import Network
import pandas as pd
import os
import random

class InteractiveDKGVisualizer:
    """生成DKG的交互式可视化图表"""
    
    def __init__(self, output_dir: str = 'dkg_mvp/visualizations'):
        """
        初始化可视化器
        
        Args:
            output_dir: 可视化文件输出目录
        """
        self.output_dir = output_dir
        self.colors = {
            'student': '#636EFA',
            'problem': '#EF553B',
            'skill': '#00CC96',
            'prerequisite': '#FFA15A',
            'similar': '#AB63FA',
            'mastery_high': '#1f77b4',
            'mastery_low': '#aec7e8'
        }
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_interactive_skill_network(self, graph: nx.MultiDiGraph, save_path: Optional[str] = None) -> str:
        """
        生成一个可拖拽的技能网络图 (使用 pyvis)
        
        Args:
            graph: DKG图谱
            save_path: HTML文件保存路径
            
        Returns:
            HTML文件路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "interactive_skill_network.html")
            
        skill_graph = nx.DiGraph()
        
        # 提取技能节点和关系
        for node, attr in graph.nodes(data=True):
            if attr.get('type') == 'skill':
                skill_graph.add_node(node, label=attr.get('skill_name', ''), title=f"ID: {attr.get('skill_id')}")
        
        for u, v, attr in graph.edges(data=True):
            if attr.get('type') in ['prerequisite', 'similar']:
                if u in skill_graph and v in skill_graph:
                    skill_graph.add_edge(u, v, 
                                         title=f"{attr.get('type')}: {attr.get('dependency_strength', attr.get('similarity_score', 0)):.2f}",
                                         type=attr.get('type'))

        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False, directed=True)
        net.from_nx(skill_graph)
        
        # 自定义样式
        for node in net.nodes:
            node['color'] = self.colors['skill']
        
        for edge in net.edges:
            if edge['type'] == 'prerequisite':
                edge['color'] = self.colors['prerequisite']
                edge['arrows'] = 'to'
            elif edge['type'] == 'similar':
                edge['color'] = self.colors['similar']
                edge['arrows'] = ''

        net.save_graph(save_path)

        # CSS Fix: Inject style to make the graph fill the container
        with open(save_path, 'r+', encoding='utf-8') as f:
            content = f.read()
            style_injection = """
            <style>
                html, body {
                    margin: 0;
                    padding: 0;
                }
                #mynetwork {
                    width: 100vw;
                    height: 100vh;
                }
            </style>
            """
            content = content.replace("</head>", f"{style_injection}</head>")
            f.seek(0)
            f.write(content)
            f.truncate()

        print(f"🕸️ 可拖拽技能网络图已保存到: {save_path}")
        return save_path

    def plot_interactive_student_radar(self, dkg_builder, student_id: int, top_n: int = 8, save_path: Optional[str] = None) -> go.Figure:
        """
        生成交互式学生知识雷达图
        
        Args:
            dkg_builder: DKGBuilder实例
            student_id: 学生ID
            top_n: 显示掌握度最高和最低的技能数量
            save_path: HTML文件保存路径
            
        Returns:
            Plotly Figure对象
        """
        knowledge_state = dkg_builder.get_student_knowledge_state(student_id)
        
        if not knowledge_state:
            print(f"学生 {student_id} 的知识状态为空。")
            # 返回一个空的figure, 但在dashboard中会显示一个消息
            return go.Figure()

        skills = sorted(knowledge_state.items(), key=lambda x: x[1]['mastery_level'], reverse=True)
        
        # 选取掌握度最高和最低的技能
        if len(skills) > top_n * 2:
            top_skills = skills[:top_n]
            bottom_skills = skills[-top_n:]
            display_skills = dict(top_skills + bottom_skills)
        else:
            display_skills = dict(skills)
            
        labels = [info['skill_name'] for info in display_skills.values()]
        values = [info['mastery_level'] for info in display_skills.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name='掌握度',
            marker_color=self.colors['student']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
            ),
            title=f"学生 {student_id} 知识状态雷达图",
            title_x=0.5,
            margin=dict(l=40, r=40, t=80, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#333'
        )
        
        if save_path:
            # 只写入body部分, 以便嵌入iframe
            fig.write_html(save_path, full_html=False, include_plotlyjs='cdn')
            # print(f"📈 交互式学生雷达图已保存到: {save_path}")

        return fig
    
    def _create_student_subgraph_pyvis(self, graph: nx.MultiDiGraph, student_id: int, max_interactions: int = 30):
        """为单个学生创建可拖拽的局部知识图谱 (pyvis), 只显示部分交互以提高性能"""
        student_node = f"student_{student_id}"
        if student_node not in graph:
            return None
        
        net = Network(height="100%", width="100%", bgcolor="#ffffff", font_color="#333", notebook=False, directed=True)
        
        # 添加学生节点
        net.add_node(student_node, 
                     label=f"学生 {student_id}", 
                     title=f"ID: {student_id}", 
                     color=self.colors['student'], 
                     size=25)

        # 提取所有 'master' 和 'solve' 关系
        master_edges = []
        solve_edges = []
        for u, v, attr in graph.edges(student_node, data=True):
            if attr.get('type') == 'master':
                master_edges.append((u, v, attr))
            elif attr.get('type') == 'solve':
                solve_edges.append((u, v, attr))

        # 如果交互太多，则进行采样
        if len(solve_edges) > max_interactions:
            solve_edges = random.sample(solve_edges, max_interactions)

        # 处理 'master' 关系 (全部显示)
        for u, v, attr in master_edges:
            skill_attr = graph.nodes[v]
            net.add_node(v, label=skill_attr.get('skill_name'), title=f"技能ID: {skill_attr.get('skill_id')}", color=self.colors['skill'], size=15)
            
            mastery_level = attr.get('mastery_level', 0)
            title = f"最终掌握度: {mastery_level:.2f}"
            net.add_edge(u, v, title=title, color=self.colors['prerequisite'], dashes=True, width=1 + mastery_level * 2)

        # 处理 'solve' 关系 (采样后)
        for u, v, attr in solve_edges:
            problem_attr = graph.nodes[v]
            # 确保题目节点存在
            if v not in net.get_nodes():
                 net.add_node(v, label=f"题目 {problem_attr.get('problem_id')}", title=f"题目ID: {problem_attr.get('problem_id')}", color=self.colors['problem'], size=10)
            
            is_correct = attr.get('correct')
            title = f"解题: {'正确' if is_correct else '错误'}"
            edge_color = '#00CC96' if is_correct else '#EF553B'
            net.add_edge(u, v, title=title, color=edge_color, width=2)
            
            # 从题目找到其需要的技能
            for _, p_node, p_attr in graph.edges(v, data=True):
                if p_attr.get('type') == 'require':
                    # 确保技能节点已经添加, 避免与master关系重复添加
                    if p_node not in net.get_nodes():
                        skill_attr = graph.nodes[p_node]
                        net.add_node(p_node, label=skill_attr.get('skill_name'), title=f"技能ID: {skill_attr.get('skill_id')}", color=self.colors['skill'], size=15)
                    
                    net.add_edge(v, p_node, title="要求技能", color="#AB63FA", width=1)

        return net

    def plot_dkg_snapshot(self, builder, student_id: int, step_number: int, save_path: Optional[str] = None):
        """
        为学生在特定模拟步骤生成一个DKG快照可视化。
        
        Args:
            builder: DKGBuilder的实例
            student_id: 学生ID
            step_number: 模拟的步骤编号
            save_path: HTML文件保存路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"dkg_snapshot_student_{student_id}_step_{step_number}.html")

        student_node = f"student_{student_id}"
        graph = builder.graph
        
        if student_node not in graph:
            print(f"学生 {student_id} 不存在于图谱中。")
            return

        net = Network(height="750px", width="100%", bgcolor="#f0f2f5", font_color="#333", notebook=False, directed=True)

        # 添加学生节点
        net.add_node(student_node, label=f"学生 {student_id}", color=self.colors['student'], size=30, shape='star')

        # 添加所有技能节点，并根据掌握度设置样式
        knowledge_state = builder.get_student_knowledge_state(student_id)
        
        for skill_key, info in knowledge_state.items():
            skill_node = skill_key # e.g., "skill_1"
            mastery = info['mastery_level']
            
            # 掌握度越高，节点越大越不透明
            size = 15 + mastery * 20
            color_rgba = f"rgba(0, 204, 150, {0.3 + mastery * 0.7})" # 'skill' color #00CC96

            net.add_node(skill_node, 
                         label=info['skill_name'], 
                         title=f"掌握度: {mastery:.2f}",
                         size=size,
                         color=color_rgba)
            
            # 添加 master 关系边
            net.add_edge(student_node, skill_node, color="#cccccc", width=1 + mastery * 3)
            
        # 添加技能间的先修关系
        for u, v, attr in graph.edges(data=True):
            if attr.get('type') == 'prerequisite':
                if u in net.get_nodes() and v in net.get_nodes():
                    net.add_edge(u, v, color=self.colors['prerequisite'], arrows="to", width=2)
                    
        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3000,
              "centralGravity": 0.1,
              "springLength": 150
            },
            "minVelocity": 0.75
          }
        }
        """)
        net.save_graph(save_path)
        print(f"📸 DKG 快照已保存到: {save_path}")

    def create_dashboard(self, builder, graph: nx.MultiDiGraph, student_ids: List[int]) -> str:
        """
        创建包含多个交互式视图的HTML仪表盘
        """
        print("🚀 开始创建交互式仪表盘...")
        if not student_ids:
            print("⚠️ 警告: 未提供学生ID，无法创建仪表盘。")
            return ""

        components_dir = os.path.join(self.output_dir, "dkg_dashboard_components")
        os.makedirs(components_dir, exist_ok=True)

        for student_id in student_ids:
            # print(f"  - 正在处理学生 {student_id}...")
            
            # 1. 生成雷达图
            radar_save_path = os.path.join(components_dir, f"radar_student_{student_id}.html")
            fig = self.plot_interactive_student_radar(builder, student_id, save_path=radar_save_path)
            if not fig.data: # 如果figure为空
                 with open(radar_save_path, 'w', encoding='utf-8') as f:
                    f.write(f"<p>无法为学生 {student_id} 生成雷达图。</p>")

            # 2. 生成推荐列表
            recs = builder.recommend_next_problems(student_id, num_recommendations=5)
            rec_save_path = os.path.join(components_dir, f"recs_student_{student_id}.html")
            if recs:
                rec_df = pd.DataFrame(recs)
                rec_fig = go.Figure(data=[go.Table(
                    header=dict(values=['题目ID', '推荐理由', '适合度'],
                                fill_color=self.colors['student'], align='left', font=dict(color='white')),
                    cells=dict(values=[rec_df.problem_id, rec_df.recommendation_reason, rec_df.suitability_score.round(3)],
                               fill_color='#F8F9FA', align='left'))
                ])
                rec_fig.update_layout(title_text=f"为学生 {student_id} 的学习推荐", title_x=0.5, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                rec_fig.write_html(rec_save_path, full_html=False, include_plotlyjs='cdn')
            else:
                with open(rec_save_path, 'w', encoding='utf-8') as f:
                    f.write(f"<html><body style='color:#333;font-family:sans-serif;text-align:center;padding-top:20px;'><p>没有为学生 {student_id} 生成推荐。</p></body></html>")
            
            # 3. 生成可拖拽的局部图谱
            pyvis_net = self._create_student_subgraph_pyvis(graph, student_id)
            if pyvis_net:
                save_path = os.path.join(components_dir, f"subgraph_student_{student_id}.html")
                pyvis_net.save_graph(save_path)
                
                # CSS Fix: Inject style to make the graph fill the iframe
                with open(save_path, 'r+', encoding='utf-8') as f:
                    content = f.read()
                    style_injection = """
                    <style>
                        html, body {
                            margin: 0;
                            padding: 0;
                        }
                        #mynetwork {
                            width: 100vw;
                            height: 100vh;
                        }
                    </style>
                    """
                    content = content.replace("</head>", f"{style_injection}</head>")
                    f.seek(0)
                    f.write(content)
                    f.truncate()
            else:
                save_path = os.path.join(components_dir, f"subgraph_student_{student_id}.html")
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(f"<html><body style='color:#333;font-family:sans-serif;text-align:center;padding-top:20px;'><p>无法为学生 {student_id} 生成局部知识图谱。</p></body></html>")


        dashboard_path = os.path.join(self.output_dir, "dkg_dashboard.html")
        default_student = student_ids[0]

        html_template = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>学生知识图谱仪表盘</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; margin: 0; padding: 20px; background-color: #f0f2f5; }}
                h1 {{ text-align: center; color: #333; }}
                .container {{ display: flex; flex-direction: column; height: 90vh; }}
                .controls {{ margin-bottom: 20px; text-align: center; background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
                .panel {{ display: flex; flex: 1; gap: 20px; }}
                .left-panel {{ flex: 2; }}
                .right-panel {{ flex: 3; display: flex; flex-direction: column; gap: 20px; }}
                .card {{ background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 15px; overflow: hidden; display: flex; flex-direction: column;}}
                .card.full-height {{ flex: 1; }}
                .iframe-container {{ width: 100%; height: 100%; min-height: 300px; border: none; }}
                select {{ padding: 8px 12px; border-radius: 4px; border: 1px solid #ccc; }}
            </style>
        </head>
        <body>
            <h1>学生知识图谱仪表盘</h1>
            <div class="controls">
                <label for="student-select">选择学生:</label>
                <select id="student-select" onchange="updateDashboard()">
                    {"".join([f'<option value="{sid}">{sid}</option>' for sid in student_ids])}
                </select>
                </div>
    <div class="container">
                <div class="panel">
                    <div class="left-panel card">
                        <iframe id="subgraph-frame" class="iframe-container" src="dkg_dashboard_components/subgraph_student_{default_student}.html"></iframe>
                    </div>
                    <div class="right-panel">
                        <div class="card">
                            <iframe id="radar-frame" class="iframe-container" src="dkg_dashboard_components/radar_student_{default_student}.html"></iframe>
                        </div>
                        <div class="card full-height">
                            <iframe id="recs-frame" class="iframe-container" src="dkg_dashboard_components/recs_student_{default_student}.html"></iframe>
                        </div>
                        </div>
                </div>
            </div>
            <script>
                function updateDashboard() {{
                    const studentId = document.getElementById('student-select').value;
                    const basePath = 'dkg_dashboard_components/';
                    document.getElementById('subgraph-frame').src = `${{basePath}}subgraph_student_${{studentId}}.html`;
                    document.getElementById('radar-frame').src = `${{basePath}}radar_student_${{studentId}}.html`;
                    document.getElementById('recs-frame').src = `${{basePath}}recs_student_${{studentId}}.html`;
                }}
            </script>
        </body>
        </html>
        """
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"🎉 交互式仪表盘已成功创建: {dashboard_path}")
        return dashboard_path

    def plot_simulation_history(self, history_df: pd.DataFrame, student_id: int, save_path: Optional[str] = None):
        """
        可视化学习模拟过程中学生知识掌握度的变化。

        Args:
            history_df: 由SimulationEngine生成的历史记录DataFrame。
            student_id: 学生ID。
            save_path: HTML文件保存路径。
        """
        if history_df.empty:
            print("⚠️ 历史记录为空，无法生成可视化图表。")
            return

        fig = go.Figure()
        
        # 为了图表清晰，只选择变化最明显的几个技能进行展示
        # 计算每个技能的掌握度变化范围
        delta = history_df.max() - history_df.min()
        top_n_skills = delta.nlargest(10).index # 选择变化最大的10个技能

        for skill_name in top_n_skills:
            if skill_name in history_df.columns:
                fig.add_trace(go.Scatter(
                    x=history_df.index, 
                    y=history_df[skill_name],
                    mode='lines+markers',
                    name=skill_name
                ))
            
        fig.update_layout(
            title=f"学生 {student_id} 知识掌握度演进模拟",
            xaxis_title="模拟步骤",
            yaxis_title="技能掌握度",
            yaxis_range=[0, 1],
            legend_title="技能",
            hovermode="x unified"
        )
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"simulation_history_student_{student_id}.html")
        
        fig.write_html(save_path)
        print(f"📈 模拟历史可视化图表已保存到: {save_path}")
        
        return fig