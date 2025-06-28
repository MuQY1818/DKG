import pandas as pd
from typing import Dict, List, Tuple, Optional
from .dkg_builder import DKGBuilder
import random
import numpy as np
from .interactive_visualization import InteractiveDKGVisualizer

class SimulationEngine:
    """自主学习智能体模拟引擎"""
    
    def __init__(self, builder: DKGBuilder, all_problems: Dict):
        self.builder = builder
        self.all_problems = all_problems # {problem_id: [skill_id]}
        self.student_state = {}

    def _update_internal_student_state(self, student_id: int):
        """缓存并更新智能体内部对自身状态的认知"""
        self.student_state = self.builder.get_student_knowledge_state(student_id)

    def _passive_knowledge_decay(self, student_id: int, practiced_skill_id: Optional[int] = None):
        """模拟被动知识遗忘"""
        student_node = f"student_{student_id}"
        decay_factor = 0.998
        
        for skill_key, info in self.student_state.items():
            skill_id = int(skill_key.split('_')[1])
            if skill_id == practiced_skill_id:
                continue # 刚练习过的技能不遗忘
                
            skill_node = f"skill_{skill_id}"
            if self.builder.graph.has_edge(student_node, skill_node):
                 edge_attr = next(iter(self.builder.graph[student_node][skill_node].values()))
                 edge_attr['mastery_level'] *= decay_factor

    def _choose_next_action(self, student_id: int) -> int:
        """智能体根据心智模型自主选择下一个要做的题目"""
        student_node = f"student_{student_id}"
        student_attrs = self.builder.graph.nodes[student_node]
        
        # 70% 常规模式, 20% 复习模式, 10% 探索模式 (受好奇心影响)
        mode_roll = random.random()
        
        # 1. 常规模式: 挑战最薄弱的、可学习的技能
        if mode_roll < (0.7 - student_attrs.get('curiosity', 0.1)):
            weakest_skills = sorted(self.student_state.items(), key=lambda x: x[1]['mastery_level'])
            
            for skill_key, _ in weakest_skills:
                skill_id = int(skill_key.split('_')[1])
                # 检查先修条件是否满足
                prereqs = self.builder.graph.predecessors(f"skill_{skill_id}")
                prereqs_met = all(self.student_state.get(f"skill_{p_id}", {}).get('mastery_level', 0) > 0.5 
                                  for p in prereqs if self.builder.graph.nodes[p].get('type') == 'skill'
                                  for p_id in [self.builder.graph.nodes[p].get('skill_id')])
                
                if prereqs_met:
                    # 找到该技能对应的所有题目
                    candidate_problems = [p_id for p_id, s_ids in self.all_problems.items() if skill_id in s_ids]
                    if candidate_problems:
                        return random.choice(candidate_problems)

        # 2. 复习模式: 巩固已学习但可能遗忘的技能
        elif mode_roll < 0.9:
            learned_skills = [s for s, i in self.student_state.items() if 0.1 < i['mastery_level'] < 0.95]
            if learned_skills:
                skill_to_review_key = random.choice(learned_skills)
                skill_id = int(skill_to_review_key.split('_')[1])
                candidate_problems = [p_id for p_id, s_ids in self.all_problems.items() if skill_id in s_ids]
                if candidate_problems:
                    return random.choice(candidate_problems)
        
        # 3. 探索模式 (或所有其他情况的回退)
        return random.choice(list(self.all_problems.keys()))


    def run_simulation(self, student_id: int, num_steps: int = 90) -> pd.DataFrame:
        history = []
        student_node = f"student_{student_id}"

        for step in range(num_steps):
            self._update_internal_student_state(student_id)
            
            # 模拟被动遗忘
            self._passive_knowledge_decay(student_id)

            # 智能体自主选择题目
            problem_id = self._choose_next_action(student_id)
            
            # ... (接下来的逻辑与之前类似：概率判断、顿悟、更新图谱)
            problem_node = f"problem_{problem_id}"
            required_skill_ids = self.all_problems.get(problem_id, [])
            
            # (为简化, 我们只考虑第一个技能)
            mastery = 0.0
            if required_skill_ids:
                skill_id = required_skill_ids[0]
                mastery = self.student_state.get(f"skill_{skill_id}", {}).get('mastery_level', 0)

            base_success_rate = 0.05
            correct = random.random() < (base_success_rate + mastery * (1 - base_success_rate))

            is_epiphany = correct and mastery < 0.4 and random.random() < 0.2
            
            interaction = {
                'problem_id': problem_id, 'correct': correct, 'score': float(correct), 'is_epiphany': is_epiphany
            }
            self.builder.update_student_state(student_id, interaction)
            
            # 记录历史
            current_state_for_history = self.builder.get_student_knowledge_state(student_id)
            flat_state = {info['skill_name']: info['mastery_level'] for _, info in current_state_for_history.items()}
            flat_state['step'] = step
            history.append(flat_state)

        return pd.DataFrame(history).set_index('step') 