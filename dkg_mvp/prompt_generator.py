"""
LLM Prompt Generator

该模块负责将学生的结构化数据（如知识画像、行为分析等）
转换为一个信息丰富、结构清晰的文本提示（Prompt），
以便大型语言模型（LLM）能够根据此提示为学生进行诊断或规划。
"""

from typing import Dict, List, Any

def format_skill_profile(profile: List[Dict[str, Any]]) -> str:
    """格式化学生的技能画像，分为强项和弱项。"""
    
    strong_skills = sorted(
        [s for s in profile if s['predicted_mastery'] >= 0.7], 
        key=lambda x: x['predicted_mastery'], 
        reverse=True
    )
    weak_skills = sorted(
        [s for s in profile if s['predicted_mastery'] < 0.7], 
        key=lambda x: x['predicted_mastery']
    )

    prompt_parts = []

    if strong_skills:
        prompt_parts.append("学生掌握较好的知识点 (按掌握度从高到低):")
        for skill in strong_skills:
            prompt_parts.append(f"- {skill['skill_name']} (预测掌握度: {skill['predicted_mastery']:.2%})")
    
    if weak_skills:
        prompt_parts.append("\n需要重点关注和提升的知识点 (按掌握度从低到高):")
        for skill in weak_skills:
            prompt_parts.append(f"- {skill['skill_name']} (预测掌握度: {skill['predicted_mastery']:.2%})")
            
    return "\n".join(prompt_parts)

def generate_learning_path_prompt(
    student_id: int, 
    student_analytics: Dict[str, Any], 
    skill_profile: List[Dict[str, Any]]
) -> str:
    """
    根据学生的分析数据和技能画像，生成用于学习路径规划的LLM提示。

    Args:
        student_id (int): 学生ID。
        student_analytics (dict): 学生的行为分析数据。
        skill_profile (list): 学生的详细技能画像列表。

    Returns:
        str: 一个结构化的、信息丰富的文本提示。
    """
    
    # 格式化技能画像文本
    formatted_profile = format_skill_profile(skill_profile)
    weakest_skill = sorted([s for s in skill_profile if s['predicted_mastery'] < 0.7], key=lambda x: x['predicted_mastery'])
    weakest_skill_name = weakest_skill[0]['skill_name'] if weakest_skill else "暂无明显薄弱知识点"

    # 构建提示的各个部分
    prompt = f"""
# AI私教任务：为学生规划下一步学习路径

## 1. 角色定义
你是一位专业的AI私教，擅长根据学生的认知诊断数据，为他们提供个性化的学习建议和路径规划。你的任务是清晰、专业、有建设性。

## 2. 学生背景数据
- **学生ID**: {student_id}
- **学生整体学习行为分析**:
  - 总体正确率: {student_analytics.get('overall_accuracy', 'N/A'):.2%}
  - 平均首次作答时间: {student_analytics.get('avg_response_time', 'N/A'):.2f} 毫秒
  - 平均提示使用次数: {student_analytics.get('avg_hint_usage', 'N/A'):.2f} 次/题

## 3. 学生知识掌握度画像
{formatted_profile}

## 4. 你的任务
基于以上数据，请完成以下任务：

1.  **综合诊断**: 简要总结该学生的整体学习状况，包括他们的学习风格（例如，是粗心但反应快，还是谨慎但速度慢？）和知识结构特点。
2.  **确定核心问题**: 明确指出当前学生最需要提升的核心知识点是什么。根据数据，目前最薄弱的知识点是 **"{weakest_skill_name}"**。
3.  **制定学习计划**: 
    - 为该学生设计一个简短、可执行的下一步学习计划，专注于攻克最薄弱的知识点。
    - 提出2-3条具体的、有针对性的学习建议。
    - 推荐3个不同难度层次（简单、中等、巩固）的相关练习题的**题型**（注意：你不需要知道具体的题目ID，只需要描述题型即可），来帮助学生巩愈这个知识点。

请以专业、鼓励的语气，直接输出你的诊断和学习计划。
"""
    return prompt.strip()

if __name__ == '__main__':
    # --- 用于测试的伪数据 ---
    mock_student_id = 78011
    mock_analytics = {
        'overall_accuracy': 0.65,
        'avg_response_time': 25000.0,
        'avg_hint_usage': 1.5
    }
    mock_profile = [
        {'skill_id': 101, 'skill_name': '一元二次方程求解', 'predicted_mastery': 0.92},
        {'skill_id': 102, 'skill_name': '韦达定理的应用', 'predicted_mastery': 0.85},
        {'skill_id': 103, 'skill_name': '相似三角形判定', 'predicted_mastery': 0.65},
        {'skill_id': 104, 'skill_name': '勾股定理逆定理', 'predicted_mastery': 0.45},
        {'skill_id': 105, 'skill_name': '辅助线的作法', 'predicted_mastery': 0.30},
    ]

    # 生成并打印提示
    generated_prompt = generate_learning_path_prompt(mock_student_id, mock_analytics, mock_profile)
    print("--- Generated LLM Prompt ---")
    print(generated_prompt) 