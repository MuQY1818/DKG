import os
import sys
import uvicorn
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

import torch
import numpy as np

# 将项目根目录添加到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dkg_mvp.data_loader import DataLoader
from dkg_mvp.orcdf.model import ORCDF
from dkg_mvp.train_orcdf import to_sparse_tensor
from dkg_mvp.analytics import StudentAnalytics # 引入分析模块
from dkg_mvp.prompt_generator import generate_learning_path_prompt # 引入提示生成器

# --- 全局变量 ---
MODELS_DIR = "models"
MODEL_NAME = "orcdf_best_model_seed42.pt" # 训练好的模型文件名
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

app = FastAPI(
    title="ORCDF 认知诊断 API",
    description="基于ORCDF（抗过平滑认知诊断框架）的API，用于预测学生的答题表现，并提供丰富的学习分析。",
    version="2.1.0",
)

# 使用 app.state 来管理全局资源
# app.state.model -> ORCDF 模型
# app.state.data_matrices (a, ia, q) -> 模型的图输入
# app.state.id_maps (student, problem) -> ID 映射
# app.state.device -> torch 设备
# app.state.analytics_data -> 预计算的分析数据
# app.state.raw_data_for_analytics -> 用于实时计算分析的原始数据

# --- Pydantic 模型定义 ---
class PredictionRequest(BaseModel):
    student_id: int = Field(..., description="学生的原始ID")
    problem_id: int = Field(..., description="练习的原始ID")

class PredictionResponse(BaseModel):
    student_id: int
    problem_id: int
    predicted_correct_probability: float = Field(..., description="模型预测的答对概率")

class SkillMastery(BaseModel):
    skill_id: int
    skill_name: str
    predicted_mastery: float = Field(..., description="对该技能的预测掌握度（平均答对率）")

class StudentProfileResponse(BaseModel):
    student_id: int
    skill_mastery_profile: List[SkillMastery] = Field(..., description="按掌握度从低到高排序的技能列表")

class RecommendedProblem(BaseModel):
    problem_id: int
    predicted_success_rate: float
    problem_text: Optional[str] = None

class RecommendationResponse(BaseModel):
    student_id: int
    recommended_for_skill_id: int
    recommended_for_skill_name: str
    recommendations: List[RecommendedProblem]

# 新增：学习分析相关的模型
class QuestionAnalyticsResponse(BaseModel):
    difficulty: Optional[float] = Field(None, description="难度 (P-value), 范围 [0, 1], 越高越简单")
    discrimination: Optional[float] = Field(None, description="区分度, 范围 [-1, 1], 越高越能区分不同水平的学生")
    avg_ms_first_response: Optional[float] = Field(None, description="平均首次作答时间 (ms)")
    avg_hint_count: Optional[float] = Field(None, description="平均使用提示次数")

class StudentAnalyticsResponse(BaseModel):
    overall_accuracy: Optional[float] = Field(None, description="总体正确率")
    avg_response_time: Optional[float] = Field(None, description="平均作答时间 (ms)")
    avg_hint_usage: Optional[float] = Field(None, description="平均提示使用次数")

class LearningVelocityResponse(BaseModel):
    student_id: int
    skill_id: int
    mastery_trend: Optional[List[float]] = Field(None, description="在连续相关练习中的掌握度变化趋势")
    learning_velocity: Optional[float] = Field(None, description="学习速度 (掌握度趋势线的斜率)")
    error: Optional[str] = None

class LLMPromptResponse(BaseModel):
    student_id: int
    prompt: str = Field(..., description="为LLM生成的、用于学习路径规划的完整文本提示")


# --- 事件处理器 ---
@app.on_event("startup")
def startup_event():
    """在服务器启动时加载模型和所需数据"""
    print("--- 服务器启动中，正在加载ORCDF模型及数据... ---")
    
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {app.state.device}")

    # 1. 加载数据以获取矩阵和ID映射
    print("加载数据以构建图矩阵和ID映射...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, 'dataset')
    
    loader = DataLoader(dataset_path)
    # 加载完整数据以构建完整的图，并为实时分析保存
    raw_skill_builder_data = loader.load_skill_builder_data()
    orcdf_data = loader.load_orcdf_data() 
    if not orcdf_data or not raw_skill_builder_data:
        print("错误：无法加载数据，服务无法启动。")
        # 在实际应用中，这里应该有更健壮的错误处理
        return

    # 将实时分析所需的数据存储到 state
    orcdf_data['skill_builder_interactions'] = raw_skill_builder_data['interactions']
    app.state.raw_data_for_analytics = orcdf_data

    app.state.id_maps = {
        "student": orcdf_data['student_map'],
        "problem": orcdf_data['problem_map'],
        "skill": orcdf_data['skill_map'], # This is {raw_id: index}
        "skill_idx_to_name": orcdf_data['skills'],
        # Correctly named maps
        "skill_id_to_idx": orcdf_data['skill_map'], # Direct mapping: {skill_id: skill_idx}
        "skill_idx_to_id": {v: k for k, v in orcdf_data['skill_map'].items()} # Reverse mapping: {skill_idx: skill_id}
    }
    app.state.problem_descriptions = orcdf_data['problem_descriptions']
    
    # 调试：打印一些学生ID
    print("--- 样本学生ID ---")
    try:
        student_map_sample = list(app.state.id_maps['student'].keys())[:5]
        print(f"可用的学生ID示例: {student_map_sample}")
    except Exception as e:
        print(f"打印学生ID时出错: {e}")
    print("--------------------")

    # 将矩阵转换为稀疏张量并存储
    app.state.data_matrices = {
        "a_matrix": to_sparse_tensor(orcdf_data['a_matrix'], app.state.device),
        "ia_matrix": to_sparse_tensor(orcdf_data['ia_matrix'], app.state.device),
        "q_matrix": to_sparse_tensor(orcdf_data['q_matrix'], app.state.device),
    }
    print("数据矩阵和ID映射加载完毕。")

    # 2. 加载预计算的分析数据
    analytics_data_path = os.path.join(project_root, 'dkg_mvp', 'analytics_data.json')
    if os.path.exists(analytics_data_path):
        print(f"加载预计算的分析数据从 {analytics_data_path}...")
        with open(analytics_data_path, 'r', encoding='utf-8') as f:
            app.state.analytics_data = json.load(f)
        print("分析数据加载成功。")
    else:
        print(f"警告: 在 '{analytics_data_path}' 未找到分析数据文件。分析类API将不可用。")
        app.state.analytics_data = None

    # 3. 加载训练好的ORCDF模型
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 在 '{MODEL_PATH}' 找不到模型文件。请先运行 train_orcdf.py 进行训练。")
        return
        
    print(f"从 {MODEL_PATH} 加载 ORCDF 模型...")
    # 注意：这里的超参数需要与训练时使用的相匹配
    model = ORCDF(
        num_students=orcdf_data['num_students'],
        num_problems=orcdf_data['num_problems'],
        num_skills=orcdf_data['num_skills'],
        embed_dim=64, # 需与训练时一致
        num_layers=2  # 需与训练时一致
    ).to(app.state.device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=app.state.device))
    model.eval() # 设置为评估模式
    
    app.state.model = model
    print("ORCDF 模型加载成功。")
    print("--- 服务器启动完成 ---")

# --- 辅助函数 ---
def _run_prediction(student_ids: List[int], problem_ids: List[int]) -> torch.Tensor:
    """
    内部辅助函数，用于运行模型预测。
    接收内部索引ID列表。
    """
    model = app.state.model
    device = app.state.device
    
    student_tensor = torch.LongTensor(student_ids).to(device)
    problem_tensor = torch.LongTensor(problem_ids).to(device)
    
    with torch.no_grad():
        preds = model(
            student_tensor,
            problem_tensor,
            app.state.data_matrices['a_matrix'],
            app.state.data_matrices['ia_matrix'],
            app.state.data_matrices['q_matrix']
        )
    return preds

def _get_student_skill_profile(student_id: int) -> List[SkillMastery]:
    """
    内部辅助函数，计算并返回单个学生的完整技能画像。
    此函数提取了原 /api/student/{student_id}/profile 的核心逻辑，以便复用。
    """
    if student_id not in app.state.id_maps['student']:
        raise HTTPException(status_code=404, detail=f"学生ID {student_id} 不存在。")

    student_idx = app.state.id_maps['student'][student_id]
    skill_idx_to_name = app.state.id_maps['skill_idx_to_name']
    q_matrix_np = app.state.data_matrices['q_matrix'].to_dense().cpu().numpy()
    skill_idx_to_id_map = app.state.id_maps['skill_idx_to_id']

    mastery_profile = []
    for skill_idx, skill_name in skill_idx_to_name.items():
        problem_indices = np.where(q_matrix_np[:, skill_idx] == 1)[0]
        
        if len(problem_indices) == 0:
            continue

        student_indices = [student_idx] * len(problem_indices)
        preds = _run_prediction(student_indices, list(problem_indices))
        
        avg_mastery = preds.mean().item()
        skill_id = skill_idx_to_id_map.get(skill_idx, -1)
        
        mastery_profile.append(
            SkillMastery(
                skill_id=skill_id,
                skill_name=skill_name,
                predicted_mastery=avg_mastery
            )
        )
    
    # 按掌握度从低到高排序
    return sorted(mastery_profile, key=lambda x: x.predicted_mastery)


# --- API Endpoints ---
@app.get("/", tags=["通用"])
def read_root():
    """欢迎信息和API文档链接"""
    return {
        "message": "欢迎使用 ORCDF 认知诊断 API",
        "documentation_url": "/docs"
    }

@app.get("/api/status", tags=["通用"])
def get_status():
    """检查API服务和模型的状态"""
    model_loaded = hasattr(app.state, 'model') and app.state.model is not None
    data_loaded = hasattr(app.state, 'data_matrices') and app.state.data_matrices is not None
    analytics_loaded = hasattr(app.state, 'analytics_data') and app.state.analytics_data is not None
    return {
        "status": "ok" if model_loaded and data_loaded else "error",
        "message": "服务正在运行。" if model_loaded and data_loaded else "服务遇到问题，模型或数据未加载。",
        "model_loaded": model_loaded,
        "data_loaded": data_loaded,
        "analytics_data_loaded": analytics_loaded,
        "device": str(app.state.device) if hasattr(app.state, 'device') else "N/A"
    }

@app.post("/api/predict", response_model=List[PredictionResponse], tags=["核心功能"])
def predict_correctness(requests: List[PredictionRequest]):
    """
    预测一个或多个学生-练习交互的答对概率。
    """
    if not hasattr(app.state, 'model') or app.state.model is None:
        raise HTTPException(status_code=503, detail="模型当前不可用，请稍后重试。")

    student_map = app.state.id_maps['student']
    problem_map = app.state.id_maps['problem']
    
    batch_student_ids = []
    batch_problem_ids = []
    
    # 验证ID并转换为内部索引
    for req in requests:
        if req.student_id not in student_map:
            raise HTTPException(status_code=404, detail=f"学生ID {req.student_id} 不存在。")
        if req.problem_id not in problem_map:
            raise HTTPException(status_code=404, detail=f"练习ID {req.problem_id} 不存在。")
        
        batch_student_ids.append(student_map[req.student_id])
        batch_problem_ids.append(problem_map[req.problem_id])

    # 准备模型输入
    preds = _run_prediction(batch_student_ids, batch_problem_ids)
    
    # 格式化响应
    responses = []
    for i, req in enumerate(requests):
        prob = preds[i].item() if torch.is_tensor(preds) and preds.ndim > 0 else preds.item()
        responses.append(
            PredictionResponse(
                student_id=req.student_id,
                problem_id=req.problem_id,
                predicted_correct_probability=prob
            )
        )
        
    return responses

@app.get("/api/student/{student_id}/profile", response_model=StudentProfileResponse, tags=["核心功能"])
def get_student_profile(student_id: int):
    """
    获取学生的完整知识画像，即对所有技能的预测掌握度。
    """
    if not hasattr(app.state, 'model'):
        raise HTTPException(status_code=503, detail="模型当前不可用")

    # 直接调用重构后的辅助函数
    skill_mastery_list = _get_student_skill_profile(student_id)
    
    return StudentProfileResponse(
        student_id=student_id,
        skill_mastery_profile=skill_mastery_list
    )


@app.get("/api/student/{student_id}/recommendations", response_model=RecommendationResponse, tags=["核心功能"])
def get_recommendations(student_id: int, count: int = 5):
    """
    为学生推荐下一步的练习题。
    默认推荐其最薄弱知识点中，最适合他当前水平的题目。
    """
    # 复用profile接口逻辑找到最薄弱的技能
    profile_data = get_student_profile(student_id)
    if not profile_data.skill_mastery_profile:
        raise HTTPException(status_code=404, detail="无法为该学生生成技能画像，无法推荐。")
        
    weakest_skill = profile_data.skill_mastery_profile[0]
    target_skill_id = weakest_skill.skill_id
    target_skill_idx = app.state.id_maps['skill_id_to_idx'].get(target_skill_id)

    if target_skill_idx is None:
        raise HTTPException(status_code=500, detail=f"Internal error: Cannot find index for skill ID {target_skill_id}")
    
    student_idx = app.state.id_maps['student'][student_id]
    problem_map_rev = {v: k for k, v in app.state.id_maps['problem'].items()}

    # 1. 找到目标技能的所有问题
    q_matrix_np = app.state.data_matrices['q_matrix'].to_dense().cpu().numpy()
    problem_indices_for_skill = np.where(q_matrix_np[:, target_skill_idx] == 1)[0]
    
    # 2. 找到学生已尝试过的所有问题
    a_matrix_np = app.state.data_matrices['a_matrix'].to_dense().cpu().numpy()
    ia_matrix_np = app.state.data_matrices['ia_matrix'].to_dense().cpu().numpy()
    attempted_mask = (a_matrix_np[student_idx, :] + ia_matrix_np[student_idx, :]) > 0
    attempted_indices = np.where(attempted_mask)[0]

    # 3. 筛选出未尝试过的候选问题
    candidate_indices = np.setdiff1d(problem_indices_for_skill, attempted_indices, assume_unique=True)

    if len(candidate_indices) == 0:
         return RecommendationResponse(
            student_id=student_id,
            recommended_for_skill_id=target_skill_id,
            recommended_for_skill_name=weakest_skill.skill_name,
            recommendations=[]
        )

    # 4. 预测这些候选问题的成功率
    student_indices_batch = [student_idx] * len(candidate_indices)
    preds = _run_prediction(student_indices_batch, list(candidate_indices))
    
    # 5. 找到最接近ZPD (最近发展区, 0.65) 的题目
    target_prob = 0.65
    problem_with_preds = sorted(
        zip(candidate_indices, preds.cpu().numpy()),
        key=lambda x: abs(x[1] - target_prob)
    )

    # 6. 格式化并返回top N个推荐
    recommendations = []
    for p_idx, p_prob in problem_with_preds[:count]:
        p_id = problem_map_rev.get(p_idx, -1)
        recommendations.append(RecommendedProblem(
            problem_id=p_id,
            predicted_success_rate=p_prob
        ))

    return RecommendationResponse(
        student_id=student_id,
        recommended_for_skill_id=target_skill_id,
        recommended_for_skill_name=weakest_skill.skill_name,
        recommendations=recommendations
    )

# --- Analytics API Endpoints ---

@app.get("/api/analytics/questions", response_model=Dict[str, QuestionAnalyticsResponse], tags=["学习分析"])
def get_all_question_analytics():
    """获取所有问题的预计算分析画像。"""
    if not app.state.analytics_data:
        raise HTTPException(status_code=404, detail="分析数据未加载。")
    return app.state.analytics_data.get("questions", {})

@app.get("/api/analytics/question/{question_id}", response_model=QuestionAnalyticsResponse, tags=["学习分析"])
def get_question_analytics(question_id: int):
    """根据ID获取单个问题的预计算分析画像。"""
    if not app.state.analytics_data:
        raise HTTPException(status_code=404, detail="分析数据未加载。")
    
    # 注意：JSON的key是字符串
    analytics = app.state.analytics_data.get("questions", {}).get(str(question_id))
    if not analytics:
        raise HTTPException(status_code=404, detail=f"找不到ID为 {question_id} 的问题的分析数据。")
    return analytics

@app.get("/api/analytics/students", response_model=Dict[str, StudentAnalyticsResponse], tags=["学习分析"])
def get_all_student_analytics():
    """获取所有学生的预计算行为分析。"""
    if not app.state.analytics_data:
        raise HTTPException(status_code=404, detail="分析数据未加载。")
    return app.state.analytics_data.get("students", {})

@app.get("/api/analytics/student/{student_id}", response_model=StudentAnalyticsResponse, tags=["学习分析"])
def get_student_analytics(student_id: int):
    """根据ID获取单个学生的预计算行为分析。"""
    if not app.state.analytics_data:
        raise HTTPException(status_code=404, detail="分析数据未加载。")
    
    analytics = app.state.analytics_data.get("students", {}).get(str(student_id))
    if not analytics:
        raise HTTPException(status_code=404, detail=f"找不到ID为 {student_id} 的学生的分析数据。")
    return analytics

@app.get("/api/analytics/student/{student_id}/learning_velocity/{skill_id}", response_model=LearningVelocityResponse, tags=["学习分析"])
def get_learning_velocity(student_id: int, skill_id: int):
    """
    实时计算并获取指定学生在特定技能上的学习速度。
    这是一个计算密集型操作。
    """
    if not hasattr(app.state, 'model') or not hasattr(app.state, 'raw_data_for_analytics'):
        raise HTTPException(status_code=503, detail="模型或实时分析所需的数据未加载。")

    try:
        # 每次请求时动态创建分析器实例
        student_analytics = StudentAnalytics(
            raw_data=app.state.raw_data_for_analytics,
            model=app.state.model,
            device=app.state.device
        )
        # 实时计算需要A/IA矩阵
        student_analytics.set_graph_matrices(
            app.state.raw_data_for_analytics['a_matrix'],
            app.state.raw_data_for_analytics['ia_matrix']
        )
        
        result = student_analytics.calculate_learning_velocity(student_id, skill_id)
        return result

    except Exception as e:
        # 捕获可预见的错误和未知错误
        raise HTTPException(status_code=500, detail=f"计算学习速度时发生内部错误: {str(e)}")


@app.get("/api/student/{student_id}/llm_learning_prompt", response_model=LLMPromptResponse, tags=["LLM集成"])
def get_llm_learning_prompt(student_id: int):
    """
    为指定学生生成一个用于学习路径规划的、给大语言模型(LLM)的详细提示(Prompt)。
    """
    # 1. 获取学生的静态行为分析数据
    if not app.state.analytics_data:
        raise HTTPException(status_code=404, detail="分析数据未加载，无法生成提示。")
    
    student_analytics = app.state.analytics_data.get("students", {}).get(str(student_id))
    if not student_analytics:
        raise HTTPException(status_code=404, detail=f"找不到ID为 {student_id} 的学生的分析数据，无法生成提示。")

    # 2. 获取学生的动态知识画像
    if not hasattr(app.state, 'model'):
        raise HTTPException(status_code=503, detail="模型当前不可用，无法生成提示。")
    
    skill_profile_models = _get_student_skill_profile(student_id)
    # 将Pydantic模型列表转换为字典列表，以供prompt_generator使用
    skill_profile_dicts = [profile.dict() for profile in skill_profile_models]

    # 3. 调用生成器生成提示
    prompt = generate_learning_path_prompt(
        student_id=student_id,
        student_analytics=student_analytics,
        skill_profile=skill_profile_dicts
    )

    return LLMPromptResponse(student_id=student_id, prompt=prompt)


# --- 主程序入口 ---
if __name__ == '__main__':
    uvicorn.run("api_server:app", host='0.0.0.0', port=5000, reload=True) 