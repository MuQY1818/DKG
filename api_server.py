import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 将项目根目录添加到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dkg_mvp.dkg_builder import DKGBuilder
from dkg_mvp.data_loader import DataLoader # 新增导入

# --- 全局变量 ---
MODELS_DIR = "models"
# DKG_SAVE_PATH 不再需要
EMBEDDINGS_DIR = os.path.join(MODELS_DIR, "embeddings")
# 新增数据集路径
DATASET_DIR = "dataset"

app = FastAPI(
    title="动态知识图谱 (DKG) API",
    description="用于学生建模的动态知识图谱API，集成了GNN嵌入相似度查询功能。",
    version="1.1.0",
)

# 使用一个字典来管理全局资源，而不是多个全局变量
app.state.dkg_builder = None
app.state.student_embeddings = None
app.state.problem_embeddings = None
app.state.skill_embeddings = None

# --- Pydantic 模型定义 ---
class Interaction(BaseModel):
    student_id: int
    problem_id: int
    correct: int
    score: Optional[float] = None
    time_taken: Optional[int] = None
    
class Embedding(BaseModel):
    id: int
    vector: List[float]

# --- 事件处理器 ---
@app.on_event("startup")
def startup_event():
    """在服务器启动时加载所有模型和数据"""
    print("--- 服务器启动中，正在加载模型... ---")
    
    # 1. 实时构建DKG主图谱，不再从文件加载
    print("实时构建DKG...")
    try:
        # 指向数据集根目录
        loader = DataLoader(DATASET_DIR)
        data_dict = loader.load_skill_builder_data()
        
        if data_dict:
            builder = DKGBuilder()
            builder.build_from_data(data_dict)
            app.state.dkg_builder = builder
            print(f"DKG实时构建完成，包含 {builder.graph.number_of_nodes()} 个节点和 {builder.graph.number_of_edges()} 条边。")
        else:
            print("错误：加载数据失败，无法构建DKG。")
            app.state.dkg_builder = DKGBuilder() # 创建一个空的builder以避免崩溃

    except Exception as e:
        print(f"在构建DKG时发生严重错误: {e}")
        # 在启动失败时创建一个空的builder实例，以允许服务启动并报告错误
        app.state.dkg_builder = DKGBuilder()

    # 2. 加载GNN生成的嵌入向量
    try:
        print("正在加载GNN嵌入向量...")
        app.state.student_embeddings = pd.read_csv(os.path.join(EMBEDDINGS_DIR, "student_embeddings.csv"), index_col=0)
        app.state.problem_embeddings = pd.read_csv(os.path.join(EMBEDDINGS_DIR, "problem_embeddings.csv"), index_col=0)
        app.state.skill_embeddings = pd.read_csv(os.path.join(EMBEDDINGS_DIR, "skill_embeddings.csv"), index_col=0)
        
        # 为索引命名，以便后续操作
        app.state.student_embeddings.index.name = 'id'
        app.state.problem_embeddings.index.name = 'id'
        app.state.skill_embeddings.index.name = 'id'

        print("所有嵌入向量加载完毕。")
    except FileNotFoundError:
        print(f"警告：在 {EMBEDDINGS_DIR} 中找不到嵌入文件。相似度查询API将不可用。")
        print("请先运行 gnn_trainer.py 来生成嵌入文件。")

    print("--- 服务器启动完成 ---")

# --- 辅助函数 ---
def get_builder() -> DKGBuilder:
    """依赖注入函数，用于获取DKG实例"""
    builder = app.state.dkg_builder
    if builder is None or builder.graph.number_of_nodes() == 0:
        raise HTTPException(status_code=503, detail="DKG服务当前不可用，模型未加载。")
    return builder

def find_similar_items(item_id: int, embeddings_df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    """通用的相似度计算函数"""
    if embeddings_df is None:
        raise HTTPException(status_code=501, detail="嵌入向量未加载，该功能不可用。")
    if item_id not in embeddings_df.index:
        raise HTTPException(status_code=404, detail=f"ID {item_id} 的嵌入向量不存在。")
        
    item_vec = embeddings_df.loc[[item_id]]
    similarities = cosine_similarity(item_vec, embeddings_df)
    
    # 获取最相似的 top_n+1 个结果（包含自身）
    similar_indices = np.argsort(similarities[0])[::-1][1:top_n+1]
    
    results = []
    for i in similar_indices:
        sim_id = embeddings_df.index[i]
        sim_score = similarities[0][i]
        results.append({"similar_id": int(sim_id), "similarity_score": float(sim_score)})
        
    return results

# --- API Endpoints ---

@app.get("/", tags=["通用"])
def read_root():
    """欢迎信息和API文档链接"""
    return {
        "message": "欢迎使用DKG API",
        "documentation_url": "/docs"
    }

@app.get("/api/status", tags=["通用"])
def get_status():
    """检查API服务和DKG的状态"""
    builder = get_builder()
    return {
        "status": "ok",
        "message": "DKG服务正在运行。",
        "graph_stats": {
            "nodes": builder.graph.number_of_nodes(),
            "edges": builder.graph.number_of_edges()
        },
        "embeddings_loaded": {
            "student": app.state.student_embeddings is not None,
            "problem": app.state.problem_embeddings is not None,
            "skill": app.state.skill_embeddings is not None,
        }
    }

# ... (其他 DKG API 端点) ...
@app.get("/api/student/{student_id}/profile", tags=["DKG查询"])
def get_student_profile_route(student_id: int):
    """获取学生画像"""
    builder = get_builder()
    profile = builder.get_student_profile(student_id)
    if "error" in profile:
        raise HTTPException(status_code=404, detail=profile["error"])
    return profile

@app.get("/api/skill/{skill_id}/details", tags=["DKG查询"])
def get_skill_details_route(skill_id: int):
    """获取技能详情"""
    builder = get_builder()
    details = builder.get_skill_details(skill_id)
    if "error" in details:
        raise HTTPException(status_code=404, detail=details["error"])
    return details

@app.get("/api/problem/{problem_id}/details", tags=["DKG查询"])
def get_problem_details_route(problem_id: int):
    """获取题目详情"""
    builder = get_builder()
    details = builder.get_problem_details(problem_id)
    if "error" in details:
        raise HTTPException(status_code=404, detail=details["error"])
    return details

@app.get("/api/student/{student_id}/recommendations", tags=["DKG查询"])
def get_recommendations_route(student_id: int, count: int = 5):
    """获取题目推荐"""
    builder = get_builder()
    recommendations = builder.recommend_next_problems(student_id, count)
    return recommendations

@app.post("/api/interaction", status_code=201, tags=["DKG更新"])
def record_interaction_route(interaction: Interaction):
    """记录一次新的交互"""
    builder = get_builder()
    interaction_dict = interaction.dict()
    interaction_dict['timestamp'] = pd.Timestamp.now()
    builder.record_interaction(interaction_dict)
    return {"status": "success", "message": "交互已成功记录。"}

# --- GNN 嵌入相似度查询 API ---
@app.get("/api/problem/{problem_id}/similar", tags=["GNN相似度查询"])
def get_similar_problems(problem_id: int, top_n: int = 5):
    """根据GNN嵌入查找相似的题目"""
    return find_similar_items(problem_id, app.state.problem_embeddings, top_n)

@app.get("/api/skill/{skill_id}/similar", tags=["GNN相似度查询"])
def get_similar_skills(skill_id: int, top_n: int = 5):
    """根据GNN嵌入查找相似的技能"""
    return find_similar_items(skill_id, app.state.skill_embeddings, top_n)

# --- 主程序入口 ---
if __name__ == '__main__':
    uvicorn.run("api_server:app", host='0.0.0.0', port=5000, reload=True) 