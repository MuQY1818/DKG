# DKG (动态知识图谱) API 使用指南

本文档旨在帮助开发者理解和使用本项目的API接口，以便查询和更新动态知识图谱，并利用图神经网络（GNN）的嵌入向量进行高级查询。

---

## 核心概念

所有的操作都围绕一个通过Web服务暴露的API展开。该服务在后台运行着一个`DKGBuilder`实例，并加载了GNN模型生成的节点嵌入向量。

- **`DKGBuilder`**: 知识图谱的容器和主要交互逻辑。
- **GNN Embeddings**: 提供了节点（如题目、技能）在知识空间中的向量表示，用于计算相似度。

---

## 1. 启动API服务

本项目使用FastAPI构建，提供了一个高性能的、带有交互式文档的API服务。

1.  **确保依赖已安装**:
    ```bash
    pip install -r dkg_mvp/requirements.txt
    pip install fastapi "uvicorn[standard]"
    ```

2.  **构建必要的模型文件**:
    在首次运行前，请确保`models/`目录下已经有构建好的DKG模型和GNN嵌入。
    - 运行 `python dkg_mvp/dkg_builder.py` 或 `python run_api_example.py` 来生成 `dkg_skill_builder.graphml`。
    - 运行 `python dkg_mvp/gnn_trainer.py` 来生成 `models/embeddings/` 目录下的嵌入文件。

3.  **启动服务器**:
    在项目根目录运行以下命令：
    ```bash
    uvicorn api_server:app --host 0.0.0.0 --port 5000 --reload
    ```
    - `--reload` 参数使服务器在代码更改后自动重启，非常适合开发环境。

4.  **访问交互式API文档**:
    服务器启动后，在浏览器中打开 **`http://127.0.0.1:5000/docs`**。
    您会看到一个Swagger UI界面，可以在其中直接测试所有API端点。

---

## 2. API 接口参考

### DKG核心API

这些API用于查询和更新基础的知识图谱状态。

#### `GET /api/status`
检查服务的健康状况，以及DKG图谱和GNN嵌入是否成功加载。

#### `GET /api/student/{student_id}/profile`
获取某个学生的完整画像，包括其知识强弱项等。
- **`student_id`**: 学生的**原始ID**。

#### `POST /api/interaction`
记录一次新的学习交互，这是更新学生知识状态的核心入口。
- **请求体 (JSON)**:
    ```json
    {
      "student_id": 73963,
      "problem_id": 76429,
      "correct": 1
    }
    ```

#### 其他DKG查询API
- `GET /api/skill/{skill_id}/details`: 获取技能详情。
- `GET /api/problem/{problem_id}/details`: 获取题目详情。
- `GET /api/student/{student_id}/recommendations`: 获取题目推荐。

---

### GNN相似度查询API

这些API利用预先训练好的GNN模型生成的嵌入向量，来寻找知识概念上的相似实体。

#### `GET /api/problem/{problem_id}/similar`
查找与给定题目在知识结构上最相似的其他题目。
- **`problem_id`**: 题目的**原始ID**。
- **查询参数 `top_n`**: 返回结果的数量，默认为5。
- **示例响应**:
    ```json
    [
      {
        "similar_id": 76430,
        "similarity_score": 0.98
      },
      {
        "similar_id": 76501,
        "similarity_score": 0.95
      }
    ]
    ```

#### `GET /api/skill/{skill_id}/similar`
查找与给定技能在知识结构上最相似的其他技能。
- **`skill_id`**: 技能的**原始ID**。
- **查询参数 `top_n`**: 返回结果的数量，默认为5。

---

## 3. 使用 `curl` 进行测试的示例

您也可以使用命令行工具如 `curl` 来与API交互。

```bash
# 获取学生73963的画像
curl http://127.0.0.1:5000/api/student/73963/profile

# 查找与题目76429相似的题目
curl http://127.0.0.1:5000/api/problem/76429/similar?top_n=3

# 记录一次交互
curl -X POST http://127.0.0.1:5000/api/interaction \
-H "Content-Type: application/json" \
-d '{"student_id": 73963, "problem_id": 76429, "correct": 1}'
``` 