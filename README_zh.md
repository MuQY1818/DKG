[Read in English](./README_en.md)

# DKG：用于学生建模的动态知识图谱

本项目提供了一个专为学生建模设计的动态知识图谱（DKG）后端引擎。它能够将学生的学习交互日志数据，实时转化为一个动态更新的知识图谱。项目提供了一套清晰、强大的Python API，用于记录学生行为、查询学生知识状态，并为上层应用（如集成大语言模型LLM）提供个性化学习建议和决策支持。

## ✨ 核心功能

- **动态图谱构建**: 从原始学生学习数据中构建一个包含学生、题目和技能的综合知识图谱。
- **状态持久化**: 支持将构建好的图谱保存到磁盘并快速加载，避免重复构建。
- **实时更新**: 根据新的学习交互，实时更新学生的知识状态和掌握程度。
- **学生画像**: 提供详细的学生画像，包括知识技能的强项和弱项。
- **智能推荐**: 根据学生的薄弱点，为他们推荐下一步的练习题。
- **LLM集成**: 生成结构化的提示（Prompt），为大语言模型规划个性化学习路径提供上下文。

## 📂 项目结构

```
.
├── DKG_API_使用指南.md    # 详细的API中文文档 (首选参考) 🌟
├── api_server.py            # FastAPI 服务器入口 🚀
├── dkg.pkl                  # 序列化后的DKG模型文件 💾
├── README.md                # 主README，提供语言选择
├── README_en.md             # 英文版README
├── README_zh.md             # 中文版README (本文)
├── models/                  # 存放GNN嵌入模型的目录
├── dataset/                 # 存放原始数据集
│   └── clear_dataset/       # 清洗后的数据集
├── dkg_mvp/
│   ├── dkg_builder.py       # DKG核心类，提供所有API ⭐
│   ├── gnn_trainer.py       # GNN模型训练器
│   ├── data_loader.py       # 数据加载与预处理
│   └── requirements.txt     # 依赖包列表
└── .gitignore               # Git忽略文件
```

## 🚀 快速开始

### 1. 安装

首先，克隆本仓库并安装所需的依赖包。

```bash
git clone https://github.com/MuQY1818/DKG.git
cd DKG
# 注意：请先根据 dkg_mvp/requirements.txt 文件顶部的说明，安装匹配的PyTorch和PyG
pip install -r dkg_mvp/requirements.txt
```

### 2. 准备模型文件 (首次运行)

在启动服务器前，请确保已生成了必要的模型文件。
- **DKG图谱**: 运行 `python -m dkg_mvp.dkg_builder` 会使用数据集构建并保存 `dkg.pkl` 文件到项目根目录。
- **GNN嵌入**: 运行 `python -m dkg_mvp.gnn_trainer` 会训练GNN并保存嵌入向量到 `models/embeddings/` 目录。

### 3. 启动API服务器

执行以下命令来启动后端API服务：

```bash
python api_server.py
```

服务器启动后，在浏览器中打开 **`http://127.0.0.1:5000/docs`** 即可访问交互式的API文档，您可以在该页面直接测试所有API。

## 🛠️ API 参考

所有核心功能都通过FastAPI服务暴露。要了解详细的端点、参数和请求/响应格式，**强烈推荐直接查阅上述交互式API文档**。`DKG_API_使用指南.md` 提供了更偏向于工作流和概念的解释。

### 主要API端点概览

- **`GET /api/status`**: 检查服务健康状况。
- **`GET /api/student/{student_id}/profile`**: 检索学生的完整画像。
- **`POST /api/interaction`**: 记录单次学生交互并实时更新图谱。
- **`GET /api/problem/{problem_id}/similar`**: (GNN功能) 查找相似题目。
- **`GET /api/skill/{skill_id}/similar`**: (GNN功能) 查找相似技能。
- ...以及更多，请参考API文档。

## 📊 数据集

系统当前主要使用 **ASSISTments 2009-2010 Skill Builder** 数据集的公开子集。相关的数据文件已经包含在项目的 `dataset/clear_dataset` 目录中，无需额外下载。