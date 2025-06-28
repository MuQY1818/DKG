# DKG-MVP: 动态知识图谱后端服务

## 项目简介

本项目是一个动态知识图谱（DKG）的后端服务引擎。它旨在将学生的学习交互日志数据，实时地转化为一个动态更新的知识图谱。项目提供了一套清晰、强大的Python API，用于记录学生行为、查询学生知识状态、并为上层应用（如集成大语言模型LLM）提供个性化学习建议和决策支持。

## 核心数据集

系统当前主要使用 **ASSISTments 2009-2010 Skill Builder** 数据集，这是一款包含了丰富学生答题日志的真实教育数据。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 核心功能与API

本项目的所有核心功能均通过 `dkg_mvp.dkg_builder.DKGBuilder` 类提供的API进行暴露。详细使用方法请查阅项目根目录下的 **`DKG_API_使用指南.md`**。

- **图谱持久化**:
  - `save_graph()`: 将内存中的图谱保存到文件。
  - `load_graph()`: 从文件快速加载图谱，无需重新构建。

- **动态更新**:
  - `record_interaction()`: 记录一次学生答题交互，实时更新学生知识状态。

- **状态查询**:
  - `get_student_profile()`: 获取学生的完整画像，包括知识强弱项。
  - `get_skill_details()`: 查询特定技能的详情及其在知识网络中的位置。
  - `get_problem_details()`: 查询特定题目的详情。

- **智能推荐**:
  - `recommend_next_problems()`: 根据学生薄弱点，推荐下一步练习。

- **LLM集成**:
  - `generate_llm_prompt()`: 生成结构化Prompt，为LLM规划学习路径提供上下文。

## 文件结构

```
.
├── DKG_API_使用指南.md    # 核心API文档 (首选阅读) 🌟
├── run_api_example.py       # API使用方法演示脚本
├── models/                  # 存放持久化图谱模型的目录
├── dataset/                 # 存放原始数据集
├── dkg_mvp/
│   ├── dkg_builder.py       # DKG核心类，提供所有API ⭐
│   ├── data_loader.py       # 数据加载与预处理
│   ├── simulation.py        # (可选) 模拟学习过程的引擎
│   ├── api_tests.py         # 单元测试与API验证
│   └── README.md            # (本文档) 模块简介
└── requirements.txt         # 依赖包列表
```

## 快速开始

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **运行演示**:
    执行项目根目录下的 `run_api_example.py` 脚本来查看完整的API工作流。
    ```bash
    python run_api_example.py
    ```
    该脚本会首次构建DKG并保存，后续运行则会直接加载，并演示如何更新和查询学生状态。

3.  **深入了解**:
    请详细阅读项目根目录下的 **`DKG_API_使用指南.md`**，它包含了所有API的详细说明和使用示例。