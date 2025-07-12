# DKG-MVP: 核心模块

## 模块简介

本目录 (`dkg_mvp`) 包含了动态知识图谱（DKG）项目的核心后端逻辑。它负责数据的加载、图谱的构建与更新、以及所有核心算法的实现。

## 核心组件

- **`dkg_builder.py`**: 模块的核心，定义了 `DKGBuilder` 类。此类封装了所有与DKG图谱交互的API，包括构建、更新、查询和推荐。
- **`data_loader.py`**: 负责从原始CSV文件加载数据，并进行预处理，为 `DKGBuilder` 提供格式化的输入。
- **`gnn_trainer.py`**: 一个独立的GNN模型训练器，用于从DKG图谱中学习节点嵌入，以支持相似度查询等高级功能。
- **`simulation.py` / `visualization.py`**: 用于模拟和可视化学习过程的辅助工具。

## 功能与API

`DKGBuilder` 类提供了所有核心功能。要了解如何通过Web服务使用这些功能，请参考项目根目录的 **`DKG_API_使用指南.md`**。

- **图谱持久化**:
  - `save_with_pickle()`: 将内存中的DKG实例完整地序列化到文件 (`.pkl`)。
  - `load_with_pickle()`: 从 pickle 文件中快速、可靠地恢复DKG实例。

- **动态更新**:
  - `record_interaction()`: 记录一次学生答题交互，实时更新学生知识状态。

- **状态查询**:
  - `get_student_profile()`: 获取学生的完整画像。
  - `get_skill_details()`: 查询特定技能的详情。
  - `get_problem_details()`: 查询特定题目的详情。

- **智能推荐**:
  - `recommend_next_problems()`: 根据学生薄弱点，推荐下一步练习。

- **LLM集成**:
  - `generate_llm_prompt()`: 生成结构化Prompt，为大语言模型(LLM)规划学习路径提供上下文。

## 基本工作流

本模块的典型使用流程被项目根目录的 `api_server.py` 所封装和展示：

1.  **离线构建**: 运行 `python -m dkg_mvp.dkg_builder`，它会调用 `DataLoader` 加载数据，然后使用 `DKGBuilder` 构建完整的图谱，并最终通过 `save_with_pickle` 将其保存为 `dkg.pkl`。
2.  **在线服务**: `api_server.py` 在启动时，会调用 `DKGBuilder.load_with_pickle` 将 `dkg.pkl` 加载到内存中，然后通过RESTful API将DKG的各项功能暴露出去。

要深入了解，请直接阅读项目根目录的 **`README.md`** 和 **`DKG_API_使用指南.md`**。