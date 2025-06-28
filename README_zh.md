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
├── README.md                # 主README，提供语言选择
├── README_en.md             # 英文版README
├── README_zh.md             # 中文版README (本文)
├── run_api_example.py       # API使用方法演示脚本
├── models/                  # 存放持久化图谱模型的目录
├── dataset/                 # 存放原始数据集
│   └── skill_builder_data09-10.csv # 主要使用的数据集
├── dkg_mvp/
│   ├── dkg_builder.py       # DKG核心类，提供所有API ⭐
│   ├── data_loader.py       # 数据加载与预处理
│   ├── simulation.py        # (可选) 模拟学习过程的引擎
│   ├── api_tests.py         # 单元测试与API验证
│   └── requirements.txt     # 依赖包列表
└── .gitignore               # Git忽略文件
```

## 🚀 快速开始

### 1. 安装

首先，克隆本仓库并安装所需的依赖包。

```bash
git clone https://github.com/MuQY1818/DKG.git
cd DKG
pip install -r dkg_mvp/requirements.txt
```

### 2. 运行示例

执行项目根目录下的 `run_api_example.py` 脚本来查看一个完整的API工作流。

```bash
python run_api_example.py
```

该脚本会执行以下操作：
- 首次运行时，它会从 `skill_builder_data09-10.csv` 数据集构建DKG，并将其保存到 `models/` 目录中。
- 后续运行时，它会从文件加载已存在的DKG。
- 接着，它会演示：
    1.  获取一个学生在交互前的知识画像。
    2.  模拟一次新的学习交互（例如，学生正确回答了一个问题）。
    3.  获取学生更新后的画像以展示变化。
    4.  基于新的画像为学生推荐下一步的练习题。

## 🛠️ API 参考

所有核心功能都通过 `dkg_mvp.dkg_builder.DKGBuilder` 类暴露。要了解详细说明和更多示例，请参阅 `DKG_API_使用指南.md`。

### 初始化与持久化

- `DKGBuilder.build_from_data(log_data)`: 从预处理过的数据构建图谱。
- `builder.save_graph(path)`: 将内存中的图谱保存到文件。
- `DKGBuilder.load_graph(path)`: 从文件加载图谱。

### 动态更新

- `builder.record_interaction(interaction: Dict)`: 记录单次学生交互并实时更新图谱。交互字典必须包含 `student_id`、`problem_id` 和 `correct`。

### 查询

- `builder.get_student_profile(student_id: int)`: 检索学生的完整画像，包括他们的知识点总结。
- `builder.get_skill_details(skill_id: int)`: 获取特定技能的详细信息。
- `builder.get_problem_details(problem_id: int)`: 获取特定题目的详细信息。

### 推荐与LLM集成

- `builder.recommend_next_problems(student_id: int)`: 根据学生的薄弱点推荐合适的练习题。
- `builder.generate_llm_prompt(...)`: 组装一个结构化的、信息丰富的提示，发送给LLM用于生成个性化学习路径等任务。

## 📊 数据集

系统当前主要使用 **ASSISTments 2009-2010 Skill Builder** 数据集，该数据集包含了丰富的真实学生解题活动日志。

**注意：** 数据集文件已通过 GitHub Releases 托管。请从以下链接下载，并将其放置在 `dataset/` 目录下，然后再运行示例脚本。

- **[下载 `skill_builder_data09-10.csv`](https://github.com/MuQY1818/DKG/releases/download/dataset/skill_builder_data09-10.csv)** (79.8 MB)
- **[下载 `assistments_2009_2010.csv`](https://github.com/MuQY1818/DKG/releases/download/dataset/assistments_2009_2010.csv)** (116 MB)