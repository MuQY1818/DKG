# DKG: Dynamic Knowledge Graph for Student Modeling

[![zh](https://img.shields.io/badge/language-简体中文-blue.svg)](./README_zh.md)
[![en](https://img.shields.io/badge/language-English-blue.svg)](./README_en.md)

This project provides a backend engine for a Dynamic Knowledge Graph (DKG) designed for student modeling. It transforms student interaction log data into a dynamically updated knowledge graph in real-time, offering a powerful Python API for real-time analysis and personalized recommendations.

本项目是一个专为学生建模设计的动态知识图谱（DKG）后端引擎。它能将学生的学习交互日志实时转化为一个动态更新的知识图谱，并提供强大的Python API进行实时分析和个性化推荐。

---

## System Architecture / 系统架构

```mermaid
graph TD
    subgraph "数据层 (Data Layer)"
        A["原始CSV文件 <br> student_logs.csv"]
    end

    subgraph "处理层 (Processing Layer)"
        B["data_loader.py <br> (加载 & 预处理)"]
    end

    subgraph "核心引擎 (Core DKG Engine)"
        C["DKGBuilder (dkg_builder.py)"]
        D["内存中的知识图谱 <br> (networkx.MultiDiGraph)"]
    end

    subgraph "应用层 (Application Layer)"
        E["API 使用示例 <br> run_api_example.py"]
        F["大语言模型 (LLM) 集成"]
        G["可视化 & 分析"]
    end

    A --> B;
    B --> C;
    C -- "构建/更新" --> D;
    D -- "查询" --> C;
    C -- "提供API" --> E;
    C -- "提供API" --> F;
    C -- "提供API" --> G;
```

## ✨ Core Features / 核心功能

- **Dynamic Graph Construction**: Builds a comprehensive knowledge graph from raw student learning data.
- **Real-time Updates**: Updates student knowledge states in real-time based on new learning interactions.
- **Student Profiling**: Provides detailed student profiles, including strengths and weaknesses.
- **Intelligent Recommendations**: Recommends the next practice problems for students based on their weak points.
- **LLM Integration**: Generates structured prompts to provide context for LLMs.

---

## 🚀 Quick Start / 快速开始

1.  **Clone Repository**:
    ```bash
    git clone https://github.com/MuQY1818/DKG.git
    cd DKG
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r dkg_mvp/requirements.txt
    ```
    
3.  **Download Datasets**:
    Download the required CSV files from the link below and place them in the `dataset/` directory.
    - **[Go to Dataset Download Page](https://github.com/MuQY1818/DKG/releases/tag/dataset)**
    > (从下方链接前往数据集下载页面，将所需CSV文件下载并放入 `dataset/` 目录。)

4.  **Run Example**:
    ```bash
    python run_api_example.py
    ```

---

## For More Information / 详细信息

For detailed documentation, please choose your preferred language:
要获取更详细的文档，请选择您的语言：

- **[English](./README_en.md)**
- **[简体中文](./README_zh.md)**
- **[Technical Documentation (English)](./TECHNICAL_DOCUMENTATION_en.md)**
- **[技术文档 (中文)](./TECHNICAL_DOCUMENTATION.md)** 