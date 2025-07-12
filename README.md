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
        H["持久化模型 <br> dkg.pkl"]
    end

    subgraph "应用层 (Application Layer)"
        E["API 服务器 <br> api_server.py"]
        F["大语言模型 (LLM) 集成"]
        G["可视化 & 分析"]
    end

    A --> B;
    B --> C;
    C -- "构建/更新" --> D;
    C -- "序列化" --> H;
    H -- "启动时加载" --> E;
    D -- "查询" --> C;
    C -- "提供API" --> E;
    E -- "提供端点" --> F;
    E -- "提供端点" --> G;
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
    > **Note**: For PyTorch and PyG, please follow the manual installation instructions at the top of the `requirements.txt` file to match your CUDA version.
    
3.  **Generate Model Files**:
    Before the first run, you must generate the necessary model files:
    ```bash
    # 1. Generate the DKG model file (dkg.pkl)
    python -m dkg_mvp.dkg_builder

    # 2. Generate the GNN embedding files
    python -m dkg_mvp.gnn_trainer
    ```

4.  **Run API Server**:
    ```bash
    python api_server.py
    ```
    Once the server is running, visit `http://127.0.0.1:5000/docs` in your browser to see the interactive API documentation.

---

## For More Information / 详细信息

For detailed documentation, please choose your preferred language:
要获取更详细的文档，请选择您的语言：

- **[English](./README_en.md)**
- **[简体中文](./README_zh.md)**
- **[Technical Documentation (English)](./TECHNICAL_DOCUMENTATION_en.md)**
- **[技术文档 (中文)](./TECHNICAL_DOCUMENTATION.md)** 