[Read in Chinese](./README_zh.md)

# DKG: Dynamic Knowledge Graph for Student Modeling

This project provides a backend engine for a Dynamic Knowledge Graph (DKG) designed for student modeling. It transforms student interaction log data into a dynamically updated knowledge graph in real-time. The project offers a clear and powerful Python API for recording student behavior, querying student knowledge states, and providing personalized learning recommendations and decision support for higher-level applications, such as integration with Large Language Models (LLMs).

## ✨ Core Features

- **Dynamic Graph Construction**: Builds a comprehensive knowledge graph from raw student learning data, including students, problems, and skills.
- **State Persistence**: Supports saving the constructed graph to disk and loading it quickly, avoiding the need for repeated construction.
- **Real-time Updates**: Updates student knowledge states and mastery levels in real-time based on new learning interactions.
- **Student Profiling**: Provides detailed student profiles, including strengths and weaknesses in knowledge skills.
- **Intelligent Recommendations**: Recommends the next practice problems for students based on their weak points.
- **LLM Integration**: Generates structured prompts to provide context for LLMs to plan personalized learning paths.

## 📂 Project Structure

```
.
├── DKG_API_使用指南.md    # Detailed API documentation in Chinese (Primary Reference) 🌟
├── api_server.py            # FastAPI Server Entrypoint 🚀
├── dkg.pkl                  # Serialized DKG model file 💾
├── README.md                # Main README for language selection
├── README_en.md             # English README (this file)
├── README_zh.md             # Chinese README
├── models/                  # Directory for GNN embedding models
├── dataset/                 # Directory for datasets
│   └── clear_dataset/       # Cleaned datasets
├── dkg_mvp/
│   ├── dkg_builder.py       # Core DKG class, provides all APIs ⭐
│   ├── gnn_trainer.py       # GNN model trainer
│   ├── data_loader.py       # Data loading and preprocessing
│   └── requirements.txt     # List of dependencies
└── .gitignore               # Git ignore file
```

## 🚀 Quick Start

### 1. Installation

First, clone the repository and install the required dependencies.

```bash
git clone https://github.com/MuQY1818/DKG.git
cd DKG
# Note: Please first follow the instructions at the top of dkg_mvp/requirements.txt
# to install versions of PyTorch and PyG that match your environment.
pip install -r dkg_mvp/requirements.txt
```

### 2. Prepare Model Files (First-time Run)

Before starting the server, ensure you have generated the necessary model files.
- **DKG Graph**: Running `python -m dkg_mvp.dkg_builder` will build and save the `dkg.pkl` file to the project root directory.
- **GNN Embeddings**: Running `python -m dkg_mvp.gnn_trainer` will train the GNN and save the embedding vectors to the `models/embeddings/` directory.

### 3. Start the API Server

Execute the following command to start the backend API service:

```bash
python api_server.py
```

After the server starts, open your browser and navigate to **`http://127.0.0.1:5000/docs`** to access the interactive API documentation, where you can test all endpoints directly.

## 🛠️ API Reference

All core functionalities are exposed through the FastAPI service. For detailed information about endpoints, parameters, and request/response formats, **it is highly recommended to consult the interactive API documentation directly**. The `DKG_API_使用指南.md` (in Chinese) provides a more workflow-oriented and conceptual explanation.

### Key API Endpoint Overview

- **`GET /api/status`**: Checks the health of the service.
- **`GET /api/student/{student_id}/profile`**: Retrieves a complete student profile.
- **`POST /api/interaction`**: Records a single student interaction to update the graph in real-time.
- **`GET /api/problem/{problem_id}/similar`**: (GNN Feature) Finds similar problems.
- **`GET /api/skill/{skill_id}/similar`**: (GNN Feature) Finds similar skills.
- ...and more, please refer to the API documentation.

## 📊 Datasets

The system currently uses a public subset of the **ASSISTments 2009-2010 Skill Builder** dataset. The relevant data files are already included in the `dataset/clear_dataset` directory of this repository and do not require separate downloads. 