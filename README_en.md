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
├── README.md                # This file, project overview
├── run_api_example.py       # Script demonstrating API usage
├── models/                  # Directory for storing persisted graph models
├── dataset/                 # Directory for raw datasets
│   └── skill_builder_data09-10.csv # Primary dataset used
├── dkg_mvp/
│   ├── dkg_builder.py       # Core DKG class, provides all APIs ⭐
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── simulation.py        # (Optional) Engine for simulating learning processes
│   ├── api_tests.py         # Unit tests and API validation
│   └── requirements.txt     # List of dependencies
└── .gitignore               # Git ignore file
```

## 🚀 Quick Start

### 1. Installation

First, clone the repository and install the required dependencies.

```bash
git clone https://github.com/MuQY1818/DKG.git
cd DKG
pip install -r dkg_mvp/requirements.txt
```

### 2. Running the Example

Execute the `run_api_example.py` script in the project root to see a complete API workflow.

```bash
python run_api_example.py
```

The script will perform the following actions:
- On the first run, it builds the DKG from the `skill_builder_data09-10.csv` dataset and saves it to the `models/` directory.
- On subsequent runs, it loads the existing DKG from the file.
- It then demonstrates:
    1.  Fetching a student's knowledge profile before an interaction.
    2.  Simulating a new learning interaction (e.g., the student answers a problem correctly).
    3.  Fetching the student's updated profile to show the changes.
    4.  Recommending the next problems for the student based on the new profile.

## 🛠️ API Reference

All core functionalities are exposed through the `dkg_mvp.dkg_builder.DKGBuilder` class. For detailed explanations and more examples, please refer to `DKG_API_使用指南.md`.

### Initialization and Persistence

- `DKGBuilder.build_from_data(log_data)`: Builds the graph from preprocessed data.
- `builder.save_graph(path)`: Saves the in-memory graph to a file.
- `DKGBuilder.load_graph(path)`: Loads the graph from a file.

### Dynamic Updates

- `builder.record_interaction(interaction: Dict)`: Records a single student interaction and updates the graph in real-time. The interaction dictionary must include `student_id`, `problem_id`, and `correct`.

### Querying

- `builder.get_student_profile(student_id: int)`: Retrieves a complete profile for a student, including their knowledge summary.
- `builder.get_skill_details(skill_id: int)`: Gets details for a specific skill.
- `builder.get_problem_details(problem_id: int)`: Gets details for a specific problem.

### Recommendations and LLM Integration

- `builder.recommend_next_problems(student_id: int)`: Recommends suitable practice problems based on the student's weaknesses.
- `builder.generate_llm_prompt(...)`: Assembles a structured, informative prompt to be sent to an LLM for tasks like personalized learning path generation.

## 📊 Datasets

The system currently primarily uses the **ASSISTments 2009-2010 Skill Builder** dataset, which contains rich log data of real student problem-solving activities.

**Note:** Due to its size, the `skill_builder_data09-10.csv` dataset is not included in this repository. You will need to download it separately and place it in the `dataset/` directory before running the example script. 