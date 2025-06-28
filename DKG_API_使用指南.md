# DKG (动态知识图谱) API 使用指南

本文档旨在帮助开发者理解和使用 `DKGBuilder` 提供的API接口，以便查询和更新动态知识图谱。这些接口特别为与大语言模型（LLM）的集成作了优化。

---

## 核心概念：`DKGBuilder`

所有的操作都围绕 `dkg_mvp.dkg_builder.DKGBuilder` 这个核心类展开。它既是图谱的构建器，也是图谱在内存中的容器和交互入口。

一个 `DKGBuilder` 实例持有一个 `networkx.MultiDiGraph` 对象，该对象存储了所有学生、题目、技能及其之间的关系。

---

## 1. DKG的初始化与生命周期管理

一个典型的DKG工作流包含三个阶段：**构建、保存和加载**。

### a) 从原始数据首次构建DKG

当您第一次使用一个数据集时，需要从原始数据构建图谱。

```python
from dkg_mvp.data_loader import DataLoader
from dkg_mvp.dkg_builder import DKGBuilder

# 1. 初始化数据加载器
loader = DataLoader('dataset') 
# 'dataset' 是存放所有数据文件的根目录

# 2. 加载特定数据集的日志数据
# 这会处理排序、去重、ID映射等所有预处理步骤
log_data = loader.load_assistments_log_data(dataset_name='skill_builder')

# 3. 初始化构建器并从数据构建图谱
if log_data:
    builder = DKGBuilder()
    builder.build_from_data(log_data)
    print("DKG 构建完成!")
    print(f"图中有 {builder.graph.number_of_nodes()} 个节点和 {builder.graph.number_of_edges()} 条边。")
```

### b) 保存构建好的DKG

为了避免每次都重复构建，您可以将内存中的图谱对象持久化到硬盘。我们推荐使用 `.graphml` 格式。

```python
# 假设 builder 是上一步构建好的实例
DKG_SAVE_PATH = "models/dkg_skill_builder.graphml" 
# (请确保 'models' 目录已存在)

builder.save_graph(DKG_SAVE_PATH)
# 输出: DKG successfully saved to models/dkg_skill_builder.graphml
```

### c) 从文件加载已存在的DKG

在后续的程序运行中，您可以直接从文件加载图谱，这比重新构建要快得多。

```python
from dkg_mvp.dkg_builder import DKGBuilder

DKG_SAVE_PATH = "models/dkg_skill_builder.graphml"

# 使用类方法加载，无需先实例化
builder = DKGBuilder.load_graph(DKG_SAVE_PATH)

print("DKG 加载完成!")
print(f"图中有 {builder.graph.number_of_nodes()} 个节点和 {builder.graph.number_of_edges()} 条边。")
```
---

## 2. API 接口参考

一旦您有了一个加载了图谱的 `builder` 实例，就可以使用以下接口进行交互。

### 更新接口 (Update APIs)

#### `record_interaction(interaction: Dict)`
这是最核心的更新接口。当学生有一次新的答题行为时，调用此接口来更新DKG。

- **参数**: `interaction` (dict) - 包含单次交互记录的字典。必须包含 `student_id`, `problem_id`, `correct`。
- **示例**:
```python
new_interaction = {
    'student_id': 10,       # 学生的索引ID
    'problem_id': 5,        # 题目的索引ID
    'correct': 1,           # 答题结果 (1:正确, 0:错误)
    'score': 1.0,           # 分数 (可选)
    'time_taken': 35000,    # 答题耗时(ms) (可选)
    'timestamp': pd.Timestamp.now() # 交互时间戳 (可选, 但建议提供)
}

builder.record_interaction(new_interaction)
# 这将自动更新学生-题目(solve)关系，并级联更新学生-技能(master)的掌握度
```

### 查询接口 (Query APIs)

这些接口用于提取信息，非常适合用于构建给LLM的Prompt。

#### `get_student_profile(student_id: int) -> Dict`
获取某个学生的完整画像。

- **参数**: `student_id` (int) - 学生的索引ID。
- **返回**: 包含学生强弱项、个性化参数和最近活动的字典。
- **示例**:
```python
student_profile = builder.get_student_profile(student_id=10)
import json
print(json.dumps(student_profile, indent=2))
```

#### `get_skill_details(skill_id: int) -> Dict`
获取某个技能的详细信息，包括其在知识图谱中的位置。

- **参数**: `skill_id` (int) - 技能的索引ID (从1开始)。
- **返回**: 包含技能名称、先修/后继技能、相关题目等信息的字典。
- **示例**:
```python
skill_info = builder.get_skill_details(skill_id=25) # 查询ID为25的技能
print(json.dumps(skill_info, indent=2))
```

#### `get_problem_details(problem_id: int) -> Dict`
获取某个题目的详细信息。

- **参数**: `problem_id` (int) - 题目的索引ID。
- **返回**: 包含题目类型、难度、所需技能等信息的字典。
- **示例**:
```python
problem_info = builder.get_problem_details(problem_id=5)
print(json.dumps(problem_info, indent=2))
```

#### `recommend_next_problems(student_id: int, num_recommendations: int = 5) -> List[Dict]`
基于学生的知识薄弱点，为其推荐最合适的练习题。

- **参数**: `student_id` (int), `num_recommendations` (int, 可选)。
- **返回**: 一个题目推荐列表，每个元素都是包含推荐理由和适合度分数的字典。
- **示例**:
```python
recommendations = builder.recommend_next_problems(student_id=10)
print(json.dumps(recommendations, indent=2))
```

#### `generate_llm_prompt(...)`
这是一个高级接口，它将多个低层级的查询结果组装成一个结构化的、信息丰富的Prompt，可以直接发送给大语言模型，用于生成个性化学习路径等任务。

- **示例**:
```python
prompt_for_llm = builder.generate_llm_prompt(
    student_id=10, 
    target_skill_ids=[50, 52] # 假设学生想学习技能50和52
)
print(prompt_for_llm)
```

---

## 3. 完整工作流示例

下面是一个典型的应用场景，展示了如何将上述API串联起来。

```python
import os
from dkg_mvp.dkg_builder import DKGBuilder
import pandas as pd
import json

MODELS_DIR = "models"
DKG_SAVE_PATH = os.path.join(MODELS_DIR, "dkg_skill_builder.graphml")

# --- 第一部分: 系统初始化 ---
# 检查是否存在已保存的DKG模型，如果否则构建并保存一个
if not os.path.exists(DKG_SAVE_PATH):
    print("未找到已保存的DKG，正在从头构建...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    from dkg_mvp.data_loader import DataLoader
    
    loader = DataLoader('dataset')
    log_data = loader.load_assistments_log_data('skill_builder')
    
    if log_data:
        builder = DKGBuilder()
        builder.build_from_data(log_data)
        builder.save_graph(DKG_SAVE_PATH)
        print("新的DKG已构建并保存。")
else:
    print("从文件加载已存在的DKG...")
    builder = DKGBuilder.load_graph(DKG_SAVE_PATH)
    print("DKG加载完毕。")

# --- 第二部分: 模拟一次新的学习交互并观察变化 ---
if 'builder' in locals():
    student_id_to_test = 20
    
    # 1. 查看交互前的学生画像
    print(f"\n--- 交互前，学生 {student_id_to_test} 的知识画像 ---")
    profile_before = builder.get_student_profile(student_id_to_test)
    print(json.dumps(profile_before['knowledge_summary'], indent=2))

    # 2. 模拟一次交互：该学生正确解答了问题15
    print(f"\n--- 模拟交互：学生 {student_id_to_test} 做对了题目 15 ---")
    interaction = {
        'student_id': student_id_to_test,
        'problem_id': 15,
        'correct': 1,
        'timestamp': pd.Timestamp.now()
    }
    builder.record_interaction(interaction)
    
    # 3. 查看交互后的学生画像，对比变化
    print(f"\n--- 交互后，学生 {student_id_to_test} 的新知识画像 ---")
    profile_after = builder.get_student_profile(student_id_to_test)
    print(json.dumps(profile_after['knowledge_summary'], indent=2))
    
    # 4. 基于更新后的状态，为学生推荐下一步练习
    print(f"\n--- 为学生 {student_id_to_test} 推荐下一步练习 ---")
    recommendations = builder.recommend_next_problems(student_id_to_test)
    print(json.dumps(recommendations, indent=2, ensure_ascii=False))

``` 