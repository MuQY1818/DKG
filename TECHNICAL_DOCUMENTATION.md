# DKG 项目技术详解

## 1. 系统核心架构

本项目的核心是一个基于 `networkx.MultiDiGraph` 构建的动态知识图谱。其架构可分为以下几个层次：

- **数据层 (Data Layer)**: 位于最底层，由原始的CSV日志文件组成，例如 `skill_builder_data09-10.csv`。这些文件记录了最基本的学生-题目交互事件。
- **加载与预处理层 (Loading & Preprocessing Layer)**: 由 `data_loader.py` 负责。它读取原始CSV文件，进行数据清洗、格式转换、ID映射等预处理工作，并将其封装成统一格式的数据字典，供上层使用。
- **图谱构建与更新层 (Graph Construction & Update Layer)**: 这是系统的核心，由 `dkg_builder.py` 实现。它负责：
    1.  **静态构建**: 根据预处理后的数据，初始化整个知识图谱，创建包含学生、题目、技能的节点，以及它们之间的初始关系。
    2.  **动态更新**: 提供 `record_interaction` 等接口，在接收到新的学习行为后，实时更新图谱中的节点属性和关系权重，最核心的是更新学生对技能的"掌握度"。
- **API与应用层 (API & Application Layer)**: `DKGBuilder` 类同样暴露了一系列查询和推荐接口（如 `get_student_profile`, `recommend_next_problems`），供上层应用（例如 `run_api_example.py` 或未来的LLM集成应用）调用，以获取决策支持。

数据流向：
`CSV文件` -> `DataLoader` -> `结构化字典` -> `DKGBuilder` -> `动态知识图谱 (内存中)` -> `API调用` -> `应用结果 (画像, 推荐等)`

---

## 2. 核心数据模型详解

图谱采用 `networkx.MultiDiGraph`，因为它允许节点间存在多种不同类型的有向边，非常适合描述教育场景中的复杂关系。

### 2.1 节点 (Nodes)

图谱中主要有三种类型的节点：`student`, `problem`, `skill`。

#### a) 学生节点 (`student`)
- **唯一标识**: `student_{student_id}` (例如: `student_0`)
- **类型属性**: `type='student'`
- **核心属性**:
    - `student_id` (int): 学生的数字索引ID。
    - `learning_rate` (float): 学生的学习效率，一个通过正态分布随机生成的个性化参数。
    - `perseverance` (int): 学生的毅力参数，代表能容忍的连续失败次数。
    - `curiosity` (float): 学生的好奇心或探索欲，影响其探索新知识领域的倾向。

#### b) 题目节点 (`problem`)
- **唯一标识**: `problem_{problem_id}` (例如: `problem_15`)
- **类型属性**: `type='problem'`
- **核心属性**:
    - `problem_id` (int): 题目的数字索引ID。
    - `problem_type` (str): 题目类型（如 `objective`）。
    - `max_score` (float): 题目的满分。
    - `difficulty` (float): 题目的难度系数，会根据学生的答题情况动态更新。
    - `discrimination` (float): 题目的区分度参数。

#### c) 技能节点 (`skill`)
- **唯一标识**: `skill_{skill_id}` (例如: `skill_25`)
- **类型属性**: `type='skill'`
- **核心属性**:
    - `skill_id` (int): 技能的数字索引ID。
    - `skill_name` (str): 技能的文本名称 (例如: "Addition and Subtraction of Integers")。
    - `difficulty_level` (float): 技能本身的难度。
    - `importance_weight` (float): 技能的重要性权重。

### 2.2 边 (Edges / Relations)

边代表了节点之间的关系，是图谱动态性的核心体现。

#### a) `solve` (学生 -> 题目)
- **描述**: 表示一个学生完成了一道题目。这是图谱中最基础、最频繁发生的事件。
- **类型属性**: `type='solve'`
- **核心属性**:
    - `correct` (int): 是否正确 (1或0)。
    - `score` (float): 本次作答的分数。
    - `attempts` (int): 尝试次数。
    - `time_taken` (int): 答题耗时（毫秒）。
    - `hints_used` (int): 使用提示的数量。

#### b) `require` (题目 -> 技能)
- **描述**: 表示一道题目考察了一个或多个技能。这个关系通常是静态的，在图谱构建之初就根据Q矩阵（Problem-Skill Matrix）确定。
- **类型属性**: `type='require'`
- **核心属性**:
    - `weight` (float): 该技能在题目中的重要性权重。

#### c) `master` (学生 -> 技能)
- **描述**: 这是整个学生模型的核心。它表示一个学生对某个技能的掌握程度。这条边的属性是动态更新的。
- **类型属性**: `type='master'`
- **核心属性**:
    - `mastery_score` (float): **核心指标**，一个在 [0, 1] 区间内的浮点数，代表学生对该技能的当前掌握水平。
    - `history` (list): 记录了与该技能相关的历次交互结果（例如 `[1, 0, 1]` 表示三次交互，两次正确一次错误）。
    - `last_updated` (timestamp): 最近一次更新的时间。

#### d) `prerequisite` (技能 -> 技能)
- **描述**: 表示技能之间的前置依赖关系（例如，必须先学"一元一次方程"才能学"一元二次方程"）。这是通过分析学生学习序列的共现关系推断出来的。
- **类型属性**: `type='prerequisite'`
- **核心属性**:
    - `confidence` (float): 该依赖关系的置信度。

#### e) `similar` (技能 -> 技能)
- **描述**: 表示两个技能在考察内容上具有相似性。这是通过计算技能在题目上的向量表示（基于Q矩阵）的余弦相似度得出的。
- **类型属性**: `type='similar'`
- **核心属性**:
    - `similarity` (float): 两个技能的相似度分数。

---
*下一部分将深入讲解关键的动态流程：图谱构建、交互更新和问题推荐。*

## 3. 关键流程详解

这部分将深入剖析DKG的核心动态流程。

### 3.1 图谱构建 (`build_from_data`)

图谱的初始构建是一个静态过程，它将预处理好的数据转化为一个结构化的图。主要步骤如下：
1.  **创建节点**:
    -   `_create_nodes` 方法被调用。
    -   根据传入的学生、题目、技能数量，批量创建 `student`, `problem`, `skill` 节点，并赋予它们在"2.1 节点"部分描述的初始属性。
2.  **创建基础关系**:
    -   `_create_basic_relations` 方法被调用，它会接着调用三个子方法：
    -   `_create_solve_relations`: 遍历所有交互日志，为每一次学生与题目的交互，在对应的节点间添加一条 `solve` 边，并记录下答题结果、耗时等信息。
    -   `_create_require_relations`: 基于Q矩阵（`problem_skill_matrix`），在题目和其所需的技能之间建立 `require` 边。
    -   `_create_master_relations`: 基于已建立的 `solve` 和 `require` 关系，为每个学生初始化他们与每个技能的 `master` 关系。初始的 `mastery_score` 是通过计算学生在某个技能下的所有相关题目的平均正确率得出的。
3.  **推导隐含关系**:
    -   `_infer_skill_prerequisites`: 分析所有学生的答题序列，通过计算技能对的出现顺序和频率，推断出技能间的 `prerequisite`（前置）关系。
    -   `_compute_skill_similarities`: 将每个技能表示为一个向量（基于它在Q矩阵中的行），然后通过计算余弦相似度，建立技能间的 `similar` 关系。

### 3.2 交互更新 (`record_interaction`)

这是DKG"动态"特性的核心。当一个新交互发生时，此方法被调用，执行以下连锁更新：

1.  **查找相关节点**: 根据传入的 `student_id` 和 `problem_id`，在图谱中定位到相应的学生和题目节点。
2.  **更新`solve`关系**:
    -   在学生和题目节点之间添加一条新的 `solve` 边，记录本次交互的 `correct`, `timestamp` 等详细信息。`MultiDiGraph` 允许多条同类边，因此历史交互记录得以保留。
3.  **触发`mastery`更新**:
    -   调用内部方法 `_update_skill_mastery`。
    -   此方法首先找到该题目 `require` 的所有技能。
    -   对于每一个相关技能，它会更新学生到该技能的 `master` 边：
        -   将本次答题结果（1或0）追加到 `history` 列表中。
        -   基于更新后的 `history` 列表，重新计算 `mastery_score`。当前的计算方法是 **取 `history` 列表的简单移动平均值（Simple Moving Average）**，例如，只考虑最近的N次交互结果的平均值，这使得模型能更快地反映学生近期的状态变化。
4.  **（可选）强化传播**:
    -   `_propagate_reinforcement` 方法被调用。
    -   当一个技能的掌握度发生变化时，该变化会以一定的衰减因子（`decay_factor`）传播到其`prerequisite`（前置）和`similar`（相似）的技能上，对它们的掌握度也产生微小的正面或负面影响。这模拟了知识的"举一反三"和"温故知新"效应。

### 3.3 问题推荐 (`recommend_next_problems`)

推荐算法的目标是为学生找到最合适的"下一个"练习题，以最高效率提升其知识水平。

1.  **获取知识状态**:
    -   调用 `get_student_knowledge_state` 方法，获取该学生所有 `master` 关系边的列表，即学生对所有技能的掌握度分数。
2.  **识别薄弱技能**:
    -   对技能列表按 `mastery_score`进行升序排序，找到学生掌握最差的N个技能。
3.  **搜寻候选题目**:
    -   遍历这些薄弱技能，找到所有 `require` 这些技能的题目。
    -   同时，排除掉该学生近期已经做过或做对过的题目，避免重复练习。
4.  **计算题目适合度**:
    -   对每一个候选题目，调用 `_calculate_problem_suitability` 方法计算其"适合度"分数。
    -   这个适合度是一个综合性指标，它考虑了：
        -   **知识匹配度**: 题目所需技能是否与学生的当前薄弱点高度匹配。
        -   **难度匹配度 (ZPD理论)**: 题目的难度 (`difficulty`) 是否略高于学生当前的知识水平 (`mastery_score`)，即处在学生的"最近发展区"(Zone of Proximal Development)。太简单或太难的题目适合度会降低。
        -   **多样性/探索性**: 引入学生的 `curiosity` 参数，偶尔推荐一些全新领域的题目，以避免学生陷入局部最优。
5.  **返回最终推荐**:
    -   根据适合度分数对所有候选题目进行降序排序，返回得分最高的 `num_recommendations` 个题目作为最终推荐列表。 

---

## 4. 模块功能拆解

除了核心的 `dkg_builder.py`，`dkg_mvp` 目录下还包含以下辅助模块：

-   **`data_loader.py`**:
    -   **功能**: 数据的入口。负责从不同的数据源（如Assistments, Math2015等）加载原始数据。
    -   **职责**: 执行数据清洗、缺失值处理、ID到索引的映射、数据结构统一化等预处理任务，为 `DKGBuilder` 提供格式标准的输入。

-   **`api_tests.py`**:
    -   **功能**: 单元测试和API验证。
    -   **职责**: 包含一系列针对 `DKGBuilder` 各个公开API的测试用例，确保图谱的构建、更新、查询、推荐等功能符合预期，是保证代码质量和稳定性的关键。

-   **`simulation.py`**:
    -   **功能**: 学习过程模拟器。
    -   **职责**: 提供一个可以模拟学生学习行为的引擎。可以用于在没有真实数据流的情况下，测试DKG的动态更新和推荐逻辑是否闭环，也可用于进行算法效果的对比实验。

-   **`visualization.py` & `interactive_visualization.py`**:
    -   **功能**: 可视化工具。
    -   **职责**: `visualization.py` 提供静态的图谱可视化方法（如将整个图或其子图绘制成图片）。`interactive_visualization.py` 则利用 `pyvis` 等库，生成可交互的HTML文件，允许用户在浏览器中缩放、拖动、点击节点来探索图谱结构。

-   **`convert_data.py` & `examples.py`**:
    -   **功能**: 工具和示例脚本。
    -   **职责**: `convert_data.py` 可能包含一些一次性的数据格式转换脚本。`examples.py` 则可能提供了一些更具体的、针对特定功能点的API使用范例。

-   **`requirements.txt`**:
    -   **功能**: 项目依赖列表。
    -   **职责**: 列出了运行本项目所需的所有Python第三方库及其版本，方便一键安装。 