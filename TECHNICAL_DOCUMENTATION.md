# DKG 项目技术详解

本文档提供对DKG项目的深度技术剖析，包括其核心架构、数据模型、关键算法及其数学实现。

---

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
    - `learning_rate` (float): **[个性化参数]** 学生的学习效率，影响其掌握度更新速度。
    - `perseverance` (int): **[个性化参数]** 学生的毅力，影响其面对挫折的行为。
    - `curiosity` (float): **[个性化参数]** 学生的好奇心，影响其对新知识的探索倾向。

#### b) 题目节点 (`problem`)
- **唯一标识**: `problem_{problem_id}` (例如: `problem_15`)
- **类型属性**: `type='problem'`
- **核心属性**:
    - `problem_id` (int): 题目的数字索引ID。
    - `problem_type` (str): 题目类型（如 `objective`）。
    - `max_score` (float): 题目的满分。
    - `difficulty` (float): **[动态属性]** 题目的难度系数。
        - **初始值**: 0.5。
        - **更新机制**: *在当前版本中，该属性暂未实现动态更新。未来的版本可以基于所有学生的作答正确率进行调整。*
    - `discrimination` (float): **[静态属性]** 题目的区分度，衡量其区分高低水平学生的能力。

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
    - `mastery_score` (float): **[核心动态指标]** 对该技能的当前掌握水平。
        - **计算方法**: 详见 3.2.2 节。
    - `history` (list): 记录了与该技能相关的历次交互正误（`[1, 0, 1]`）。
    - `last_updated` (timestamp): 最近一次更新的时间。

#### d) `prerequisite` (技能 -> 技能)
- **描述**: 表示技能之间的前置依赖关系（例如，必须先学"一元一次方程"才能学"一元二次方程"）。这是通过分析学生学习序列的共现关系推断出来的。
- **类型属性**: `type='prerequisite'`
- **核心属性**:
    - `confidence` (float): 该依赖关系的置信度。
        - **计算方法**: 基于学生学习路径中共现技能对的条件概率。

#### e) `similar` (技能 -> 技能)
- **描述**: 表示两个技能在考察内容上具有相似性。这是通过计算技能在题目上的向量表示（基于Q矩阵）的余弦相似度得出的。
- **类型属性**: `type='similar'`
- **核心属性**:
    - `similarity` (float): 相似度分数。
        - **计算方法**: 详见 3.1.2 节。

---
*下一部分将深入讲解关键的动态流程：图谱构建、交互更新和问题推荐。*

## 3. 关键算法与流程详解

### 3.1 静态图谱构建 (`build_from_data`)

此过程在首次加载数据时执行，建立图谱的骨架。

#### 3.1.1 初始掌握度计算
在创建 `master` 关系时，学生的初始掌握度 $M_{initial}$ 是根据其在相关历史题目上的平均表现计算的：
$$
M_{initial}(s, k) = \frac{\sum_{p \in P_k} C(s, p)}{|P_k|}
$$
其中，$s$ 是学生，$k$ 是技能，$P_k$ 是学生 $s$ 已作答的、考察技能 $k$ 的题目集合，$C(s, p)$ 是学生在题目 $p$ 上的作答结果（1为正确，0为错误）。如果 $|P_k|=0$，则初始值默认为0.5。

#### 3.1.2 技能相似度计算
技能间的 `similar` 关系是通过计算其向量表示的余弦相似度得出的。首先，根据Q矩阵（problem-skill matrix）为每个技能 $k$ 构建一个向量 $\vec{v_k}$，该向量的维度等于总题目数，如果题目 $p$ 考察技能 $k$，则向量的第 $p$ 个分量为1，否则为0。
技能 $k_i$ 和 $k_j$ 的相似度 $\text{sim}(k_i, k_j)$ 计算如下：
$$
\text{sim}(k_i, k_j) = \frac{\vec{v_{k_i}} \cdot \vec{v_{k_j}}}{||\vec{v_{k_i}}|| \cdot ||\vec{v_{k_j}}||}
$$

### 3.2 动态交互更新 (`record_interaction`)

这是系统的核心动态过程。

#### 3.2.1 知识状态更新
当学生 $s$ 与题目 $p$ 发生一次新的交互，其结果为 $c_t$ (1或0) 时，系统会找到该题目所需的所有技能 $\{k_1, k_2, ...\}$。对其中每一个技能 $k_i$，系统会更新学生与该技能的 `master` 关系。

#### 3.2.2 掌握度分数 (`mastery_score`) 计算
`mastery_score` 的更新采用**带遗忘因子的移动平均法**，以更好地反映学生当前的状态，而非遥远的历史。
首先，更新交互历史 $H_{new}(s, k) = H_{old}(s, k) \cup \{c_t\}$。
然后，取最近的 $N$ 次历史记录（$N$ 为窗口大小，例如10），计算其均值：
$$
M_{t}(s, k) = \frac{1}{N} \sum_{i=t-N+1}^{t} c_i
$$
这个值将成为新的 `mastery_score`。这种方法使得近期的表现比远期的表现有更大的影响力。

#### 3.2.3 知识强化传播 (`_propagate_reinforcement`)
当一个技能 $k_{primary}$ 的掌握度发生变化 $\Delta M$ 时，该变化会以一定衰减 $\delta$ (例如, `decay_factor=0.4`) 传播到其相邻的技能（前置或相似技能 $k_{related}$）上，实现"举一反三"的效果。
$$
\Delta M_{propagated} = \Delta M_{primary} \times \delta \times w_{relation}
$$
$$
M_{new}(s, k_{related}) = \text{clip}(M_{old}(s, k_{related}) + \Delta M_{propagated}, 0, 1)
$$
其中 $w_{relation}$ 是关系权重（如 `similarity` 或 `confidence`）。

### 3.3 问题推荐 (`recommend_next_problems`)

推荐算法的核心是为学生找到"最适合"的下一道题。

#### 3.3.1 题目适合度 (`suitability`) 计算
对每个候选题目 $p$，系统会计算一个适合度分数 $S(s, p)$，该分数由三个部分加权组成：
$$
S(s, p) = w_{knowledge} \cdot F_{knowledge} + w_{zpd} \cdot F_{zpd} + B_{novelty}
$$
-   **知识匹配度 $F_{knowledge}$**: 衡量题目是否针对学生的薄弱点。
    $$
    F_{knowledge} = 1 - M(s, k_p)
    $$
    其中 $M(s, k_p)$ 是学生对该题目所考察技能的掌握度。掌握度越低，匹配度越高。

-   **难度匹配度 $F_{zpd}$**: 基于维果茨基的"最近发展区"理论，认为最合适的题目其难度应略高于学生当前水平。这通过一个高斯函数来实现：
    $$
    F_{zpd} = e^{-\frac{(D_p - M(s, k_p) - \mu)^2}{2\sigma^2}}
    $$
    其中 $D_p$ 是题目难度，$\mu$ 是最佳难度差（例如0.15），$\sigma$ 是容忍范围。当 $D_p - M(s, k_p)$ 约等于 $\mu$ 时，该项得分最高。

-   **新颖度奖励 $B_{novelty}$**: 如果题目考察了学生从未接触过的新技能，则给与一个基于其 `curiosity` 属性的奖励分，鼓励探索。
    $$
    B_{novelty} = \begin{cases} \text{curiosity}_s & \text{if } p \text{ is novel} \\ 0 & \text{otherwise} \end{cases}
    $$

最后，系统会选择 $S(s, p)$ 分数最高的题目进行推荐。

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