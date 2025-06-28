# DKG Project Technical Deep Dive

This document provides a deep technical analysis of the DKG project, including its core architecture, data model, key algorithms, and their mathematical implementations.

---

## 1. Core System Architecture

(This section remains unchanged from v1.0)

The core of this project is a dynamic knowledge graph built upon `networkx.MultiDiGraph`. Its architecture can be divided into the following layers:

- **Data Layer**: The lowest layer, consisting of raw CSV log files.
- **Loading & Preprocessing Layer**: Handled by `data_loader.py`, responsible for data cleaning, format conversion, and ID mapping.
- **Graph Construction & Update Layer**: The system's core, implemented in `dkg_builder.py`, responsible for the static construction and dynamic updates of the graph.
- **API & Application Layer**: The `DKGBuilder` class exposes a series of query and recommendation APIs for higher-level applications.

---

## 2. Core Data Model Explained

The graph uses `networkx.MultiDiGraph` to allow multiple types of directed edges between nodes.

### 2.1 Nodes

#### a) Student Node
- **Key Attributes**:
    - `student_id` (int): The numerical index ID of the student.
    - `learning_rate` (float): **[Personalization Param]** The student's learning efficiency, affecting their mastery update speed.
    - `perseverance` (int): **[Personalization Param]** The student's persistence, affecting their behavior when facing setbacks.
    - `curiosity` (float): **[Personalization Param]** The student's curiosity, influencing their tendency to explore new knowledge areas.

#### b) Problem Node
- **Key Attributes**:
    - `problem_id` (int): The numerical index ID of the problem.
    - `difficulty` (float): **[Dynamic Attribute]** The difficulty coefficient of the problem.
        - **Initial Value**: 0.5.
        - **Update Mechanism**: *In the current version, this attribute's dynamic update is not yet implemented. Future versions could adjust it based on the overall correctness rate from all students.*
    - `discrimination` (float): **[Static Attribute]** The problem's discrimination parameter, measuring its ability to distinguish between high and low-level students.

#### c) Skill Node
- **Key Attributes**:
    - `skill_id` (int): The numerical index ID of the skill.
    - `skill_name` (str): The text name of the skill.

### 2.2 Edges / Relations

#### a) `solve` (Student -> Problem)
- **Description**: Records a student-problem interaction event.

#### b) `require` (Problem -> Skill)
- **Description**: Statically represents the skills tested by a problem.

#### c) `master` (Student -> Skill)
- **Description**: **Core of the model**, dynamically represents a student's mastery of a skill.
- **Key Attributes**:
    - `mastery_score` (float): **[Core Dynamic Metric]** The current mastery level of the skill.
        - **Calculation Method**: See Section 3.2.2.
    - `history` (list): Records the history of correctness for interactions related to this skill (e.g., `[1, 0, 1]`).

#### d) `prerequisite` (Skill -> Skill)
- **Description**: Prerequisite dependency relationship between skills.
- **Key Attributes**:
    - `confidence` (float): The confidence level of the dependency.
        - **Calculation Method**: Based on the conditional probability of co-occurring skill pairs in student learning paths.

#### e) `similar` (Skill -> Skill)
- **Description**: Similarity between skills.
- **Key Attributes**:
    - `similarity` (float): The similarity score.
        - **Calculation Method**: See Section 3.1.2.

---

## 3. Key Algorithms and Processes Explained

### 3.1 Static Graph Construction (`build_from_data`)

This process is executed when loading data for the first time, building the graph's skeleton.

#### 3.1.1 Initial Mastery Calculation
When creating a `master` relationship, a student's initial mastery $M_{initial}$ is calculated based on their average performance on related historical problems:
$$
M_{initial}(s, k) = \frac{\sum_{p \in P_k} C(s, p)}{|P_k|}
$$
where $s$ is the student, $k$ is the skill, $P_k$ is the set of problems testing skill $k$ that student $s$ has attempted, and $C(s, p)$ is the student's result on problem $p$ (1 for correct, 0 for incorrect). If $|P_k|=0$, the initial value defaults to 0.5.

#### 3.1.2 Skill Similarity Calculation
The `similar` relationship between skills is derived by calculating the cosine similarity of their vector representations. First, for each skill $k$, a vector $\vec{v_k}$ is constructed from the Q-matrix (problem-skill matrix). The vector's dimension equals the total number of problems, and the $p$-th component is 1 if problem $p$ tests skill $k$, and 0 otherwise.
The similarity $\text{sim}(k_i, k_j)$ between skill $k_i$ and $k_j$ is calculated as follows:
$$
\text{sim}(k_i, k_j) = \frac{\vec{v_{k_i}} \cdot \vec{v_{k_j}}}{||\vec{v_{k_i}}|| \cdot ||\vec{v_{k_j}}||}
$$

#### 3.1.3 Skill Prerequisite Inference (`_infer_skill_prerequisites`)
The `prerequisite` relationship is automatically discovered by analyzing the structure of the **Question-Skill Matrix (Q-matrix)**, rather than analyzing the temporal sequence of student learning. The core idea is that if the presence of Skill B almost always implies the presence of Skill A, but the reverse is not true, then A is likely a prerequisite for B.

1.  **Calculate Skill Co-occurrence Matrix**: The system first computes a skill-skill co-occurrence matrix, where each element $(i, j)$ represents the number of questions that require both skill $i$ and skill $j$.

2.  **Calculate Conditional Probabilities**: For any pair of skills $(A, B)$, the system calculates the conditional probabilities in both directions:
    -   $P(A|B)$: The probability that a question involving skill B also involves skill A.
    -   $P(B|A)$: The probability that a question involving skill A also involves skill B.

3.  **Establish Relationship**: If $P(A|B)$ is significantly greater than $P(B|A)$ and $P(A|B)$ itself exceeds a high confidence threshold (e.g., 0.7), the system determines that A is a prerequisite for B and establishes a `prerequisite` edge from A to B. The `confidence` attribute of this edge is set to the value of $P(A|B)$.

### 3.2 Dynamic Interaction Updates (`record_interaction`)
When the system records a new learning interaction, it triggers a series of dynamic updates to reflect the changes in the student's cognitive state in real-time. This process is more than just updating a single data point; it simulates a complex cognitive evolution.

The core flow of this function is as follows:
1.  **Update 'solve' Relationship**: It creates or updates the `solve` relationship between the student and problem nodes, recording details of the interaction such as score, attempts, timestamp, etc.

2.  **Update Directly Related Skill Mastery (`_update_skill_mastery`)**:
    - The system identifies all skills directly required by the problem.
    - For each skill, it adjusts the student's `mastery_level` based on the interaction outcome. The core update logic is as follows:
      $$
      \Delta M_{base} = \begin{cases} 0.1 \times \text{learning\_rate} & \text{if correct} \\ -0.02 & \text{if incorrect} \end{cases}
      $$
      $$
      M_{new} = \text{clip}(M_{current} + \Delta M_{base} + \text{bonus}_{epiphany}, 0, 1)
      $$
      Here, `learning_rate` is a personalized student parameter, enabling differential modeling. The code also includes an `is_epiphany` interface, allowing for a significant non-linear increase in mastery under specific conditions.

3.  **Knowledge Reinforcement Propagation (`_propagate_reinforcement`)**: This is the most sophisticated part of the dynamic update.
    - After a skill's mastery level is increased from the interaction, the system initiates a recursive reinforcement process.
    - This process takes the **total change in mastery** ($\Delta M_{total} = \Delta M_{base} + \text{bonus}_{epiphany}$) and propagates it backward to all **direct prerequisite skills**, attenuated by a predefined `decay_factor`.
      $$
      \Delta M_{propagated} = \Delta M_{total} \times \text{decay\_factor}
      $$
      $$
      M_{new\_prereq} = \text{clip}(M_{current\_prereq} + \Delta M_{propagated}, 0, 1)
      $$
    - This propagation continues recursively until there are no further underlying prerequisite skills.
    - **Significance of this mechanism**: It perfectly simulates the "knowledge consolidation" phenomenon in pedagogy—that is, **practicing and applying advanced knowledge serves as an effective review and reinforcement of the relevant foundational knowledge**.

### 3.3 Recommendations and Diagnostics

#### 3.3.1 Problem Recommendation (`recommend_next_problems`)
The system's recommendation module is designed to plan an efficient and user-friendly learning path for students. The design of its core algorithm is not a simple random selection or difficulty ranking but is deeply integrated with the **"Zone of Proximal Development" (ZPD) theory** from educational psychology.

This theory states that the most effective learning occurs in the zone just beyond a student's current ability level but not so far as to be unattainable. The system implements this theory through the following steps:

1.  **Identify Weak Skills**: First, the system identifies skills for which the student's current `mastery_level` is below a certain threshold (e.g., 0.7), marking them as potential learning targets.
2.  **Calculate Problem Suitability (`_calculate_problem_suitability`)**: For each candidate problem under a weak skill, the system calculates a "suitability" score. The core of this score is to find the optimal match between problem difficulty and the student's ability. This is calculated in several steps:
    - **A. Difficulty Suitability**: First, based on the student's mastery of the skill ($M$) and the problem's difficulty ($D$), a base difficulty suitability score ($S_{diff}$) is calculated. The ideal problem difficulty is set to be slightly above the student's current level ($M+0.1$).
      $$
      S_{diff} = \max(0, 1 - 2 \times |D - (M + 0.1)|)
      $$
    - **B. Challenge Factor**: Next, to avoid problems that are too easy or too hard, a challenge factor ($F_{challenge}$) is introduced.
      $$
      F_{challenge} = \begin{cases} 0.5 & \text{if } D < M - 0.2 \text{ (Too Easy)} \\ 0.3 & \text{if } D > M + 0.3 \text{ (Too Hard)} \\ 1.0 & \text{otherwise (Optimal)} \end{cases}
      $$
    - **C. Final Suitability**: The final score is the product of the two.
      $$
      S_{total} = S_{diff} \times F_{challenge}
      $$
    - **Avoid Ineffective Practice**: If a student has already answered a problem correctly, its suitability score is directly set to a low value (e.g., 0.3) to prevent redundant practice.
3.  **Generate Recommendation List**: The system aggregates the suitability scores for all candidate problems and sorts them in descending order, ultimately producing a personalized recommendation list that both targets weak knowledge points and offers an appropriate level of challenge.

#### 3.3.2 Prompt Generation for LLMs (`generate_llm_prompt`)
This function serves as a bridge between the DKG and Large Language Models (LLMs), designed to leverage DKG's structured knowledge to empower LLMs for generating high-quality learning path plans. It does not directly output a learning plan but rather constructs a well-structured, information-rich **Prompt**.

This prompt "translates" the analytical results of the DKG into a context that LLMs can understand and utilize, in the following ways:
1.  **Assigning Role and Task**: It explicitly instructs the LLM to act as an "AI Education Planning Expert".
2.  **Providing Student Profile**:
    - **Knowledge Strengths and Weaknesses**: It extracts the student's skills with the highest and lowest mastery levels, giving the LLM a clear picture of the student's current knowledge state.
    - **Learning Goals**: It clearly informs the LLM of the learning objectives the student wishes to achieve.
3.  **Injecting Core Rules**:
    - **Prerequisite Constraints**: It provides all relevant `prerequisite` relationships extracted from the graph as hard constraints, ensuring the learning path generated by the LLM is logically sound and feasible.

In this way, the DKG injects the most critical structured knowledge (student state, knowledge dependencies) into the LLM's reasoning process, significantly enhancing the accuracy and quality of the personalized, effective learning plans it generates.

<details>
<summary>Click to see a Prompt Generation Example</summary>

```
# **Role:**
You are a top-tier AI Education Planning expert, proficient in cognitive science and instructional design.

# **Task:**
Based on the provided student knowledge profile and knowledge graph rules, design a personalized, step-by-step learning path to help the student master the target skills.

---

# **Student Profile:**

## **1. Basic Information:**
- **Student ID:** 3

## **2. Knowledge Strengths (Mastered Skills):**
- **Linear Equations**: Mastery 0.85
- **Algebraic Foundations**: Mastery 0.92

## **3. Knowledge Weaknesses (Skills to Improve):**
- **Factorization**: Mastery 0.33
- **Introduction to Functions**: Mastery 0.41

---

# **Goal and Constraints:**

## **1. Learning Goal:**
The student wants to systematically learn and master the following skill:
- **Quadratic Equations**

## **2. Mandatory Rules (Knowledge Structure):**
The learning path must strictly adhere to the following prerequisite relationships between knowledge points. A skill can only be learned after all of its prerequisites have been mastered.
- **'Algebraic Foundations'** is a prerequisite for **'Linear Equations'**.
- **'Linear Equations'** is a prerequisite for **'Quadratic Equations'**.
- **'Factorization'** is a prerequisite for **'Quadratic Equations'**.

---

# **Output Format:**

Please provide a clear, step-by-step learning plan. Each step should include:
1.  **The name of the skill to be learned.**
2.  **The reason for recommending this skill** (e.g., because it is a necessary prerequisite for mastering target skill XXX, or it is one of the student's current weaknesses).
3.  **Learning Order:** Please ensure the sequence of the entire plan strictly follows the prerequisite relationships and starts with the knowledge points the student needs to address most urgently.

Please begin generating the learning plan:
```

</details>

## 4. Module Breakdown

(This section remains unchanged from v1.0)

## 4. Appendix

### 4.1 Data Source Compatibility
The `DKGBuilder` is designed with compatibility for various educational data formats in mind. During the `_create_solve_relations` phase of graph construction, the system can automatically detect and process the following three mainstream data types:
- **Log-based Data**: Each row is a complete learning interaction record (e.g., the ASSISTments dataset).
- **Binary Matrix**: A (student x problem) matrix of 0s and 1s, representing correctness (e.g., the FrcSub dataset).
- **Continuous Score Matrix**: A (student x problem) matrix of floating-point numbers, representing score ratios (e.g., the Math1/Math2 datasets).

This flexibility allows the tool to be conveniently applied to a broader range of educational scenarios.
... 