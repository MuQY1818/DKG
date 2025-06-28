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

### 3.2 Dynamic Interaction Update (`record_interaction`)

This is the core dynamic process of the system.

#### 3.2.1 Knowledge State Update
When a student $s$ has a new interaction with a problem $p$ with an outcome of $c_t$ (1 or 0), the system identifies all skills $\{k_1, k_2, ...\}$ required by that problem. For each of these skills $k_i$, the system updates the `master` relationship between the student and the skill.

#### 3.2.2 Mastery Score (`mastery_score`) Calculation
The `mastery_score` is updated using a **moving average method with a forgetting factor** to better reflect the student's current state rather than distant history.
First, the interaction history is updated: $H_{new}(s, k) = H_{old}(s, k) \cup \{c_t\}$.
Then, the mean of the most recent $N$ history records is calculated ($N$ is the window size, e.g., 10):
$$
M_{t}(s, k) = \frac{1}{N} \sum_{i=t-N+1}^{t} c_i
$$
This value becomes the new `mastery_score`. This method gives more weight to recent performance.

#### 3.2.3 Knowledge Reinforcement Propagation (`_propagate_reinforcement`)
When the mastery of a skill $k_{primary}$ changes by $\Delta M$, this change propagates with a certain decay $\delta$ (e.g., `decay_factor=0.4`) to its adjacent skills (prerequisite or similar skills $k_{related}$), simulating a "transfer of learning" effect.
$$
\Delta M_{propagated} = \Delta M_{primary} \times \delta \times w_{relation}
$$
$$
M_{new}(s, k_{related}) = \text{clip}(M_{old}(s, k_{related}) + \Delta M_{propagated}, 0, 1)
$$
where $w_{relation}$ is the relationship weight (e.g., `similarity` or `confidence`).

### 3.3 Problem Recommendation (`recommend_next_problems`)

The core of the recommendation algorithm is to find the "most suitable" next problem for a student.

#### 3.3.1 Problem Suitability (`suitability`) Calculation
For each candidate problem $p$, the system calculates a suitability score $S(s, p)$, which is a weighted sum of three components:
$$
S(s, p) = w_{knowledge} \cdot F_{knowledge} + w_{zpd} \cdot F_{zpd} + B_{novelty}
$$
-   **Knowledge Fit $F_{knowledge}$**: Measures if the problem targets the student's weak points.
    $$
    F_{knowledge} = 1 - M(s, k_p)
    $$
    where $M(s, k_p)$ is the student's mastery of the skill tested by the problem. The lower the mastery, the higher the fit.

-   **Difficulty Fit $F_{zpd}$**: Based on Vygotsky's "Zone of Proximal Development" theory, the most suitable problem should be slightly more difficult than the student's current level. This is implemented using a Gaussian function:
    $$
    F_{zpd} = e^{-\frac{(D_p - M(s, k_p) - \mu)^2}{2\sigma^2}}
    $$
    where $D_p$ is the problem difficulty, $\mu$ is the optimal difficulty gap (e.g., 0.15), and $\sigma$ is the tolerance. The score is highest when $D_p - M(s, k_p)$ is close to $\mu$.

-   **Novelty Bonus $B_{novelty}$**: If the problem tests a skill the student has never encountered, a bonus score based on their `curiosity` attribute is given to encourage exploration.
    $$
    B_{novelty} = \begin{cases} \text{curiosity}_s & \text{if } p \text{ is novel} \\ 0 & \text{otherwise} \end{cases}
    $$

Finally, the system recommends the problems with the highest $S(s, p)$ scores.

---

## 4. Module Breakdown

(This section remains unchanged from v1.0)
... 