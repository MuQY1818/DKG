document.addEventListener('DOMContentLoaded', () => {
    const studentSelect = document.getElementById('student-select');
    const problemSelect = document.getElementById('problem-select');
    const predictBtn = document.getElementById('predict-btn');
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const errorContainer = document.getElementById('error-container');
    const errorText = document.getElementById('error-text');

    const API_BASE_URL = window.location.origin; // 使用相对路径，适用于 ngrok 和本地

    // --- 数据加载函数 ---
    async function fetchData(endpoint) {
        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`);
            if (!response.ok) {
                throw new Error(`网络响应错误: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            showError(`加载数据失败: ${error.message}`);
            console.error(`Error fetching from ${endpoint}:`, error);
        }
    }

    async function populateStudents() {
        const students = await fetchData('/api/students');
        if (!students) return;

        studentSelect.innerHTML = '<option value="">-- 请选择一个学生 --</option>';
        students.forEach(student => {
            const option = document.createElement('option');
            option.value = student.student_id;
            option.textContent = `学生 ${student.student_id}`;
            studentSelect.appendChild(option);
        });
    }

    async function populateProblems() {
        const problems = await fetchData('/api/problems');
        if (!problems) return;

        problemSelect.innerHTML = '<option value="">-- 请选择一个练习 --</option>';
        problems.forEach(problem => {
            const option = document.createElement('option');
            option.value = problem.problem_id;
            // 如果有题目描述，就用它，否则用ID
            const description = problem.problem_text ? `: ${problem.problem_text.substring(0, 50)}...` : '';
            option.textContent = `练习 ${problem.problem_id}${description}`;
            problemSelect.appendChild(option);
        });
    }

    // --- UI 更新函数 ---
    function showResult(probability) {
        const percentage = (probability * 100).toFixed(2);
        resultText.textContent = `该学生答对这道题的概率是 ${percentage}%。`;
        resultContainer.classList.remove('hidden');
        errorContainer.classList.add('hidden');
    }

    function showError(message) {
        errorText.textContent = message;
        errorContainer.classList.remove('hidden');
        resultContainer.classList.add('hidden');
    }
    
    function hideMessages() {
        errorContainer.classList.add('hidden');
        resultContainer.classList.add('hidden');
    }

    // --- 事件处理函数 ---
    async function handlePrediction() {
        const studentId = studentSelect.value;
        const problemId = problemSelect.value;

        hideMessages();

        if (!studentId || !problemId) {
            showError('请先选择一个学生和一道练习题。');
            return;
        }

        const requestBody = [{
            student_id: parseInt(studentId, 10),
            problem_id: parseInt(problemId, 10)
        }];

        try {
            const response = await fetch(`${API_BASE_URL}/api/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `API 请求失败，状态码: ${response.status}`);
            }

            const data = await response.json();
            if (data && data.length > 0) {
                showResult(data[0].predicted_correct_probability);
            } else {
                showError('API返回了非预期的空数据。');
            }

        } catch (error) {
            showError(`预测时发生错误: ${error.message}`);
            console.error('Prediction error:', error);
        }
    }

    // --- 初始化 ---
    predictBtn.addEventListener('click', handlePrediction);
    
    // 并行加载学生和练习列表
    Promise.all([
        populateStudents(),
        populateProblems()
    ]);
}); 