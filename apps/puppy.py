import streamlit as st
import PyPDF2
import io
from openai import OpenAI
import time
import json
from datetime import datetime

# 设置页面配置
st.set_page_config(
    page_title="AI模拟面试",
    layout="wide",
    initial_sidebar_state="expanded"  # 默认展开侧边栏
)


def get_model_config(model_name):
    """返回模型的配置信息"""
    configs = {
        "claude-3-5-sonnet": {
            "api_key": "sk-trbFwocx3hwvWWrS7eA4770d89564207Be2046F4956eA896",
            "base_url": "https://aihubmix.com/v1",
            "model": "claude-3-5-haiku-20241022"
        },
        "deepseek-chat": {
            "api_key": "sk-trbFwocx3hwvWWrS7eA4770d89564207Be2046F4956eA896",
            "base_url": "https://aihubmix.com/v1",
            "model": "deepseek-chat"
        },
        "qwen-max": {
            "api_key": "sk-trbFwocx3hwvWWrS7eA4770d89564207Be2046F4956eA896",
            "base_url": "https://aihubmix.com/v1",
            "model": "qwen-max"
        }
    }
    return configs.get(model_name)


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def generate_questions_by_type(resume_text, question_type, count, jd_text=None):
    """根据不同类型生成面试问题"""
    model_config = get_model_config(st.session_state.selected_model)
    client = OpenAI(
        api_key=model_config['api_key'],
        base_url=model_config['base_url']
    )

    type_prompts = {
        'knowledge': f"""
        请生成{count}个针对性的技术知识题库面试问题。问题应该覆盖基础概念、原理理解等。

        简历内容: {resume_text}
        {'招聘JD: ' + jd_text if jd_text else ''}

        要求：
        1. 每个问题都要编号
        2. 问题应该考察核心技术概念和原理
        3. 难度要适中，有区分度
        4. 使用专业但清晰的语言
        """,
        'leetcode': f"""
        请生成{count}个算法编程面试问题，类似LeetCode风格。

        简历内容: {resume_text}
        {'招聘JD: ' + jd_text if jd_text else ''}

        要求：
        1. 每个问题都要编号
        2. 包含具体的问题描述和示例
        3. 难度要标注（简单/中等/困难）
        4. 问题类型多样（数组、字符串、动态规划等）
        """,
        'project': f"""
        请生成{count}个项目经验相关的面试问题。

        简历内容: {resume_text}
        {'招聘JD: ' + jd_text if jd_text else ''}

        要求：
        1. 每个问题都要编号
        2. 深入询问简历中提到的具体项目
        3. 关注技术选型、架构设计、问题解决等
        4. 考察项目经验的深度和广度
        """
    }

    try:
        response = client.chat.completions.create(
            model=model_config['model'],
            messages=[
                {"role": "system", "content": "你是一位专业的技术面试官，请根据要求生成合适的面试问题。"},
                {"role": "user", "content": type_prompts[question_type]}
            ],
            temperature=0.7
        )

        questions = response.choices[0].message.content.strip().split('\n')
        questions = [q.strip() for q in questions if q.strip() and any(str(i) in q[:4] for i in range(1, count + 1))]
        return questions[:count]
    except Exception as e:
        st.error(f"生成{question_type}类型问题时出错: {str(e)}")
        return [f"{i}. 示例{question_type}问题 {i}" for i in range(1, count + 1)]


def calculate_type_questions(total_questions, type_percentages):
    """计算每种类型的具体题目数量"""
    type_counts = {}
    total_percentage = sum(type_percentages.values())

    if total_percentage == 0:
        return {'knowledge': total_questions}

    # 首先按比例计算
    remaining = total_questions
    for type_key, percentage in type_percentages.items():
        count = int(round(total_questions * percentage / total_percentage))
        type_counts[type_key] = min(count, remaining)
        remaining -= type_counts[type_key]

    # 如果还有剩余的题目，分配给百分比最高的类型
    if remaining > 0:
        max_type = max(type_percentages.items(), key=lambda x: x[1])[0]
        type_counts[max_type] += remaining

    return type_counts


def generate_questions(resume_text, jd_text=None):
    """整合不同类型的问题"""
    try:
        total_questions = st.session_state.question_count
        questions = []

        # 获取每种类型的百分比
        type_percentages = {
            type_key: type_info['percentage']
            for type_key, type_info in st.session_state.interview_types.items()
        }

        # 计算每种类型的具体题目数量
        type_counts = calculate_type_questions(total_questions, type_percentages)

        # 生成每种类型的问题
        for type_key, count in type_counts.items():
            if count > 0:
                type_questions = generate_questions_by_type(resume_text, type_key, count, jd_text)
                type_name = st.session_state.interview_types[type_key]['name']
                questions.extend([f"[{type_name}] {q}" for q in type_questions])

        # 如果生成的问题数量不够，补充题库类型的问题
        while len(questions) < total_questions:
            questions.append(f"[题库] {len(questions) + 1}. 请描述一个相关的技术概念？")

        # 如果问题数量超出，截取需要的数量
        questions = questions[:total_questions]

        # 如果包含自我介绍，将其添加到开头
        if st.session_state.include_self_intro:
            intro_prompt = """
                请用3-5分钟的时间进行一个简洁的自我介绍，包括：
                1. 您的教育背景和专业领域
                2. 相关的工作经验和项目经历
                3. 您的技术特长和专业优势
                4. 为什么您认为自己适合这个职位
                请注意把控时间，突出重点，展现您的专业能力。
            """
            questions.insert(0, intro_prompt)

        return questions
    except Exception as e:
        st.error(f"生成问题时出错: {str(e)}")
        base_questions = [f"{i}. 请描述您的相关经验？" for i in range(1, total_questions + 1)]
        if st.session_state.include_self_intro:
            base_questions.insert(0, intro_prompt)
        return base_questions


def evaluate_answer(question, answer, jd_text=None):
    try:
        model_config = get_model_config(st.session_state.selected_model)

        # 判断问题类型
        is_self_intro = "自我介绍" in question and "教育背景" in question
        is_leetcode = "[LeetCode]" in question
        is_project = "[项目经验]" in question
        is_knowledge = "[题库]" in question

        if is_self_intro:
            prompt = f"""
            请评估以下自我介绍的回答。

            自我介绍要求：{question}
            回答: {answer}

            请按以下格式评估：
            分数：[X/10分]
            评价：[对表达清晰度、内容完整性、时间控制的评价]
            优点：[自我介绍中的亮点]
            建议：[改进建议]
            """
        elif is_leetcode:
            prompt = f"""
            请评估以下算法编程问题的回答。

            问题: {question}
            回答: {answer}

            请按以下格式评估：
            分数：[X/10分]
            评价：[对解题思路、时间复杂度、代码质量的评价]
            优化建议：[可能的优化方向]
            知识点：[涉及的算法知识点]
            """
        elif is_project:
            prompt = f"""
            请评估以下项目经验相关问题的回答。

            问题: {question}
            回答: {answer}
            {'招聘JD: ' + jd_text if jd_text else ''}

            请按以下格式评估：
            分数：[X/10分]
            评价：[对项目理解深度、技术选型、解决方案的评价]
            亮点：[回答中的亮点]
            建议：[改进建议]
            """
        else:  # 知识题库问题
            prompt = f"""
            请评估以下技术知识问题的回答。

            问题: {question}
            回答: {answer}

            请按以下格式评估：
            分数：[X/10分]
            评价：[对概念理解、专业深度的评价]
            正确点：[回答中的正确观点]
            补充：[需要补充的知识点]
            """

        client = OpenAI(
            api_key=model_config['api_key'],
            base_url=model_config['base_url']
        )

        response = client.chat.completions.create(
            model=model_config['model'],
            messages=[
                {"role": "system",
                 "content": "你是一位专业的技术面试官，正在评估候选人的回答。请用中文提供专业、建设性的反馈。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content
    except Exception as e:
        st.error(f"评估回答时出错: {str(e)}")
        return "分数：N/A\n评价：由于技术原因无法完成评估\n建议：请稍后重试"


# 初始化session state
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = False
if 'answer_submitted' not in st.session_state:
    st.session_state.answer_submitted = False
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'time_limit' not in st.session_state:
    st.session_state.time_limit = 30
if 'interview_history' not in st.session_state:
    st.session_state.interview_history = []
if 'interview_count' not in st.session_state:
    st.session_state.interview_count = 0
if 'question_count' not in st.session_state:
    st.session_state.question_count = 5
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'claude-3-5-sonnet'
if 'include_self_intro' not in st.session_state:
    st.session_state.include_self_intro = False
if 'interview_types' not in st.session_state:
    st.session_state.interview_types = {
        'knowledge': {'name': '题库', 'percentage': 100},
        'leetcode': {'name': 'LeetCode', 'percentage': 0},
        'project': {'name': '项目经验', 'percentage': 20}
    }

# 添加CSS样式
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 4px;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stTextArea textarea {
        border-radius: 4px;
        border: 1px solid #ccc;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    #timer {
        position: fixed;
        top: 60px;
        right: 30px;
        background-color: rgba(240, 242, 246, 0.9);
        padding: 10px 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        font-size: 1.1em;
        font-weight: bold;
        color: #333;
    }
    .warning {
        color: #ff4b4b !important;
    }
    .question-area {
        margin-top: 20px;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    .feedback-area {
        margin-top: 20px;
        padding: 20px;
        background-color: #e8f5e9;
        border-radius: 8px;
    }
    .summary-area {
        margin-top: 30px;
        padding: 25px;
        background-color: #f3e5f5;
        border-radius: 8px;
    }
    .sidebar-content {
        padding: 20px;
    }
    .history-item {
        margin: 10px 0;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
    }
    .type-percentage {
        padding: 10px;
        background-color: #f1f3f4;
        border-radius: 4px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.title("面试系统设置")

    # 配置部分
    st.header("配置选项")
    with st.expander("面试配置", expanded=True):
        # 自我介绍选项
        st.session_state.include_self_intro = st.checkbox(
            "包含自我介绍",
            value=False,
            help="选择是否在面试开始时添加自我介绍环节"
        )

        # 面试类型占比配置
        st.markdown("### 面试问题类型配置")
        st.write("设置各类型问题的占比（0-100%）")

        # 为每种面试类型创建滑动条
        col1, col2 = st.columns(2)
        with col1:
            for type_key in ['knowledge', 'leetcode']:
                type_info = st.session_state.interview_types[type_key]
                type_info['percentage'] = st.slider(
                    f"{type_info['name']}占比",
                    min_value=0,
                    max_value=100,
                    value=type_info['percentage'],
                    help=f"设置{type_info['name']}类型问题的百分比"
                )

        with col2:
            type_info = st.session_state.interview_types['project']
            type_info['percentage'] = st.slider(
                f"{type_info['name']}占比",
                min_value=0,
                max_value=100,
                value=type_info['percentage'],
                help=f"设置{type_info['name']}类型问题的百分比"
            )

        # 显示当前配置的预览
        st.markdown("### 当前配置预览")
        total_percentage = sum(t['percentage'] for t in st.session_state.interview_types.values())
        if total_percentage > 0:
            for type_key, type_info in st.session_state.interview_types.items():
                actual_questions = int(
                    round(st.session_state.question_count * type_info['percentage'] / total_percentage))
                st.markdown(f"""
                    <div class="type-percentage">
                        {type_info['name']}: {actual_questions}题 ({type_info['percentage']}%)
                    </div>
                """, unsafe_allow_html=True)

            if total_percentage > 100:
                st.warning("注意：总占比超过100%，系统会按比例自动调整")

        # 问题数量选择
        st.markdown("### 其他配置")
        st.session_state.question_count = st.slider(
            "技术问题数量",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="选择本次模拟面试的技术问题数量（不包含自我介绍）"
        )

        # 面试时长选择
        time_options = {
            "15分钟": 15,
            "30分钟": 30,
            "45分钟": 45,
            "60分钟": 60,
            "90分钟": 90,
            "120分钟": 120
        }
        selected_time = st.selectbox(
            "面试时长",
            options=list(time_options.keys()),
            index=1,
            help="选择本次模拟面试的总时长"
        )
        st.session_state.time_limit = time_options[selected_time]

        # 模型选择
        model_options = {
            "Claude 3.5 Sonnet": "claude-3-5-sonnet",
            "Deepseek": "deepseek-chat",
            "Qwen": "qwen-max"
        }
        selected_model = st.selectbox(
            "选择面试评估模型",
            options=list(model_options.keys()),
            help="选择用于生成问题和评估答案的AI模型"
        )
        st.session_state.selected_model = model_options[selected_model]

    st.markdown("---")

    # 历史记录部分
    st.header("面试历史记录")
    st.write(f"已完成模拟面试总数: {st.session_state.interview_count}")

    if len(st.session_state.interview_history) > 0:
        for idx, interview in enumerate(st.session_state.interview_history):
            with st.expander(f"面试 #{idx + 1} - {interview['date']}"):
                st.write(f"总分: {interview['total_score']}/{interview['question_count'] * 10}")
                st.write(f"平均分: {interview['average_score']:.1f}/10")
                st.write(f"时长: {interview['duration']}分钟")
                st.write(f"问题数量: {interview['question_count']}题")
                if interview.get('include_self_intro', False):
                    st.write("✓ 包含自我介绍")
                st.write(f"使用模型: {interview['model']}")

                # 显示问题类型分布
                st.write("问题类型分布:")
                if 'interview_types' in interview:
                    for type_key, type_info in interview['interview_types'].items():
                        if type_info['percentage'] > 0:
                            st.write(f"- {type_info['name']}: {type_info['percentage']}%")

                st.markdown("---")
                st.write("问题回顾:")
                for q_idx, (question, score) in enumerate(zip(interview['questions'], interview['scores'])):
                    st.write(f"问题 {q_idx + 1}:")
                    st.write(question)
                    st.write(f"得分: {score}/10")
                    st.markdown("---")

# 主界面
st.title("AI模拟面试")

st.write("上传您的简历开始模拟面试！您也可以粘贴招聘JD来获得更有针对性的面试体验。")

# 创建两列布局用于文件上传
col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader("上传您的简历（PDF格式）", type="pdf")

with col2:
    jd_text_input = st.text_area("粘贴招聘JD（可选）", height=150,
                                 placeholder="请将招聘职位要求粘贴到这里...")

# 开始面试按钮
if uploaded_resume and not st.session_state.interview_started:
    if st.button("开始模拟面试"):
        with st.spinner("正在分析简历并生成面试问题..."):
            resume_text = extract_text_from_pdf(uploaded_resume)

            if jd_text_input.strip():
                st.session_state.jd_text = jd_text_input.strip()

            st.session_state.questions = generate_questions(resume_text, st.session_state.jd_text)
            st.session_state.interview_started = True
            st.session_state.current_question = 0
            st.session_state.scores = []
            st.session_state.show_feedback = False
            st.session_state.answer_submitted = False
            st.session_state.start_time = time.time()
            st.rerun()

# 显示计时器和面试内容
if st.session_state.interview_started:
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    elapsed_time = int(time.time() - st.session_state.start_time)
    initial_remaining = max(0, st.session_state.time_limit * 60 - elapsed_time)

    st.markdown(f"""
        <div id="timer">剩余时间: {initial_remaining // 60:02d}:{initial_remaining % 60:02d}</div>

        <script>
            var remainingTime = {initial_remaining};
            var timerElement = document.getElementById('timer');

            function updateTimer() {{
                if (remainingTime > 0) {{
                    remainingTime -= 1;
                    var minutes = Math.floor(remainingTime / 60);
                    var seconds = remainingTime % 60;
                    timerElement.innerHTML = '剩余时间: ' + 
                        String(minutes).padStart(2, '0') + ':' + 
                        String(seconds).padStart(2, '0');

                    if (remainingTime < 300) {{
                        timerElement.classList.add('warning');
                    }}

                    if (remainingTime <= 0) {{
                        location.reload();
                    }}
                }}
            }}

            setInterval(updateTimer, 1000);
        </script>
    """, unsafe_allow_html=True)

    # 显示进度
    total_questions = st.session_state.question_count + (1 if st.session_state.include_self_intro else 0)
    st.progress((st.session_state.current_question) / total_questions)

    # 显示当前问题
    with st.container():
        st.markdown('<div class="question-area">', unsafe_allow_html=True)
        if st.session_state.include_self_intro and st.session_state.current_question == 0:
            st.subheader("自我介绍")
        else:
            actual_question_num = st.session_state.current_question - (1 if st.session_state.include_self_intro else 0)
            st.subheader(f"问题 {actual_question_num + 1}/{st.session_state.question_count}")
        st.write(st.session_state.questions[st.session_state.current_question])
        st.markdown('</div>', unsafe_allow_html=True)

    # 获取答案
    answer = st.text_area("您的回答:", key=f"answer_{st.session_state.current_question}",
                          height=150)

    # 提交答案按钮
    if not st.session_state.answer_submitted and st.button("提交答案"):
        if answer:
            with st.spinner("正在评估您的回答..."):
                feedback = evaluate_answer(
                    st.session_state.questions[st.session_state.current_question],
                    answer,
                    st.session_state.jd_text
                )
                st.session_state.scores.append(feedback)
                st.session_state.show_feedback = True
                st.session_state.answer_submitted = True
                st.rerun()
        else:
            st.warning("请先输入您的回答再提交")

    # 显示反馈和下一题按钮
    if st.session_state.show_feedback:
        with st.container():
            st.markdown('<div class="feedback-area">', unsafe_allow_html=True)
            st.subheader("评估反馈:")
            st.write(st.session_state.scores[-1])
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.current_question < total_questions - 1:
            if st.button("继续下一题"):
                st.session_state.current_question += 1
                st.session_state.show_feedback = False
                st.session_state.answer_submitted = False
                st.rerun()
        else:
            if st.button("查看面试总结"):
                st.success("面试完成！")

                # 计算总分和记录历史
                total_score = 0
                question_scores = []
                for i, (q, s) in enumerate(zip(st.session_state.questions, st.session_state.scores)):
                    try:
                        score_line = [line for line in s.split('\n') if '分数：' in line][0]
                        score = float(score_line.split('分数：')[1].split('/')[0])
                        total_score += score
                        question_scores.append(score)
                    except:
                        question_scores.append(0)

                # 保存本次面试记录
                interview_record = {
                    'date': time.strftime("%Y-%m-%d %H:%M"),
                    'total_score': total_score,
                    'average_score': total_score / total_questions,
                    'duration': st.session_state.time_limit,
                    'questions': st.session_state.questions,
                    'scores': question_scores,
                    'jd_provided': st.session_state.jd_text is not None,
                    'question_count': st.session_state.question_count,
                    'model': st.session_state.selected_model,
                    'include_self_intro': st.session_state.include_self_intro,
                    'interview_types': st.session_state.interview_types.copy()
                }

                st.session_state.interview_history.insert(0, interview_record)
                st.session_state.interview_count += 1

                # 显示总结
                with st.container():
                    st.markdown('<div class="summary-area">', unsafe_allow_html=True)
                    st.subheader("面试总结")
                    st.write(f"使用模型: {st.session_state.selected_model}")
                    st.write(f"问题数量: {total_questions}题")

                    # 显示问题类型分布
                    st.subheader("问题类型分布")
                    for type_key, type_info in st.session_state.interview_types.items():
                        if type_info['percentage'] > 0:
                            st.write(f"{type_info['name']}: {type_info['percentage']}%")

                    # 按类型统计得分
                    type_scores = {}
                    for i, (q, s) in enumerate(zip(st.session_state.questions, st.session_state.scores)):
                        for type_key, type_info in st.session_state.interview_types.items():
                            if f"[{type_info['name']}]" in q:
                                if type_key not in type_scores:
                                    type_scores[type_key] = {'total': 0, 'count': 0}
                                try:
                                    score_line = [line for line in s.split('\n') if '分数：' in line][0]
                                    score = float(score_line.split('分数：')[1].split('/')[0])
                                    type_scores[type_key]['total'] += score
                                    type_scores[type_key]['count'] += 1
                                except:
                                    pass

                    st.subheader("各类型得分分析")
                    for type_key, scores in type_scores.items():
                        if scores['count'] > 0:
                            avg_score = scores['total'] / scores['count']
                            st.write(f"{st.session_state.interview_types[type_key]['name']}: {avg_score:.1f}/10")

                    st.subheader("详细问答回顾")
                    for i, (q, s) in enumerate(zip(st.session_state.questions, st.session_state.scores)):
                        if st.session_state.include_self_intro and i == 0:
                            st.write("\n自我介绍:")
                        else:
                            st.write(f"\n问题 {i if not st.session_state.include_self_intro else i}:")
                        st.write(q)
                        st.write("反馈:")
                        st.write(s)

                    st.subheader("总体评分")
                    st.write(f"总分：{total_score}/{total_questions * 10}")
                    st.write(f"平均分：{total_score / total_questions:.1f}/10")
                    st.markdown('</div>', unsafe_allow_html=True)

                if st.button("开始新的面试"):
                    # 保留历史记录相关的状态
                    history = st.session_state.interview_history
                    count = st.session_state.interview_count
                    question_count = st.session_state.question_count
                    selected_model = st.session_state.selected_model
                    time_limit = st.session_state.time_limit
                    include_self_intro = st.session_state.include_self_intro
                    interview_types = st.session_state.interview_types.copy()

                    # 重置其他状态
                    st.session_state.clear()

                    # 恢复保存的状态
                    st.session_state.interview_history = history
                    st.session_state.interview_count = count
                    st.session_state.question_count = question_count
                    st.session_state.selected_model = selected_model
                    st.session_state.time_limit = time_limit
                    st.session_state.include_self_intro = include_self_intro
                    st.session_state.interview_types = interview_types

                    # 初始化新面试的状态
                    st.session_state.current_question = 0
                    st.session_state.questions = []
                    st.session_state.scores = []
                    st.session_state.interview_started = False
                    st.session_state.show_feedback = False
                    st.session_state.answer_submitted = False
                    st.session_state.jd_text = None
                    st.session_state.start_time = None

                    st.rerun()

    # 检查是否超时
    if initial_remaining <= 0 and not st.session_state.show_feedback:
        st.error("面试时间已到！系统将自动提交当前答案。")
        if not st.session_state.answer_submitted:
            current_answer = st.session_state.get(f"answer_{st.session_state.current_question}", "")
            if current_answer:
                feedback = evaluate_answer(
                    st.session_state.questions[st.session_state.current_question],
                    current_answer,
                    st.session_state.jd_text
                )
                st.session_state.scores.append(feedback)
            else:
                st.session_state.scores.append(
                    "未作答\n分数：0/10\n评价：未在规定时间内完成答题\n建议：请注意把控答题时间")

            st.session_state.show_feedback = True
            st.session_state.answer_submitted = True
            st.rerun()