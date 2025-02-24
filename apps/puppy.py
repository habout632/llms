import streamlit as st
import PyPDF2
import io
from openai import OpenAI
import time

# 设置页面配置
st.set_page_config(page_title="AI模拟面试", layout="wide")

# API配置
llm_model_claude_3_5_haiku = 'claude-3-5-haiku-20241022'
api_key = "sk-trbFwocx3hwvWWrS7eA4770d89564207Be2046F4956eA896"
base_url = "https://aihubmix.com/v1"


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def generate_questions(resume_text, jd_text=None):
    try:
        if jd_text:
            prompt = f"""
            请根据以下简历内容和招聘JD，生成5个针对性的技术面试问题。

            简历内容: {resume_text}

            招聘JD: {jd_text}

            要求：
            1. 每个问题都要编号（1-5）
            2. 问题要紧密结合JD中的要求和简历中的经验
            3. 测试候选人是否符合岗位的具体要求
            4. 关注技术细节和项目实现
            5. 使用专业但清晰的语言
            """
        else:
            prompt = f"""
            请根据以下简历内容，生成5个技术面试问题，重点关注AI项目和经验。

            简历内容: {resume_text}

            要求：
            1. 每个问题都要编号（1-5）
            2. 问题要测试候选人的理论理解和实践经验
            3. 关注技术细节和项目实现
            4. 使用专业但清晰的语言
            """

        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位专业的AI和机器学习技术面试官，请用中文提供专业的技术面试问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        questions = response.choices[0].message.content.strip().split('\n')
        questions = [q.strip() for q in questions if
                     q.strip() and any(q.strip().startswith(str(i)) for i in range(1, 6))]
        while len(questions) < 5:
            questions.append(f"{len(questions) + 1}. 请详细描述您的一个相关项目经验？")
        return questions[:5]

    except Exception as e:
        st.error(f"生成问题时出错: {str(e)}")
        return [f"{i}. 请描述您的相关项目经验？" for i in range(1, 6)]


def evaluate_answer(question, answer, jd_text=None):
    try:
        if jd_text:
            prompt = f"""
            请根据招聘JD的要求，评估以下技术面试问题的回答。

            招聘JD: {jd_text}

            问题: {question}
            回答: {answer}

            请按以下格式回复：
            分数：[X/10分]
            评价：[结合JD要求的详细反馈意见]
            建议：[针对岗位要求的改进建议]
            匹配度：[回答与JD要求的匹配程度分析]
            """
        else:
            prompt = f"""
            请评估以下技术面试问题的回答。
            请用中文提供详细的反馈和评分。

            问题: {question}
            回答: {answer}

            请按以下格式回复：
            分数：[X/10分]
            评价：[详细的反馈意见]
            建议：[改进建议]
            """

        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
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
    st.session_state.time_limit = 30  # 默认30分钟

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
    .timer-container {
        position: fixed;
        top: 60px;
        right: 30px;
        background-color: rgba(240, 242, 246, 0.9);
        padding: 10px 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        z-index: 1000;
    }
    .timer-text {
        font-size: 1.1em;
        font-weight: bold;
        color: #333;
    }
    .timer-warning {
        color: #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit界面
st.title("AI模拟面试")

# 在标题下方添加时间选择
if not st.session_state.interview_started:
    time_options = {
        "15分钟": 15,
        "30分钟": 30,
        "45分钟": 45,
        "60分钟": 60
    }
    selected_time = st.selectbox(
        "选择面试时长",
        options=list(time_options.keys()),
        index=1  # 默认选择30分钟
    )
    st.session_state.time_limit = time_options[selected_time]

# 显示计时器
if st.session_state.interview_started:
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    elapsed_time = int(time.time() - st.session_state.start_time)
    remaining_time = max(0, st.session_state.time_limit * 60 - elapsed_time)

    minutes = remaining_time // 60
    seconds = remaining_time % 60

    # 根据剩余时间设置样式
    time_style = "timer-text"
    if remaining_time < 300:  # 小于5分钟显示红色
        time_style += " timer-warning"

    # 显示计时器
    st.markdown(f"""
        <div class="timer-container">
            <div class="{time_style}">
                剩余时间: {minutes:02d}:{seconds:02d}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 检查是否超时
    if remaining_time <= 0 and not st.session_state.show_feedback:
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

st.write("上传您的简历开始模拟面试！您也可以粘贴招聘JD来获得更有针对性的面试体验。")

# 创建两列布局用于文件上传
col1, col2 = st.columns(2)

with col1:
    # 简历上传
    uploaded_resume = st.file_uploader("上传您的简历（PDF格式）", type="pdf")

with col2:
    # JD文本输入（可选）
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
            st.session_state.start_time = time.time()  # 开始计时

if st.session_state.interview_started:
    # 显示进度
    st.progress((st.session_state.current_question) / 5)

    # 显示当前问题
    st.subheader(f"问题 {st.session_state.current_question + 1}/5")
    st.write(st.session_state.questions[st.session_state.current_question])

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
        st.write("评估反馈:")
        st.write(st.session_state.scores[-1])

        if st.session_state.current_question < 4:
            if st.button("继续下一题"):
                st.session_state.current_question += 1
                st.session_state.show_feedback = False
                st.session_state.answer_submitted = False
                st.rerun()
        else:
            if st.button("查看面试总结"):
                st.success("面试完成！")

                # 显示总结
                st.subheader("面试总结")
                total_score = 0
                for i, (q, s) in enumerate(zip(st.session_state.questions, st.session_state.scores)):
                    st.write(f"\n问题 {i + 1}:")
                    st.write(q)
                    st.write("反馈:")
                    st.write(s)

                    # 提取分数并计算总分
                    try:
                        score_line = [line for line in s.split('\n') if '分数：' in line][0]
                        score = float(score_line.split('分数：')[1].split('/')[0])
                        total_score += score
                    except:
                        pass

                # 显示总分和平均分
                st.subheader("总体评分")
                st.write(f"总分：{total_score}/50")
                st.write(f"平均分：{total_score / 5:.1f}/10")

                # 重置按钮
                if st.button("开始新的面试"):
                    st.session_state.interview_started = False
                    st.session_state.current_question = 0
                    st.session_state.questions = []
                    st.session_state.scores = []
                    st.session_state.show_feedback = False
                    st.session_state.answer_submitted = False
                    st.session_state.jd_text = None
                    st.session_state.start_time = None  # 重置计时器
                    st.rerun()