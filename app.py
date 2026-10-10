
import re
import html
import streamlit as st
import pymupdf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ResumeIQ | AI Resume Analyzer",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #17252b;
        --muted: #718087;
        --teal: #197487;
        --teal-light: #e7f4f3;
        --cream: #fafaf6;
        --line: #e4eae7;
        --green: #26764b;
        --red: #b64b47;
    }

    .stApp {
        background: var(--cream);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.5px;
    }

    .block-container {
        max-width: 1220px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 28px;
    }

    .brand-icon {
        background: #197487;
        color: white;
        width: 43px;
        height: 43px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        font-size: 23px;
        font-weight: 700;
    }

    .brand-name {
        font-family: 'Outfit', sans-serif;
        font-size: 23px;
        font-weight: 800;
        color: #17252b;
        line-height: 1.1;
    }

    .brand-sub {
        color: #718087;
        font-size: 11px;
        margin-top: 4px;
    }

    .eyebrow {
        text-transform: uppercase;
        color: var(--teal);
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero {
        background: linear-gradient(120deg, #ffffff 10%, #edf7f5 100%);
        border: 1px solid #dcebe6;
        border-radius: 25px;
        padding: 32px;
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
    }

    .hero h1 {
        font-size: clamp(30px, 4vw, 44px);
        line-height: 1.12;
        margin: 0 0 13px 0;
        font-weight: 800;
    }

    .hero p {
        color: #66767a;
        max-width: 650px;
        line-height: 1.8;
        margin: 0;
        font-size: 15px;
    }

    .hero-pill {
        display: inline-block;
        background: #dff0eb;
        color: #176653;
        border-radius: 100px;
        padding: 7px 12px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 17px;
    }

    .section-label {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 21px;
        margin: 25px 0 5px;
    }

    .section-description {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 18px;
    }

    .panel {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 19px;
        padding: 21px;
        margin-bottom: 15px;
    }

    .panel-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 17px;
        margin-bottom: 7px;
    }

    .panel-subtitle {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.6;
    }

    .score-card {
        background: linear-gradient(135deg, #173d43, #197487);
        color: white;
        border-radius: 23px;
        padding: 27px;
        min-height: 220px;
        margin-bottom: 15px;
        box-shadow: 0 12px 30px rgba(25, 116, 135, 0.10);
    }

    .score-label {
        font-size: 12px;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: #d4e9e8;
        font-weight: 700;
    }

    .score-number {
        font-family: 'Outfit', sans-serif;
        font-size: 58px;
        font-weight: 800;
        line-height: 1.35;
        letter-spacing: -2px;
    }

    .score-note {
        color: #e0eeee;
        font-size: 12px;
        line-height: 1.7;
    }

    .metric-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 19px;
        min-height: 112px;
        margin-bottom: 12px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 12px;
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.1;
    }

    .skill-chip {
        display: inline-block;
        padding: 8px 11px;
        border-radius: 9px;
        margin: 4px 5px 4px 0;
        font-size: 12px;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    .chip-good {
        color: #236b49;
        background: #e7f4e9;
        border: 1px solid #d3ead8;
    }

    .chip-missing {
        color: #994b43;
        background: #fff0eb;
        border: 1px solid #f2d8d1;
    }

    .chip-neutral {
        color: #4e6067;
        background: #f0f4f3;
        border: 1px solid #e0e8e5;
    }

    .empty-state {
        background: white;
        border: 1px dashed #cbdad6;
        border-radius: 22px;
        text-align: center;
        padding: 38px 22px;
    }

    .empty-icon {
        font-size: 38px;
        margin-bottom: 12px;
    }

    .empty-title {
        font-family: 'Outfit', sans-serif;
        font-size: 23px;
        font-weight: 700;
        margin-bottom: 9px;
    }

    .empty-text {
        color: var(--muted);
        max-width: 450px;
        margin: auto;
        font-size: 13px;
        line-height: 1.8;
    }

    .footer {
        text-align: center;
        color: #82908e;
        font-size: 11px;
        padding: 25px 0 0;
    }

    div.stButton > button[kind="primary"] {
        background: #197487;
        border: 1px solid #197487;
        border-radius: 11px;
        min-height: 47px;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #145d6c;
        border-color: #145d6c;
        transform: translateY(-1px);
    }

    div.stDownloadButton > button {
        background: #ffffff;
        color: #197487;
        border: 1px solid #c9dedb;
        border-radius: 11px;
        font-weight: 700;
        min-height: 43px;
    }

    div.stDownloadButton > button:hover {
        border-color: #197487;
        color: #145d6c;
    }

    div[data-testid="stFileUploader"] {
        background: #f8fbfa;
        border: 1px dashed #c5d9d4;
        border-radius: 13px;
        padding: 10px;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 11px;
        border-color: #dce6e3;
        background: #ffffff;
        font-size: 13px;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #197487;
        box-shadow: 0 0 0 1px #197487;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 16px;
    }

    hr {
        border-color: var(--line);
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }

        .hero {
            padding: 23px 19px;
            border-radius: 19px;
        }

        .score-number {
            font-size: 48px;
        }

        .panel {
            padding: 16px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
            scroll-behavior: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SKILL DATABASE
# --------------------------------------------------

SKILLS = [
    "python", "java", "javascript", "typescript", "c++",
    "sql", "mongodb", "mysql", "postgresql", "html", "css",
    "react", "node.js", "express", "flask", "django", "fastapi",
    "streamlit", "machine learning", "deep learning",
    "natural language processing", "nlp",
    "artificial intelligence", "data analysis", "pandas",
    "numpy", "scikit-learn", "tensorflow", "pytorch",
    "git", "github", "docker", "aws", "excel", "power bi",
    "communication", "problem solving", "rest api",
    "api development", "data visualization", "statistics",
    "leadership", "teamwork", "agile", "linux",
    "cybersecurity", "computer vision", "generative ai",
]


# --------------------------------------------------
# CORE FUNCTIONS
# --------------------------------------------------

def extract_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


def extract_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    ) as document:
        if document.needs_pass:
            raise ValueError(
                "This PDF is password-protected. "
                "Please upload an unlocked PDF."
            )

        pages = [page.get_text() for page in document]

    return "\n".join(pages).strip()


def calculate_match(resume, job_description):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(
        [resume, job_description]
    )

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2],
    )[0][0]

    return round(float(similarity) * 100)


def render_skill_chips(skills, chip_class):
    if not skills:
        st.caption("No skills detected in this category.")
        return

    chips = "".join(
        '<span class="skill-chip '
        + chip_class
        + '">'
        + html.escape(skill)
        + "</span>"
        for skill in skills
    )

    st.markdown(
        chips,
        unsafe_allow_html=True,
    )


def build_report(score, matching, missing, resume_skills):
    return f"""RESUMEIQ — RESUME ANALYSIS REPORT

Text similarity score: {score}%

MATCHING SKILLS
{chr(10).join('- ' + s for s in matching) if matching else '- None detected'}

SKILLS TO CONSIDER LEARNING
{chr(10).join('- ' + s for s in missing) if missing else '- None detected'}

SKILLS DETECTED IN RESUME
{chr(10).join('- ' + s for s in resume_skills) if resume_skills else '- None detected'}

IMPORTANT
The score is a TF-IDF text similarity estimate.
It is not a hiring probability or a measure of candidate quality.
Skill extraction uses a predefined vocabulary.
"""


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">✳</div>
            <div>
                <div class="brand-name">ResumeIQ</div>
                <div class="brand-sub">SMART CAREER INSIGHTS</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Workspace")
    st.caption("Prepare your resume for the role you want.")

    input_method = st.radio(
        "Resume source",
        ["Upload PDF", "Paste text"],
        help="Choose how to provide your resume.",
    )

    resume_text = ""

    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader(
            "Your resume",
            type=["pdf"],
            help="Upload a text-based PDF, up to 10 MB.",
        )

        if uploaded_file is not None:
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("The PDF must be smaller than 10 MB.")
            else:
                try:
                    resume_text = extract_pdf_text(uploaded_file)

                    if resume_text:
                        st.success("Resume text extracted")

                        with st.expander("Preview resume"):
                            st.text_area(
                                "Extracted content",
                                value=resume_text,
                                height=180,
                                disabled=True,
                                label_visibility="collapsed",
                            )
                    else:
                        st.warning(
                            "No selectable text found. "
                            "This PDF may require OCR."
                        )

                except ValueError as error:
                    st.error(str(error))
                except Exception:
                    st.error(
                        "Unable to read this PDF. "
                        "Try another valid PDF."
                    )

    else:
        resume_text = st.text_area(
            "Your resume",
            height=220,
            placeholder=(
                "Paste your resume, education, "
                "projects, skills and experience..."
            ),
        )

    st.markdown("### Target role")

    job_description = st.text_area(
        "Job description",
        height=210,
        placeholder=(
            "Paste the job description for "
            "the role you want..."
        ),
    )

    st.caption(
        "Tip: A detailed job description usually "
        "provides more useful comparisons."
    )

    analyze_clicked = st.button(
        "✳  Analyze my resume",
        type="primary",
        use_container_width=True,
    )

    if st.button(
        "Clear analysis",
        use_container_width=True,
    ):
        st.session_state.analysis_result = None
        st.session_state.resume_text = ""
        st.rerun()

    st.markdown("---")
    st.caption("ResumeIQ · Built with Python and NLP")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-pill">✳ &nbsp; YOUR CAREER, IN FOCUS</div>
        <h1>Make your next application<br>your strongest one.</h1>
        <p>
            Understand how your resume aligns with a job.
            Discover matching skills, identify potential gaps,
            and make more informed improvements before applying.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-label">Your analysis workspace</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Provide your resume and a target job description using '
    'the controls in the sidebar, then run your analysis.'
    '</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# RUN ANALYSIS
# --------------------------------------------------

if analyze_clicked:
    if not resume_text.strip():
        st.session_state.analysis_result = None
        st.warning(
            "Please upload a readable PDF or paste your resume "
            "in the sidebar."
        )

    elif not job_description.strip():
        st.session_state.analysis_result = None
        st.warning(
            "Please add the target job description in the sidebar."
        )

    else:
        try:
            with st.spinner("Analyzing your resume..."):
                resume_skills = extract_skills(resume_text)
                job_skills = extract_skills(job_description)

                matching = sorted(
                    set(resume_skills) & set(job_skills)
                )

                missing = sorted(
                    set(job_skills) - set(resume_skills)
                )

                score = calculate_match(
                    resume_text,
                    job_description,
                )

                st.session_state.resume_text = resume_text

                st.session_state.analysis_result = {
                    "score": score,
                    "matching": matching,
                    "missing": missing,
                    "resume_skills": resume_skills,
                    "job_skills": job_skills,
                }

        except ValueError:
            st.session_state.analysis_result = None
            st.error(
                "There is not enough meaningful text to compare. "
                "Please provide a more detailed resume and "
                "job description."
            )

        except Exception:
            st.session_state.analysis_result = None
            st.error(
                "Analysis failed. Check your inputs and try again."
            )


# --------------------------------------------------
# RESULTS DASHBOARD
# --------------------------------------------------

result = st.session_state.analysis_result

if result:
    score = result["score"]
    matching = result["matching"]
    missing = result["missing"]
    resume_skills = result["resume_skills"]
    job_skills = result["job_skills"]

    st.markdown(
        '<div class="section-label">Your results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Here is the text similarity estimate and the skill '
        'overlap detected in your resume and target role.'
        '</div>',
        unsafe_allow_html=True,
    )

    score_col, metrics_col = st.columns([1.05, 1.7])

    with score_col:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">Resume text match</div>
                <div class="score-number">{score}%</div>
                <div class="score-note">
                    Estimated textual similarity between your
                    resume and the job description.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metrics_col:
        metric_a, metric_b = st.columns(2)

        with metric_a:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Matching skills</div>
                    <div class="metric-value">{len(matching)}</div>
                    <div class="panel-subtitle">
                        Skills found in both texts
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric_b:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Potential skill gaps</div>
                    <div class="metric-value">{len(missing)}</div>
                    <div class="panel-subtitle">
                        Job skills not detected in your resume
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Skills detected in your resume</div>
                <div class="metric-value">{len(resume_skills)}</div>
                <div class="panel-subtitle">
                    Based on the application's predefined skill list
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "A high text similarity score does not guarantee a job "
        "interview. A low score does not mean you are unqualified."
    )

    st.markdown("---")

    tab_skills, tab_advice, tab_resume = st.tabs(
        ["Skill breakdown", "Recommendations", "Resume preview"]
    )

    with tab_skills:
        skill_col1, skill_col2 = st.columns(2)

        with skill_col1:
            st.markdown(
                """
                <div class="panel">
                    <div class="panel-title">✓ Matching skills</div>
                    <div class="panel-subtitle">
                        Detected in both the resume and job description.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_skill_chips(matching, "chip-good")

        with skill_col2:
            st.markdown(
                """
                <div class="panel">
                    <div class="panel-title">↗ Potential skill gaps</div>
                    <div class="panel-subtitle">
                        Mentioned in the job description but not
                        detected in the resume.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_skill_chips(missing, "chip-missing")

        st.markdown("#### Skills found in your resume")
        render_skill_chips(resume_skills, "chip-neutral")

        st.markdown("#### Skills identified in the job description")
        render_skill_chips(job_skills, "chip-neutral")

    with tab_advice:
        st.markdown("### Your improvement checklist")

        if missing:
            st.markdown(
                "**1. Review the potential gaps**"
            )
            st.write(
                "Choose the missing skills that are relevant "
                "to your target role. Learn them through practice "
                "and projects where appropriate."
            )

            st.markdown(
                "**2. Show evidence of relevant skills**"
            )
            st.write(
                "If you already possess a skill, make your "
                "projects, coursework, or experience demonstrate it."
            )
        else:
            st.markdown(
                "**1. Strengthen your evidence**"
            )
            st.write(
                "Describe how you applied your detected skills "
                "in projects, internships, or coursework."
            )

        st.markdown("**3. Improve your project descriptions**")
        st.write(
            "Explain the problem, technologies used, your "
            "contribution, and measurable results where available."
        )

        st.markdown("**4. Tailor your resume honestly**")
        st.write(
            "Prioritize relevant experience and skills, but "
            "never claim qualifications you do not have."
        )

    with tab_resume:
        st.markdown("### Extracted resume text")
        st.caption(
            "Check the extracted content for missing characters "
            "or formatting issues before relying on the analysis."
        )

        st.text_area(
            "Resume contents",
            value=st.session_state.resume_text,
            height=350,
            disabled=True,
            label_visibility="collapsed",
        )

    st.markdown("---")
    st.markdown("### Take your results with you")

    report = build_report(
        score,
        matching,
        missing,
        resume_skills,
    )

    st.download_button(
        "↓  Download analysis report",
        data=report,
        file_name="resumeiq_analysis.txt",
        mime="text/plain",
        use_container_width=True,
    )

else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">✳️</div>
            <div class="empty-title">
                Your next opportunity starts here.
            </div>
            <div class="empty-text">
                Add your resume and the job description in the sidebar.
                Your analysis will appear here, including a text
                similarity estimate, skill overlap, and recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">01 · Add resume</div>
                <div class="panel-subtitle">
                    Upload a PDF or paste your resume text.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step2:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">02 · Compare</div>
                <div class="panel-subtitle">
                    Provide a job description for comparison.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step3:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">03 · Improve</div>
                <div class="panel-subtitle">
                    Review matching skills and possible gaps.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        RESUMEIQ &nbsp;·&nbsp; BUILT WITH PYTHON, STREAMLIT
        AND SCIKIT-LEARN<br><br>
        Your results are estimates to support your own review,
        not guarantees of hiring outcomes.
    </div>
    """,
    unsafe_allow_html=True,
)
