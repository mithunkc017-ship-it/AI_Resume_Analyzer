
import re

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

SKILLS = [
    "python", "java", "javascript", "typescript",
    "c++", "sql", "mongodb", "mysql", "postgresql",
    "html", "css", "react", "node.js", "express",
    "flask", "django", "fastapi", "streamlit",
    "machine learning", "deep learning",
    "natural language processing", "nlp",
    "artificial intelligence", "data analysis",
    "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "git", "github",
    "docker", "aws", "excel", "power bi",
    "communication", "problem solving",
]


def extract_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


def calculate_match(resume, job_description):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(
        [resume, job_description]
    )

    similarity = cosine_similarity(
        matrix[0:1], matrix[1:2]
    )[0][0]

    return round(similarity * 100)


st.title("📄 AI Resume Analyzer")
st.write(
    "Compare your resume with a job description, "
    "discover relevant skills, and identify skill gaps."
)

st.info(
    "Your text is analyzed in this application. "
    "The match score is a text-similarity estimate, "
    "not a prediction of hiring success."
)

left, right = st.columns(2)

with left:
    st.subheader("Your Resume")
    resume = st.text_area(
        "Paste your resume text",
        height=280,
        placeholder=(
            "Example: Python developer with experience "
            "in machine learning, SQL and data analysis..."
        ),
    )

with right:
    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the target job description",
        height=280,
        placeholder=(
            "Example: Looking for a Python developer "
            "with SQL, machine learning and Git skills..."
        ),
    )

if st.button("Analyze Resume", type="primary"):
    if not resume.strip() or not job_description.strip():
        st.warning("Please enter both text fields.")
    else:
        resume_skills = extract_skills(resume)
        job_skills = extract_skills(job_description)

        matching = sorted(set(resume_skills) & set(job_skills))
        missing = sorted(set(job_skills) - set(resume_skills))

        score = calculate_match(resume, job_description)

        st.divider()
        st.subheader("Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Text Match", f"{score}%")
        col2.metric("Matching Skills", len(matching))
        col3.metric("Missing Skills", len(missing))

        st.progress(score / 100)

        st.subheader("✅ Matching Skills")
        if matching:
            st.write(", ".join(matching))
        else:
            st.write("No matching skills detected.")

        st.subheader("📚 Skills to Consider Learning")
        if missing:
            st.write(", ".join(missing))
        else:
            st.success("All detected job skills appear in your resume.")

        st.subheader("💡 Suggestions")

        if missing:
            st.write(
                "1. Review the missing skills and learn the "
                "ones relevant to your career goals."
            )
            st.write(
                "2. If you genuinely have these skills, "
                "describe your experience with them in your resume."
            )

        if score < 30:
            st.write(
                "3. Check whether your resume targets the "
                "same role as the job description."
            )
        else:
            st.write(
                "3. Tailor your project descriptions and "
                "achievements to the target role."
            )

        st.caption(
            "Note: Skill extraction uses a predefined skill list. "
            "The tool may miss synonyms and context."
        )
