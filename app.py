
import re
import streamlit as st
import pymupdf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

SKILLS = [
    "python", "java", "javascript", "typescript", "c++",
    "sql", "mongodb", "mysql", "postgresql", "html", "css",
    "react", "node.js", "express", "flask", "django", "fastapi",
    "streamlit", "machine learning", "deep learning",
    "natural language processing", "nlp",
    "artificial intelligence", "data analysis", "pandas",
    "numpy", "scikit-learn", "tensorflow", "pytorch",
    "git", "github", "docker", "aws", "excel", "power bi",
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


def extract_pdf_text(uploaded_file):
    """Extract selectable text from an uploaded PDF."""
    pdf_bytes = uploaded_file.getvalue()

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.needs_pass:
            raise ValueError(
                "This PDF is password-protected. "
                "Please upload an unlocked PDF."
            )

        extracted_pages = []

        for page in document:
            extracted_pages.append(page.get_text())

        return "\n".join(extracted_pages).strip()


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
    "Upload your resume or paste its text. "
    "Compare it with a job description and discover "
    "matching skills and potential skill gaps."
)

st.info(
    "Your resume is processed locally by this application "
    "and is not saved to a database by this code."
)

st.subheader("1. Add Your Resume")

resume_method = st.radio(
    "Choose an input method",
    ["Upload PDF", "Paste resume text"],
    horizontal=True,
)

resume_text = ""

if resume_method == "Upload PDF":
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=["pdf"],
        help="Maximum file size: 10 MB.",
    )

    if uploaded_file is not None:
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("Your PDF exceeds the 10 MB limit.")
        else:
            try:
                resume_text = extract_pdf_text(uploaded_file)

                if resume_text:
                    st.success("PDF text extracted successfully!")

                    with st.expander("Preview extracted resume text"):
                        st.text_area(
                            "Extracted text",
                            value=resume_text,
                            height=220,
                            disabled=True,
                        )
                else:
                    st.warning(
                        "No selectable text was found. "
                        "This may be a scanned PDF. "
                        "Try a text-based PDF instead."
                    )

            except (ValueError, pymupdf.FileDataError) as error:
                st.error(str(error))
            except Exception:
                st.error(
                    "Could not read this PDF. "
                    "Please check that it is a valid PDF file."
                )

else:
    resume_text = st.text_area(
        "Paste your resume",
        height=250,
        placeholder=(
            "Example: Python developer with knowledge "
            "of machine learning, SQL, Git and data analysis..."
        ),
    )

st.divider()

st.subheader("2. Add the Job Description")

job_description = st.text_area(
    "Paste the job description",
    height=220,
    placeholder=(
        "Example: We are looking for a Python developer "
        "with SQL, machine learning and Git skills..."
    ),
)

analyze_clicked = st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True,
)

if analyze_clicked:
    if not resume_text.strip():
        st.warning("Please upload a readable PDF or enter resume text.")

    elif not job_description.strip():
        st.warning("Please enter a job description.")

    else:
        try:
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

            st.divider()
            st.header("3. Your Analysis Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Text Match", f"{score}%")
            col2.metric("Matching Skills", len(matching))
            col3.metric("Missing Skills", len(missing))

            st.progress(score / 100)

            st.caption(
                "Text similarity is an estimate, not a hiring "
                "probability or a measure of candidate quality."
            )

            st.subheader("✅ Matching Skills")

            if matching:
                st.write(", ".join(matching))
            else:
                st.write("No matching skills detected.")

            st.subheader("📚 Skills to Consider Learning")

            if missing:
                st.write(", ".join(missing))
                st.write(
                    "Focus on relevant gaps. Only add a skill "
                    "to your resume if you genuinely have it."
                )
            else:
                st.success(
                    "All skills detected in the job description "
                    "also appear in your resume."
                )

            st.subheader("💡 Suggestions")

            if score < 30:
                st.write(
                    "- Check whether your resume targets the "
                    "same role as the job description."
                )
            else:
                st.write(
                    "- Tailor your projects and experience "
                    "to the role you're applying for."
                )

            if missing:
                st.write(
                    "- Build projects or gain experience in "
                    "relevant missing skills."
                )

            st.write(
                "- Include measurable achievements where "
                "you can support them with real evidence."
            )

            report = f"""AI RESUME ANALYSIS REPORT

Text similarity score: {score}%

Matching skills:
{', '.join(matching) if matching else 'None detected'}

Skills to consider learning:
{', '.join(missing) if missing else 'None detected'}

Note: The score is based on text similarity.
It does not predict hiring success.
"""

            st.download_button(
                "Download Analysis Report",
                data=report,
                file_name="resume_analysis.txt",
                mime="text/plain",
            )

        except ValueError:
            st.error(
                "The text is too short or contains insufficient "
                "distinctive words for similarity analysis. "
                "Please provide more detailed text."
            )

        except Exception:
            st.error(
                "Analysis failed. Please check the resume and "
                "job description, then try again."
            )

st.divider()

st.caption(
    "Built with Python, Streamlit, PyMuPDF and Scikit-learn."
)
