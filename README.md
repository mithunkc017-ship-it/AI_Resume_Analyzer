# AI Resume Analyzer

An AI-powered tool that analyzes resumes, extracts key candidate information, and provides smart insights or feedback to match job descriptions.

A beginner-friendly Python NLP project that compares
resume text with job descriptions.

## Features

- TF-IDF text similarity score
- Resume skill extraction
- Matching and missing skill detection
- Personalized improvement suggestions
- Interactive Streamlit interface

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Natural Language Processing (NLP)

## Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Run the application:

   streamlit run app.py

## How It Works

The application uses TF-IDF vectorization and cosine
similarity to estimate textual similarity between a resume
and a job description. It also compares detected skills
against a predefined skill list.

The score is not a hiring probability or a measure of
professional competence.
