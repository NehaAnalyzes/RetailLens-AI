import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
vectorizer = joblib.load("skills_vectorizer.pkl")
kmeans = joblib.load("karkidi_cluster_model.pkl")

# Load clustered job data
job_data = pd.read_csv("karkidi_clustered_jobs.csv")

# Title
st.set_page_config(page_title="Karkidi Job Recommender", layout="wide")
st.title("🔍 Job Recommender Based on Your Skills")

# Login Simulation via GitHub (done automatically via Streamlit sharing)
st.markdown("**Logged in via GitHub (Streamlit Sharing handles this automatically).**")

# User skill input
user_input = st.text_input("Enter your skills (comma separated):", "python, machine learning, sql")

if st.button("Find Matching Jobs"):
    def clean_skills(skill_str):
        skills = [skill.strip().lower() for skill in skill_str.split(',')]
        return ' '.join(skills)

    cleaned = clean_skills(user_input)
    user_vector = vectorizer.transform([cleaned])
    user_cluster = kmeans.predict(user_vector)[0]

    st.success(f"📌 Based on your skills, you match **Cluster {user_cluster}**.")

    matching_jobs = job_data[job_data['cluster'] == user_cluster]
    st.subheader(f"📄 Found {len(matching_jobs)} matching jobs:")

    for _, row in matching_jobs.iterrows():
        st.markdown(f"""
        ---
        ### {row['Title']}
        - **Company**: {row['Company']}
        - **Location**: {row['Location']}
        - **Experience**: {row['Experience']}
        - **Skills**: {row['Skills']}
        - **Summary**: {row['Summary']}
        """)

