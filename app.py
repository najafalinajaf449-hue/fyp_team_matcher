import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="FYP Team Matcher", layout="centered")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .card {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODELS ------------------
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
kmeans = joblib.load("kmeans.pkl")
df = joblib.load("data.pkl")

# ------------------ HEADER ------------------
st.title("🎓 Smart FYP Team Matcher")
st.markdown("### Find your perfect teammates using AI 🤖")
st.write("---")

# ------------------ INPUT SECTION ------------------
st.subheader("Enter Your Skills")

col1, col2 = st.columns(2)

with col1:
    python = st.slider("Python", 1, 10)
    ml = st.slider("Machine Learning", 1, 10)
    web = st.slider("Web Development", 1, 10)
    ai = st.slider("AI", 1, 10)

with col2:
    comm = st.slider("Communication", 1, 10)
    lead = st.slider("Leadership", 1, 10)
    db = st.slider("Database", 1, 10)
    cgpa = st.slider("CGPA", 1.0, 4.0)

# ------------------ MATCHING LOGIC ------------------
if st.button("🚀 Find My Team"):
    
    # User input
    user_data = [[python, ml, web, ai, comm, lead, db, cgpa]]

    # Transform
    scaled = scaler.transform(user_data)
    pca_data = pca.transform(scaled)

    # Predict cluster
    cluster = kmeans.predict(pca_data)[0]

    # Assign clusters to dataset
    df['Cluster'] = kmeans.labels_

    # Filter cluster
    cluster_data = df[df['Cluster'] == cluster].copy()

    # Compute distances
    features = cluster_data.drop(['Student', 'Cluster'], axis=1)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    distances = np.linalg.norm(features_pca - pca_data, axis=1)
    cluster_data['Distance'] = distances

    # Get top 3 matches
    top_matches = cluster_data.sort_values(by='Distance').head(3)

    # ------------------ OUTPUT ------------------
    st.write("---")
    st.subheader("🎯 Your Best Teammates")

    for _, row in top_matches.iterrows():
        st.markdown(f"""
        <div class="card">
            <h4>👤 {row['Student']}</h4>
            <p>💻 Python: {row['Python']} | 🤖 ML: {row['ML']} | 🌐 Web: {row['Web']}</p>
            <p>🧠 AI: {row['AI']} | 🗣 Communication: {row['Communication']}</p>
            <p>📊 CGPA: {row['CGPA']}</p>
        </div>
        """, unsafe_allow_html=True)