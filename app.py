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

    # Scale
    scaled = scaler.transform(user_data)

    # 🔥 APPLY SAME CGPA WEIGHT
    scaled[0, -1] = scaled[0, -1] * 3

    # PCA transform
    pca_data = pca.transform(scaled)

    # Predict cluster
    cluster = kmeans.predict(pca_data)[0]

    # Assign cluster labels to dataset
    df['Cluster'] = kmeans.labels_

    # Filter same cluster
    cluster_data = df[df['Cluster'] == cluster].copy()

    # Feature transform for dataset
    features = cluster_data.drop(['Student', 'Cluster'], axis=1)
    features_scaled = scaler.transform(features)

    # 🔥 APPLY SAME CGPA WEIGHT TO DATASET
    features_scaled[:, -1] = features_scaled[:, -1] * 3

    features_pca = pca.transform(features_scaled)

    # Distance calculation
    distances = np.linalg.norm(features_pca - pca_data, axis=1)
    cluster_data['Distance'] = distances

    # Sort by closest
    cluster_data = cluster_data.sort_values(by='Distance')

    # Take top 5 → then random 3 (to avoid repetition)
    top_candidates = cluster_data.head(5)
    final_matches = top_candidates.sample(n=min(3, len(top_candidates)))

    # ------------------ OUTPUT ------------------
    st.write("---")
    st.subheader("🎯 Your Best Teammates")

    for _, row in final_matches.iterrows():
        st.markdown(f"""
        <div class="card">
            <h4>👤 {row['Student']}</h4>
            <p>💻 Python: {row['Python']} | 🤖 ML: {row['ML']} | 🌐 Web: {row['Web']}</p>
            <p>🧠 AI: {row['AI']} | 🗣 Communication: {row['Communication']}</p>
            <p>📊 CGPA: {row['CGPA']}</p>
        </div>
        """, unsafe_allow_html=True)