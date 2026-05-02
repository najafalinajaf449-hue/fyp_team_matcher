import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# ---------------- LOAD DATA ----------------
df = pd.read_csv("students_skills_large.csv")

students = df["Student"]
X = df.drop("Student", axis=1)

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔥 GIVE MORE IMPORTANCE TO CGPA
X_scaled[:, -1] = X_scaled[:, -1] * 3

# ---------------- PCA ----------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ---------------- KMEANS ----------------
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_pca)

# ---------------- SAVE ----------------
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(kmeans, "kmeans.pkl")
joblib.dump(df, "data.pkl")

print("✅ Model trained and saved successfully!")