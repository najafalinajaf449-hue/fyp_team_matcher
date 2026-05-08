import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ---------------- LOAD DATA ----------------
df = pd.read_csv("cs_students.csv")

# Remove duplicates
df = df.drop_duplicates()

# ---------------- SKILL MAPPING ----------------
skill_map = {
    "Weak": 1,
    "Average": 2,
    "Strong": 3
}

df["Python_num"] = df["Python"].map(skill_map)
df["SQL_num"] = df["SQL"].map(skill_map)
df["Java_num"] = df["Java"].map(skill_map)

# ---------------- DOMAIN ENCODING ----------------
domain_encoder = LabelEncoder()

df["Domain_num"] = domain_encoder.fit_transform(
    df["Interested Domain"]
)

# ---------------- FEATURE SELECTION ----------------
features = df[[
    "GPA",
    "Domain_num",
    "Python_num",
    "SQL_num",
    "Java_num"
]]

# ---------------- SCALING ----------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(features)

# ---------------- KNN MODEL ----------------
knn = NearestNeighbors(
    n_neighbors=5,
    metric='euclidean'
)

knn.fit(X_scaled)

# ---------------- SAVE FILES ----------------
joblib.dump(knn, "knn.pkl")

joblib.dump(scaler, "scaler.pkl")

joblib.dump(domain_encoder, "domain_encoder.pkl")

joblib.dump(df, "data.pkl")

print("✅ KNN Team Matcher Model Trained Successfully!")