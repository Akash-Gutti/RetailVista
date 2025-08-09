import pandas as pd
import numpy as np
import os
import hdbscan
import joblib
from sklearn.preprocessing import StandardScaler

# Paths
input_path = "data/processed/customer_rfm_features.csv"
model_output_path = "models/customer_cluster_model.pkl"
os.makedirs("models", exist_ok=True)

# Load RFM data
df = pd.read_csv(input_path)

# Select numeric features
features = ["recency", "frequency", "monetary", "avg_txn_amt", "total_points"]
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Missing required features in {input_path}: {missing}")

X = df[features].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=20, prediction_data=True)
labels = clusterer.fit_predict(X_scaled)

# Optional: quick visibility (does not write any new CSV)
noise_ratio = (labels == -1).mean()
print(f"Trained HDBSCAN. Noise (-1) ratio: {noise_ratio:.2%}")

# -------- Save bundle (for Streamlit & FastAPI inference) --------
bundle = {
    "features": features,
    "scaler": scaler,
    "clusterer": clusterer
}

# Save model
joblib.dump(bundle, model_output_path)
print(f"Cluster bundle saved to: {model_output_path}")
