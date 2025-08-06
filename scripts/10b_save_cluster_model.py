import pandas as pd
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
X = df[features].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=20, prediction_data=True)
clusterer.fit(X_scaled)

# Save model
joblib.dump(clusterer, model_output_path)
print(f"✅ Cluster model saved to: {model_output_path}")
