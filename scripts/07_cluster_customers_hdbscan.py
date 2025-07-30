import pandas as pd
import os
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Paths
input_path = "data/processed/customer_rfm_features.csv"
output_dir = "data/processed"
clustered_path = os.path.join(output_dir, "customer_clusters.csv")
plot_path = "reports/figures/customer_clusters_heatmap.png"
os.makedirs("reports/figures", exist_ok=True)

# Load RFM data
df = pd.read_csv(input_path)

# Select numeric features only
features = ["recency", "frequency", "monetary", "avg_txn_amt", "total_points"]
X = df[features].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=20, prediction_data=True)
cluster_labels = clusterer.fit_predict(X_scaled)

# Attach cluster labels to original DataFrame
df["cluster"] = cluster_labels

# Save clustered data
df.to_csv(clustered_path, index=False)
print(f"Clustered data saved to {clustered_path}")

# Visualize cluster means (heatmap)
cluster_profile = df.groupby("cluster")[features].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_profile, cmap="YlGnBu", annot=True, fmt=".1f")
plt.title("Customer Cluster Profiles (HDBSCAN)")
plt.tight_layout()
plt.savefig(plot_path)
print(f"Heatmap saved to {plot_path}")
