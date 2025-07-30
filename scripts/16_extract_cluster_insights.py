# scripts/16_extract_cluster_insights.py

import pandas as pd
import os

# Load clustered customer data
df = pd.read_csv("data/processed/customer_clusters.csv")

# Drop noise points (HDBSCAN labels them as -1)
df = df[df['cluster'] != -1]

# Group by cluster and compute aggregate stats
summary = df.groupby("cluster").agg({
    "recency": ["mean", "min", "max"],
    "frequency": ["mean", "min", "max"],
    "monetary": ["mean", "min", "max"],
    "customer_id": "count"
}).reset_index()

# Flatten MultiIndex
summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

# Rename for clarity
summary = summary.rename(columns={"customer_id_count": "num_customers"})

# Save summary for GPT input
os.makedirs("data/processed", exist_ok=True)
summary.to_csv("data/processed/cluster_insights_summary.csv", index=False)
print("Cluster insights saved to data/processed/cluster_insights_summary.csv")
