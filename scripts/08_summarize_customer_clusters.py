import pandas as pd
import os

# Input
input_file = "data/processed/customer_clusters.csv"
output_file = "data/processed/cluster_personas.csv"

# Load clustered data
df = pd.read_csv(input_file)

# Replace -1 with 'Noise' for readability
df["cluster"] = df["cluster"].apply(lambda x: "Noise" if x == -1 else f"Cluster {x}")

# Features to summarize
group_cols = [
    "gender", "country", "segment"
]
agg_cols = [
    "recency", "frequency", "monetary",
    "avg_txn_amt", "total_points"
]

# Summarize demographics
demographic_summary = df.groupby("cluster")[group_cols].agg(lambda x: x.mode()[0])

# Summarize RFM behavior
rfm_summary = df.groupby("cluster")[agg_cols].mean().round(1)

# Merge summaries
summary = pd.concat([demographic_summary, rfm_summary], axis=1).reset_index()

# Save
os.makedirs("data/processed", exist_ok=True)
summary.to_csv(output_file, index=False)
print(f"Cluster personas exported to {output_file}")
