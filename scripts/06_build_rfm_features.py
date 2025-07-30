import pandas as pd
import os
from datetime import datetime

# Paths
cust_path = "data/raw/customers.csv"
loyalty_path = "data/raw/loyalty_transactions.csv"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "customer_rfm_features.csv")

# Load data
customers = pd.read_csv(cust_path)
loyalty = pd.read_csv(loyalty_path, parse_dates=["txn_date"])

# Set reference date
snapshot_date = loyalty["txn_date"].max() + pd.Timedelta(days=1)

# RFM aggregation
rfm = (
    loyalty.groupby("customer_id")
    .agg(
        recency=("txn_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("txn_date", "count"),
        monetary=("txn_amount", "sum"),
        avg_txn_amt=("txn_amount", "mean"),
        total_points=("points_earned", "sum"),
    )
    .reset_index()
)

# Merge with static customer attributes
df = pd.merge(customers, rfm, on="customer_id", how="inner")

# Save
df.to_csv(output_path, index=False)
print(f"RFM features saved to {output_path}")
