import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# Paths
cust_path = "data/processed/customer_clusters.csv"
output_file = "data/processed/promo_uplift_data.csv"
os.makedirs("data/processed", exist_ok=True)

# Load customers
df = pd.read_csv(cust_path)

# Simulate campaign exposure (binary treatment)
df["treatment"] = np.random.binomial(1, 0.5, len(df))

# Simulate baseline purchase behavior
df["base_purchase"] = np.random.normal(50, 15, len(df)).clip(0)

# Simulate uplift effect (some clusters respond more)
df["uplift_factor"] = df["cluster"].apply(
    lambda x: 0.2 if x == 0 else (0.4 if x == 1 else 0.1)
)

# Simulate observed purchase with treatment effect
df["purchase"] = df["base_purchase"] + (df["treatment"] * df["uplift_factor"] * df["base_purchase"])

# Clip purchase to realistic values
df["purchase"] = df["purchase"].round(2).clip(0, 200)

# Drop intermediate columns
df.drop(columns=["base_purchase", "uplift_factor"], inplace=True)

# Save
df.to_csv(output_file, index=False)
print(f"Promo uplift simulation saved to {output_file}")
