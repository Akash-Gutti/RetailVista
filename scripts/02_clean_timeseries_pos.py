import pandas as pd
import os

# Input & output paths
input_file = "data/raw/pos_data.csv"
output_dir = "data/processed"
output_file = os.path.join(output_dir, "weekly_sku_sales.csv")

os.makedirs(output_dir, exist_ok=True)

# Load raw POS data
df = pd.read_csv(input_file, parse_dates=["date"])

# Ensure columns are standardized
assert {"sku", "units_sold", "date"}.issubset(df.columns), "Missing required columns."

# Resample to weekly sales per SKU
df["week"] = df["date"].dt.to_period("W").dt.start_time
weekly_df = (
    df.groupby(["sku", "week"])
    .agg({"units_sold": "sum"})
    .reset_index()
    .sort_values(["sku", "week"])
)

weekly_df.rename(columns={"week": "date"}, inplace=True)

# Save cleaned dataset
weekly_df.to_csv(output_file, index=False)
print(f"Weekly SKU sales saved to {output_file}")
