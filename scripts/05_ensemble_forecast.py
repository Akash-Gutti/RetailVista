import os
import pandas as pd

# File paths
prophet_path = "data/forecasts/prophet_forecast.csv"
neural_path = "data/forecasts/neuralprophet_forecast.csv"
output_path = "data/forecasts/ensemble_forecast.csv"

# Load forecasts
df_prophet = pd.read_csv(prophet_path, parse_dates=["date"])
df_neural = pd.read_csv(neural_path, parse_dates=["date"])

# Rename forecast columns
df_prophet.rename(columns={"forecast_units": "prophet_units"}, inplace=True)
df_neural.rename(columns={"forecast_units": "neural_units"}, inplace=True)

# Merge on SKU and date
df_merged = pd.merge(
    df_prophet, df_neural, on=["sku", "date"], how="inner"
)

# Compute ensemble average
df_merged["ensemble_units"] = df_merged[["prophet_units", "neural_units"]].mean(axis=1)

# Save to CSV
df_merged.to_csv(output_path, index=False)
print(f"Ensemble forecast saved to {output_path}")
