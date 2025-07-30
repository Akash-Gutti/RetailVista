import os
import pandas as pd
from prophet import Prophet

# Parameters
input_file = "data/processed/weekly_sku_sales.csv"
output_dir = "data/forecasts"
os.makedirs(output_dir, exist_ok=True)

# Load weekly time series
df = pd.read_csv(input_file, parse_dates=["date"])
sku_list = df["sku"].unique()
forecast_horizon = 12  # weeks

def forecast_sku(sku, data):
    sku_df = data[data["sku"] == sku].copy()
    sku_df = sku_df.rename(columns={"date": "ds", "units_sold": "y"})

    # Skip SKUs with too few data points
    if len(sku_df) < 15:
        return None

    model = Prophet()
    model.fit(sku_df)

    future = model.make_future_dataframe(periods=forecast_horizon, freq="W")
    forecast = model.predict(future)

    result = forecast[["ds", "yhat"]].copy()
    result["sku"] = sku
    return result

# Forecast for each SKU
results = []
for sku in sku_list:
    print(f"Forecasting for {sku} ...")
    forecast_df = forecast_sku(sku, df)
    if forecast_df is not None:
        results.append(forecast_df)

# Concatenate all SKU forecasts
final_df = pd.concat(results, axis=0)
final_df.rename(columns={"ds": "date", "yhat": "forecast_units"}, inplace=True)

# Save forecast
output_file = os.path.join(output_dir, "prophet_forecast.csv")
final_df.to_csv(output_file, index=False)
print(f"Forecasts saved to {output_file}")
