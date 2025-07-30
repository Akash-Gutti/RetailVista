import os
import pandas as pd
from neuralprophet import NeuralProphet

# Parameters
input_file = "data/processed/weekly_sku_sales.csv"
output_dir = "data/forecasts"
os.makedirs(output_dir, exist_ok=True)

# Load weekly time series
df = pd.read_csv(input_file, parse_dates=["date"])
sku_list = df["sku"].unique()
forecast_horizon = 12  # weeks

def forecast_with_neuralprophet(sku, data):
    sku_df = data[data["sku"] == sku].copy()
    sku_df = sku_df.rename(columns={"date": "ds", "units_sold": "y"})
    sku_df = sku_df[["ds", "y"]]  # Drop unexpected columns

    if len(sku_df) < 15:
        return None

    model = NeuralProphet(n_changepoints=10, yearly_seasonality=False, weekly_seasonality=True)
    model.fit(sku_df, freq="W", progress="off")

    future = model.make_future_dataframe(sku_df, periods=forecast_horizon)
    forecast = model.predict(future)

    result = forecast[["ds", "yhat1"]].copy()
    result["sku"] = sku
    return result

# Forecast each SKU
results = []
for sku in sku_list:
    print(f"Forecasting (NeuralProphet) for {sku} ...")
    forecast_df = forecast_with_neuralprophet(sku, df)
    if forecast_df is not None:
        results.append(forecast_df)

# Concatenate all results
final_df = pd.concat(results, axis=0)
final_df.rename(columns={"ds": "date", "yhat1": "forecast_units"}, inplace=True)

# Save output
output_file = os.path.join(output_dir, "neuralprophet_forecast.csv")
final_df.to_csv(output_file, index=False)
print(f"NeuralProphet forecasts saved to {output_file}")
