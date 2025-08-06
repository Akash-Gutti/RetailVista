import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# ---------------- Setup ----------------
app = FastAPI(title="Static Forecast API", version="1.0")

# Load CSVs
try:
    prophet_df = pd.read_csv("data/forecasts/prophet_forecast.csv", parse_dates=["date"])
    neural_df = pd.read_csv("data/forecasts/neuralprophet_forecast.csv", parse_dates=["date"])
    ensemble_df = pd.read_csv("data/forecasts/ensemble_forecast.csv", parse_dates=["date"])
except Exception as e:
    raise RuntimeError(f"❌ Failed to load forecast CSVs: {e}")

# ---------------- Request Model ----------------
class ForecastRequest(BaseModel):
    sku: str
    from_date: str  # YYYY-MM-DD
    to_date: str    # YYYY-MM-DD

# ---------------- Endpoints ----------------
@app.post("/forecast/prophet")
def get_prophet_forecast(req: ForecastRequest):
    df = prophet_df.query("sku == @req.sku and date >= @req.from_date and date <= @req.to_date")
    if df.empty:
        raise HTTPException(status_code=404, detail="No matching Prophet forecast found.")
    return df.to_dict(orient="records")

@app.post("/forecast/neuralprophet")
def get_neural_forecast(req: ForecastRequest):
    df = neural_df.query("sku == @req.sku and date >= @req.from_date and date <= @req.to_date")
    if df.empty:
        raise HTTPException(status_code=404, detail="No matching NeuralProphet forecast found.")
    return df.to_dict(orient="records")

@app.post("/forecast/ensemble")
def get_ensemble_forecast(req: ForecastRequest):
    df = ensemble_df.query("sku == @req.sku and date >= @req.from_date and date <= @req.to_date")
    if df.empty:
        raise HTTPException(status_code=404, detail="No matching Ensemble forecast found.")
    return df.to_dict(orient="records")

# ---------------- Run Locally ----------------
if __name__ == "__main__":
    uvicorn.run("scripts.22_fastapi_forecasting_server:app", host="0.0.0.0", port=8000, reload=True)
