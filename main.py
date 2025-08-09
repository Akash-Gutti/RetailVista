from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import os
import json
import pandas as pd
import numpy as np
import joblib
import hdbscan

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================================================
# Paths & Globals
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"

# Forecast CSVs
PROPHET_CSV   = DATA_DIR / "forecasts" / "prophet_forecast.csv"
NEURAL_CSV    = DATA_DIR / "forecasts" / "neuralprophet_forecast.csv"
ENSEMBLE_CSV  = DATA_DIR / "forecasts" / "ensemble_forecast.csv"

# Campaign summaries (CSV preferred; fallback to JSON)
CAMPAIGN_CSV  = DATA_DIR / "processed" / "campaign_summaries.csv"
CAMPAIGN_JSON = DATA_DIR / "processed" / "campaign_summaries.json"

# Models
CLUSTER_BUNDLE_PKL = MODEL_DIR / "customer_cluster_model.pkl"  # dict: {'features','scaler','clusterer'}
SENTIMENT_PKL      = MODEL_DIR / "sentiment_model.pkl"         # 3-class scikit pipeline (TF-IDF + LR)

# In-memory holders
prophet_df: Optional[pd.DataFrame]  = None
neural_df: Optional[pd.DataFrame]   = None
ensemble_df: Optional[pd.DataFrame] = None
campaign_df: Optional[pd.DataFrame] = None

cluster_bundle: Optional[Dict[str, Any]] = None
sentiment_model = None  # scikit pipeline (predict_proba)

SEGMENT_MAPPING = {0: "Bronze", 1: "Silver", 2: "Gold"}
SENT_LABELS     = {0: "Negative", 1: "Neutral", 2: "Positive"}

# =========================================================
# FastAPI App
# =========================================================
app = FastAPI(
    title="RetailVista Unified API",
    description="Unified API for forecasts, customer clustering, sentiment analysis, "
    "and GPT-powered campaign summaries.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (loosen if you want to call from Streamlit Cloud, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Models / Schemas
# =========================================================
class ForecastRequest(BaseModel):
    sku: str = Field(..., description="SKU code, e.g., 'SKU_100'")
    from_date: str = Field(..., description="YYYY-MM-DD")
    to_date: str   = Field(..., description="YYYY-MM-DD")

class ClusterInput(BaseModel):
    # Accepts arbitrary mapping, but must cover the bundle features
    features: Dict[str, float] = Field(..., description="Feature mapping matching the trained cluster bundle.")

class ClusterOutput(BaseModel):
    cluster_id: int
    segment: str
    strength: Optional[float] = None
    description: Optional[str] = None

class SentimentInput(BaseModel):
    review: str

class SentimentOutput(BaseModel):
    label: str
    label_id: int
    confidence: float

class CampaignResponse(BaseModel):
    matched_summaries: List[Dict[str, Any]]

# =========================================================
# Loaders
# =========================================================
def _load_forecast_csv(path: Path, y_col_candidates: List[str]) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # normalize one forecast column name to 'forecast_units'
    for cand in y_col_candidates:
        if cand in df.columns:
            if cand != "forecast_units":
                df = df.rename(columns={cand: "forecast_units"})
            break
    return df

def load_all_forecasts() -> None:
    global prophet_df, neural_df, ensemble_df
    prophet_df  = _load_forecast_csv(PROPHET_CSV,  ["forecast_units", "prophet_units"])
    neural_df   = _load_forecast_csv(NEURAL_CSV,   ["forecast_units", "neural_units"])
    ensemble_df = _load_forecast_csv(ENSEMBLE_CSV, ["forecast_units", "ensemble_units"])

def load_campaigns() -> None:
    """Load campaigns from CSV (preferred) or JSON."""
    global campaign_df
    if CAMPAIGN_CSV.exists():
        campaign_df = pd.read_csv(CAMPAIGN_CSV)
    elif CAMPAIGN_JSON.exists():
        data = json.loads(CAMPAIGN_JSON.read_text(encoding="utf-8"))
        campaign_df = pd.DataFrame(data)
    else:
        campaign_df = None

def load_cluster_bundle() -> None:
    global cluster_bundle
    if CLUSTER_BUNDLE_PKL.exists():
        obj = joblib.load(CLUSTER_BUNDLE_PKL)
        # allow legacy model file (just estimator)
        if isinstance(obj, dict) and {"features", "scaler", "clusterer"}.issubset(obj.keys()):
            cluster_bundle = obj
        else:
            cluster_bundle = None  # won't support inference without the scaler/features
    else:
        cluster_bundle = None

def load_sentiment_model() -> None:
    global sentiment_model
    sentiment_model = joblib.load(SENTIMENT_PKL) if SENTIMENT_PKL.exists() else None

# =========================================================
# Utilities
# =========================================================
def _filter_forecast(df: pd.DataFrame, sku: str, from_date: str, to_date: str) -> List[Dict[str, Any]]:
    if df is None:
        return []
    mask = (df["sku"].astype(str) == str(sku)) if "sku" in df.columns else pd.Series(True, index=df.index)
    sub = df.loc[mask].copy()
    sub = sub[(sub["date"] >= pd.to_datetime(from_date)) & (sub["date"] <= pd.to_datetime(to_date))]
    if sub.empty:
        return []
    out = sub[["date", "forecast_units"] + (["sku"] if "sku" in sub.columns else [])].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")

def _predict_cluster(features_map: Dict[str, float]) -> Tuple[int, Optional[float]]:
    """Use HDBSCAN approximate_predict via the saved bundle. Returns (label, strength)."""
    if not cluster_bundle:
        raise HTTPException(status_code=503, detail="Cluster bundle not loaded.")
    feats  = cluster_bundle["features"]
    scaler = cluster_bundle["scaler"]
    model  = cluster_bundle["clusterer"]

    # Validate required features
    missing = [f for f in feats if f not in features_map]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}. Expected: {feats}")

    vec = np.array([[float(features_map[f]) for f in feats]], dtype=float)
    Xs  = scaler.transform(vec)

    # module-level function (not a method on the estimator)
    labels, strengths = hdbscan.approximate_predict(model, Xs)
    return int(labels[0]), float(strengths[0])

def _predict_sentiment(review: str) -> Tuple[int, float]:
    """Return (label_id, confidence) using 3-class scikit model."""
    if sentiment_model is None:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded.")
    if not review or not review.strip():
        raise HTTPException(status_code=400, detail="Empty review.")
    proba = sentiment_model.predict_proba([review])[0]
    label_id = int(np.argmax(proba))
    confidence = float(proba[label_id])
    return label_id, confidence

# =========================================================
# Startup
# =========================================================
@app.on_event("startup")
def _startup():
    load_all_forecasts()
    load_campaigns()
    load_cluster_bundle()
    load_sentiment_model()

# =========================================================
# Pretty home page
# =========================================================
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>RetailVista Unified API</title>
  <style>
    body {font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; background: #0f172a; color: #e2e8f0; margin: 0;}
    .wrapper {max-width: 960px; margin: 48px auto; padding: 0 20px;}
    .card {background: #0b1220; border: 1px solid #1f2937; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.25);}
    h1 {margin: 0 0 8px 0; font-size: 28px;}
    p  {margin: 8px 0 16px 0; color: #9ca3af;}
    a.button {display:inline-block; padding:10px 14px; border:1px solid #334155; border-radius:10px; color:#e2e8f0; text-decoration:none; margin-right:10px;}
    code {background:#111827; padding:2px 6px; border-radius:6px; color:#a5b4fc;}
    .grid {display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top:16px;}
    .tile {background:#0b1220; border:1px solid #1f2937; border-radius:12px; padding:16px;}
    .tile h3 {margin:0 0 8px 0; font-size:16px;}
    .tile p {margin:0; font-size:14px; color:#94a3b8;}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="card">
      <h1>RetailVista Unified API</h1>
      <p>Forecasts · Clustering · Sentiment · Campaign Summaries</p>
      <a class="button" href="/docs">OpenAPI Docs</a>
      <a class="button" href="/health">Health</a>
    </div>

    <div class="grid">
      <div class="tile">
        <h3>/forecast/{source}</h3>
        <p>POST JSON: <code>{"sku":"SKU_100","from_date":"2024-06-01","to_date":"2024-07-31"}</code><br/>source ∈ prophet | neuralprophet | ensemble</p>
      </div>
      <div class="tile">
        <h3>/cluster/infer</h3>
        <p>POST JSON: <code>{"features": {"recency":30,"frequency":5,"monetary":800,"avg_txn_amt":160,"total_points":1200}}</code></p>
      </div>
      <div class="tile">
        <h3>/sentiment/predict</h3>
        <p>POST JSON: <code>{"review": "نص المراجعة..."}</code> &nbsp; → 3-class (negative/neutral/positive)</p>
      </div>
      <div class="tile">
        <h3>/campaigns/by-segment</h3>
        <p>GET params: <code>?segment=Gold&top_k=10</code> or <code>?cluster_id=3</code></p>
      </div>
    </div>
  </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return HOME_HTML

@app.get("/health")
def health():
    return {
        "status": "ok",
        "forecasts": {
            "prophet": PROPHET_CSV.exists(),
            "neuralprophet": NEURAL_CSV.exists(),
            "ensemble": ENSEMBLE_CSV.exists(),
        },
        "campaigns_loaded": campaign_df is not None,
        "cluster_bundle_loaded": cluster_bundle is not None,
        "sentiment_model_loaded": sentiment_model is not None,
    }

# =========================================================
# Forecast Endpoints (Unified + Back-Compat)
# =========================================================
@app.post("/forecast/{source}")
def get_forecast(source: str, req: ForecastRequest):
    src = source.lower().strip()
    if src not in {"prophet", "neuralprophet", "ensemble"}:
        raise HTTPException(status_code=400, detail="source must be one of: prophet | neuralprophet | ensemble")

    if src == "prophet":
        rows = _filter_forecast(prophet_df, req.sku, req.from_date, req.to_date)
    elif src == "neuralprophet":
        rows = _filter_forecast(neural_df, req.sku, req.from_date, req.to_date)
    else:
        rows = _filter_forecast(ensemble_df, req.sku, req.from_date, req.to_date)

    if not rows:
        raise HTTPException(status_code=404, detail="No matching forecast rows found.")
    return rows

# Back-compat routes (optional)
@app.post("/forecast/prophet")
def forecast_prophet(req: ForecastRequest):   return get_forecast("prophet", req)
@app.post("/forecast/neuralprophet")
def forecast_neural(req: ForecastRequest):    return get_forecast("neuralprophet", req)
@app.post("/forecast/ensemble")
def forecast_ensemble(req: ForecastRequest):  return get_forecast("ensemble", req)

# =========================================================
# Clustering Endpoint
# =========================================================
@app.post("/cluster/infer", response_model=ClusterOutput)
def infer_cluster(payload: ClusterInput):
    if not cluster_bundle:
        raise HTTPException(status_code=503, detail="Cluster bundle not loaded on server.")
    label, strength = _predict_cluster(payload.features)

    description = {
        0: "Price-sensitive but consistent",
        1: "Moderate value, moderate loyalty",
        2: "High value, high loyalty",
    }.get(label, "N/A")

    return ClusterOutput(
        cluster_id=label,
        segment=SEGMENT_MAPPING.get(label, "Unknown"),
        strength=strength,
        description=description
    )

# =========================================================
# Sentiment Endpoint (3-class)
# =========================================================
@app.post("/sentiment/predict", response_model=SentimentOutput)
def predict_sentiment(payload: SentimentInput):
    label_id, conf = _predict_sentiment(payload.review)
    return SentimentOutput(
        label=SENT_LABELS[label_id],
        label_id=label_id,
        confidence=round(conf, 4)
    )

# =========================================================
# Campaign Summaries
# =========================================================
@app.get("/campaigns/by-segment", response_model=CampaignResponse)
def campaigns_by_segment(
    segment: Optional[str] = Query(None, description="Bronze | Silver | Gold"),
    cluster_id: Optional[int] = Query(None, description="Cluster number"),
    top_k: int = Query(20, ge=1, le=200, description="Limit results"),
):
    if campaign_df is None:
        raise HTTPException(status_code=503, detail="Campaign summaries not loaded.")

    sub = campaign_df.copy()
    if segment:
        sub = sub[sub["segment"].astype(str).str.lower() == segment.lower()]
    if cluster_id is not None and "cluster" in sub.columns:
        sub = sub[sub["cluster"] == cluster_id]

    if sub.empty:
        raise HTTPException(status_code=404, detail="No matching summaries found.")

    # If uplift_score columns exist, sort by it
    sort_col = "uplift_score"
    if sort_col in sub.columns:
        sub["uplift_score_num"] = pd.to_numeric(sub[sort_col], errors="coerce")
        sub = sub.sort_values("uplift_score_num", ascending=False)

    cols = [c for c in ["cluster", "segment", "uplift_score", "campaign_brief"] if c in sub.columns]
    out = sub[cols].head(top_k).to_dict(orient="records")
    return CampaignResponse(matched_summaries=out)

# =========================================================
# Local dev entry (optional)
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
