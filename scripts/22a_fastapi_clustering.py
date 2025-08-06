import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# ----------- Config -----------
MODEL_PATH = "models/customer_cluster_model.pkl"
SEGMENT_MAPPING = {
    0: "Bronze",
    1: "Silver",
    2: "Gold"
}
CLUSTER_DESCRIPTIONS = {
    0: "Price-sensitive but consistent",
    1: "Moderate value, moderate loyalty",
    2: "High value, high loyalty"
}

# ----------- Load Model -----------
try:
    cluster_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Please train and save it first.")

# ----------- FastAPI App -----------
app = FastAPI(
    title="Customer Clustering Inference API",
    description="Infer cluster, segment, and description from customer profile",
    version="1.0"
)

class CustomerInput(BaseModel):
    age: float
    recency: float  # days since last purchase
    frequency: float  # purchases/month
    monetary: float  # monthly spend

@app.post("/cluster/infer")
def infer_cluster(customer: CustomerInput):
    input_df = pd.DataFrame([{
        "age": customer.age,
        "recency": customer.recency,
        "frequency": customer.frequency,
        "monetary": customer.monetary
    }])

    cluster_id = cluster_model.fit_predict(input_df)[0]  # Using fit_predict for HDBSCAN
    segment = SEGMENT_MAPPING.get(cluster_id, "Unknown")
    description = CLUSTER_DESCRIPTIONS.get(cluster_id, "No description available")

    return {
        "cluster_id": int(cluster_id),
        "segment": segment,
        "description": description
    }

# Optional: local run
if __name__ == "__main__":
    uvicorn.run("scripts.22afastapi_clustering:app", host="127.0.0.1", port=8001, reload=True)
