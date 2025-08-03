import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load uplift predictions
input_path = "data/processed/uplift_predictions.csv"
df = pd.read_csv(input_path)

# Features & target
categorical = ["gender", "country", "segment"]
numerical = ["age", "recency", "frequency", "monetary", "avg_txn_amt", "total_points"]
target = "uplift_score"

# Encode categorical features
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder='passthrough')

# Pipeline with RandomForestRegressor (can be changed)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit model
X = df[categorical + numerical]
y = df[target]
pipeline.fit(X, y)

# Save pipeline
joblib.dump(pipeline, "models/lime_pipeline.pkl")

# Sample 50 customers for LIME explanations
lime_sample = df.sample(n=50, random_state=42)
lime_sample.to_csv("data/processed/lime_sample.csv", index=False)

print(f"Pipeline saved to models/lime_pipeline.pkl")
print(f"LIME sample saved to data/processed/lime_sample.csv")
