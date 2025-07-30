import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from econml.metalearners import TLearner

# Paths
input_file = "data/processed/promo_uplift_data.csv"
output_file = "data/processed/uplift_predictions.csv"

# Load data
df = pd.read_csv(input_file)

# Features
X = pd.get_dummies(df[["gender", "cluster"]], drop_first=True)
T = df["treatment"].values
Y = df["purchase"].values

# Split for validation (not mandatory)
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, T, Y, test_size=0.2, random_state=42
)

# T-Learner with Gradient Boosting
model_t = GradientBoostingRegressor(n_estimators=100, max_depth=3)
model_c = GradientBoostingRegressor(n_estimators=100, max_depth=3)
learner = TLearner(models=[model_t, model_c])
learner.fit(Y_train, T_train, X=X_train)

# Predict treatment effect on all customers
te_pred = learner.effect(X)

# Save predictions
df_out = df.copy()
df_out["uplift_score"] = te_pred.round(2)
df_out.to_csv(output_file, index=False)

print(f"Uplift predictions saved to {output_file}")
