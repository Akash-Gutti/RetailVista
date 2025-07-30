import pandas as pd
import numpy as np
import os
from dowhy import CausalModel
import warnings

warnings.filterwarnings("ignore")

# Paths
input_file = "data/processed/promo_uplift_data.csv"
output_file = "data/processed/causal_audit_summary.txt"

# Load data
df = pd.read_csv(input_file)

# Encode categorical vars
df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
df["cluster"] = df["cluster"].replace("Noise", -1)

# Build causal model
model = CausalModel(
    data=df,
    treatment="treatment",
    outcome="purchase",
    common_causes=["gender", "cluster"],
)

identified_estimand = model.identify_effect()
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

# Do refutation (placebo test)
refute = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter"
)

# Save summary
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Causal Estimate:\n")
    f.write(str(estimate))
    f.write("\n\nPlacebo Test:\n")
    f.write(str(refute))

print(f"Causal audit summary saved to {output_file}")
