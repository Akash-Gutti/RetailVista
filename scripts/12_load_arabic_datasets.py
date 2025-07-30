import pandas as pd
import os

# File paths
labr_train = "data/raw/labr/2class-balanced-train.txt"
labr_test = "data/raw/labr/2class-balanced-test.txt"
sarcasm_v1_train = "data/raw/arsarcasm_v1/ArSarcasm_train.csv"
sarcasm_v1_test = "data/raw/arsarcasm_v1/ArSarcasm_test.csv"
sarcasm_v2_train = "data/raw/arsarcasm_v2/training_data.csv"
sarcasm_v2_test = "data/raw/arsarcasm_v2/testing_data.csv"

# Load LABR (Tab-separated)
df_labr_train = pd.read_csv(labr_train, sep="\t", header=None, names=["review", "label"])
df_labr_test = pd.read_csv(labr_test, sep="\t", header=None, names=["review", "label"])

# Load ArSarcasm
df_sarcasm_v1_train = pd.read_csv(sarcasm_v1_train)
df_sarcasm_v1_test = pd.read_csv(sarcasm_v1_test)
df_sarcasm_v2_train = pd.read_csv(sarcasm_v2_train)
df_sarcasm_v2_test = pd.read_csv(sarcasm_v2_test)

# Inspection
print("LABR Train:", df_labr_train.shape)
print("LABR Test:", df_labr_test.shape)
print("ArSarcasm v1 (train):", df_sarcasm_v1_train.shape)
print("ArSarcasm v1 (test):", df_sarcasm_v1_test.shape)
print("ArSarcasm v2 (train):", df_sarcasm_v2_train.shape)
print("ArSarcasm v2 (test):", df_sarcasm_v2_test.shape)

# Optional: Save 5-sample previews
os.makedirs("data/processed", exist_ok=True)
df_labr_train.sample(5).to_csv("data/processed/labr_sample.csv", index=False)
df_sarcasm_v1_train.sample(5).to_csv("data/processed/arsarcasm_v1_sample.csv", index=False)
df_sarcasm_v2_train.sample(5).to_csv("data/processed/arsarcasm_v2_sample.csv", index=False)
