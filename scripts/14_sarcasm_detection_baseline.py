import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os

# File paths
train_path = "data/raw/arsarcasm_v1/ArSarcasm_train.csv"
test_path = "data/raw/arsarcasm_v1/ArSarcasm_test.csv"
output_path = "data/processed/arsarcasm_predictions.csv"

# Load data
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Select only text and label columns
text_col = "tweet"
label_col = "sarcasm"  # 1 = Sarcasm, 0 = Non-sarcasm

# Drop nulls and ensure strings
df_train = df_train.dropna(subset=[text_col, label_col])
df_test = df_test.dropna(subset=[text_col, label_col])
df_train[text_col] = df_train[text_col].astype(str)
df_test[text_col] = df_test[text_col].astype(str)

# Build TF-IDF + LR pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=200)),
])

# Fit model
pipeline.fit(df_train[text_col], df_train[label_col])

# Predict on test set
df_test["predicted"] = pipeline.predict(df_test[text_col])
df_test["prob_sarcasm"] = pipeline.predict_proba(df_test[text_col])[:, 1]

# Save predictions
os.makedirs("data/processed", exist_ok=True)
df_test.to_csv(output_path, index=False)
print(f"Sarcasm predictions saved to {output_path}")

# Print metrics
print("\nClassification Report:\n")
print(classification_report(df_test[label_col], df_test["predicted"]))
