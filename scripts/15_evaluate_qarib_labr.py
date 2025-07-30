import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from torch.nn.functional import softmax
import os

# File paths
raw_reviews_path = "data/raw/labr/reviews.tsv"
output_file = "data/processed/labr_qarib_evaluation.csv"
model_path = "llm_models/qarib"

# Step 1: Load raw reviews.tsv (no header, custom columns)
df = pd.read_csv(raw_reviews_path, sep="\t", header=None)
df.columns = ["rating", "user_id", "book_id", "review_id", "review"]
df = df[["review", "rating"]].dropna()

# Filter to binary sentiment: 1,2 → 0; 4,5 → 1
df = df[df["rating"].isin([1, 2, 4, 5])]
df["label"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

# Sample 500 pos, 500 neg
df_pos = df[df["label"] == 1].sample(500, random_state=42)
df_neg = df[df["label"] == 0].sample(500, random_state=42)
df_sample = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

# Load QARiB model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Inference
results = []
true_labels = []
pred_labels = []

for i, row in df_sample.iterrows():
    text = str(row["review"])
    label = int(row["label"])

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()

    results.append({
        "review": text,
        "true_label": label,
        "pred_label": pred,
        "confidence": round(probs[pred].item(), 3)
    })
    true_labels.append(label)
    pred_labels.append(pred)

# Save output
os.makedirs("data/processed", exist_ok=True)
df_out = pd.DataFrame(results)
df_out.to_csv(output_file, index=False)
print(f"QARiB sentiment predictions saved to {output_file}")

# Print metrics
print("\nClassification Report (QARiB on LABR sample):\n")
print(classification_report(true_labels, pred_labels, digits=3))
