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

# Step 1: Load raw reviews.tsv (no header)
df = pd.read_csv(raw_reviews_path, sep="\t", header=None)
df.columns = ["rating", "user_id", "book_id", "review_id", "review"]
df = df[["review", "rating"]].dropna()
df["review"] = df["review"].astype(str)

# Step 2: Map to 3 sentiment classes
# 1–2 = Negative (0), 3 = Neutral (1), 4–5 = Positive (2)
df = df[df["rating"].isin([1, 2, 3, 4, 5])]
df["label"] = df["rating"].apply(lambda x: 0 if x in [1, 2] else (1 if x == 3 else 2))

# Step 3: Balance sample (300 per class)
df_neg = df[df["label"] == 0].sample(n=300, random_state=42)
df_neu = df[df["label"] == 1].sample(n=300, random_state=42)
df_pos = df[df["label"] == 2].sample(n=300, random_state=42)
df_sample = pd.concat([df_neg, df_neu, df_pos]).sample(frac=1, random_state=42).reset_index(drop=True)

# Step 4: Load model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Step 5: Predict
results = []
true_labels = []
pred_labels = []

for i, row in df_sample.iterrows():
    text = row["review"]
    true = int(row["label"])

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()

    results.append({
        "review": text,
        "true_label": true,
        "pred_label": pred,
        "confidence": round(probs[pred].item(), 3)
    })
    true_labels.append(true)
    pred_labels.append(pred)

# Step 6: Save results
os.makedirs("data/processed", exist_ok=True)
df_out = pd.DataFrame(results)
df_out.to_csv(output_file, index=False)
print(f"QARiB 3-class predictions saved to {output_file}")

# Step 7: Show metrics
print("\nClassification Report (3-class):\n")
print(classification_report(true_labels, pred_labels, digits=3, target_names=["Negative", "Neutral", "Positive"]))
