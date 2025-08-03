import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import os

# Paths
model_path = "llm_models/qarib"
input_file = "data/raw/labr/reviews.tsv"
output_file = "data/processed/labr_qarib_sentiment.csv"

# Load + Clean
df = pd.read_csv(input_file, sep="\t", header=None)
df.columns = ["rating", "user_id", "book_id", "review_id", "review"]
df = df[["review", "rating"]].dropna()
df["review"] = df["review"].astype(str).str.strip()

# Convert to 3-Class
def map_label(rating):
    if rating in [1, 2]:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    elif rating in [4, 5]:
        return 2  # Positive
    return None

df["label"] = df["rating"].apply(map_label)
df = df[df["label"].isin([0, 1, 2])].dropna()

# Optional balance: sample ~150 per class
df_balanced = (
    df.groupby("label")
    .apply(lambda x: x.sample(n=150, random_state=42) if len(x) >= 150 else x)
    .reset_index(drop=True)
)

# Load QARiB
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Inference 
results = []

for i, row in df_balanced.iterrows():
    text = row["review"]
    true_label = int(row["label"])
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).squeeze()
        pred_label = torch.argmax(probs).item()

    results.append({
        "review": text,
        "true_label": true_label,
        "pred_label": pred_label,
        "confidence": round(probs[pred_label].item(), 3)
    })

# ---------- Step 5: Save ----------
os.makedirs("data/processed", exist_ok=True)
df_out = pd.DataFrame(results)
df_out.to_csv(output_file, index=False)
print(f"Saved 3-class sentiment predictions to: {output_file}")
