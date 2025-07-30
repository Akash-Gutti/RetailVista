import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import os

# Paths
model_path = "llm_models/qarib"
input_file = "data/processed/labr_sample.csv"
output_file = "data/processed/labr_qarib_sentiment.csv"

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load sample LABR
df = pd.read_csv(input_file)
df = df.dropna(subset=["review"])
df["review"] = df["review"].astype(str)
df = df[df["review"].str.strip().astype(bool)].reset_index(drop=True)

# Classify reviews
results = []
for i, row in df.iterrows():
    text = row["review"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = softmax(outputs.logits, dim=1).squeeze()
        pred_label = torch.argmax(scores).item()
    results.append({
        "review": row["review"],
        "true_label": row["label"],
        "pred_label": pred_label,
        "confidence": round(scores[pred_label].item(), 3)
    })

# Save output
df_out = pd.DataFrame(results)
df_out.to_csv(output_file, index=False)
print(f"Sentiment predictions saved to {output_file}")
