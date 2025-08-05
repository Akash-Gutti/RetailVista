# scripts/20_lime_explainer_visualizer.py

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer

# ------------------- Config -------------------
input_path = "data/processed/labr_qarib_evaluation.csv"
output_dir = "outputs/lime_outputs"
output_json = os.path.join(output_dir, "lime_sentiment_explanations.json")
translation_path = "utils/token_translation_dict.json"

os.makedirs(output_dir, exist_ok=True)

# ------------------- Load Data -------------------
df = pd.read_csv(input_path)
df = df[df["pred_label"].isin([0, 1, 2])].dropna(subset=["review"])
df = df.rename(columns={"true_label": "true", "pred_label": "pred"})

class_names = ["Negative", "Neutral", "Positive"]

# ------------------- Load Translation Dictionary -------------------
with open(translation_path, "r", encoding="utf-8") as f:
    token_translation = json.load(f)

def translate_token(token):
    token_clean = token.strip().strip("“”‘’\"'")
    return f"{token_clean} ({token_translation.get(token_clean, 'N/A')})"

# ------------------- Train TF-IDF Classifier -------------------
pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),
    LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
)
pipeline.fit(df["review"], df["pred"])  # 🔄 TRAINED ON PREDICTED LABELS NOW

# ------------------- Sample 10 Balanced Reviews -------------------
samples = []
class_sample_counts = {0: 4, 1: 3, 2: 3}  # per class

for label, n_samples in class_sample_counts.items():
    class_df = df[df["pred"] == label]
    if len(class_df) >= n_samples:
        samples.append(class_df.sample(n_samples, random_state=label))
    elif len(class_df) > 0:
        print(f"⚠️ Only {len(class_df)} samples found for class {label}. Using all.")
        samples.append(class_df)
    else:
        print(f"❌ No samples found for class {label}. Skipping.")

samples = pd.concat(samples).reset_index(drop=True)

# ------------------- Initialize LIME -------------------
explainer = LimeTextExplainer(class_names=class_names)
results = []

# ------------------- Generate Explanations -------------------
for i, row in samples.iterrows():
    review = row["review"]
    true = int(row["true"])
    pred = int(row["pred"])

    try:
        explanation = explainer.explain_instance(
            review,
            pipeline.predict_proba,
            num_features=10,
            labels=(pred,)  # ✅ Directly specify the correct label
        )

        tokens = explanation.as_list(label=pred)
        translated = [(translate_token(tok), weight) for tok, weight in tokens]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        words, weights = zip(*translated)
        ax.barh(words, weights, color=['green' if w > 0 else 'red' for w in weights])
        ax.set_title(f"Sentiment: {class_names[pred]} (Conf: ~{max(pipeline.predict_proba([review])[0]):.2f})")
        ax.invert_yaxis()
        plt.tight_layout()
        img_path = os.path.join(output_dir, f"lime_expl_{i}.png")
        plt.savefig(img_path)
        plt.close()

        results.append({
            "id": i,
            "true_label": class_names[true],
            "predicted_label": class_names[pred],
            "review": review,
            "tokens": [t[0] for t in translated],
            "weights": [round(t[1], 4) for t in translated],
            "image_path": img_path
        })

    except Exception as e:
        print(f"❌ Failed on sample {i}: {e}")
        continue

# ------------------- Save Explanations -------------------
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ LIME explanations saved to folder: {output_dir}")
print(f"📄 Summary saved to: {output_json}")
