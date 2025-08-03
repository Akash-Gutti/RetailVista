# scripts/19_prepare_lime_dataset.py

import pandas as pd
import os
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import lime
import lime.lime_text
import matplotlib.pyplot as plt

# Load sentiment predictions
input_path = "data/processed/labr_qarib_evaluation.csv"
df = pd.read_csv(input_path)

# Use review and predicted sentiment label
texts = df["review"].astype(str).tolist()
labels = df["pred_label"].tolist()

# Build text classification pipeline
pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),
    LogisticRegression(max_iter=1000)
)
pipeline.fit(texts, labels)

# Define dynamic class names
label_set = sorted(df["pred_label"].unique())
class_names = ["Negative", "Neutral", "Positive"]
class_map = {i: class_names[i] for i in label_set}

# Initialize LIME explainer
explainer = lime.lime_text.LimeTextExplainer(class_names=[class_map[i] for i in label_set])

# Output folder
output_dir = "outputs/lime_sentiment"
os.makedirs(output_dir, exist_ok=True)

# Explain a few examples from each class
samples_per_class = 3
for class_label in label_set:
    class_df = df[df["pred_label"] == class_label]
    if len(class_df) == 0:
        print(f"No samples found for class {class_map[class_label]}. Skipping.")
        continue

    sampled = class_df.sample(n=min(samples_per_class, len(class_df)), random_state=42)
    for idx, row in sampled.iterrows():
        review_text = row["review"]
        try:
            exp = explainer.explain_instance(review_text, pipeline.predict_proba, num_features=10)
            fig = exp.as_pyplot_figure()
            fig.suptitle(f"LIME: Pred = {class_map[class_label]}", fontsize=12)
            fig.tight_layout()
            fig.savefig(f"{output_dir}/lime_class{class_label}_sample{idx}.png")
            plt.close()
        except Exception as e:
            print(f"Failed on sample {idx} (class {class_label}): {e}")

print("LIME sentiment explanations saved in outputs/lime_sentiment/")
