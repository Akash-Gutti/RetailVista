import pandas as pd
import os
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==== Paths ====
input_path = "data/processed/labr_qarib_evaluation.csv"  # Your 3-class dataset
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# ==== Load Dataset ====
df = pd.read_csv(input_path)

# Expected columns: 'review' (text), 'pred_label' (0=Neg, 1=Neutral, 2=Positive)
if not {"review", "pred_label"}.issubset(df.columns):
    raise ValueError(f"Dataset must contain 'review' and 'pred_label' columns, found: {df.columns}")

texts = df["review"].astype(str).tolist()
labels = df["pred_label"].tolist()

# ==== Train Pipeline ====
pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
    LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="multinomial")
)

pipeline.fit(texts, labels)

# ==== Save Model ====
model_path = os.path.join(model_dir, "sentiment_model.pkl")
joblib.dump(pipeline, model_path)

print(f"3-class sentiment model saved to {model_path}")
