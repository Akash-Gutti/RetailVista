import pandas as pd
import json
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# === CONFIG ===
HF_TOKEN = "your_hf_token_here"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
CLUSTER_CSV = "/workspace/data/cluster_insights_summary.csv"
UPLIFT_CSV = "/workspace/data/uplift_predictions.csv"
OUTPUT_JSON = "/workspace/campaign_summaries_test.json"

# === STEP 1: Load Data ===
print("🔄 Loading CSVs...")
df_cluster = pd.read_csv(CLUSTER_CSV)
df_uplift = pd.read_csv(UPLIFT_CSV)
merged_df = pd.merge(df_cluster, df_uplift, on="cluster", how="inner")
print(f"✅ Merged {len(merged_df)} rows.")

# === STEP 2: Authenticate HuggingFace ===
print("🔐 Logging into Hugging Face...")
login(token=HF_TOKEN)

# === STEP 3: Load Model ===
print(f"🚀 Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === STEP 4: Prompt Template ===
def build_prompt(row):
    return f"""You are a marketing strategist at a leading hypermarket chain in the UAE/MENA region.

Craft a short, emotionally resonant campaign brief (3–5 sentences) for the following customer segment. The goal is to increase loyalty, visit frequency, and monthly spend — without repeating the raw data directly.

Customer Profile:
- Segment: {row['segment']}
- Avg Age: {row['age']:.1f}
- Recency: {row['recency_mean']:.1f} days
- Frequency: {row['frequency_mean']:.1f} per month
- Spend: ${row['monetary_mean']:.2f} monthly
- Size: {int(row['num_customers'])} customers

Campaign Objective:
- Treatment Strategy: {row['treatment']}
- Observed Uplift Score: {row['uplift_score']:.3f}

🟢 Now write the final campaign message directly, with:
- No stats or segments repeated
- Emotionally engaging language
- Actionable campaign message — Suggested promotional offers, product categories, and expected customer reaction
- Tone: Compelling, Warm, community-driven, benefit-first

Begin:
""".strip()

# === STEP 5: Output Cleaner ===
def clean_output(text):
    text = text.replace("\\n", " ").replace("\n", " ").replace("\\", "")
    text = text.replace("\"", "'")
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"(Sincerely,.*|Warm Regards.*|See you.*|#.*|---.*)", "", text)

    # Remove placeholders or meta
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"(Note:|P\.?S\.?:).*?$", "", text, flags=re.IGNORECASE)

    # Vary repetitive openers
    openers = {
        "Dear valued": "Welcome back",
        "Dear Valued": "We've missed you",
        "Dear Gold Members": "To our treasured Gold community",
        "Dear Silver Members": "To our loyal Silver shoppers",
        "Dear Bronze Members": "To our vibrant Bronze family",
    }
    for k, v in openers.items():
        text = text.replace(k, v)
    
    return text.strip()

# === STEP 6: Generate Summaries ===
print("🧠 Generating test summaries...")

results = []
for i, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    try:
        prompt = build_prompt(row)
        output = pipe(prompt, max_new_tokens=220, temperature=0.7)[0]["generated_text"]

        # Remove prompt prefix from output
        campaign_brief = output[len(prompt):].strip() if output.startswith(prompt) else output.strip()

        campaign_brief = clean_output(campaign_brief)

        if not campaign_brief:
            raise ValueError("Empty or leaked response.")

        results.append({
            "cluster": int(row["cluster"]),
            "segment": row["segment"],
            "uplift_score": round(row["uplift_score"], 3),
            "campaign_brief": campaign_brief
        })

    except Exception as e:
        print(f"❌ Error on cluster {row['cluster']}: {e}")
        results.append({
            "cluster": int(row["cluster"]),
            "segment": row["segment"],
            "uplift_score": round(row["uplift_score"], 3),
            "campaign_brief": f"[Error: {e}]"
        })

# === STEP 7: Save Output ===
print(f"💾 Saving to: {OUTPUT_JSON}")
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Done! Generated {len(results)} test summaries.")
