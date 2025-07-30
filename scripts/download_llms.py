import os
import shutil
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

# === PATHS ===
base_dir = "C:/Users/akash/projects/TradeSense"
llm_path = os.path.join(base_dir, "llm_models")
os.makedirs(llm_path, exist_ok=True)

# === 1. Move Mistral from HF cache ===
print("Copying Mistral v0.3 from HuggingFace cache...")

mistral_cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3")
mistral_target_path = os.path.join(llm_path, "mistral")
os.makedirs(mistral_target_path, exist_ok=True)

for file in os.listdir(mistral_cache_path):
    src = os.path.join(mistral_cache_path, file)
    dst = os.path.join(mistral_target_path, file)
    if os.path.isfile(src):
        shutil.copy2(src, dst)

print(f"Mistral v0.3 copied to {mistral_target_path}")

# === 2. Download QARiB sentiment model ===
print("Downloading QARiB (CAMeL Arabic Sentiment BERT)...")

qarib_repo = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
qarib_path = os.path.join(llm_path, "qarib")
os.makedirs(qarib_path, exist_ok=True)

qarib_tokenizer = AutoTokenizer.from_pretrained(qarib_repo)
qarib_model = AutoModelForSequenceClassification.from_pretrained(qarib_repo)

qarib_tokenizer.save_pretrained(qarib_path)
qarib_model.save_pretrained(qarib_path)

print(f"QARiB model saved to {qarib_path}")
print("All models are now ready for offline use.")
