import os
import json
import re
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load summaries
with open("data/processed/campaign_summaries.json", "r", encoding="utf-8") as f:
    summaries = json.load(f)

# Group by segment
segment_data = defaultdict(list)
for item in summaries:
    segment = item.get("segment", "Unknown")
    if "campaign_brief" in item:
        segment_data[segment].append(item["campaign_brief"])

# Tokenization
def clean_tokens(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return [w for w in text.split() if len(w) > 3]

# Extract top keywords
segment_keywords = {}
for seg, texts in segment_data.items():
    all_words = []
    for t in texts:
        all_words.extend(clean_tokens(t))
    segment_keywords[seg] = Counter(all_words).most_common(25)

# Save as JSON
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/gpt_output_comparison.json", "w", encoding="utf-8") as f:
    json.dump(segment_keywords, f, indent=2, ensure_ascii=False)

# Word clouds
os.makedirs("assets/wordclouds", exist_ok=True)
for segment, keywords in segment_keywords.items():
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(keywords))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Segment: {segment}")
    plt.tight_layout()
    plt.savefig(f"assets/wordclouds/{segment}_wordcloud.png")
    plt.close()

print("Sub-step 5.4 complete. Files generated:")
print(" - data/processed/gpt_output_comparison.json")
print(" - assets/wordclouds/*.png")
