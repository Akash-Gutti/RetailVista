from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn

# Load Model
MODEL_PATH = "llm_models/qarib"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

class InputText(BaseModel):
    review: str

class OutputSentiment(BaseModel):
    label: str
    confidence: float

app = FastAPI(
    title="TradeSense - Arabic Sentiment API",
    description="API for 3-class sentiment analysis using QARiB model",
    version="1.0"
)

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.post("/sentiment", response_model=OutputSentiment)
def analyze_sentiment(input: InputText):
    if not input.review.strip():
        raise HTTPException(status_code=400, detail="Empty review provided.")

    inputs = tokenizer(input.review, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {"label": label_map[pred], "confidence": round(confidence, 4)}

if __name__ == "__main__":
    uvicorn.run("scripts.22b_fastapi_sentiment:app", host="127.0.0.1", port=8000, reload=True)
