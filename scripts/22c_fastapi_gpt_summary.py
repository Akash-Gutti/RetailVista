from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import json
import os

app = FastAPI(
    title="TradeSense - GPT Campaign Summary API",
    description="Returns LLM-generated marketing briefs for a customer cluster or segment",
    version="1.0"
)

# Load campaign summaries
with open("data/processed/campaign_summaries.json", "r", encoding="utf-8") as f:
    summaries = json.load(f)

@app.get("/campaign_summary/")
def get_summary(
    cluster_id: Optional[int] = Query(None),
    segment: Optional[str] = Query(None)
):
    if not cluster_id and not segment:
        raise HTTPException(status_code=400, detail="Provide at least cluster_id or segment")

    matches = []
    for entry in summaries:
        if cluster_id is not None and entry.get("cluster") == cluster_id:
            matches.append(entry)
        elif segment is not None and entry.get("segment", "").lower() == segment.lower():
            matches.append(entry)

    if not matches:
        raise HTTPException(status_code=404, detail="No summary found for given cluster_id or segment.")

    return {
        "matched_summaries": [
            {
                "cluster": m["cluster"],
                "segment": m["segment"],
                "uplift_score": m["uplift_score"],
                "campaign_brief": m["campaign_brief"]
            } for m in matches
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("scripts.22c_fastapi_gpt_summary:app", host="127.0.0.1", port=8000, reload=True)
