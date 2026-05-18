from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_loader import get_sentiment_pipeline
from sentiment_logic import analyze_text

app = FastAPI(title="Sentiment API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class AnalyzeResponse(BaseModel):
    sentiment_type: str
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    try:
        pipeline_fn = get_sentiment_pipeline()
        sentiment_type, message = analyze_text(request.text, pipeline_fn)
        return AnalyzeResponse(sentiment_type=sentiment_type, message=message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
