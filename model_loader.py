import os
from functools import lru_cache

from transformers import pipeline

DEFAULT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"


@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    """Загружает предобученную модель с Hugging Face или из локальной папки."""
    local_path = os.getenv("SENTIMENT_MODEL_PATH", "./local_model")
    if os.path.isdir(local_path) and os.listdir(local_path):
        return pipeline("sentiment-analysis", model=local_path)

    model_id = os.getenv("SENTIMENT_MODEL", DEFAULT_MODEL)
    return pipeline("sentiment-analysis", model=model_id)
