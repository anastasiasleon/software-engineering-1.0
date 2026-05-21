---
title: Sentiment Analyzer
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# Многоязычный анализатор тональности

Streamlit-интерфейс. Backend (FastAPI) — отдельный Space или локально.

**Модель:** [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

## Секрет (Settings → Variables and secrets)

```
API_URL = https://ВАШ-ЛОГИН-sentiment-api.hf.space
```

Без `API_URL` интерфейс покажет подсказку по настройке.

## Локальный запуск

```bash
pip install -r requirements.txt
# backend: uvicorn api:app --port 8000
export API_URL=http://localhost:8000
streamlit run app.py
```
