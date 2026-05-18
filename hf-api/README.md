---
title: Sentiment API
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Backend: FastAPI

REST API для анализа тональности текста.

- `GET /health` — проверка работы
- `POST /analyze` — тело: `{"text": "ваш текст"}`

Модель: [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
