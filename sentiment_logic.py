"""
Логика анализа тональности (без зависимостей от Streamlit и модели).
"""


def get_sentiment_display(sentiment_label: str, score: float) -> tuple[str, str]:
    """
    Возвращает (тип_тональности, сообщение) по результату модели.
    """
    if sentiment_label in ["POSITIVE", "Very Positive"]:
        return "positive", f"Тональность: Положительная (Уверенность: {score:.2f})"
    elif sentiment_label in ["NEGATIVE", "Very Negative"]:
        return "negative", f"Тональность: Отрицательная (Уверенность: {score:.2f})"
    else:
        return "neutral", f"Тональность: Нейтральная (Уверенность: {score:.2f})"


def analyze_text(text: str, pipeline_func) -> tuple[str, str]:
    """
    Анализирует текст и возвращает (тип_тональности, сообщение).
    """
    result = pipeline_func(text)
    sentiment = result[0]["label"]
    score = result[0]["score"]
    return get_sentiment_display(sentiment, score)
