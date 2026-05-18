"""
Тесты для многоязычного анализатора тональности.
"""
import pytest
from unittest.mock import MagicMock

from sentiment_logic import get_sentiment_display, analyze_text


def test_get_sentiment_display_positive():
    """Тест отображения положительной тональности."""
    sentiment_type, message = get_sentiment_display("POSITIVE", 0.95)
    assert sentiment_type == "positive"
    assert "Положительная" in message
    assert "0.95" in message


def test_get_sentiment_display_very_positive():
    """Тест отображения очень положительной тональности."""
    sentiment_type, message = get_sentiment_display("Very Positive", 0.99)
    assert sentiment_type == "positive"
    assert "Положительная" in message


def test_get_sentiment_display_negative():
    """Тест отображения отрицательной тональности."""
    sentiment_type, message = get_sentiment_display("NEGATIVE", 0.87)
    assert sentiment_type == "negative"
    assert "Отрицательная" in message
    assert "0.87" in message


def test_get_sentiment_display_very_negative():
    """Тест отображения очень отрицательной тональности."""
    sentiment_type, message = get_sentiment_display("Very Negative", 0.92)
    assert sentiment_type == "negative"
    assert "Отрицательная" in message


def test_get_sentiment_display_neutral():
    """Тест отображения нейтральной тональности."""
    sentiment_type, message = get_sentiment_display("NEUTRAL", 0.5)
    assert sentiment_type == "neutral"
    assert "Нейтральная" in message


def test_get_sentiment_display_unknown_label():
    """Тест для неизвестной метки — должна интерпретироваться как нейтральная."""
    sentiment_type, message = get_sentiment_display("SOME_UNKNOWN_LABEL", 0.33)
    assert sentiment_type == "neutral"
    assert "Нейтральная" in message


def test_get_sentiment_display_score_formatting():
    """Тест форматирования уверенности (2 знака после запятой)."""
    _, message = get_sentiment_display("POSITIVE", 0.12345)
    assert "0.12" in message


def test_analyze_text_with_mock_pipeline():
    """Тест analyze_text с мок-пайплайном."""
    mock_pipeline = MagicMock(return_value=[
        {"label": "POSITIVE", "score": 0.98}
    ])

    sentiment_type, message = analyze_text("Отличный продукт!", mock_pipeline)

    mock_pipeline.assert_called_once_with("Отличный продукт!")
    assert sentiment_type == "positive"
    assert "Положительная" in message


def test_analyze_text_negative_result():
    """Тест analyze_text для отрицательного результата."""
    mock_pipeline = MagicMock(return_value=[
        {"label": "NEGATIVE", "score": 0.85}
    ])

    sentiment_type, message = analyze_text("Ужасное качество", mock_pipeline)

    assert sentiment_type == "negative"
    assert "Отрицательная" in message


def test_analyze_text_neutral_result():
    """Тест analyze_text для нейтрального результата."""
    mock_pipeline = MagicMock(return_value=[
        {"label": "NEUTRAL", "score": 0.55}
    ])

    sentiment_type, message = analyze_text("Обычный день", mock_pipeline)

    assert sentiment_type == "neutral"
    assert "Нейтральная" in message


def test_analyze_text_pipeline_raises():
    """Тест что исключения пайплайна пробрасываются наружу."""
    mock_pipeline = MagicMock(side_effect=RuntimeError("Ошибка модели"))

    with pytest.raises(RuntimeError, match="Ошибка модели"):
        analyze_text("Текст", mock_pipeline)


def test_analyze_text_empty_result():
    """Тест обработки пустого результата от пайплайна."""
    mock_pipeline = MagicMock(return_value=[])

    with pytest.raises(IndexError):
        analyze_text("Текст", mock_pipeline)
