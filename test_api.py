from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    mock_pipeline = MagicMock(
        return_value=[{"label": "POSITIVE", "score": 0.95}]
    )
    with patch("api.get_sentiment_pipeline", return_value=mock_pipeline):
        from api import app

        yield TestClient(app), mock_pipeline


def test_health(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_positive(client):
    test_client, mock_pipeline = client
    response = test_client.post("/analyze", json={"text": "Отличный продукт!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_type"] == "positive"
    assert "Положительная" in data["message"]
    mock_pipeline.assert_called_once_with("Отличный продукт!")


def test_analyze_negative(client):
    test_client, _ = client
    with patch("api.get_sentiment_pipeline") as mock_get:
        mock_get.return_value = MagicMock(
            return_value=[{"label": "NEGATIVE", "score": 0.88}]
        )
        from api import app

        test_client = TestClient(app)
        response = test_client.post("/analyze", json={"text": "Ужасно"})
    assert response.status_code == 200
    assert response.json()["sentiment_type"] == "negative"


def test_analyze_empty_text(client):
    test_client, _ = client
    response = test_client.post("/analyze", json={"text": ""})
    assert response.status_code == 422


def test_analyze_pipeline_error(client):
    test_client, mock_pipeline = client
    mock_pipeline.side_effect = RuntimeError("Ошибка модели")
    response = test_client.post("/analyze", json={"text": "Текст"})
    assert response.status_code == 500
    assert "Ошибка модели" in response.json()["detail"]
