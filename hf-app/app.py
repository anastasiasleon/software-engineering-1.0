import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("API_TIMEOUT", "120"))

st.title("Многоязычный Анализатор Тональности")
st.write("Введите текст, чтобы определить его тональность (положительная или отрицательная).")

if not API_URL:
    st.error(
        "Не задан URL backend. В настройках Space (Settings → Variables and secrets) "
        "добавьте переменную **API_URL** — адрес Space с FastAPI, например: "
        "`https://username-sentiment-api.hf.space`"
    )
    st.stop()

st.caption(f"Backend API: `{API_URL}`")


def analyze_via_api(text: str) -> dict:
    response = requests.post(
        f"{API_URL}/analyze",
        json={"text": text},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


user_input = st.text_area("Ваш текст для анализа:", "Я очень люблю это приложение!")

if st.button("Анализировать тональность"):
    if not user_input.strip():
        st.warning("Пожалуйста, введите текст для анализа.")
    else:
        try:
            with st.spinner("Отправка запроса на сервер..."):
                result = analyze_via_api(user_input)
            sentiment_type = result["sentiment_type"]
            message = result["message"]
            st.subheader("Результат анализа:")
            if sentiment_type == "positive":
                st.success(message)
            elif sentiment_type == "negative":
                st.error(message)
            else:
                st.info(message)
        except requests.exceptions.ConnectionError:
            st.error(
                f"Не удалось подключиться к API ({API_URL}). "
                "Убедитесь, что Space с backend запущен."
            )
        except requests.exceptions.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = exc.response.text if exc.response is not None else ""
            st.error(f"Ошибка API ({exc.response.status_code}): {detail}")
        except requests.exceptions.Timeout:
            st.error("Превышено время ожидания ответа от API.")
        except Exception as exc:
            st.error(f"Произошла непредвиденная ошибка: {exc}")
