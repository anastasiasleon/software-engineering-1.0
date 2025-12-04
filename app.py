import streamlit as st
from transformers import pipeline

# Загрузка модели для анализа тональности
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

sentiment_pipeline = load_sentiment_pipeline()

st.title("Многоязычный Анализатор Тональности")
st.write("Введите текст, чтобы определить его тональность (положительная или отрицательная).")

user_input = st.text_area("Ваш текст для анализа:", "Я очень люблю это приложение!")

if st.button("Анализировать тональность"):
    if user_input:
        try:
            result = sentiment_pipeline(user_input)
            sentiment = result[0]['label']
            score = result[0]['score']
            st.subheader("Результат анализа:")
            if sentiment == "POSITIVE":
                st.success(f"Тональность: Положительная (Уверенность: {score:.2f})")
            else:
                st.error(f"Тональность: Отрицательная (Уверенность: {score:.2f})")
        except Exception as e:
            st.error(f"Произошла непредвиденная ошибка: {e}")
    else:
        st.warning("Пожалуйста, введите текст для анализа.")
