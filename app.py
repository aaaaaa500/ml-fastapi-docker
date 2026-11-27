import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Iris Flower Classifier")
st.write("Enter the flower sizes in cm and press Predict")

sepal_length = st.number_input("Sepal length", min_value=0.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal width", min_value=0.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal length", min_value=0.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal width", min_value=0.0, value=0.2, step=0.1)

if st.button("Predict"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        r.raise_for_status()
        st.success(f"Predicted class: {r.json()['class_name']}")
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")