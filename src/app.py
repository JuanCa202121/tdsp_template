# app.py
import streamlit as st
import json
import pandas as pd
import requests

st.title("Demo de Inferencia XGBoost")

# 1. Uploader de JSON
uploaded = st.file_uploader("Selecciona tu payload.json", type="json")
if uploaded is not None:
    payload = json.load(uploaded)
    st.subheader("Contenido del payload")
    st.json(payload)

    # 2. URL del endpoint
    url = st.text_input(
        "URL de inferencia",
        "http://localhost:5001/invocations"
    )

    # 3. Botón para predecir
    if st.button("Predecir"):
        with st.spinner("Enviando petición…"):
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
        if resp.status_code == 200:
            result = resp.json()
            st.success(f"🏷️ Predicción: {result['predictions']}")
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")

