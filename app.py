import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# LOAD MODEL DARI PILIHAN USER
# =========================================================
@st.cache_resource
def load_model(model_name):
    return joblib.load(model_name)

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Prediksi Popularitas Lagu", layout="centered")

st.title("🎵 Prediksi Popularitas Lagu")
st.write("Gunakan salah satu model Machine Learning untuk prediksi popularitas lagu.")

# =========================================================
# PILIH MODEL
# =========================================================
available_models = []

# Cek file di folder
for file in os.listdir():
    if file.startswith("pipeline") or file.endswith(".pkl"):
        available_models.append(file)

model_choice = st.selectbox("Pilih Model yang Akan Digunakan:", available_models)

model = load_model(model_choice)

st.success(f"Model aktif: **{model_choice}**")

# =========================================================
# MODE INPUT
# =========================================================
menu = st.radio("Pilih Mode Input:", ["Upload CSV", "Input Manual"])

# =========================================================
# MODE 1: UPLOAD CSV
# =========================================================
if menu == "Upload CSV":
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Data input:")
        st.dataframe(df)

        try:
            pred = model.predict(df)
            df["Prediksi"] = pred
            st.success("Prediksi selesai!")
            st.dataframe(df)

            csv_output = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Hasil Prediksi",
                csv_output,
                "hasil_prediksi.csv",
                "text/csv"
            )
        except Exception as e:
            st.error(f"Error saat prediksi: {e}")

# =========================================================
# MODE 2: INPUT MANUAL
# =========================================================
else:
    st.subheader("Input Fitur Lagu")

    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.number_input("Loudness", -60.0, 0.0, -10.0)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.1)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.15)
    valence = st.slider("Valence", 0.0, 1.0, 0.4)
    tempo = st.number_input("Tempo", 60.0, 220.0, 120.0)

    if st.button("Prediksi"):
        input_data = pd.DataFrame([[
            danceability, energy, loudness, speechiness,
            acousticness, instrumentalness, liveness, valence, tempo
        ]], columns=[
            "danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness",
            "valence", "tempo"
        ])

        try:
            pred = model.predict(input_data)[0]
            st.success(f"Hasil Prediksi ({model_choice}): **{pred}**")
        except Exception as e:
            st.error(f"Terjadi error: {e}")
