import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Prediksi Popularitas Lagu", layout="centered")

st.title("🎵 Prediksi Popularitas Lagu")
st.write("Upload file atau input fitur secara manual untuk memprediksi popularitas lagu.")

menu = st.radio("Pilih Mode Input:", ["Upload CSV", "Input Manual"])

# =========================================================
# MODE 1: UPLOAD CSV
# =========================================================
if menu == "Upload CSV":
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Data yang diupload:")
        st.dataframe(df)

        try:
            pred = model.predict(df)
            df["Prediksi Popularitas"] = pred
            st.success("Prediksi berhasil!")
            st.dataframe(df)

            # Download
            csv_output = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Hasil Prediksi",
                csv_output,
                "hasil_prediksi.csv",
                "text/csv"
            )
        except Exception as e:
            st.error(f"Terjadi error saat memprediksi: {e}")

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
            danceability, energy, loudness, speechiness, acousticness,
            instrumentalness, liveness, valence, tempo
        ]], columns=[
            "danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness",
            "valence", "tempo"
        ])

        pred = model.predict(input_data)[0]
        st.success(f"Prediksi Popularitas Lagu: **{pred}**")
