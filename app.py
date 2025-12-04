import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# CONFIG STREAMLIT
# =========================================================
st.set_page_config(page_title="Spotify Popularity Predictor", layout="centered")

st.title("🎵 Spotify Popularity Predictor")
st.write("Prediksi popularitas lagu menggunakan model Machine Learning (Pipeline).")


# =========================================================
# LOAD PIPELINE MODEL 
# =========================================================
@st.cache_resource
def load_pipeline():
    with gzip.open("pipeline_reg.pkl", "rb") as f:
        return joblib.load(f)

pipeline = load_pipeline()


# =========================================================
# MODE PILIHAN
# =========================================================
menu = st.radio("Pilih Mode Input:", ["Upload CSV", "Input Manual"])


# =========================================================
# MODE 1 — UPLOAD CSV
# =========================================================
if menu == "Upload CSV":
    st.subheader("📂 Upload CSV untuk Prediksi Batch")

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("📄 Data yang diupload:")
        st.dataframe(df)

        try:
            pred = pipeline.predict(df)
            df["Prediksi Popularitas"] = pred

            st.success("Prediksi berhasil!")
            st.dataframe(df)

            # Download hasil
            csv_output = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Hasil Prediksi",
                csv_output,
                "hasil_prediksi.csv",
                "text/csv"
            )
        except Exception as e:
            st.error(f"❌ Error saat memprediksi: {e}")


# =========================================================
# MODE 2 — INPUT MANUAL
# =========================================================
else:
    st.subheader("🎯 Input Manual Fitur Lagu")

    # --- INPUT NUMERIC ---
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.number_input("Loudness (dB)", -60.0, 0.0, -10.0)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.1)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.15)
    valence = st.slider("Valence", 0.0, 1.0, 0.4)
    tempo = st.number_input("Tempo", 60.0, 220.0, 120.0)

    if st.button("Prediksi Popularitas 🎵"):
        input_df = pd.DataFrame([[
            danceability, energy, loudness, speechiness,
            acousticness, instrumentalness, liveness, valence, tempo
        ]], columns=[
            "danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness",
            "valence", "tempo"
        ])

        try:
            pred = pipeline.predict(input_df)[0]
            st.success(f"🎧 Prediksi Popularitas Lagu: **{pred:.2f} / 100**")
        except Exception as e:
            st.error(f"❌ Error saat memprediksi: {e}")
