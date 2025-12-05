import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# LOAD MODEL PIPELINE
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # pipeline full: preprocess + model

model = load_model()

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Prediksi Popularitas Lagu", layout="centered")

st.title("🎵 Prediksi Popularitas Lagu")
st.write("Gunakan CSV atau input manual untuk memprediksi popularitas lagu.")

mode = st.radio("Pilih Mode Input:", ["Upload CSV", "Input Manual"])

# =========================================================
# MODE 1: UPLOAD CSV
# =========================================================
if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    
    if uploaded is not None:
        df = pd.read_csv(uploaded)

        st.write("📄 Data yang diupload:")
        st.dataframe(df)

        try:
            pred = model.predict(df)

            df["Prediksi Popularitas"] = pred
            st.success("Prediksi berhasil!")
            st.dataframe(df)

            st.download_button(
                "Download Hasil Prediksi",
                df.to_csv(index=False).encode("utf-8"),
                "hasil_prediksi.csv",
                "text/csv"
            )

        except Exception as e:
            st.error("❌ CSV tidak cocok dengan struktur dataset saat training.")
            st.write(e)

# =========================================================
# MODE 2: INPUT MANUAL
# =========================================================
else:
    st.subheader("Input Fitur Manual")

    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.number_input("Loudness", -60.0, 0.0, -10.0)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.1)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.15)
    valence = st.slider("Valence", 0.0, 1.0, 0.4)
    tempo = st.number_input("Tempo", 60.0, 220.0, 120.0)

    # NOTE:
    # Pipeline akan otomatis encode + scale + preprocessing
    # Jadi cukup kirim kolom yang sama dengan dataset original.

    if st.button("Prediksi"):
        # Buat dataframe 1 baris sesuai dataset original
        data = pd.DataFrame([{
            "danceability": danceability,
            "energy": energy,
            "loudness": loudness,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo
        }])

        pred = model.predict(data)[0]

        st.success(f"Prediksi Popularitas Lagu: **{pred:.2f}**")
