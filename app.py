import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip

# =========================================================
# LOAD MODEL (Membaca model terkompres .pkl.gz)
# =========================================================
@st.cache_resource
def load_model():
    with gzip.open("model.pkl.gz", "rb") as f:
        return joblib.load(f)

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

            # Download hasil prediksi
            csv_output = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Hasil Prediksi",
