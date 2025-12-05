import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ================================
# 1. Load Model PKL (3 model)
# ================================
@st.cache_resource
def load_models():
    with open("model.pkl", "rb") as f:
        models = pickle.load(f)
    return models

models = load_models()

regressor   = models["regressor"]
classifier  = models["classifier"]
clustering  = models["clustering"]

# ================================
# 2. Judul Aplikasi
# ================================
st.title("🎵 Prediksi Popularitas Lagu Spotify")
st.write("Aplikasi ini memprediksi popularitas lagu menggunakan 3 model ML: Regressor, Classifier, dan Clustering.")

st.write("---")

# ================================
# 3. Input Fitur
# ================================
st.header("🔧 Input Fitur Lagu")

danceability   = st.number_input("Danceability", 0.0, 1.0, 0.5)
energy         = st.number_input("Energy", 0.0, 1.0, 0.5)
loudness       = st.number_input("Loudness", -60.0, 5.0, -8.0)
speechiness    = st.number_input("Speechiness", 0.0, 1.0, 0.05)
acousticness   = st.number_input("Acousticness", 0.0, 1.0, 0.1)
instrumentalness = st.number_input("Instrumentalness", 0.0, 1.0, 0.0)
liveness       = st.number_input("Liveness", 0.0, 1.0, 0.2)
valence        = st.number_input("Valence", 0.0, 1.0, 0.5)
tempo          = st.number_input("Tempo", 40.0, 250.0, 120.0)
duration_ms    = st.number_input("Duration (ms)", 10000, 500000, 200000)

# Feature engineering (harus sama dengan training)
TrackLength_min = duration_ms / 60000
EnergyLevel = "High" if energy > 0.6 else "Low"

# Convert kategorikal jadi numeric dummy
df_input = pd.DataFrame({
    "danceability": [danceability],
    "energy": [energy],
    "loudness": [loudness],
    "speechiness": [speechiness],
    "acousticness": [acousticness],
    "instrumentalness": [instrumentalness],
    "liveness": [liveness],
    "valence": [valence],
    "tempo": [tempo],
    "TrackLength_min": [TrackLength_min],
    "EnergyLevel_High": [1 if EnergyLevel == "High" else 0],
})

st.write("---")

# ================================
# 4. Tombol Prediksi
# ================================
if st.button("🎯 Prediksi Sekarang"):
    
    # Pastikan input sesuai kolom model
    input_reg = df_input.reindex(columns=regressor.feature_names_in_, fill_value=0)
    input_clf = df_input.reindex(columns=classifier.feature_names_in_, fill_value=0)
    input_cluster = df_input.reindex(columns=clustering.feature_names_in_, fill_value=0)

    # Prediksi
    reg_pred = regressor.predict(input_reg)[0]
    clf_pred = classifier.predict(input_clf)[0]
    cluster_pred = clustering.predict(input_cluster)[0]

    # ================================
    # 5. Output
    # ================================
    st.subheader("📌 Hasil Prediksi")
    
    st.success(f"⭐ **Prediksi Popularitas (Regresi): `{reg_pred:.2f}`**")
    st.info(f"📊 **Kategori Popularitas (Classifier): `{clf_pred}`**")
    st.warning(f"🎧 **Cluster Lagu (KMeans): `{cluster_pred}`**")

    st.write("Prediksi berhasil menggunakan 3 model Machine Learning dari model.pkl")

