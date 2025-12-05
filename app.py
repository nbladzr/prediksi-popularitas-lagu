import streamlit as st
import joblib
import pandas as pd

# ================================
# 1. Load Model PKL (Hanya Classifier)
# ================================
@st.cache_resource
def load_model():
    models = joblib.load("model.pkl")
    return models["classifier"], models["columns"]

classifier, feature_columns = load_model()

# ================================
# 2. Judul Aplikasi
# ================================
st.title("🎵 Prediksi Popularitas Lagu Spotify")
st.write("Aplikasi ini memprediksi kategori popularitas lagu menggunakan RandomForestClassifier.")

st.write("---")

# ================================
# 3. Input Fitur
# ================================
st.header("🔧 Input Fitur Lagu")

danceability     = st.number_input("Danceability", 0.0, 1.0, 0.5)
energy           = st.number_input("Energy", 0.0, 1.0, 0.5)
loudness         = st.number_input("Loudness", -60.0, 5.0, -8.0)
speechiness      = st.number_input("Speechiness", 0.0, 1.0, 0.05)
acousticness     = st.number_input("Acousticness", 0.0, 1.0, 0.1)
instrumentalness = st.number_input("Instrumentalness", 0.0, 1.0, 0.0)
liveness         = st.number_input("Liveness", 0.0, 1.0, 0.2)
valence          = st.number_input("Valence", 0.0, 1.0, 0.5)
tempo            = st.number_input("Tempo", 40.0, 250.0, 120.0)
duration_ms      = st.number_input("Duration (ms)", 10000, 500000, 200000)

# Feature engineering (sama seperti saat training)
TrackLength_min = duration_ms / 60000
EnergyLevel = "High" if energy > 0.6 else "Low"

# Buat dataframe input sesuai kolom model
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

# ================================
# 4. Tombol Prediksi
# ================================
if st.button("🎯 Prediksi Sekarang"):
    
    # Pastikan input sesuai kolom model
    input_clf = df_input.reindex(columns=feature_columns, fill_value=0)

    # Prediksi
    clf_pred = classifier.predict(input_clf)[0]

    # ================================
    # 5. Output
    # ================================
    st.subheader("📌 Hasil Prediksi")
    st.success(f"📊 **Kategori Popularitas Lagu: `{clf_pred}`**")
    st.write("Prediksi berhasil menggunakan RandomForestClassifier dari model.pkl")
