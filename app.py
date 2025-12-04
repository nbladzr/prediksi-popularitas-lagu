import streamlit as st
import joblib
import numpy as np

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Popularitas Lagu",
    page_icon="🎵",
    layout="centered"
)

# =========================
# CUSTOM CSS AGAR MIRIP UI DI SCREENSHOT
# =========================
st.markdown("""
<style>
    .title-center {
        text-align: center;
        font-size: 42px !important;
        font-weight: 700 !important;
        margin-top: -30px;
    }

    .subtitle-center {
        text-align: center;
        font-size: 18px;
        color: #cccccc;
        margin-bottom: 30px;
    }

    .upload-box {
        border: 2px dashed #444 !important;
        padding: 40px;
        border-radius: 10px;
        background-color: #1e1e1e;
    }

    .block-container {
        padding-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER PERSIS SEPERTI UI GAMBAR
# =========================
st.markdown("<h1 class='title-center'>🎵 Prediksi Popularitas Lagu</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle-center'>Upload file atau input fitur secara manual untuk memprediksi popularitas lagu.</p>",
    unsafe_allow_html=True
)

# =========================
# PILIH MODE INPUT
# =========================
mode = st.radio("Pilih Mode Input:", ["Upload CSV", "Input Manual"])

# =========================
# MODE 1: UPLOAD CSV
# =========================
if mode == "Upload CSV":
    st.write("Upload file CSV")

    uploaded_file = st.file_uploader(
        " ",
        type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        import pandas as pd
        df = pd.read_csv(uploaded_file)

        try:
            preds = model.predict(df)
            st.success("Berhasil memproses CSV!")
            st.write("Hasil Prediksi:")
            st.write(preds)
        except:
            st.error("Format CSV tidak sesuai fitur model!")

# =========================
# MODE 2: INPUT MANUAL
# =========================
else:
    st.write("Isi fitur secara manual:")

    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    tempo = st.slider("Tempo", 50.0, 200.0, 120.0)

    input_data = np.array([[
        danceability,
        energy,
        loudness,
        acousticness,
        speechiness,
        instrumentalness,
        liveness,
        valence,
        tempo
    ]])

    if st.button("Prediksi"):
        pred = model.predict(input_data)[0]
        st.success(f"Hasil Prediksi Popularitas Lagu: {pred}")
