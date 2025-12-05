import streamlit as st
import joblib
import pandas as pd

# ================================
# Load model
# ================================
@st.cache_resource
def load_model():
    models = joblib.load("model.pkl")
    return models["classifier"], models["columns"]

classifier, feature_columns = load_model()

st.title("🎵 Prediksi Popularitas Lagu Spotify")
st.write("Pilih cara input data: CSV atau Manual")

# ================================
# Pilih mode input
# ================================
mode = st.radio("Pilih mode input:", ("Manual", "CSV"))

if mode == "CSV":
    uploaded_file = st.file_uploader("📂 Upload CSV lagu", type=["csv"])
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success("CSV berhasil dibaca!")
            st.dataframe(df_input.head())
        except Exception as e:
            st.error(f"❌ Gagal membaca CSV: {e}")
            df_input = None
    else:
        df_input = None

elif mode == "Manual":
    st.header("🔧 Input Manual Fitur Lagu")
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

    TrackLength_min = duration_ms / 60000
    EnergyLevel = "High" if energy > 0.6 else "Low"

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
        "EnergyLevel_High": [1 if EnergyLevel=="High" else 0],
    })

st.write("---")

# ================================
# Prediksi
# ================================
if st.button("🎯 Prediksi Popularitas"):
    if df_input is not None:
        df_pred = df_input.reindex(columns=feature_columns, fill_value=0)
        pred = classifier.predict(df_pred)

        st.subheader("📊 Hasil Prediksi Popularitas")
        st.dataframe(pred)
    else:
        st.warning("❌ Tidak ada data untuk diprediksi")
