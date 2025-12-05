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
MODEL_FEATURES = list(model.feature_names_in_)

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Prediksi Popularitas Lagu", layout="centered")

st.title("🎵 Prediksi Popularitas Lagu")
st.write("Upload CSV atau input manual untuk memprediksi popularitas lagu.")

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

        # Cek apakah kolom cocok dengan model
        missing_features = [col for col in MODEL_FEATURES if col not in df.columns]

        if missing_features:
            st.error("❌ Kolom pada CSV tidak cocok dengan model!")
            st.warning(f"Kolom yang hilang: {missing_features[:20]} ...")
        else:
            pred = model.predict(df)
            df["Prediksi Popularitas"] = pred
            st.success("Prediksi berhasil!")
            st.dataframe(df)

            csv_output = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Hasil Prediksi",
                               csv_output,
                               "hasil_prediksi.csv",
                               "text/csv")


# =========================================================
# MODE 2: INPUT MANUAL
# =========================================================
else:
    st.subheader("Input Fitur Lagu (Manual)")

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
        # DataFrame kosong sesuai fitur model
        input_df = pd.DataFrame([np.zeros(len(MODEL_FEATURES))], columns=MODEL_FEATURES)

        # Isi 9 fitur utama
        for col, val in {
            "danceability": danceability,
            "energy": energy,
            "loudness": loudness,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo,
        }.items():
            if col in input_df.columns:
                input_df[col] = val

        pred = model.predict(input_df)[0]
        st.success(f"Prediksi Popularitas Lagu: **{pred}**")
