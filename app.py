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
# FUNGSI PREPROCESS (SAMA DENGAN SAAT TRAINING)
# =========================================================
def preprocess(df):
    # 1. Drop kolom ID unik
    drop_cols = ["id", "uri", "track_href", "analysis_url", "track_id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # 2. Imputasi numerik
    num_cols = [
        "time_signature", "speechiness", "danceability", "duration_ms", "energy",
        "mode", "instrumentalness", "valence", "key", "tempo", "loudness",
        "acousticness", "liveness", "track_popularity"
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 3. Imputasi kategorikal
    cat_cols = [
        "playlist_name", "track_artist", "playlist_genre", "playlist_subgenre",
        "track_name", "type", "track_album_release_date",
        "track_album_id", "playlist_id", "track_album_name"
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Fitur baru
    if "duration_ms" in df.columns:
        df["TrackLength_min"] = df["duration_ms"] / 60000

    if "energy" in df.columns:
        df["EnergyLevel"] = df["energy"].apply(lambda x: "High" if x > 0.6 else "Low")

    # 5. TitleWord
    if "track_name" in df.columns:
        df["TitleWord"] = df["track_name"].str.extract(r"^([A-Za-z]+)", expand=False).fillna("Unknown")

    # 6. Binning
    if "track_popularity" in df.columns:
        df["PopularityBin"] = pd.qcut(df["track_popularity"], 4,
                                      labels=["Low", "Mid", "High", "VeryHigh"])

    if "tempo" in df.columns:
        df["TempoBin"] = pd.cut(
            df["tempo"],
            bins=[0, 80, 120, 160, 200, 300],
            labels=["Slow", "Moderate", "Fast", "VeryFast", "UltraFast"]
        )

    # 7. One Hot Encoding
    cat_to_encode = [
        "playlist_genre", "playlist_subgenre", "EnergyLevel",
        "PopularityBin", "TempoBin", "TitleWord",
        "playlist_name", "track_artist", "track_album_name", "track_album_id",
        "track_name", "type", "track_album_release_date", "playlist_id"
    ]

    cat_to_encode = [c for c in cat_to_encode if c in df.columns]

    df = pd.get_dummies(df, columns=cat_to_encode, drop_first=True)

    # ⛔ Pastikan kolom cocok dengan model feature
    model_features = model.feature_names_in_

    # Column missing → create empty column
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Extra column → drop
    df = df[model_features]

    return df


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
        df_raw = pd.read_csv(uploaded_file)
        st.write("📄 Data asli yang di-upload:")
        st.dataframe(df_raw)

        try:
            df_clean = preprocess(df_raw)
            preds = model.predict(df_clean)

            df_raw["Prediksi Popularitas"] = preds

            st.success("Prediksi berhasil!")
            st.dataframe(df_raw)

            csv_output = df_raw.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Hasil Prediksi",
                csv_output,
                "hasil_prediksi.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Terjadi error saat memprediksi: {e}")

# =========================================================
# MODE 2: INPUT MANUAL (TETAP BISA)
# =========================================================
else:
    st.subheader("Input Fitur Lagu (Versi Sederhana)")

    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.number_input("Loudness", -60.0, 0.0, -10.0)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.5)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    tempo = st.number_input("Tempo", 60.0, 220.0, 120.0)

    if st.button("Prediksi"):
        df_manual = pd.DataFrame([[
            danceability, energy, loudness, speechiness, acousticness,
            instrumentalness, liveness, valence, tempo
        ]], columns=[
            "danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"
        ])

        df_clean = preprocess(df_manual)
        pred = model.predict(df_clean)[0]

        st.success(f"Prediksi Popularitas Lagu: **{pred}**")
