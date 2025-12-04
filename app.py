import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import joblib

# ============================
# (NEW) 0. EXTRACT ZIP MODEL
# ============================
def extract_if_needed(zip_path, target_pkl):
    """Extract if PKL not found."""
    if not os.path.exists(target_pkl):
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall()
            print(f"Extracted: {zip_path}")
        else:
            print(f"ZIP not found: {zip_path}")
    else:
        print(f"PKL already exists: {target_pkl}")

# Sesuaikan nama ZIP dengan file kamu
extract_if_needed("pipeline_reg.zip", "pipeline_reg.pkl")

# Try load model
loaded_pipeline = None
if os.path.exists("pipeline_reg.pkl"):
    loaded_pipeline = joblib.load("pipeline_reg.pkl")


# ============================
# 1. JUDUL APLIKASI
# ============================
st.title("🎵 Spotify Track Popularity Predictor")
st.write("Aplikasi untuk memprediksi popularitas lagu berdasarkan fitur audio Spotify.")


# ============================
# 2. LOAD DATASET
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("low_popularity_spotify_data.csv")

    # Drop kolom yang tidak dipakai
    identifier_cols = ["id", "uri", "track_href", "analysis_url"]
    df = df.drop(columns=[c for c in identifier_cols if c in df.columns], errors="ignore")

    # Imputasi nilai hilang
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

df = load_data()

st.subheader("📌 Sample Dataset")
st.dataframe(df.head())


# ============================
# 3. DEFINE FEATURES & TARGET
# ============================
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Hilangkan target dari fitur kategorikal
if "track_popularity" in numeric_features:
    numeric_features.remove("track_popularity")

target = "track_popularity"

X = df[numeric_features + categorical_features]
y = df[target]


# ============================
# 4. BUILD PIPELINE
# ============================
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])


# ============================
# 5. TRAIN MODEL (only if no ZIP model)
# ============================
if loaded_pipeline is None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
else:
    pipeline = loaded_pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_pred = pipeline.predict(X_test)


# ============================
# 6. TAMPILKAN METRIK
# ============================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.subheader("📊 Model Performance")

st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.3f}")
st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")


# ============================
# 7. FORM INPUT PREDIKSI
# ============================
st.subheader("🎯 Prediksi Popularitas Lagu Baru")

user_input = {}

# Form numeric
for col in numeric_features:
    user_input[col] = st.number_input(f"{col}", value=float(df[col].median()))

# Form categorical
for col in categorical_features:
    unique_vals = df[col].unique().tolist()
    user_input[col] = st.selectbox(f"{col}", unique_vals)

# Convert ke DataFrame 1 row
input_df = pd.DataFrame([user_input])


# ============================
# 8. PREDIKSI
# ============================
if st.button("Prediksi Popularitas 🎵"):
    pred = pipeline.predict(input_df)[0]
    st.success(f"Perkiraan Popularitas Lagu: **{pred:.2f} / 100**")
