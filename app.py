import os
import io
import pickle
from pathlib import Path
from flask import Flask, request, render_template_string, redirect, url_for, send_file
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------
PIPELINE_REG_PATH = "pipeline_reg.pkl"
PIPELINE_CLF_PATH = "pipeline_clf.pkl"
DEFAULT_DATA_LOW = "/content/low_popularity_spotify_data.csv"
DEFAULT_DATA_HIGH = "/content/high_popularity_spotify_data.csv"
ALLOWED_EXT = {"csv"}

# ---------- FLASK SETUP ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB upload limit

# ---------- Utility functions ----------
def safe_read_csv_file_storage(fs):
    """Read uploaded FileStorage to pandas DataFrame."""
    content = fs.read()
    return pd.read_csv(io.BytesIO(content))

def read_csv_path_if_exists(path):
    """Return DataFrame if path exists, else None."""
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None

def preprocess_feature_engineering(df):
    """
    Apply deterministic feature engineering used for both training & inference:
    - TrackLength_min
    - EnergyLevel (High if energy > 0.6 else Low)
    - TempoBin (bin tempo into categories)
    """
    # ensure safe copies
    df = df.copy()

    # TrackLength_min
    if "duration_ms" in df.columns:
        df["TrackLength_min"] = df["duration_ms"] / 60000.0

    # EnergyLevel
    if "energy" in df.columns:
        df["EnergyLevel"] = df["energy"].apply(lambda x: "High" if pd.notnull(x) and x > 0.6 else "Low")

    # TempoBin
    if "tempo" in df.columns:
        df["TempoBin"] = pd.cut(df["tempo"],
                                bins=[0, 80, 120, 160, 200, 300],
                                labels=["Slow","Moderate","Fast","VeryFast","UltraFast"])
    # fill simple NaNs for required numeric columns later in pipeline via SimpleImputer
    return df

def build_and_save_pipelines(df, save_dir="."):
    """
    Build preprocessing + model pipelines for:
     - regression (predict track_popularity)
     - classification (PopularityBin)
    Save pipelines to disk.
    Returns (pipeline_reg, pipeline_clf)
    """
    df = df.copy()

    # Drop identifier-ish columns if present
    id_cols = ["id", "uri", "track_href", "analysis_url", "track_id", "track_album_id", "playlist_id"]
    df = df.drop(columns=[c for c in id_cols if c in df.columns], errors='ignore')

    # Impute basic missing values for numeric columns
    numeric_defaults = ['danceability','energy','loudness','speechiness','acousticness',
                        'instrumentalness','liveness','valence','tempo','duration_ms',
                        'key','mode','time_signature']
    for c in numeric_defaults:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Fill categorical with mode
    cat_defaults = ['playlist_genre','playlist_subgenre','playlist_name','track_name','track_artist','track_album_name','type']
    for c in cat_defaults:
        if c in df.columns:
            try:
                df[c] = df[c].fillna(df[c].mode()[0])
            except Exception:
                df[c] = df[c].fillna("Unknown")

    # Feature engineering (use helper)
    df = preprocess_feature_engineering(df)

    # Targets
    if "track_popularity" not in df.columns:
        raise ValueError("Dataset must contain column 'track_popularity' for training pipelines.")

    # classification target
    df["PopularityBin"] = pd.qcut(df["track_popularity"], 4, labels=["Low","Mid","High","VeryHigh"])

    # Choose features
    numeric_features = [c for c in ["danceability","energy","loudness","speechiness",
                                   "acousticness","instrumentalness","liveness","valence",
                                   "tempo","TrackLength_min","key","mode","time_signature"] if c in df.columns]

    categorical_features = [c for c in ["playlist_genre","playlist_subgenre","EnergyLevel","TempoBin"] if c in df.columns]

    # Prepare matrix
    X = df[numeric_features + categorical_features]
    y_reg = df["track_popularity"]
    y_clf = df["PopularityBin"]

    # Preprocessors
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")

    # Pipelines
    pipeline_reg = Pipeline([
        ("preproc", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    pipeline_clf = Pipeline([
        ("preproc", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    # Train
    # Note: we don't need large hold-outs for the app; train on full data to produce a usable model for demo.
    pipeline_reg.fit(X, y_reg)
    pipeline_clf.fit(X, y_clf)

    # Save
    with open(os.path.join(save_dir, PIPELINE_REG_PATH), "wb") as f:
        pickle.dump(pipeline_reg, f)
    with open(os.path.join(save_dir, PIPELINE_CLF_PATH), "wb") as f:
        pickle.dump(pipeline_clf, f)

    return pipeline_reg, pipeline_clf

def load_pipelines_if_exist():
    if Path(PIPELINE_REG_PATH).exists() and Path(PIPELINE_CLF_PATH).exists():
        with open(PIPELINE_REG_PATH, "rb") as f:
            pipe_reg = pickle.load(f)
        with open(PIPELINE_CLF_PATH, "rb") as f:
            pipe_clf = pickle.load(f)
        return pipe_reg, pipe_clf
    return None, None

# Try load pipelines on startup (if exist)
PIPE_REG, PIPE_CLF = load_pipelines_if_exist()

# ---------- Minimal HTML templates (render_template_string used to keep single-file) ----------
HOME_HTML = """
<!doctype html>
<title>Spotify Popularity Predictor</title>
<h2>Spotify Popularity - Demo App</h2>
<p>
- Pipelines: {{ has_pipelines }}<br>
- Upload CSV to train pipelines (if pipelines not found) or use existing pipelines for prediction.
</p>
<hr>
<h3>1) Single song prediction (fill values)</h3>
<form action="{{ url_for('predict_single') }}" method="post">
  <label>danceability: <input name="danceability" step="any" required></label><br>
  <label>energy: <input name="energy" step="any" required></label><br>
  <label>loudness: <input name="loudness" step="any"></label><br>
  <label>speechiness: <input name="speechiness" step="any"></label><br>
  <label>acousticness: <input name="acousticness" step="any"></label><br>
  <label>instrumentalness: <input name="instrumentalness" step="any"></label><br>
  <label>liveness: <input name="liveness" step="any"></label><br>
  <label>valence: <input name="valence" step="any"></label><br>
  <label>tempo: <input name="tempo" step="any" required></label><br>
  <label>duration_ms: <input name="duration_ms" step="any" required></label><br>
  <label>key: <input name="key" step='any'></label><br>
  <label>mode: <input name="mode" step='any'></label><br>
  <label>time_signature: <input name="time_signature" step='any'></label><br>
  <label>playlist_genre: <input name="playlist_genre"></label><br>
  <label>playlist_subgenre: <input name="playlist_subgenre"></label><br>
  <button type="submit">Predict single</button>
</form>

<hr>
<h3>2) Upload CSV (one or two files)</h3>
<form action="{{ url_for('predict_file') }}" method="post" enctype="multipart/form-data">
  <p>Upload CSV file(s). If you have two CSVs (low & high), upload both (they will be concatenated).</p>
  <input type="file" name="file1" accept=".csv"><br>
  <input type="file" name="file2" accept=".csv"><br>
  <button type="submit">Upload & Predict</button>
</form>

<hr>
<h3>3) (Re)Train pipelines from default CSV paths on server</h3>
<form action="{{ url_for('train_from_default') }}" method="post">
  <p>Will look for default files on server:</p>
  <pre>{{ default_low }}\n{{ default_high }}</pre>
  <button type="submit">Train pipelines from server CSV(s)</button>
</form>

<hr>
<p><em>Notes:</em> If pipelines do not exist, the app will ask you to train them first (use upload CSV or train from default).</p>
"""

RESULT_HTML = """
<!doctype html>
<title>Prediction Result</title>
<h2>Prediction Result</h2>
<p>{{ info }}</p>
{% if table_html %}
<hr>
<h3>Batch predictions</h3>
<div>{{ table_html|safe }}</div>
{% endif %}
<hr>
<p><a href="{{ url_for('home') }}">Back</a></p>
"""

# ---------- Flask routes ----------
@app.route("/", methods=["GET"])
def home():
    has_pipes = Path(PIPELINE_REG_PATH).exists() and Path(PIPELINE_CLF_PATH).exists()
    return render_template_string(HOME_HTML,
                                  has_pipelines="Yes" if has_pipes else "No",
                                  default_low=DEFAULT_DATA_LOW,
                                  default_high=DEFAULT_DATA_HIGH)

@app.route("/train_default", methods=["POST"])
def train_from_default():
    global PIPE_REG, PIPE_CLF
    # try read default CSVs if exist
    df_low = read_csv_path_if_exists(DEFAULT_DATA_LOW)
    df_high = read_csv_path_if_exists(DEFAULT_DATA_HIGH)

    if df_low is None and df_high is None:
        return render_template_string(RESULT_HTML, info="No default CSVs found on server.", table_html=None)

    if df_low is not None and df_high is not None:
        df = pd.concat([df_low, df_high], ignore_index=True)
    else:
        df = df_low if df_low is not None else df_high

    # build pipelines and save
    PIPE_REG, PIPE_CLF = build_and_save_pipelines(df)
    return render_template_string(RESULT_HTML, info="Pipelines trained and saved from default CSV(s).", table_html=None)

@app.route("/predict_single", methods=["POST"])
def predict_single():
    global PIPE_REG, PIPE_CLF
    # Ensure pipelines exist
    if PIPE_REG is None or PIPE_CLF is None:
        return render_template_string(RESULT_HTML, info="No pipelines found. Train pipelines first (upload CSV or train from default).", table_html=None)

    # collect form values into dataframe
    payload = {}
    for k, v in request.form.items():
        try:
            # convert numeric if possible
            if v == "":
                payload[k] = np.nan
            else:
                payload[k] = float(v)
        except ValueError:
            payload[k] = v  # keep string (e.g., playlist_genre)

    new_df = pd.DataFrame([payload])
    new_df = preprocess_feature_engineering(new_df)

    # predict
    try:
        pred_reg = PIPE_REG.predict(new_df)[0]
        pred_clf = PIPE_CLF.predict(new_df)[0]
        info = f"Predicted popularity (regression): {pred_reg:.3f} | Predicted popularity bin (classifier): {pred_clf}"
    except Exception as e:
        info = f"Prediction error: {e}"

    return render_template_string(RESULT_HTML, info=info, table_html=None)

@app.route("/predict_file", methods=["POST"])
def predict_file():
    global PIPE_REG, PIPE_CLF

    # read uploaded files (0,1 or 2)
    files = []
    if "file1" in request.files and request.files["file1"].filename:
        files.append(request.files["file1"])
    if "file2" in request.files and request.files["file2"].filename:
        files.append(request.files["file2"])

    if len(files) == 0:
        return render_template_string(RESULT_HTML, info="No files uploaded.", table_html=None)

    # read files into dataframes
    dfs = []
    try:
        for f in files:
            # ensure csv
            if not f.filename.lower().endswith(".csv"):
                return render_template_string(RESULT_HTML, info=f"File {f.filename} is not CSV.", table_html=None)
            df_tmp = safe_read_csv_file_storage(f)
            dfs.append(df_tmp)
    except Exception as e:
        return render_template_string(RESULT_HTML, info=f"Failed reading uploaded CSV(s): {e}", table_html=None)

    # concat if multiple
    df_all = pd.concat(dfs, ignore_index=True)

    # If pipelines not exist, build them using uploaded data (this trains on uploaded file(s))
    if PIPE_REG is None or PIPE_CLF is None:
        try:
            PIPE_REG, PIPE_CLF = build_and_save_pipelines(df_all)
        except Exception as e:
            return render_template_string(RESULT_HTML, info=f"Training pipelines failed: {e}", table_html=None)

    # Prepare for prediction: feature engineering
    df_pe = preprocess_feature_engineering(df_all)

    # For prediction we must feed the raw features that pipelines expect.
    # Our pipeline expects numeric_features + categorical_features defined in build function.
    # We'll attempt to predict directly and let pipeline handle missing columns via imputers.
    try:
        preds_reg = PIPE_REG.predict(df_pe)
        preds_clf = PIPE_CLF.predict(df_pe)
    except Exception as e:
        return render_template_string(RESULT_HTML, info=f"Prediction error: {e}", table_html=None)

    # Attach predictions
    out = df_all.copy()
    out["pred_popularity_reg"] = preds_reg
    out["pred_popularity_bin"] = preds_clf

    # show first 20 rows as HTML
    table_html = out.head(20).to_html(classes="table table-striped", index=False)
    return render_template_string(RESULT_HTML, info=f"Batch prediction done on {len(out)} rows.", table_html=table_html)

@app.route("/download_pipeline/<which>", methods=["GET"])
def download_pipeline(which):
    """Optional: download pipeline files."""
    if which == "reg":
        p = PIPELINE_REG_PATH
    else:
        p = PIPELINE_CLF_PATH
    if not Path(p).exists():
        return f"File {p} not found on server.", 404
    return send_file(p, as_attachment=True)

# ---------- Run ----------
if __name__ == "__main__":
    # If app started and pipelines missing, user can train via UI (upload) or train_default
    port = int(os.environ.get("PORT", 5000))
    print("Starting app on http://127.0.0.1:%d" % port)
    app.run(host="0.0.0.0", port=port, debug=True)
