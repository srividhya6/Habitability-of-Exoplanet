from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import traceback

app = Flask(__name__)
CORS(app)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # notebook/api
NOTEBOOK_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # notebook

MODEL_PATH = os.path.join(NOTEBOOK_DIR, "final_habitability_model.pkl")
DATASET_PATH = os.path.join(NOTEBOOK_DIR, "exo_habitability_final.csv")

# ---------------- LOAD MODEL ----------------
try:
    pipeline = joblib.load(MODEL_PATH)
    app.logger.info("Model loaded successfully")
except Exception as e:
    pipeline = None
    app.logger.error(f"Failed to load model: {e}")

# ---------------- FEATURES ----------------
EXPECTED_FEATURES = [
    "pl_rade",
    "pl_bmasse",
    "pl_orbper",
    "pl_orbsmax",
    "pl_eqt",
    "pl_insol",
    "st_teff",
    "st_rad",
    "st_met",
    "planet_density",
    "star_luminosity",
    "HSI",
    "SCI",
    "star_type_G",
    "star_type_K",
    "star_type_M",
    "star_type_F",
]

# ---------------- MEDIANS ----------------
df_sample = pd.read_csv(DATASET_PATH)
MEDIANS = {
    col: float(df_sample[col].median()) if col in df_sample.columns else 0.0
    for col in EXPECTED_FEATURES
}

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return jsonify({"message": "Exoplanet habitability prediction api is running"})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/info")
def info():
    return jsonify({
        "model_loaded": pipeline is not None,
        "expected_features": EXPECTED_FEATURES
    })

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return jsonify({"error": "Model not available"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    try:
        row = []
        filled_from_median = []

        for feat in EXPECTED_FEATURES:
            if feat in data:
                row.append(float(data[feat]))
            else:
                row.append(MEDIANS.get(feat, 0.0))
                filled_from_median.append(feat)

        input_df = pd.DataFrame([row], columns=EXPECTED_FEATURES)

        prob = pipeline.predict_proba(input_df)[0][1]
        pred = int(prob >= 0.15)

        return jsonify({
            "habitability_prediction": "Habitable" if pred == 1 else "Non-Habitable",
            "habitability_probability": round(float(prob), 3),
            "filled_defaults": filled_from_median
        })

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

# ---------------- CSV HELPERS ----------------
def _read_csv_rel(filename):
    path = os.path.join(NOTEBOOK_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@app.route("/top-habitable")
def top_habitable():
    df = _read_csv_rel("Top_Habitable_Exoplanets.csv")
    return jsonify(df.to_dict(orient="records")) if df is not None else ("Not found", 404)

@app.route("/feature-importance")
def feature_importance():
    df = _read_csv_rel("feature_importance_ranking.csv")
    return jsonify(df.to_dict(orient="records")) if df is not None else ("Not found", 404)

@app.route("/model-comparisons")
def model_comparisons():
    df = _read_csv_rel("baseline_models_comparison.csv")
    return jsonify(df.to_dict(orient="records")) if df is not None else ("Not found", 404)

@app.route("/sampling-comparisons")
def sampling_comparisons():
    df = _read_csv_rel("sampling_techniques_comparison.csv")
    return jsonify(df.to_dict(orient="records")) if df is not None else ("Not found", 404)

@app.route("/exo-final")
def exo_final():
    df = _read_csv_rel("exo_habitability_final.csv")
    return jsonify(df.to_dict(orient="records")) if df is not None else ("Not found", 404)

if __name__ == "__main__":
    app.run(debug=True)
