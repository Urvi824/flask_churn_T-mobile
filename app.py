from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

# --- Config ---
FEATURES_PATH = os.path.join("config", "features.json")
MODEL_PATH = os.path.join("models", "model.pkl")

# Load feature order (used to build forms and keep input order consistent)
if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, "r") as f:
        feature_cfg = json.load(f)
    FEATURE_ORDER = feature_cfg.get("feature_order", [])
else:
    FEATURE_ORDER = []

# Load model pipeline
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"[WARN] Failed to load model at {MODEL_PATH}: {e}")

@app.route("/", methods=["GET"])
def home():
    # Render a simple form with fields driven by FEATURE_ORDER
    return render_template("index.html", features=FEATURE_ORDER)

def _coerce(value: str):
    """Attempt to coerce string inputs to float or int if possible, otherwise return the original string."""
    if value is None:
        return value
    s = str(value).strip()
    if s == "":
        return None
    # Try int
    try:
        i = int(s)
        return i
    except:
        pass
    # Try float
    try:
        f = float(s)
        return f
    except:
        pass
    # Fallback to original string (categorical)
    return s

def _predict_from_dict(payload: dict):
    if model is None:
        return {"error": "Model not loaded. Please place your trained pipeline at models/model.pkl"}, 500

    # Build a single-row DataFrame in the expected feature order (if provided)
    if FEATURE_ORDER:
        row = {k: _coerce(payload.get(k)) for k in FEATURE_ORDER}
        X = pd.DataFrame([row], columns=FEATURE_ORDER)
    else:
        # If no feature order provided, use the payload as-is (sorted for determinism)
        ordered_keys = sorted(payload.keys())
        row = {k: _coerce(payload.get(k)) for k in ordered_keys}
        X = pd.DataFrame([row], columns=ordered_keys)

    # Predict using sklearn pipeline (handles preprocessing internally if saved that way)
    try:
        pred = model.predict(X)[0]
        if hasattr(model, "predict_proba"):
            proba = float(np.max(model.predict_proba(X), axis=1)[0])
        else:
            proba = None
        return {"prediction": str(pred), "probability": proba}
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}, 400

@app.route("/predict", methods=["POST"])
def predict_form():
    # Form submission from HTML; fields named as feature names
    form_dict = {k: request.form.get(k) for k in request.form}
    result = _predict_from_dict(form_dict)
    status = 200
    if isinstance(result, tuple):
        result, status = result
    # Render back on the page
    return render_template("index.html", features=FEATURE_ORDER, result=result), status

@app.route("/api/predict", methods=["POST"])
def predict_api():
    if not request.is_json:
        return jsonify({"error": "Send JSON with Content-Type: application/json"}), 415
    payload = request.get_json(silent=True) or {}
    result = _predict_from_dict(payload)
    if isinstance(result, tuple):
        body, status = result
        return jsonify(body), status
    return jsonify(result), 200

@app.route("/health", methods=["GET"])
def health():
    ok = (model is not None)
    return jsonify({"status": "ok" if ok else "no-model", "has_model": ok}), 200

if __name__ == "__main__":
    # For local dev only. In production, use gunicorn: gunicorn -w 2 -b 0.0.0.0:8000 app:app
    app.run(host="0.0.0.0", port=5000, debug=True)
