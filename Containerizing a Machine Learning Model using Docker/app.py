import os
from pathlib import Path
import joblib
import numpy as np
from flask import Flask, jsonify, request
from sklearn.datasets import load_iris

# Initialize Flask application
app = Flask(__name__)

# Target class names from Iris dataset
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

# Load serialized model from project root
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[INFO] Model successfully loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load model from {MODEL_PATH}: {e}")
        model = None
else:
    print(f"[WARNING] Model file not found at {MODEL_PATH}")

@app.route("/", methods=["GET"])
def home():
    """
    Home root endpoint returning API status message.
    """
    return jsonify({
        "message": "Iris ML Model API is running",
        "status": "success"
    }), 200

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint returning application and model loading status.
    """
    is_loaded = model is not None
    return jsonify({
        "status": "healthy" if is_loaded else "unhealthy",
        "model_loaded": is_loaded
    }), (200 if is_loaded else 503)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint accepting flower measurements and returning predicted class.
    Expected JSON:
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    """
    if model is None:
        return jsonify({
            "error": "Model is not loaded on the server",
            "status": "error"
        }), 503

    # Validate JSON payload exists
    if not request.is_json:
        return jsonify({
            "error": "Request body must be valid JSON with Content-Type: application/json",
            "status": "error"
        }), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "error": "Malformed or empty JSON body",
            "status": "error"
        }), 400

    if "features" not in data:
        return jsonify({
            "error": "Missing 'features' key in JSON request",
            "status": "error"
        }), 400

    features = data["features"]

    # Validate features is a list
    if not isinstance(features, (list, tuple)):
        return jsonify({
            "error": "'features' must be an array of numeric values",
            "status": "error"
        }), 400

    # Validate feature count is exactly 4
    if len(features) != 4:
        return jsonify({
            "error": f"Exactly four features are required. Received {len(features)}.",
            "status": "error"
        }), 400

    # Validate numeric and finite values
    validated_features = []
    for idx, val in enumerate(features):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return jsonify({
                "error": f"Feature at index {idx} ('{val}') is not numeric",
                "status": "error"
            }), 400
        try:
            float_val = float(val)
            if not np.isfinite(float_val):
                return jsonify({
                    "error": f"Feature at index {idx} must be a finite number",
                    "status": "error"
                }), 400
            validated_features.append(float_val)
        except (ValueError, TypeError):
            return jsonify({
                "error": f"Unable to parse feature at index {idx} into float",
                "status": "error"
            }), 400

    # Perform prediction
    try:
        input_array = np.array([validated_features], dtype=float)
        pred_class_idx = int(model.predict(input_array)[0])
        pred_species = TARGET_NAMES[pred_class_idx]

        return jsonify({
            "prediction": pred_species,
            "prediction_class": pred_class_idx,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Inference execution failed: {str(e)}",
            "status": "error"
        }), 500

if __name__ == "__main__":
    # Configure Flask to run on Host: 0.0.0.0 and Port: 5000
    app.run(host="0.0.0.0", port=5000, debug=False)
