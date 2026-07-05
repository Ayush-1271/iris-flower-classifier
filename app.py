"""
app.py
------
Flask application that serves the trained Iris classifier to the general
public, both as:
  - a simple HTML form at "/"            (human-friendly)
  - a JSON REST API at "/api/predict"    (machine-friendly)

Works identically whether deployed on Render with Docker or without Docker
(the code doesn't care - only the deploy configuration differs).
"""

import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "iris_model.pkl")
bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
FEATURE_NAMES = bundle["feature_names"]
TARGET_NAMES = bundle["target_names"]

# Typical real-world ranges in the classic Iris dataset, shown as input
# guidance so a non-technical user knows what reasonable values look like.
FEATURE_RANGES = {
    "sepal_length": (4.3, 7.9),
    "sepal_width": (2.0, 4.4),
    "petal_length": (1.0, 6.9),
    "petal_width": (0.1, 2.5),
}

# Short educational blurb + a reference photo for each species, shown
# alongside the prediction result.
SPECIES_INFO = {
    "setosa": {
        "label": "Iris setosa",
        "description": (
            "The smallest of the three species, with short, wide petals and "
            "a compact flower. Native to North America, Alaska, and parts of "
            "eastern Asia."
        ),
        "image": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
    },
    "versicolor": {
        "label": "Iris versicolor (Blue Flag)",
        "description": (
            "A medium-sized species with intermediate petal and sepal "
            "proportions. Common in wetlands and marshes across eastern "
            "North America."
        ),
        "image": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    },
    "virginica": {
        "label": "Iris virginica",
        "description": (
            "The largest of the three species, with long, narrow petals and "
            "sepals. Found in wetlands of the southeastern United States."
        ),
        "image": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
    },
}


def build_features(sepal_length, sepal_width, petal_length, petal_width):
    """Recreate the same engineered features used during training."""
    petal_area = petal_length * petal_width
    sepal_area = sepal_length * sepal_width
    return np.array(
        [[sepal_length, sepal_width, petal_length, petal_width, petal_area, sepal_area]]
    )


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    X = build_features(sepal_length, sepal_width, petal_length, petal_width)
    pred_idx = int(MODEL.predict(X)[0])
    proba = MODEL.predict_proba(X)[0]
    species = TARGET_NAMES[pred_idx]
    return {
        "species": species,
        "confidence": round(float(proba[pred_idx]), 4),
        "probabilities": {
            TARGET_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)
        },
        "info": SPECIES_INFO.get(species),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    if request.method == "POST":
        try:
            form_values = {
                "sepal_length": float(request.form["sepal_length"]),
                "sepal_width": float(request.form["sepal_width"]),
                "petal_length": float(request.form["petal_length"]),
                "petal_width": float(request.form["petal_width"]),
            }
            result = predict_species(**form_values)
        except (KeyError, ValueError):
            error = "Please enter valid numeric values for all four measurements."

    return render_template(
        "index.html",
        result=result,
        error=error,
        values=form_values,
        ranges=FEATURE_RANGES,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API for the general public / other programs.

    Example request:
        POST /api/predict
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    """
    data = request.get_json(silent=True) or {}
    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        values = {f: float(data[f]) for f in required}
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric."}), 400

    prediction = predict_species(**values)
    return jsonify({"input": values, "prediction": prediction})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
