"""HeartSafe AI: Flask web interface for a heart-disease risk model.

The model and scaler are loaded relative to this file so deployment does not
depend on a developer-specific absolute path.
"""
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_results" / "heart_disease_model.pkl"
SCALER_PATH = BASE_DIR / "final_results" / "scaler.pkl"

# Fail early with a useful message if a required deployment artifact is missing.
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
if not SCALER_PATH.is_file():
    raise FileNotFoundError(f"Feature scaler not found: {SCALER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        name = request.form.get("Name", "").strip() or "Not Provided"
        email = request.form.get("Email", "").strip() or "Not Provided"

        age = float(request.form["age"])
        sex = int(request.form["sex"])
        cp_value = int(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = int(request.form["fbs"])
        thalach = float(request.form["thalach"])
        exang = int(request.form["exang"])

        if not 1 <= age <= 120:
            raise ValueError("Age must be between 1 and 120.")
        if sex not in (0, 1):
            raise ValueError("Select a valid sex value.")
        if cp_value not in (0, 1, 2, 3):
            raise ValueError("Select a valid chest-pain type.")
        if fbs not in (0, 1) or exang not in (0, 1):
            raise ValueError("Select valid Yes/No values for the clinical fields.")
        if not 50 <= trestbps <= 300:
            raise ValueError("Resting blood pressure must be between 50 and 300 mm Hg.")
        if not 50 <= chol <= 600:
            raise ValueError("Cholesterol must be between 50 and 600 mg/dL.")
        if not 60 <= thalach <= 260:
            raise ValueError("Maximum heart rate must be between 60 and 260.")

        # Keep the exact feature order used during model training.
        input_data = np.array(
            [[age, sex, cp_value, trestbps, chol, fbs, thalach, exang]],
            dtype=float,
        )
        scaled_data = scaler.transform(input_data)

        prediction = int(model.predict(scaled_data)[0])
        probabilities = model.predict_proba(scaled_data)[0]
        positive_class_index = list(model.classes_).index(1)
        probability = float(probabilities[positive_class_index] * 100)

        if prediction == 1:
            result_text = f"High predicted risk of heart disease ({probability:.2f}% model estimate)"
            risk_color = "danger"
        else:
            result_text = f"Low predicted risk of heart disease ({probability:.2f}% model estimate)"
            risk_color = "success"

        return render_template(
            "result.html",
            prediction_text=result_text,
            risk_color=risk_color,
            name=name,
            email=email,
            age=age,
            sex="Male" if sex == 1 else "Female",
            cp=cp_value,
            trestbps=trestbps,
            chol=chol,
            fbs=fbs,
            thalach=thalach,
            exang=exang,
        )
    except (KeyError, TypeError, ValueError) as exc:
        # Keep user-input errors understandable without exposing server internals.
        return render_template(
            "result.html",
            prediction_text=f"Please check the entered details: {exc}",
            risk_color="warning",
            name=request.form.get("Name", "Not Provided"),
            email=request.form.get("Email", "Not Provided"),
        ), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "result.html",
            prediction_text="The prediction could not be completed. Please try again later.",
            risk_color="warning",
            name=request.form.get("Name", "Not Provided"),
            email=request.form.get("Email", "Not Provided"),
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
