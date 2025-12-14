# ==============================
# Heart Disease Prediction Web App
# ==============================

from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# ========== Load trained XGBoost model ==========
model = joblib.load("/Users/aabidakhan/GIT/HeartSafe-AI/final_results/heart_disease_model.pkl")

# ========== Home Route ==========
@app.route('/')
def home():
    return render_template('index.html')  # main input form page


# ========== Prediction Route ==========
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- Collect form inputs safely ---
        name = request.form.get('Name', 'Not Provided')
        email = request.form.get('Email', 'Not Provided')
        age = float(request.form.get('age', 0))
        sex = float(request.form.get('sex', 0))  # 1 = Male, 0 = Female

        # --- Convert Chest Pain (Yes/No) into numeric ---
        cp_input = request.form.get('cp', '').strip().lower()
        cp_value = 1 if cp_input == 'yes' else 0

        # --- Other numeric parameters ---
        trestbps = float(request.form.get('trestbps', 0))
        chol = float(request.form.get('chol', 0))
        fbs = float(request.form.get('fbs', 0))
        thalach = float(request.form.get('thalach', 0))
        exang = float(request.form.get('exang', 0))

        # --- Prepare input for prediction ---
        input_data = np.array([[age, sex, cp_value, trestbps, chol, fbs, thalach, exang]])

        # --- Make prediction ---
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100  # Probability (%)

        # --- Interpretation text ---
        if prediction == 1:
            result_text = f"⚠️ High Risk of Heart Disease ({probability:.2f}% likelihood)"
            risk_color = "danger"
        else:
            result_text = f"💚 Low Risk of Heart Disease ({probability:.2f}% likelihood)"
            risk_color = "success"

        # --- Render the result page ---
        return render_template(
            'result.html',
            prediction_text=result_text,
            risk_color=risk_color,
            name=name,
            email=email,
            age=age,
            sex='Male' if sex == 1 else 'Female',
            cp='Yes' if cp_value == 1 else 'No',
            trestbps=trestbps,
            chol=chol,
            fbs=fbs,
            thalach=thalach,
            exang=exang
        )

    except Exception as e:
        # If any error occurs, display it on the result page
        return render_template('result.html', prediction_text=f"❌ Error: {str(e)}", risk_color="warning")


# ========== Run the Flask App ==========
if __name__ == "__main__":
    app.run(debug=True)

