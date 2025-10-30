# ==============================
# Heart Disease Prediction Web App
# ==============================

from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# ========== Load your trained model ==========
model = joblib.load("best_model_XGBoost.pkl")

# ========== Home route ==========
@app.route('/')
def home():
    return render_template('index.html')

# ========== Prediction route ==========
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- Get input values from form ---
        age = float(request.form['age'])
        sex = float(request.form['sex'])

        # Chest pain: convert yes/no → numeric
        cp_input = request.form['cp'].strip().lower()
        cp_value = 1 if cp_input == 'yes' else 0

        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])

        # --- Create input array in the same order as model training ---
        input_data = np.array([[age, sex, cp_value, trestbps, chol, fbs, thalach, exang]])

        # --- Make prediction ---
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100  # class 1 probability %

        # --- Generate readable output ---
        if prediction == 1:
            result_text = f"⚠️ High Risk of Heart Disease ({probability:.2f}% likelihood)"
        else:
            result_text = f"💚 Low Risk of Heart Disease ({probability:.2f}% likelihood)"

        # --- Render result page ---
        return render_template('result.html', prediction_text=result_text)

    except Exception as e:
        return f"Error: {e}"

# ========== Run the app ==========
if __name__ == "__main__":
    app.run(debug=True)
