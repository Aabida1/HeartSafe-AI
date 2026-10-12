# HeartSafe AI 

HeartSafe AI is a machine-learning web application that estimates heart-disease risk from selected health indicators. It uses a trained XGBoost model and a Flask interface to collect inputs and display a prediction.

> **Important medical disclaimer:** HeartSafe AI is an educational machine-learning project, not a medical device. Its output is not a diagnosis and must not be used to make treatment decisions. Consult a qualified healthcare professional for medical advice.

## Features

- Web-based form for entering health indicators
- Prediction from a trained XGBoost model
- Displays a predicted risk category and the model's estimated probability
- Result page displays the submitted input values
- Flask backend with a simple HTML template interface

## Inputs

The current prediction route accepts these eight model features:

| Feature | Description |
|---|---|
| Age | Age in years |
| Sex | Encoded as 1 for male and 0 for female |
| Chest pain | Yes/No selection, converted to a numeric value by the app |
| Resting blood pressure (`trestbps`) | Resting blood pressure |
| Serum cholesterol (`chol`) | Serum cholesterol measurement |
| Fasting blood sugar (`fbs`) | Encoded fasting blood sugar indicator |
| Maximum heart rate (`thalach`) | Maximum heart rate achieved |
| Exercise-induced angina (`exang`) | Encoded indicator |

**Note:** The feature encodings and order must match the data preprocessing used to train the saved model.

## Tech stack

- **Python**
- **Flask** — web application
- **XGBoost** — prediction model
- **NumPy** — input array preparation
- **Joblib** — loading the saved model
- **scikit-learn** — machine-learning ecosystem
- **Gunicorn** — production WSGI server dependency

## Repository structure

A typical project layout for this app is:

```text
HeartSafe-AI/
├── app.py
├── requirements.txt
├── final_results/
│   └── heart_disease_model.pkl
├── templates/
│   ├── index.html
│   └── result.html
└── static/
    ├── css/
    └── images/
```

The exact template and static-file contents depend on the files present in the repository.

## Run locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Aabida1/HeartSafe-AI.git
   cd HeartSafe-AI
   ```

2. Create and activate a virtual environment:

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows**
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the trained model file is available. In `app.py`, update the model-loading path to a path relative to the project, for example:

   ```python
   model = joblib.load("final_results/heart_disease_model.pkl")
   ```

   The current code contains a machine-specific absolute path, so this adjustment is required on another computer or deployment environment.

5. Start the Flask app:

   ```bash
   python app.py
   ```

6. Open the local address printed by Flask in your browser (usually `http://127.0.0.1:5000/`).

For deployment, configure the hosting service to start the app with Gunicorn (for example, `gunicorn app:app`) and ensure the model file is included at the configured path.

## Dataset and model

The project has previously been described as using the UCI Heart Disease dataset. The web app loads a pre-trained model from a serialized `.pkl` file; training and evaluation details should be documented alongside the model or in a separate notebook/script if available.

## Limitations

- Predictions depend on the training data, feature definitions, preprocessing, and model quality.
- The probability displayed by the application is the model's output; it should not automatically be interpreted as a clinically calibrated probability.
- Performance metrics and clinical validation are not asserted here because they are not documented in this README.
- Do not enter real patient-identifiable information into a demo deployment.

## Future improvements

- Document model training, preprocessing, and evaluation metrics
- Add automated input validation and clearer error handling
- Add tests for the prediction route and input encoding
- Provide screenshots of the interface and example outputs

## Author

**Aabida Khan**

GitHub: [@Aabida1](https://github.com/Aabida1)

---

Built as a machine-learning project to explore heart-disease risk prediction through a web application.
