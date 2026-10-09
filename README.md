# HeartSafe AI

HeartSafe AI is a Flask-based web application that demonstrates a machine-learning workflow for estimating heart-disease risk from selected clinical features.

**Live demo:** https://heartsafe-ai.onrender.com  
**Important:** This is an educational screening prototype, not a medical device. It cannot diagnose or rule out heart disease. Seek advice from a qualified healthcare professional for medical decisions or urgent symptoms.

## Start here

- `app.py` — Flask application and prediction endpoint.
- `templates/index.html` — web form and user-facing explanations.
- `templates/result.html` — prediction result and client-side PDF report download.
- `final_results/heart_disease_model.pkl` — serialized trained model used by the app.
- `final_results/scaler.pkl` — feature scaler used during model training; the app applies it before prediction.
- `final_results/model_results.csv` — saved comparison metrics from model evaluation.
- `heart.csv` — dataset used in the project.
- `main-2.ipynb` — exploratory analysis, model comparison, and training notebook.
- `static/style.css` — standalone stylesheet retained in the repository; the current landing page primarily uses inline CSS.

## Repository layout

```text
HeartSafe-AI/
├── app.py
├── requirements.txt
├── runtime.txt
├── heart.csv
├── main-2.ipynb
├── final_results/
│   ├── heart_disease_model.pkl
│   ├── scaler.pkl
│   └── model_results.csv
├── templates/
│   ├── index.html
│   └── result.html
└── static/
    └── style.css
```

## Run locally

Use Python 3.10 (the version declared in `runtime.txt`).

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# On Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

The serialized model and scaler must remain in `final_results/` unless their paths are updated in `app.py`. Do not move or rename these files independently.

## Render deployment

This repository is configured for a Flask application. In the Render Web Service settings, use:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app`
- **Root directory:** repository root (leave blank unless the service is intentionally configured otherwise)

Render deploys the branch selected in the service settings. Review a change on a separate branch before merging it into the deployed branch. Keep `app.py`, `requirements.txt`, `templates/`, and `final_results/` in the locations above so the application entry point and model paths remain valid.

## Input features

The app expects the eight features in the same order as the training dataset: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `thalach`, and `exang`. Chest-pain type (`cp`) uses the dataset's four categories (0–3); it is not a yes/no field.

## Notes

- Do not commit credentials, private patient information, or environment secrets.
- Treat predictions as model estimates, not calibrated clinical probabilities.
- Re-evaluate the model on appropriate independent data before any real-world clinical use.
