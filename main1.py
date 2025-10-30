# ===============================================
# Modern ML Comparison with XGBoost Integration
# ===============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# ===============================================
# Load & Preprocess Dataset
# ===============================================
df = pd.read_csv("heart.csv")   # change filename
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================================
# Helper Function to Evaluate Models
# ===============================================
def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "Model": name,
        "CrossVal_Acc": np.mean(cv_scores),
        "Test_Acc": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    }
    return metrics, y_pred, y_proba

# ===============================================
# Train Models
# ===============================================
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = []
preds = {}
probas = {}

for name, model in models.items():
    metrics, y_pred, y_proba = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
    results.append(metrics)
    preds[name] = y_pred
    probas[name] = y_proba

# ===============================================
# Hyperparameter Tuning for XGBoost
# ===============================================
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1
)
grid.fit(X_train_scaled, y_train)

best_xgb = grid.best_estimator_
best_xgb.fit(X_train_scaled, y_train)

y_pred_xgb = best_xgb.predict(X_test_scaled)
y_proba_xgb = best_xgb.predict_proba(X_test_scaled)[:, 1]

print("\n✅ Best XGBoost Parameters:", grid.best_params_)
print("\nClassification Report (Tuned XGBoost):\n", classification_report(y_test, y_pred_xgb))

# ===============================================
# Confusion Matrix + ROC Curve for Tuned XGBoost
# ===============================================
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title("Confusion Matrix (Tuned XGBoost)")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_xgb)
ax[1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_test, y_proba_xgb):.3f})')
ax[1].plot([0, 1], [0, 1], '--')
ax[1].set_title("ROC Curve (Tuned XGBoost)")
ax[1].set_xlabel("False Positive Rate")
ax[1].set_ylabel("True Positive Rate")
ax[1].legend()

plt.tight_layout()
plt.show()

# ===============================================
# Feature Importance (Tuned XGBoost)
# ===============================================
importances = best_xgb.feature_importances_
feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp, palette="viridis")
plt.title("Feature Importance (Tuned XGBoost)")
plt.show()

# ===============================================
# Summary Table
# ===============================================
results_df = pd.DataFrame(results)
results_df = results_df.round(4)
print("\nSUMMARY TABLE - BASE MODELS\n", "=" * 80)
print(results_df)
# ====================================================
# 🔥 Performance Comparison Bar Plots
# ====================================================
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for i, metric in enumerate(metrics):
    sns.barplot(x="Model", y=metric, data=results_df, palette="coolwarm", ax=axes[i])
    axes[i].set_title(metric)
    axes[i].set_ylim(0.6, 1.05)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

plt.suptitle("Model Performance Comparison", fontsize=16)
plt.tight_layout()
plt.show()


# ===============================================
# Comparison Bar Graphs
# ===============================================
metrics_to_plot = ["Test_Acc", "Precision", "Recall", "F1", "ROC_AUC"]

plt.figure(figsize=(10, 6))
for metric in metrics_to_plot:
    plt.bar(results_df["Model"], results_df[metric], label=metric)
plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.legend()
plt.show()
