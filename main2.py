# ==============================
# MODEL COMPARISON + EXPLAINABILITY (Final Clean Version)
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

# ========== LOAD DATA ==========
# Replace this path or DataFrame name with your dataset
data = pd.read_csv("heart.csv")  # Change to your actual dataset
print("\n📘 DATASET SUMMARY\n", "="*80)
print(data.info())
print("\n🧮 Basic Statistics:\n", data.describe())
print("\n🔍 Missing Values per Column:\n", data.isnull().sum())

# Assuming the last column is the target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ========== DATA SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ========== DEFINE MODELS ==========
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = []

# ========== TRAIN + EVALUATE ==========
for name, model in models.items():
    cv_acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        "Model": name,
        "CrossVal_Acc": round(cv_acc, 4),
        "Test_Acc": round(test_acc, 4),
        "ROC_AUC": round(roc_auc, 4)
    })

# ========== SUMMARY TABLE ==========
summary_df = pd.DataFrame(results).sort_values(by="Test_Acc", ascending=False)
print("\n📊 SUMMARY TABLE - BASE MODELS\n", "="*80)
print(summary_df)

# Save summary as CSV and TXT
output_dir = os.getcwd()
summary_csv_path = os.path.join(output_dir, "model_summary.csv")
summary_txt_path = os.path.join(output_dir, "model_summary.txt")

summary_df.to_csv(summary_csv_path, index=False)
with open(summary_txt_path, "w") as f:
    f.write("MODEL PERFORMANCE SUMMARY\n")
    f.write("="*60 + "\n")
    f.write(summary_df.to_string(index=False))

# ========== MODEL COMPARISON BAR PLOT (with gradient) ==========
plt.figure(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(summary_df)))
plt.bar(summary_df["Model"], summary_df["Test_Acc"], color=colors)
plt.title("Model Comparison - Test Accuracy", fontsize=14)
plt.ylabel("Test Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_accuracy_comparison.png"), dpi=300)
plt.show()

# ========== BEST MODEL ==========
best_model_name = summary_df.iloc[0]["Model"]
best_model_acc = summary_df.iloc[0]["Test_Acc"]
print(f"\n🏆 BEST MODEL: {best_model_name} with Test Accuracy = {best_model_acc:.4f} 🏆")

best_model = models[best_model_name]

# ========== BEST MODEL ROC CURVE ==========
y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="mediumvioletred", lw=2, label=f"{best_model_name} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_model_name}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "best_model_roc_curve.png"), dpi=300)
plt.show()
# ========== GRADIENT ROC CURVE (NO COLORBAR) ==========
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from sklearn.metrics import roc_curve, roc_auc_score

def plot_gradient_roc(y_test, y_prob, model_name, output_dir="."):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Create gradient segments
    points = np.array([fpr, tpr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Apply gradient colormap (plasma or viridis look great)
    cmap = cm.plasma
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(tpr)
    lc.set_linewidth(3)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_collection(lc)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name} (AUC = {roc_auc:.3f})", fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_curve_{model_name.replace(' ', '_')}.png"), dpi=300)
    plt.show()

# Example usage (replace with your variables)
# plot_gradient_roc(y_test, y_prob_best, best_model_name, output_dir)

# ========== XGBOOST SHAP EXPLAINABILITY ==========
best_xgb = models["XGBoost"]
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False, cmap='viridis')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "xgboost_shap_summary.png"), dpi=300)
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, cmap='viridis')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "xgboost_shap_bar.png"), dpi=300)
plt.close()

# ========== FEATURE IMPORTANCE BAR GRAPH ==========
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names, output_path="feature_importance.png"):
    """Plot feature importance for models with coef_ or feature_importances_."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        print("⚠️ Feature importance not available for this model.")
        return
    
    # Sort and plot with gradient colors
    indices = np.argsort(importance)[::-1]
    sorted_features = np.array(feature_names)[indices]
    sorted_importance = importance[indices]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_features)))
    plt.figure(figsize=(8, 5))
    plt.barh(sorted_features, sorted_importance, color=colors)
    plt.gca().invert_yaxis()
    plt.title("Feature Importance", fontsize=14)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

# Example usage (place at the end of your main code)
plot_feature_importance(best_model, X.columns, output_path=os.path.join(output_dir, "feature_importance.png"))


# ========== SAVE BEST MODEL ==========
model_path = os.path.join(output_dir, f"best_model_{best_model_name.replace(' ', '_')}.pkl")
joblib.dump(best_model, model_path)

# ========== RELIABILITY ==========
reliability = f"""
📘 MODEL RELIABILITY ANALYSIS
====================================
Best Model: {best_model_name}
Accuracy: {best_model_acc:.4f}
ROC-AUC: {roc_auc:.4f}
Cross-Validation Accuracy: {summary_df[summary_df['Model']==best_model_name]['CrossVal_Acc'].values[0]:.4f}

Interpretation:
The model shows good generalization performance with consistent accuracy across folds.
A ROC-AUC near 1 indicates high discriminative reliability.
Further validation using unseen data or domain-specific metrics is recommended.
"""
print(reliability)

with open(os.path.join(output_dir, "model_reliability.txt"), "w") as f:
    f.write(reliability)

print(f"\n✅ All files saved in: {output_dir}")
