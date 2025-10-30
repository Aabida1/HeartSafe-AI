# -------------------------------
# HEART DISEASE PREDICTION MODEL
# -------------------------------

# 1. Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# 2. Load the dataset
df = pd.read_csv('heart.csv')

# 3. Drop non-numeric columns if any (like Unnamed:0)
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# 4. Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# 5. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier()
}

# 8. Train models and store metrics
accuracy_scores = {}
precision_scores = {}
recall_scores = {}
f1_scores = {}
support_scores = {}

print("=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)

for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    # Store scores
    accuracy_scores[name] = accuracy
    precision_scores[name] = precision
    recall_scores[name] = recall
    f1_scores[name] = f1
    
    # Get support (number of instances for class 1)
    support = (y_test == 1).sum()
    support_scores[name] = support
    
    # Print detailed classification report
    print(f"\n{name} Performance:")
    print("-" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Support:   {support}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=" * 80)

# -------------------------------
# 📊 ACCURACY & F1-SCORE COMPARISON (Unique Colors per Model)
# -------------------------------

# Define fixed colors for each model (you can customize these)
model_colors = {
    "Logistic Regression": "#6112AC",  # deep magenta
    "Random Forest": "#600DAE",        # orange
    "SVM": "#7631B6",                  # blue
    "KNN": "#753FA7"                   # green
}

# ---- Accuracy Plot ----
plt.figure(figsize=(10, 6))
accuracy_values = list(accuracy_scores.values())
colors = [model_colors[name] for name in accuracy_scores.keys()]
bars = plt.bar(accuracy_scores.keys(), accuracy_values, color=colors, edgecolor='black', linewidth=0.8)

plt.title("Model Accuracy Comparison", fontsize=16, fontweight='bold')
plt.ylabel("Accuracy Score", fontsize=12)
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ---- F1-Score Plot ----
plt.figure(figsize=(10, 6))
f1_values = list(f1_scores.values())
colors = [model_colors[name] for name in f1_scores.keys()]
bars = plt.bar(f1_scores.keys(), f1_values, color=colors, edgecolor='black', linewidth=0.8)

plt.title("Model F1-Score Comparison (Class 1: Heart Disease)", fontsize=16, fontweight='bold')
plt.ylabel("F1-Score", fontsize=12)
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()


# 11. Summary Table
print("\n" + "=" * 80)
print("SUMMARY TABLE - ALL MODELS")
print("=" * 80)
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
print("-" * 80)
for name in models.keys():
    print(f"{name:<20} {accuracy_scores[name]:<10.4f} {precision_scores[name]:<10.4f} {recall_scores[name]:<10.4f} {f1_scores[name]:<10.4f} {support_scores[name]:<10}")
print("=" * 80)
import pickle

# Suppose Random Forest performed the best
best_model = models["Random Forest"]

# Save both model and scaler
with open("heart_disease_model1.pkl", "wb") as file:
    pickle.dump((scaler, best_model), file)

print("✅ Model and scaler saved as heart_disease_model1.pkl")
rf = RandomForestClassifier(random_state=42)
scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("Random Forest CV Accuracy: %.4f ± %.4f" % (scores.mean(), scores.std()))