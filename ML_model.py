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
from matplotlib.colors import LinearSegmentedColormap

# 2. Load the dataset
df = pd.read_csv('heartt.csv')

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

blue_red_cmap = LinearSegmentedColormap.from_list('blue_red', ["#E2D3D3", "#370369"])

# 🔹 Function to create gradient colors based on 0.5–1 scale
def get_gradient_colors_fixed(values, cmap=blue_red_cmap, vmin=0.5, vmax=1.0):
    """Generate gradient colors based on fixed 0.5–1 scale"""
    norm_values = [(val - vmin) / (vmax - vmin) for val in values]  # Normalize between 0 and 1
    norm_values = [max(0, min(1, v)) for v in norm_values]  # Clip to 0–1
    colors = [cmap(val) for val in norm_values]
    return colors

# 🔹 Updated normalization for colorbars (vmin=0.5, vmax=1)
norm = plt.Normalize(0.5, 1)

# -----------------------
# 9. Accuracy Comparison
# -----------------------
plt.figure(figsize=(10, 6))
accuracy_values = list(accuracy_scores.values())
colors = get_gradient_colors_fixed(accuracy_values)
bars = plt.bar(accuracy_scores.keys(), accuracy_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
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

# 🔹 Colorbar with range 0.5–1
sm = plt.cm.ScalarMappable(cmap=blue_red_cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
cbar.set_label('Accuracy (0.5–1 scale)', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()

# -----------------------
# 10. F1-Score Comparison
# -----------------------
plt.figure(figsize=(10, 6))
f1_values = list(f1_scores.values())
colors = get_gradient_colors_fixed(f1_values)
bars = plt.bar(f1_scores.keys(), f1_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
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

# 🔹 Colorbar with range 0.5–1
sm = plt.cm.ScalarMappable(cmap=blue_red_cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
cbar.set_label('F1-Score (0.5–1 scale)', rotation=270, labelpad=15)

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