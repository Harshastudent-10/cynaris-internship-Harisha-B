# Logistic Regression & Classification
# Train, Evaluate, ROC-AUC, Decision Boundary,
# Multi-class One-vs-Rest vs Softmax

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ==========================================================
# PART 1: BINARY CLASSIFICATION
# Breast Cancer Dataset
# ==========================================================

print("=" * 60)
print("BINARY CLASSIFICATION - BREAST CANCER")
print("=" * 60)

# Load dataset
cancer = load_breast_cancer()

X = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)

y = pd.Series(cancer.target)

print("\nDataset shape:", X.shape)
print("Classes:", cancer.target_names)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Logistic Regression
binary_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=5000))
])

binary_model.fit(X_train, y_train)


# Prediction
y_pred = binary_model.predict(X_test)
y_prob = binary_model.predict_proba(X_test)[:, 1]


# ==========================================================
# COEFFICIENTS AND INTERCEPT
# ==========================================================

logistic = binary_model.named_steps["logistic"]

print("\n========== COEFFICIENTS ==========")

for feature, coefficient in zip(
    cancer.feature_names,
    logistic.coef_[0]
):
    print(f"{feature}: {coefficient:.4f}")

print("\nIntercept:", logistic.intercept_[0])


# ==========================================================
# EVALUATION
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n========== EVALUATION ==========")

print("Confusion Matrix:")
print(cm)

print("\nAccuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# ==========================================================
# CONFUSION MATRIX HEATMAP
# ==========================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=cancer.target_names,
    yticklabels=cancer.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


# ==========================================================
# ROC CURVE
# ==========================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression - ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()


# ==========================================================
# DECISION BOUNDARY USING 2 FEATURES
# ==========================================================

print("\n========== 2D DECISION BOUNDARY ==========")

# Select first two features
X_2d = X.iloc[:, :2]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_2d,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

boundary_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=5000))
])

boundary_model.fit(
    X2_train,
    y2_train
)

# Create mesh
x_min = X_2d.iloc[:, 0].min() - 1
x_max = X_2d.iloc[:, 0].max() + 1

y_min = X_2d.iloc[:, 1].min() - 1
y_max = X_2d.iloc[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z = boundary_model.predict(grid)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3
)

plt.scatter(
    X_2d.iloc[:, 0],
    X_2d.iloc[:, 1],
    c=y,
    edgecolor="k",
    alpha=0.7
)

plt.xlabel(cancer.feature_names[0])
plt.ylabel(cancer.feature_names[1])

plt.title(
    "Logistic Regression - Decision Boundary"
)

plt.tight_layout()
plt.savefig("decision_boundary.png")
plt.show()


# ==========================================================
# PART 2: MULTI-CLASS CLASSIFICATION
# Iris Dataset
# ==========================================================

print("\n")
print("=" * 60)
print("MULTI-CLASS CLASSIFICATION - IRIS")
print("=" * 60)

iris = load_iris()

X_iris = iris.data
y_iris = iris.target


# Train-test split
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris,
    y_iris,
    test_size=0.20,
    random_state=42,
    stratify=y_iris
)


# ==========================================================
# ONE-VS-REST
# ==========================================================

ovr_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "ovr",
        OneVsRestClassifier(
            LogisticRegression(max_iter=5000)
        )
    )
])

ovr_model.fit(
    X_train_i,
    y_train_i
)

ovr_pred = ovr_model.predict(
    X_test_i
)

ovr_accuracy = accuracy_score(
    y_test_i,
    ovr_pred
)


# ==========================================================
# SOFTMAX
# ==========================================================

softmax_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "softmax",
        LogisticRegression(
            multi_class="multinomial",
            max_iter=5000
        )
    )
])

softmax_model.fit(
    X_train_i,
    y_train_i
)

softmax_pred = softmax_model.predict(
    X_test_i
)

softmax_accuracy = accuracy_score(
    y_test_i,
    softmax_pred
)


# ==========================================================
# MULTI-CLASS RESULTS TABLE
# ==========================================================

comparison = pd.DataFrame({
    "Model": [
        "One-vs-Rest",
        "Softmax"
    ],
    "Accuracy": [
        ovr_accuracy,
        softmax_accuracy
    ]
})

print("\n========== MULTI-CLASS RESULTS ==========")

print(
    comparison.to_string(index=False)
)


# Save results
comparison.to_csv(
    "multiclass_results.csv",
    index=False
)

print("\nResults saved to multiclass_results.csv")


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n")
print("=" * 60)
print("TASK COMPLETED")
print("=" * 60)

print("""
Binary Classification:
✓ Logistic Regression trained
✓ Coefficients and intercept printed
✓ Confusion Matrix calculated
✓ Accuracy calculated
✓ Precision calculated
✓ Recall calculated
✓ F1-Score calculated
✓ ROC-AUC calculated
✓ Confusion Matrix heatmap created
✓ ROC curve created
✓ Decision boundary created

Multi-Class Classification:
✓ One-vs-Rest implemented
✓ Softmax implemented
✓ Models compared in a table
""")