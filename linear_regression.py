# Linear Regression - Scikit-Learn
# Train, Evaluate and Compare Linear, Ridge and Lasso Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Load Real Dataset
# --------------------------------------------------

data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dataset shape:", X.shape)
print("\nFeatures:")
print(X.head())


# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 3. Create Models
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ]),

    "Lasso Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=0.1, max_iter=10000))
    ])
}


# --------------------------------------------------
# 4. Train and Evaluate Models
# --------------------------------------------------

results = []
predictions = {}

for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })


# --------------------------------------------------
# 5. Results Table
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n================ RESULTS ================")
print(results_df.to_string(index=False))


# --------------------------------------------------
# 6. Print Linear Regression Coefficients
# --------------------------------------------------

linear_model = models["Linear Regression"]

print("\n========== LINEAR REGRESSION COEFFICIENTS ==========")

for feature, coefficient in zip(
    X.columns,
    linear_model.coef_
):
    print(f"{feature}: {coefficient:.4f}")

print("\nIntercept:", round(linear_model.intercept_, 4))


# --------------------------------------------------
# 7. Predicted vs Actual Plot
# --------------------------------------------------

y_pred_linear = predictions["Linear Regression"]

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_linear,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression - Predicted vs Actual")

plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.show()


# --------------------------------------------------
# 8. Residual Plot
# --------------------------------------------------

residuals = y_test - y_pred_linear

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred_linear,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Linear Regression - Residual Plot")

plt.tight_layout()
plt.savefig("residual_plot.png")
plt.show()


# --------------------------------------------------
# 9. Compare R2 Scores
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.ylabel("R² Score")
plt.title("Comparison of Regression Models")

plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()


# --------------------------------------------------
# 10. Save Results
# --------------------------------------------------

results_df.to_csv(
    "regression_results.csv",
    index=False
)

print("\nResults saved to regression_results.csv")
print("Plots saved successfully.")