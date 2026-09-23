from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train a model and return evaluation metrics."""
    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_prediction)
    test_mse = mean_squared_error(y_test, test_prediction)

    train_r2 = r2_score(y_train, train_prediction)
    test_r2 = r2_score(y_test, test_prediction)

    return train_mse, test_mse, train_r2, test_r2


def main():
    # Load dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # High-complexity polynomial model
    polynomial_model = Pipeline([
        ("scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=3)),
        ("model", LinearRegression())
    ])

    # Ridge regularisation
    ridge_model = Pipeline([
        ("scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=3)),
        ("model", Ridge(alpha=10.0))
    ])

    # Lasso regularisation
    lasso_model = Pipeline([
        ("scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=3)),
        ("model", Lasso(alpha=0.1, max_iter=10000))
    ])

    models = {
        "Polynomial Regression": polynomial_model,
        "Ridge Regression": ridge_model,
        "Lasso Regression": lasso_model
    }

    print("Bias-Variance Tradeoff & Regularisation")
    print("=" * 50)

    for name, model in models.items():
        train_mse, test_mse, train_r2, test_r2 = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(f"\n{name}")
        print(f"Training MSE : {train_mse:.4f}")
        print(f"Testing MSE  : {test_mse:.4f}")
        print(f"Training R²  : {train_r2:.4f}")
        print(f"Testing R²   : {test_r2:.4f}")


if __name__ == "__main__":
    main()