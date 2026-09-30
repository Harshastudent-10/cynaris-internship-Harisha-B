# W3D5: Hyperparameter Tuning using GridSearchCV and RandomizedSearchCV

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load the Iris dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target


# --------------------------------------------------
# 2. Split the dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Create a pipeline
# --------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])


# --------------------------------------------------
# 4. Define hyperparameters for GridSearch
# --------------------------------------------------

grid_params = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__kernel": ["linear", "rbf", "poly"],
    "svm__gamma": ["scale", "auto"]
}


# --------------------------------------------------
# 5. GridSearchCV
# --------------------------------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=grid_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)


# --------------------------------------------------
# 6. Evaluate GridSearch model
# --------------------------------------------------

grid_predictions = grid_search.predict(X_test)

grid_accuracy = accuracy_score(
    y_test,
    grid_predictions
)

print("========== GRID SEARCH ==========")
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Score:")
print(grid_search.best_score_)

print("\nTest Accuracy:")
print(grid_accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    grid_predictions
))


# --------------------------------------------------
# 7. Define hyperparameters for RandomSearch
# --------------------------------------------------

random_params = {
    "svm__C": [0.01, 0.1, 1, 10, 100, 1000],
    "svm__kernel": ["linear", "rbf", "poly", "sigmoid"],
    "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1]
}


# --------------------------------------------------
# 8. RandomizedSearchCV
# --------------------------------------------------

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=random_params,
    n_iter=15,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)


# --------------------------------------------------
# 9. Evaluate RandomSearch model
# --------------------------------------------------

random_predictions = random_search.predict(X_test)

random_accuracy = accuracy_score(
    y_test,
    random_predictions
)

print("\n========== RANDOM SEARCH ==========")
print("Best Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validation Score:")
print(random_search.best_score_)

print("\nTest Accuracy:")
print(random_accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    random_predictions
))


# --------------------------------------------------
# 10. Compare both approaches
# --------------------------------------------------

print("\n========== COMPARISON ==========")
print("GridSearch Test Accuracy:",
      grid_accuracy)

print("RandomSearch Test Accuracy:",
      random_accuracy)