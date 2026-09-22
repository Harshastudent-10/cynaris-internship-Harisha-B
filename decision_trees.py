# Decision Tree and Random Forest Classification
# Week 3 AIML Practical Task

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

print("Dataset shape:", X.shape)
print("\nFirst five rows:")
print(X.head())


# --------------------------------------------------
# 2. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Decision Tree Model
# --------------------------------------------------

decision_tree = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

decision_tree.fit(X_train, y_train)

dt_predictions = decision_tree.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nDecision Tree Accuracy:", dt_accuracy)

print("\nDecision Tree Classification Report:")
print(classification_report(
    y_test,
    dt_predictions,
    target_names=iris.target_names
))


# --------------------------------------------------
# 4. Random Forest Model
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:", rf_accuracy)

print("\nRandom Forest Classification Report:")
print(classification_report(
    y_test,
    rf_predictions,
    target_names=iris.target_names
))


# --------------------------------------------------
# 5. Compare Models
# --------------------------------------------------

results = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [dt_accuracy, rf_accuracy]
})

print("\nModel Comparison:")
print(results)


# --------------------------------------------------
# 6. MLflow Experiment Tracking
# --------------------------------------------------

mlflow.set_experiment("Decision Tree and Random Forest")

with mlflow.start_run(run_name="Decision_Tree"):

    mlflow.log_param("model", "Decision Tree")
    mlflow.log_param("max_depth", 4)
    mlflow.log_metric("accuracy", dt_accuracy)

    mlflow.sklearn.log_model(
        decision_tree,
        "decision_tree_model"
    )


with mlflow.start_run(run_name="Random_Forest"):

    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", rf_accuracy)

    mlflow.sklearn.log_model(
        random_forest,
        "random_forest_model"
    )


print("\nExperiment tracking completed successfully.")