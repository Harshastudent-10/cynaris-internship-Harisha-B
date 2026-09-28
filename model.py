import os
import pickle
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    """Trains a simple Random Forest classifier on the Iris dataset."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[Training Complete] Model Accuracy: {acc:.4f}")
    return model, X_test, y_test

def serialize_with_pickle(model, filename="model_pickle.pkl"):
    """Serializes the trained model using Python's built-in pickle module."""
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"[Pickle] Saved model successfully to '{filename}'.")

def deserialize_with_pickle(filename="model_pickle.pkl"):
    """Deserializes and loads the model using pickle."""
    with open(filename, "rb") as f:
        loaded_model = pickle.load(f)
    print(f"[Pickle] Loaded model successfully from '{filename}'.")
    return loaded_model

def serialize_with_joblib(model, filename="model_joblib.joblib"):
    """Serializes the trained model using Joblib (optimized for NumPy arrays)."""
    joblib.dump(model, filename)
    print(f"[Joblib] Saved model successfully to '{filename}'.")

def deserialize_with_joblib(filename="model_joblib.joblib"):
    """Deserializes and loads the model using Joblib."""
    loaded_model = joblib.load(filename)
    print(f"[Joblib] Loaded model successfully from '{filename}'.")
    return loaded_model

def run_tests():
    """Runs basic validation checks on serialization and prediction consistency."""
    # Step 1: Train
    original_model, X_test, y_test = train_model()
    original_preds = original_model.predict(X_test)

    # Step 2: Pickle Serialization & Test
    pickle_file = "model_pickle.pkl"
    serialize_with_pickle(original_model, pickle_file)
    pickle_model = deserialize_with_pickle(pickle_file)
    pickle_preds = pickle_model.predict(X_test)
    assert np.array_equal(original_preds, pickle_preds), "Pickle predictions do not match!"
    print("[Test Passed] Pickle model predictions match original model.")

    # Step 3: Joblib Serialization & Test
    joblib_file = "model_joblib.joblib"
    serialize_with_joblib(original_model, joblib_file)
    joblib_model = deserialize_with_joblib(joblib_file)
    joblib_preds = joblib_model.predict(X_test)
    assert np.array_equal(original_preds, joblib_preds), "Joblib predictions do not match!"
    print("[Test Passed] Joblib model predictions match original model.")

    # Clean up generated files
    if os.path.exists(pickle_file):
        os.remove(pickle_file)
    if os.path.exists(joblib_file):
        os.remove(joblib_file)
    print("\nAll unit tests passed successfully!")

if __name__ == "__main__":
    run_tests()