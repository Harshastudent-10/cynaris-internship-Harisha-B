# SVM and KNN Classification
# Import required libraries

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = load_iris()

X = data.data
y = data.target


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------------
# SVM MODEL
# -------------------------

svm_model = SVC(kernel="rbf", random_state=42)

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)

print("SVM Accuracy:", svm_accuracy)
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))


# -------------------------
# KNN MODEL
# -------------------------

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)

print("KNN Accuracy:", knn_accuracy)
print("\nKNN Classification Report:")
print(classification_report(y_test, knn_predictions))


# -------------------------
# MODEL COMPARISON
# -------------------------

print("\nModel Comparison")
print("----------------")
print("SVM Accuracy:", svm_accuracy)
print("KNN Accuracy:", knn_accuracy)