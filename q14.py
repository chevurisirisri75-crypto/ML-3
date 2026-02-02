import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
# Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
classes = np.unique(y)
mask = (y == classes[0]) | (y == classes[1])
X = X[mask]
y = y[mask]
y_binary = np.where(y == classes[0], 0, 1)
X_bias = np.c_[np.ones(X.shape[0]), X]
X_train, X_test, y_train, y_test = train_test_split(
    X_bias, y_binary, test_size=0.3, random_state=42
)
def matrix_inversion_classifier(X_train, y_train):
    return np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
def predict_matrix_classifier(X_test, weights):
    y_pred = X_test @ weights
    return np.where(y_pred >= 0.5, 1, 0)
if __name__ == "__main__":
    weights = matrix_inversion_classifier(X_train, y_train)
    y_pred_matrix = predict_matrix_classifier(X_test, weights)
    matrix_accuracy = np.mean(y_pred_matrix == y_test)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train[:, 1:], y_train)  # remove bias for kNN
    y_pred_knn = knn.predict(X_test[:, 1:])
    knn_accuracy = np.mean(y_pred_knn == y_test)

    print("Matrix Inversion Accuracy:", matrix_accuracy)
    print("kNN Accuracy:", knn_accuracy)
