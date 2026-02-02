import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
classes = np.unique(y)
mask = (y == classes[0]) | (y == classes[1])
X = X[mask]
y = y[mask]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
def euclidean_distance(A, B):
    return np.sqrt(np.sum((A - B) ** 2))
def knn_predict(X_train, y_train, test_point, k):
    distances = []

    for i in range(len(X_train)):
        d = euclidean_distance(X_train[i], test_point)
        distances.append((d, y_train[i]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    labels = [label for _, label in neighbors]
    return max(set(labels), key=labels.count)
def knn_classifier(X_train, y_train, X_test, k):
    return np.array([knn_predict(X_train, y_train, x, k) for x in X_test])
if __name__ == "__main__":
    k_values = range(1, 12)
    accuracies = []
    for k in k_values:
        predictions = knn_classifier(X_train, y_train, X_test, k)
        cm = confusion_matrix(y_test, predictions)
        accuracy = np.trace(cm) / np.sum(cm)
        accuracies.append(accuracy)
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("k vs Accuracy for kNN Classifier")
    plt.grid(True)
    plt.show()
