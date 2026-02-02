import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
# Train kNN classifier (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
#computing the confusion matrix
def confusion_matrix_own(y_true, y_pred):
    TP = TN = FP = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1

    return np.array([[TN, FP],[FN, TP]])
# Function to compute accuracy
def accuracy(cm):
    return (cm[0,0] + cm[1,1]) / np.sum(cm)
# Function to compute precision
def precision(cm):
    return cm[1,1] / (cm[0,1] + cm[1,1])
def recall(cm):
    return cm[1,1] / (cm[1,0] + cm[1,1])
def f1_score(p, r):
    return 2 * p * r / (p + r)
if __name__ == "__main__":
    # Predictions on test data
    y_pred = knn.predict(X_test)
    # Compute confusion matrix
    cm = confusion_matrix_own(y_test, y_pred)
    # Computing the performance metrics
    acc = accuracy(cm)
    prec = precision(cm)
    rec = recall(cm)
    f1 = f1_score(prec, rec)

    print("Confusion Matrix:\n", cm)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
