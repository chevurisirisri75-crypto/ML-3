import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
classes = np.unique(y)
mask = (y == classes[0]) | (y == classes[1])
X = X[mask]
y = y[mask]

# spliting it into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Train kNN classifier when (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
if __name__ == "__main__":

    # Predictions on a training data
    y_train_pred = knn.predict(X_train)
    # Predictions on testing data
    y_test_pred = knn.predict(X_test)
    # Confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    # Training metrics
    print("Training Confusion Matrix:\n", cm_train)
    print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Training Precision:", precision_score(y_train, y_train_pred))
    print("Training Recall:", recall_score(y_train, y_train_pred))
    print("Training F1 Score:", f1_score(y_train, y_train_pred))
    # Testing metrics
    print("\nTesting Confusion Matrix:\n", cm_test)
    print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Testing Precision:", precision_score(y_test, y_test_pred))
    print("Testing Recall:", recall_score(y_test, y_test_pred))
    print("Testing F1 Score:", f1_score(y_test, y_test_pred))
