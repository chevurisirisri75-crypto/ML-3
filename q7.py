import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
classes = np.unique(y)
mask = (y == classes[0]) | (y == classes[1])
X = X[mask]
y = y[mask]
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
