import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski

# Load the dataset
data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Extract feature vectors
X = data.iloc[:, :-1].values

# Select any two feature vectors
A = X[0]
B = X[1]

# Own function to calculate Minkowski distance
def minkowski_distance(A, B, p):
    return (sum(abs(a - b) ** p for a, b in zip(A, B))) ** (1 / p)


# =========================
# MAIN PROGRAM
# =========================
if __name__ == "__main__":

    p = 3  # Example value of p

    own_distance = minkowski_distance(A, B, p)
    scipy_distance = minkowski(A, B, p)

    print("Own Minkowski Distance (p=3):", own_distance)
    print("SciPy Minkowski Distance (p=3):", scipy_distance)
