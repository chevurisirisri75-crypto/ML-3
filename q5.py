import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski
data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
X = data.iloc[:, :-1].values
A = X[0]
B = X[1]
def minkowski_distance(A, B, p):
    return (sum(abs(a - b) ** p for a, b in zip(A, B))) ** (1 / p)
if __name__ == "__main__":
    p = 3  
    own_distance = minkowski_distance(A, B, p)
    scipy_distance = minkowski(A, B, p)

    print("Own Minkowski Distance (p=3):", own_distance)
    print("SciPy Minkowski Distance (p=3):", scipy_distance)
