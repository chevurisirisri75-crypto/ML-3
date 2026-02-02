import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
X = data.iloc[:, :-1].values
A = X[0]
B = X[1]
def minkowski_distance(A, B, p):
    return (sum(abs(a - b) ** p for a, b in zip(A, B))) ** (1 / p)
if __name__ == "__main__":
    p_values = range(1, 11)
    distances = []
    for p in p_values:
        dist = minkowski_distance(A, B, p)
        distances.append(dist)
    plt.plot(p_values, distances, marker='o')
    plt.xlabel("Value of p")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance vs p")
    plt.grid(True)
    plt.show()
