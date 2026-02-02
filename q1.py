import numpy as np
import pandas as pd
data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
X = data.iloc[:, :-1].values#extracting only feature 
A = X[0]
B = X[1]
def dot_product(A, B):#product of 2 vectors
    return sum(a * b for a, b in zip(A, B))
def euclidean_norm(A):#function that is created to compute the eucledean norm
    return (sum(a * a for a in A)) ** 0.5
if __name__ == "__main__":

    own_dot = dot_product(A, B)
    numpy_dot = np.dot(A, B)

    own_norm = euclidean_norm(A)
    numpy_norm = np.linalg.norm(A)

    print("Own Dot Product:", own_dot)
    print("NumPy Dot Product:", numpy_dot)

    print("Own Norm:", own_norm)
    print("NumPy Norm:", numpy_norm)
