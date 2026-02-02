import numpy as np
import pandas as pd
data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
classes = np.unique(y)
class1 = X[y == classes[0]]
class2 = X[y == classes[1]]
def mean_vector(X):#calculate mean vector of a dataset
    return np.mean(X, axis=0)
def variance_vector(X):#calculate variance vector of a dataset
    return np.var(X, axis=0)
#calculate standard deviation vector of a dataset
def std_vector(X):
    return np.std(X, axis=0)
#calculate Euclidean distance between two vectors
def euclidean_distance(A, B):
    return np.linalg.norm(A - B)
if __name__ == "__main__":

    # Calculate class centroids (mean vectors)
    mean_class1 = mean_vector(class1)
    mean_class2 = mean_vector(class2)

    # Calculate spread (standard deviation) for each class
    std_class1 = std_vector(class1)
    std_class2 = std_vector(class2)

    # Calculate interclass distance between class centroids
    interclass_dist = euclidean_distance(mean_class1, mean_class2)

    print("Mean Vector of Class 1:\n", mean_class1)
    print("Mean Vector of Class 2:\n", mean_class2)

    print("\nStandard Deviation of Class 1:\n", std_class1)
    print("Standard Deviation of Class 2:\n", std_class2)

    print("\nInterclass Distance:", interclass_dist)
