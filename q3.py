import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
feature = data.iloc[:, 0].values
def calculate_mean(X):
    return np.mean(X)
def calculate_variance(X):# Function to calculate variance of a feature
    return np.var(X)
def plot_histogram(X, bins):# Function to plot histogram of a feature
    plt.hist(X, bins=bins)
    plt.xlabel("Feature Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Selected Feature")
    plt.show()
if __name__ == "__main__":
    mean_value = calculate_mean(feature)
    variance_value = calculate_variance(feature)
    plot_histogram(feature, bins=10)
    print("Mean of the feature:", mean_value)
    print("Variance of the feature:", variance_value)
