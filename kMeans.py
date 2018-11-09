# Author: Snehith Raj bada

import pandas as pd
import numpy as np
from copy import deepcopy
from collections import Counter


# Standardize the data(zero-mean and standard deviation = 1)
def standardization(df):
    for column in df:
        x = pd.DataFrame(data=df[column])
        x -= np.mean(x)
        x /= np.std(x)
        df[column] = x
    return df

# Implementing K-means on X:dataset with clusters:k
def mykmeans(X, k):
    X = X.values # Converting dataframe to numpy
    n = X.shape[0] # n gives number of objects
    c = X.shape[1] # c gives number of attributes

    # Generate random centers, using sigma and mean to ensure it represent the whole data
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    centers = np.random.randn(k, c) * std + mean
    print("Centers=", centers)

    # Initializing old_centres, new_centers, clusters and distances with default zeroes
    old_centers = np.zeros(centers.shape)
    new_centers = deepcopy(centers)
    clusters = np.zeros(n)
    distances = np.zeros((n, k))

    error = np.linalg.norm(new_centers - old_centers)

    # When error between new_centers and old_centers is zero stop updating centers
    while error != 0:

        # Measure the distance of every element in data set to every center
        for i in range(k):
            distances[:, i] = np.linalg.norm(X - centers[i], axis=1)

        # Assign elements to their closest center
        clusters = np.argmin(distances, axis=1)

        # Update old_centers into new_centers
        old_centers = deepcopy(new_centers)

        # Calculate mean for every cluster and update the center
        for i in range(k):
            new_centers[i] = np.mean(X[clusters == i], axis=0)
        error = np.linalg.norm(new_centers - old_centers)

    print("Centers New=", new_centers, new_centers.shape[1])
    print("Clusters=", clusters)
    print(Counter(clusters))


if __name__ == "__main__":

    def datacleaning(df):
        df = standardization(df)
        k = int(input("Enter k value\n"))
        mykmeans(df, k)

    # Column names for the dataset
    col_names = ['Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA',
                 '2P%',
                 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
    dataframe = pd.read_csv('NBAstats.csv', names=col_names, skiprows=1)  # Store the csv data into dataframe
    df = dataframe


    # Data selection and creating datasets(Training,Testing)
    def datacleaning(df):
        df = standardization(df)
        k = int(input("Enter k value\n"))
        mykmeans(dfk, k)

    c = int(input(
        "Select one of the following for Kmeans\n2 : Use all features except team\nor\n3 : Use the following set of attributes {2P%, 3P%, FT%, TRB, AST, STL, BLK}\n"))

    # 2  Use all features except team and group the players into k clusters
    if (c == 2):
        new_col = ['Pos', 'Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
                   'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
        dfk = dataframe[new_col]
        dfk["Pos"] = pd.Categorical(dfk["Pos"])
        dfk["Pos"] = dfk["Pos"].cat.codes
        datacleaning(dfk)

    # 3 Use the following set of attributes {2P%, 3P%, FT%, TRB, AST, STL, BLK} to perform k-means clustering.
    if (c == 3):
        new_col = ['2P%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK']
        dfk = dataframe[new_col]
        datacleaning(dfk)
