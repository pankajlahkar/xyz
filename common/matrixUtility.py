import numpy as np
def calcCov(x, y):
    mean_x, mean_y = x.mean(), y.mean()
    n = len(x)

    return sum((x - mean_x) * (y - mean_y)) / n


# calculates the Covariance matrix
def covMat(data):

    # get the rows and cols
    print(data.shape)
    rows, cols = data.shape

    # the covariance matroix has a shape of n_features x n_features
    # n_featurs  = cols - 1 (not including the target column)
    cov_mat = np.zeros((cols, cols))

    for i in range(cols):
        for j in range(cols):
            # store the value in the matrix
            cov_mat[i][j] = calcCov(data[:, i], data[:, j])
    return cov_mat

def addMatrix(matrixA, matrixB):
    table = []
    for r in range(len(matrixA)):
        row = []
        for c in range(len(matrixA[0])):
            row.append(matrixA[r][c] + matrixB[r][c])
        table.append(row)
    table = np.array(table)
    return table
