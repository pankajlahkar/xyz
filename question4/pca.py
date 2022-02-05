import numpy as np
import matplotlib.pyplot as plt
from question4 import giffilereader
from common import matrixUtility
from sklearn.preprocessing import StandardScaler

def calculateResidualError(eigenvalues):
    N = np.zeros(len(eigenvalues))
    for i in range(0, len(eigenvalues)):
        total = 0
        for j in range(i, len(eigenvalues)):
            total += eigenvalues[j]
        N[i] = total
    return N
def plotResidualError(eigenvalues):
    fig, ax = plt.subplots(figsize=(12, 6))
    N = np.zeros(len(eigenvalues))
    for i in range(0, len(eigenvalues)):
        N[i] = i
    ax.plot(N, calculateResidualError(eigenvalues))
    ax.set_title("Residual Error plot")
    ax.set_xlabel("i")
    ax.set_ylabel("Residual Error")
    ax.set_yticks([0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xticks(N)
def plotEigenValues(eigenvalues):
    fig, ax = plt.subplots(figsize=(12, 6))
    N = np.zeros(len(eigenvalues))
    for i in range(1, len(eigenvalues)):
        N[i] = i
    ax.plot(N, eigenvalues)
    ax.set_title("Eigen value plot")

    #ax.set_yticks([0,10, 20, 30, 40, 50, 60])
    ax.set_yticks([0,1000, 2000, 3000])
    ax.set_xticks(N)

    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2,2))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("i")
    ax.set_ylabel("eigen value")

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=1)
def apply():
    path = "Data\emotion_classification\\train"
    #
    #raw_data is NXD matrix
    #
    raw_data, happy_index, sad_index = giffilereader.readFile(path)
    #print("************Raw Data**********\n", raw_data.shape)

    N = raw_data.shape[0]
    D = raw_data.shape[1]
    #
    #direction_matrix is a NXD matrix
    #
    direction_matrix = raw_data - raw_data.mean(axis=0)
    direction_covariance_matrix = matrixUtility.covMat(raw_data.T)
    eigenvalues_u, eigenvectors_u = np.linalg.eigh(direction_covariance_matrix/N)

    eigenvalues_v = eigenvalues_u
    #
    #This is DXN eigen vector matrix
    #
    eigenvectors_v = np.dot(direction_matrix.T, eigenvectors_u)
    idx = eigenvalues_v.argsort()[::-1]
    #print("index =\t",idx)
    eigenvalues_v = eigenvalues_v[idx]
    eigenvectors_v = eigenvectors_v[:,idx]
    eigenvalues_diagonal = np.diag(1/np.sqrt(eigenvalues_v*N))
    eigenvectors_v = np.dot(eigenvectors_v, eigenvalues_diagonal)
    #eigenvectors_v = eigenvectors_v/np.linalg.norm(eigenvectors_v, axis=0)

    plotEigenValues(eigenvalues_v)
    plotResidualError(eigenvalues_v)
    #eigenvectors_d_dimension = eigenvectors_u[:2, :]
    #print(eigenvectors_v.shape)
    #print(eigenvectors_d_dimension.T.shape)
    #print(raw_data.T.shape)
    #
    #Here the sample is in column and dimension is in row
    #
    #eigenvectors_v = eigenvectors_v[:, :8]
    raw_data_reduced = np.dot(eigenvectors_v.T, raw_data.T)
    #
    #Revert the samples to row and dimension to column
    #
    raw_data_reduced = raw_data_reduced.T
    #
    #Now reduce the matrix
    #
    #raw_data_reduced = raw_data_reduced[:,:2]
    return raw_data_reduced, happy_index, sad_index
