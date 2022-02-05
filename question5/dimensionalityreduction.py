import sys
sys.path.insert(0, '/common/')
import numpy as np
from common import matrixUtility
def generateWhiteningTransform(stftLoggedData):
      Sx = matrixUtility.covMat(stftLoggedData.T)
      eigenvalues, eigenvectors = np.linalg.eigh(Sx)
      #print(eigenvalues)
      eigenvalues_diagonal = np.diag(1/np.sqrt(eigenvalues))
      #print(eigenvalues_diagonal)
      return np.dot(eigenvalues_diagonal, eigenvectors)

def averageAbsouluteNonDiagonalEntries(data):
    covarianceMatrix = matrixUtility.covMat(data.T)
    absoluteCovarianceMatrix = np.abs(covarianceMatrix)
    trace = np.trace(absoluteCovarianceMatrix)
    total = np.sum(absoluteCovarianceMatrix)
    average = (total - trace)/data.shape[0]*(data.shape[1] - 1)
    return average
