import numpy as np
import matplotlib.pyplot as plt
from question4 import pca
from common import matrixUtility

def plotProjection(eigenvalues, eigenvectors):
    soa = np.array([[-2, -2,
             10* eigenvectors[0][0],
             10 * eigenvectors[1][0]]])
    X, Y, U, V = zip(*soa)
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)

def plotData(data, happy_face_index, sad_face_index):
    colors = np.zeros(len(happy_face_index) + len(sad_face_index))
    happy_class = []
    sad_class = []
    for j in happy_face_index:
        colors[j] = 0
        happy_class.append(data[j])
    happy_class = np.array(happy_class)

    for j in sad_face_index:
        colors[j] = 1
        sad_class.append(data[j])
    sad_class = np.array(sad_class)
    color= ['red' if c == 0 else 'green' for c in colors]
    return happy_class, sad_class

def plotProjectedData(data, happy_face_index, sad_face_index):
    colors = np.zeros(len(happy_face_index) + len(sad_face_index))
    for j in happy_face_index:
        colors[j] = 0

    for j in sad_face_index:
        colors[j] = 1
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    color= ['red' if c == 0 else 'green' for c in colors]
    ax.scatter(data,np.zeros(len(happy_face_index) + len(sad_face_index)), color=color)
    ax.set_title("Data plotting")

    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel("Feature1")

def apply():
    raw_data, happy_face_index, sad_face_index = pca.apply()
    print("************Raw Data**********\n", raw_data.shape)
    N = len(happy_face_index) + len(sad_face_index)
        #
        #We got the dimensionality redueced data.
        #raw_data is NXD data
        #
        #raw_data = scaler.fit_transform(raw_data)
        #raw_data = raw_data - raw_data.mean(axis=0)
        #centered_data = raw_data - np.average(raw_data, axis = 0)
    happy_class, sad_class = plotData(raw_data, happy_face_index, sad_face_index)
    covariance_T = matrixUtility.covMat(raw_data)
    covariance_happy_class = np.cov(happy_class.T)
    covariance_sad_class =  np.cov(sad_class.T)
    covariance_W = matrixUtility.addMatrix(covariance_happy_class, covariance_sad_class)
    diff = happy_class.mean(axis = 0) - sad_class.mean(axis=0)
    diff2 = np.array([diff])
    covariance_B = np.dot(diff.T, diff)
    eigenvectors = np.dot(np.linalg.inv(covariance_W), diff.T)
    principal1D = np.array([eigenvectors])/np.linalg.norm([eigenvectors], axis=0)
    projection = np.dot(principal1D, raw_data.T)
    plotProjectedData(projection, happy_face_index, sad_face_index)
