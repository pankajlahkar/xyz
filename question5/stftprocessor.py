import wave
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import math

def stft(allFrames, shortWindowSize, hopSize, totalFrequencySize, windowLimit):
    stftData = []
    window = 1
    counter = 0
    while((window*hopSize + shortWindowSize) < len(allFrames) and window < windowLimit):
        if counter == 0:
            print("Window***********",window, "**********Done")
        counter = window % 25
        frequencyIntensity = []
        for r in range(totalFrequencySize):
            sum = 0
            for c in range(window*hopSize, window*hopSize + shortWindowSize):
                sum = sum + allFrames[c]*np.exp((-1j*2*math.pi*c*r)/shortWindowSize)
            frequencyIntensity.append(sum)
        stftData.append(frequencyIntensity)
        window = window + 1
    stftData = np.array(stftData)
    return stftData.T


def applyLog(stftData):
    stftDataMagnitude = np.zeros((stftData.shape[0], stftData.shape[1]))
    for r in range(stftData.shape[0]):
        absoluteSTFTData = np.abs(stftData[r,:])
        row = np.log(absoluteSTFTData, out=np.zeros_like(absoluteSTFTData), where=(absoluteSTFTData!=0))
        stftDataMagnitude[r] = row
    return stftDataMagnitude

def generateSpectogram(stftLogData, sampleRate, hopSize, y_axis="log"):
    plt.figure(figsize=(15, 6))
    librosa.display.specshow(stftLogData, sr=sampleRate, hop_length = hopSize, x_axis = "time", y_axis = y_axis)
    plt.colorbar(format="%+2.f")
