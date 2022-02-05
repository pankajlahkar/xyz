import audioprocessor
import stftprocessor
import dimensionalityreduction
import numpy as np

def handleFile(filename):
    #
    #We are given a window size of 25ms and hop size of 10 ms
    #We will convert it to number by multplying with frame rate
    #Total frequency window is given as 128
    #We are going to restric windows to 298
    #
    windowSize = 0.025
    hopSize = 0.010
    totalFrequencySize = 128
    windowLimit = 1
    frames, frameRate, frameCount = audioprocessor.readAudioFile(filename)
    #
    #Clean audio file detail
    #
    print("File:", filename)
    print("frames:", frames)
    print("frameRate:", frameRate)
    print("frameCount:", frameCount)
    #
    #Now we are going to do short time fourier transformation
    #
    windowSize = windowSize*frameRate
    hopSize = hopSize*frameRate
    stft_data = stftprocessor.stft(frames,int(windowSize), int(hopSize), totalFrequencySize, windowLimit + 1)
    print("stft data:",stft_data.shape)
    logged_stft_data = stftprocessor.applyLog(stft_data)
    print("Logged stft_data:", logged_stft_data.shape)
    stftprocessor.generateSpectogram(logged_stft_data, int(frameRate), int(hopSize))
    return logged_stft_data
def apply():

    clean_logged_stft_data = handleFile(cleanAudioFile)
    whiteningTransformation = dimensionalityreduction.generateWhiteningTransform(clean_logged_stft_data)
    print("Whitening Transform:", whiteningTransformation.shape)
    transformed_clean_data = np.dot(whiteningTransformation, clean_logged_stft_data)
    noisy_logged_stft_data = handleFile(noisyAudioFile)
    transformed_noisy_data = np.dot(whiteningTransformation, noisy_logged_stft_data)
    cleanAudioAverage = dimensionalityreduction.averageAbsouluteNonDiagonalEntries(transformed_clean_data)
    print("Clean Audio Average :", cleanAudioAverage)
    noisyAudioAverage = dimensionalityreduction.averageAbsouluteNonDiagonalEntries(transformed_noisy_data)
    print("Noisy Audio Average :", noisyAudioAverage)
