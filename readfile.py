import wave
import pylab
import os
import cmath
import math
import numpy as np
import matrixUtility

def decomposeFrequency(X, N, H):
    table = []
    D = 128
    window = 1
    while((window*H + N) < len(X) and window < 20):
        print("Window***********",window)
        row = []
        for r in range(D):
            #print("Row*********", r)
            sum = 0
            for c in range(window*H, window*H + N):
                #sum = sum + X[c]*cmath.exp(complex(0, ((-1)*2*math.pi*c*r)/N))
                sum = sum + X[c]*np.exp((-1j*2*math.pi*c*r)/N)
            row.append(sum)
        table.append(row)
        window = window + 1
    table = np.array(table)
    return table.T
def readwavefile(filename):
  """
  Reads a WAV file and returns all the frames converted to floats and the framerate.
  """
  assert os.path.exists(filename) and os.path.isfile(filename)
  wavefile = wave.open(filename, 'r')
  nframes = wavefile.getnframes()
  #print("nframes :", nframes)
  framerate = wavefile.getframerate()
  #print("framerate :", framerate)
  #print("channels :", wavefile.getnchannels())
  #print("sample width :", wavefile.getsampwidth())
  #time = nframes/framerate
  #print("Duration of file:", time)
  datatype = None
  # Zero converter!
  fconverter = lambda a : a
  if wavefile.getsampwidth() == 1:
      # 8-Bit format is unsigned.
      datatype = np.uint8
      fconverter = lambda a : ((a / 255.0) - 0.5) * 2
  elif wavefile.getsampwidth() == 2:
      # 16-Bit format is signed.
      datatype = np.int16
      fconverter = lambda a : a / 32767.0
  # Read and convert to float array
  frames = np.frombuffer(wavefile.readframes(nframes), dtype = datatype)
  #print("frames :", frames.shape)
  #print("actual frames :\n", frames)
  #frames = fconverter(np.asarray(frames, dtype = np.float64))
  #print("converted frames :", frames.shape)
  #print("actual frames2 :\n", frames)
  #frames = frames[:48000]
  #frames.shape = (160,300)
  #frames = frames[:, 1:]
  #print("frame shape before reshaping", frames.shape)
  #frames = frames[:, :298]
  #print("frame shape after reshaping", frames.shape)
  #print("actual frames: ", frames)
  #time_sequence = np.arange(0, time, 1/float(framerate))
  wavefile.close()
  decomposed_data = decomposeFrequency(frames, 400, 160)
  print("Decomposed data*********\n", decomposed_data.shape)
  #print("Decomposed data*********\n", decomposed_data)
  fft_magnitude = np.zeros((decomposed_data.shape[0], decomposed_data.shape[1]))
  #print(fft_magnitude)
  for r in range(decomposed_data.shape[0]):
      #print("Printing row :", r)
      m = np.abs(decomposed_data[r,:])
      row = np.log(m, out=np.zeros_like(m), where=(m!=0))
      #sum = 0
      #for c in range(col.shape[0]):
        #  sum = sum + col[c]
      #print(sum)
      #fft_magnitude[r] = math.log(sum)
      fft_magnitude[r] = row
  #print(fft_magnitude)
  #pylab.plot(time_sequence, fft_magnitude)
  #pylab.show()
  # This shouldn't go wrong...
  #assert frames.shape[0] == nframes
  matxUtility = matrixUtility.MatrixUtility()
  Sx = matxUtility.covMat(fft_magnitude.T)
  #Sx = np.cov(fft_magnitude)
  print("Sx dimension:",Sx.shape)
  eigenvalues, eigenvectors = np.linalg.eigh(Sx)
  print(eigenvalues)
  eigenvalues_diagonal = np.diag(1/np.sqrt(eigenvalues))
  print(eigenvalues_diagonal)
  whiteningTransformation = np.dot(eigenvalues_diagonal, eigenvectors)
  print(whiteningTransformation)

  noisyFile = wave.open("noisy.wav", 'r')
  noisyNFrames = noisyFile.getnframes()
  noisyFrameRate = noisyFile.getframerate()
  noisyFrames = np.frombuffer(noisyFile.readframes(noisyNFrames), dtype = datatype)
  noisyFile.close()
  noisyDecomposedData = decomposeFrequency(noisyFrames, 400, 160)
  print("Noisy Decomposed data*********\n", noisyDecomposedData)
  #print("Decomposed data*********\n", decomposed_data)
  noisy_fft_magnitude = np.zeros((noisyDecomposedData.shape[0], noisyDecomposedData.shape[1]))
  #print(fft_magnitude)
  for r in range(noisyDecomposedData.shape[0]):
      #print("Printing row :", r)
      m = np.abs(noisy_fft_magnitude[r,:])
      row = np.log(m, out=np.zeros_like(m), where=(m!=0))
      #sum = 0
      #for c in range(col.shape[0]):
        #  sum = sum + col[c]
      #print(sum)
      #fft_magnitude[r] = math.log(sum)
      noisy_fft_magnitude[r] = row
  print("Noisy :\n", noisy_fft_magnitude)
  transformed_noisy_data = np.dot(whiteningTransformation, noisy_fft_magnitude)
  print(transformed_noisy_data)
  #print(Sx)
  return frames, framerate

if __name__ == '__main__':
  frames, fps = readwavefile("noisy.wav")
  t = np.arange(0, frames.shape[0], 1)
  pylab.plot(t, frames)
  #pylab.show()
