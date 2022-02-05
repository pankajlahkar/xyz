import wave
import numpy as np

def readAudioFile(filename):
  file = wave.open(filename, 'r')
  frameCount = file.getnframes()
  frameRate = file.getframerate()
  dataType = None
  if file.getsampwidth() == 1:
      dataType = np.uint8
  elif file.getsampwidth() == 2:
      dataType = np.int16
  frames = np.frombuffer(file.readframes(frameCount), dtype = dataType)
  file.close()
  return frames, frameRate, frameCount
