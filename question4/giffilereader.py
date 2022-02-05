import os
import numpy as np
from skimage.io import imread

def readFile(directory):
    path = directory
    raw_data = []
    happy_index = []
    sad_index = []
    count = 0
    for img in os.listdir(path):
        image_array = imread(os.path.join(path, img))
        if(".happy." in img):
            happy_index.append(count)
        else:
            sad_index.append(count)
        raw_data.append(image_array.flatten())
        count = count + 1
    raw_data = np.array(raw_data)
    happy_index = np.array(happy_index)
    sad_index = np.array(sad_index)
    return raw_data, happy_index, sad_index
