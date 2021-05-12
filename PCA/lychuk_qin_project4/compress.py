import numpy as np
import matplotlib.pyplot as plt
import os


def load_data(input_dir):

    idx = 0
    for filename in os.listdir(input_dir):
        img = plt.imread(os.path.join(input_dir, filename)).astype('float')
        matrix = np.reshape(img, (-1, 1))  # takes an image and flattens it to 1 column (i think)
        if idx == 0:  # initialize a new array
            data = matrix
        else:
            data = np.hstack((data, matrix))  # adding new columns

        idx = idx + 1
    # print(np.shape(data))  # gets the shape to check if correct
    return data

print(load_data('Data/Train/'))