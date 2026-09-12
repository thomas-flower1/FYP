'''
Function to take a file an convert it into a vector grayscale vector 
This vector is what is used to train the neural network.
Creates a new class that containts the data and the correct labbel for training
'''

from enum import Enum

import numpy as np
from PIL import Image


class Classification(Enum):
    DOG = 0
    CAT = 1


class Data:
    def __init__(self, data: np.array, classification: Classification):
        self._data: np.array = data
        self._classification: Classification = classification

    def get_data(self) -> np.array:
        return self._data

    def set_data(self, vector: np.array) -> None:
        # TODO need to make sure that the shape is correct
        self._data = vector


    def get_classification(self) -> Classification:
        return self._classification

    def set_classification(self, cls: Classification):
        self._classification = cls


def get_img_vector(rel_path: str, cls: Classification) -> Data:
    '''
    Takes in the relative path and returns the grayscale vector of the image 
    This provides a single input to the nn
    '''

    img = Image.open(rel_path).convert("L") # convert the image to grayscale   
    data = np.array(img.getdata())

    return Data(data, cls)


    ### VALIDATION ###
    # WIDTH, HEIGHT = img.size

    # matrix = []

    # index = 0
    # for _ in range(HEIGHT):
    #     row = []
    #     for _ in range(WIDTH):
    #         row.append(data[index])
    #         index += 1
    #     matrix.append(row)

    # for row in matrix:
    #     for col in row:
    #         if col > 128:
    #             print(' ', end='')
    #         else:
    #             print('1', end='')

    #     print('\n')


### TODO: Need to do some iamge cleanup ###
### TODO: Need to move the image to a standalone function ###

data = get_img_vector("eggs.png", Classification.DOG)
print(type(data.get_classification()))

