'''
Function to take a file an convert it into a vector grayscale vector 
This vector is what is used to train the neural network.
Creates a new class that containts the data and the correct labbel for training
'''

import numpy as np
from PIL import Image

WIDTH = 128
HEIGHT = 128

def process(filename: str) -> None:
    '''
    Given a filenameto write to, this function will:
    - Crop the image
    - Turn the image grayscale
    - Convert this data to a (1,) arr
    - write this to a file

    Note: This was written to process the images formatted "cat.n.jpg" etc
    Note2: Writes to a csv file. The data can be read using the complimentary "load" function
    '''
    START_INDEX = 4001
    END_INDEX = 4500

    with open(filename, "w") as wf: # Outside of the loop so we don't constantly open the file
        for i in range(START_INDEX, END_INDEX):
            fn = f"data_set/cats_set/cat.{i}.jpg"
            # fn = f"data_set/dogs_set/dog.{i}.jpg"
            img = Image.open(fn).convert("L")

            # Resizing the image 
            resize = img.resize((WIDTH, HEIGHT))

            data = list(resize.getdata()) # This is the list of grayscale values
            wf.write(','.join(str(num) for num in data))
            wf.write("\n")
            

class Data:
    def __init__(self, data, target):
        self._data: list = data
        self._target: list = target


    def get_data(self):
        return self._data

    def get_target(self):
        return self._target


def read(filename: str, hot_encoded: list) -> list:
    '''Takes in the csv filename and returns a list of Data objects'''

    ### hot encoded is either [0, 1] or [1, 0 ] for our uses case
    sol = []
    with open(filename, "r") as rf:
        for line in rf:
            arr =  arr = [int(x) / 255 for x in line.rstrip().split(",")]
            data = Data(arr, hot_encoded)
            sol.append(data)

    return sol



### TESTING
# data = read("cat_dog_training_data/cat.csv", [0, 1])



### WRITING THE DATA
# process("dog.csv")
process("cat.csv")







