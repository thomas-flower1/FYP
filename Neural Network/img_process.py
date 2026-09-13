'''
Function to take a file an convert it into a vector grayscale vector 
This vector is what is used to train the neural network.
Creates a new class that containts the data and the correct labbel for training
'''

import numpy as np
from PIL import Image

WIDTH = 64
HEIGHT = 64

def validate(rel_path: str):
    img = Image.open(rel_path).convert("L") # convert the image to grayscale   
    data = np.array(img.getdata())

   
    width, height = img.size

    matrix = []

    index = 0
    for _ in range(height):
        row = []
        for _ in range(width):
            row.append(data[index])
            index += 1
        matrix.append(row)

    for row in matrix:
        for col in row:
            if col > 128:
                print(' ', end='')
            else:
                print('1', end='')

        print('\n')

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
            # fn = f"data_set/cats_set/cat.{i}.jpg"
            fn = f"data_set/dogs_set/dog.{i}.jpg"
            img = Image.open(fn).convert("L")

            # Resizing the image 
            resize = img.resize((WIDTH, HEIGHT))

            data = list(resize.getdata()) # This is the list of grayscale values
            wf.write(','.join(str(num) for num in data))
            wf.write("\n")
            

### WRITING THE DATA
# process("dog.csv")






