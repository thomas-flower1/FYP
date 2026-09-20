'''
Function to take a file an convert it into a vector grayscale vector 
This vector is what is used to train the neural network.
Creates a new class that containts the data and the correct labbel for training
'''

from PIL import Image
import numpy as np

WIDTH = 256
HEIGHT = 256

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
        for i in range(START_INDEX, END_INDEX + 1):
            fn = f"data_set/cats_set/cat.{i}.jpg"
            # fn = f"data_set/dogs_set/dog.{i}.jpg"
            img = Image.open(fn).convert("L")

            # Resizing the image 
            resize = img.resize((WIDTH, HEIGHT))
            data = list(resize.getdata()) # This is the list of grayscale values
            if len(data) != WIDTH * HEIGHT:
                raise Exception("READ ERROR")


            wf.write(','.join(str(num) for num in data))
            # DONT PRINT A NEWLINE AT THE END OF THE FILE - CAUSED ISSUES FOR TRAINING
            if i != END_INDEX:
                wf.write("\n")
            

class Data:
    def __init__(self, data, target):
        self._data: np.ndarray = data
        self._target: np.ndarray = target

    def get_data(self):
        return self._data

    def get_target(self):
        return self._target


def read(filename: str, hot_encoded: list) -> list:
    '''
    Takes in the csv filename and returns a list of Data objects

    # NOTE SHOULD NOT HAVE USED LISTS IN THE ORIGINAL IMPLEMENTATION
    '''

    ### hot encoded is either [0, 1] or [1, 0 ] for our uses case
    sol = []
    with open(filename, "r") as rf:
        for line in rf:
            arr = np.ndarray([int(x) / 255 for x in line.rstrip().split(",")])
            data = Data(arr, hot_encoded)
            sol.append(data)

    return sol


### WRITING THE DATA
if __name__ == "__main__":
    pass
    # process("dog.csv")
    # process("cat.csv")

    ### TESTING WHAT WOULD A RESONABLE IMAGE SIZE CONSIST OF ###
    # lengths = []
    # for i in range(4001, 4500, 1):
    #     sample = f"data_set/cats_set/cat.{i}.jpg"
    #     img = Image.open(sample).convert("L")
    #     lengths.append(len(img.getdata()))

    # avg = round(sum(lengths) / len(lengths))
    # print(f"The average number of pixels is: {avg}")
    # print(f"We are resizing to {WIDTH} X {HEIGHT} which is {WIDTH * HEIGHT} pixels")
    # print(f"This is {round(WIDTH * HEIGHT / avg * 100)}% of the pixels of the original image")








