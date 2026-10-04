"""
img.py

Reads image data, converts to a grayscale and writes to a csv file. Contains functionality to
read this csv data and creates a corresponding Data object.

Takes two command line arguments:
    - Name of the output file
    - Pathname to the dataset directory

"""

from PIL import Image
import numpy as np
import sys
import re
import os

WIDTH = 64
HEIGHT = 64


class Data:
    """Data class that holds an image's grayscale vector and the corresponding one hot encoded vector

    Args:
        data (np.ndarray) : Grayscale vector of the image (n,)
        target (np.ndarray) : One-hot-encoded vector (2,)

    """

    def __init__(self, data, target):
        self._data: np.ndarray = data
        self._target: np.ndarray = target

    def get_data(self):
        return self._data

    def get_target(self):
        return self._target


def process(filename: str, dir: str) -> None:
    """Creates and writes image data into a new csv file. Filename is specified by the user
    and is located in the current repository

    Args:
        filename (str) : The name of the file to write to, must contain the word "cat" or "dog"

    Returns:
        n/a

    """
    img_names = os.listdir(dir)
    SIZE = len(img_names)
    with open(filename, "w") as wf:
        for index, name in enumerate(img_names):
            path = f"{dir}/{name}"
            img = Image.open(path).convert("L")
            resize = img.resize((WIDTH, HEIGHT))
            data = list(
                resize.getdata()
            )  # Returns a 1D list with all the grayscale values of a single image

            if len(data) != WIDTH * HEIGHT:
                raise Exception("READ ERROR")
            wf.write(",".join(str(num) for num in data))

            if index != SIZE - 1:
                wf.write("\n")


def process_single(filepath, target) -> Data:
    """Helper functions for \"user_defined_image\""""

    img = Image.open(filepath).convert("L")
    resize = img.resize((WIDTH, HEIGHT))
    data = list(
        resize.getdata()
    )  # Returns a 1D list with all the grayscale values of a single image

    if len(data) != WIDTH * HEIGHT:
        raise Exception("READ ERROR")

    return Data(data, target)


def read(filename: str, target: np.ndarray) -> list:
    """Reads a full csv file and loads into memory a Data list

    Args:
        filename (str) : name of the csv to read from
        hot_encoded (np.ndarray) : one-hot-encoded target (either [1, 0] or [0, 1])

    Returns:
        list : A list of data objects

    """

    sol = []
    with open(filename, "r") as rf:
        for line in rf:
            arr = np.array([int(x) / 255 for x in line.rstrip().split(",")])
            data = Data(arr, target)
            sol.append(data)

    return sol


def user_defined_image() -> Data:
    """Getting user specified image and returning an array to pass into the nn
    Assumption: Cat is hot encoded to [1,0] and dog is [0,1]
    *Usage: To be called in the NN program*

    Args:
        n/a

    Returns:
        np.ndarray: Grayscale vector of the image, shape is (n,)

    Raises:
        System Exit: Too many command line arguments

    """
    cat_hot_encoded = [1, 0]
    dog_hot_encoded = [0, 1]

    # Checking command line arguments
    if len(sys.argv) != 2:
        raise SystemExit("Too many / little  command line arguments")

    filepath = sys.argv[1].lower()  # should be the complete pathname
    print(filepath)
    if re.search(r"cat", filepath):
        return process_single(filepath, cat_hot_encoded)
    elif re.search(r"dog", filepath):
        return process_single(filepath, dog_hot_encoded)

    raise FileNotFoundError(
        "The specified file doens't contain the word dog or cat in the filepath"
    )


## TODO
def comp_pixel_percentage():
    lengths = []
    for i in range(4001, 4500, 1):
        sample = f"data_set/cats_set/cat.{i}.jpg"
        img = Image.open(sample).convert("L")
        lengths.append(len(img.getdata()))

    avg = round(sum(lengths) / len(lengths))
    print(f"The average number of pixels is: {avg}")
    print(f"We are resizing to {WIDTH} X {HEIGHT} which is {WIDTH * HEIGHT} pixels")
    print(
        f"This is {round(WIDTH * HEIGHT / avg * 100)}% of the pixels of the original image"
    )


### TODO: TESTS ###


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("No CSV file specified, goodbye")
    process(sys.argv[1], sys.argv[2])
