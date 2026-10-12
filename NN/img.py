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


def process_single(filepath) -> np.ndarray:
    """Helper functions for \"user_defined_image\""""

    img = Image.open(filepath).convert("L")
    resize = img.resize((WIDTH, HEIGHT))
    data = np.array(
        resize.getdata()
    )  # Returns a 1D list with all the grayscale values of a single image

    if len(data) != WIDTH * HEIGHT:
        raise Exception("READ ERROR")

    return data


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



### TODO: TESTS ###

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("No CSV file specified, goodbye")
    process(sys.argv[1], sys.argv[2])
