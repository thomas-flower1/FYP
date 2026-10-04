import time
from random import shuffle

import numpy as np

from img import HEIGHT, WIDTH, read

"""
Final NN implementation 


TODO:
    - Implement other activation functions
    - Have the Backpropagation for these functions too
"""


class Layer:
    """
    Layer class that keeps track of the weights and biases. Responsible for their initialization, updating and
    forwarding the input layer through to the next layer

    """

    def __init__(self, input_vector: np.ndarray, current_layer_size: int):
        self._input_vector: np.ndarray = input_vector
        self._weights: np.ndarray = np.random.randn(current_layer_size, input_vector.size) * np.sqrt(2 / input_vector.size)  # He initialization
        self._bias: np.ndarray = np.random.uniform(-1, 1, (current_layer_size,))

    def set_input_vector(self, input_vector: np.ndarray) -> None:
        self._input_vector = input_vector

    def get_input_vector(self) -> np.ndarray:
        return self._input_vector

    def forward(self) -> np.ndarray:
        dp = np.dot(self._weights, self._input_vector)
        return np.add(dp, self._bias)

    def update_weights(self, gradients: np.ndarray, learning_rate: float = 0.01) -> None:
        self._weights -= gradients * learning_rate

    def update_bias(self, gradients: np.ndarray, learning_rate: float = 0.01) -> None:
        self._bias -= gradients * learning_rate

    # TODO : need to test and validate these :
    def save(self, weights_filename: str, bias_filename: str) -> None:
        np.save(weights_filename, self._weights)
        np.save(bias_filename, self._bias)

    def load(self, weights_filename: str, bias_filename: str) -> None:
        self._weights = np.load(weights_filename)
        self._boas = np.load(bias_filename)


### ACTIVATION FUNCTIONS ###


def ReLu(vector: np.ndarray) -> np.ndarray:
    return np.maximum(vector, 0)


def Softmax(vector: np.ndarray):
    shifted = vector - np.max(vector)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


### LOSS FUNCTIONS ###


def cross_entropy_loss(prediction: np.ndarray, target: np.ndarray) -> int:
    eps = 1e-12
    prediction = np.clip(prediction, eps, 1 - eps)
    return np.sum(-(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))) / prediction.shape[0]


### BACKPROPAGATION ###
def softmax_backpropagation(prediction: np.ndarray, target: np.ndarray, output_prev: np.ndarray, w: np.ndarray) -> tuple:
    """Function that computes the gradients of the output layer, with respect to softmax and cross-entropy loss

    Args:
        prediction (np.ndarray) : Vector output of the output layer
        target (np.ndarray) : one-hot encoded vector
        output_prev (np.ndarray) : Vector output of the prev layer that is input to this layer
        w (np.ndarray) : Matrix of the weights of the current layer

    Returns:
        dc_dw (np.ndarray) : Gradient of the weights to update
        dc_da (np.ndarray) : Derivative of this layer to pass onto the prev layer
        delta (np.ndarray) : Gradient of the bias to update

    """

    delta = prediction - target  # da / dz * dc / da
    dc_dw = np.outer(delta, output_prev)  # Gradients for W

    ### DC/DA - TO PASS ONTO THE NEXT LAYER ###
    ### FIXME : get rid of the loop?
    trans_w = w.transpose()  # each row now corresponds to an output neuron
    dc_da = []  # a(L-1)
    for row in trans_w:
        dc_da.append(np.dot(row, delta))

    dc_da = np.array(dc_da)
    return (dc_dw, dc_da, delta)


# TODO: DOCSTRING
def relu_backpropagation(dc_da: np.ndarray, z: np.ndarray, w: np.ndarray, output_prev: np.ndarray) -> tuple:
    """Function that computes the gradients with respect to the layer in-front of it

    Args:
        prediction (np.ndarray) : Vector output of the output layer
        target (np.ndarray) : one-hot encoded vector
        output_prev (np.ndarray) : Vector output of the prev layer that is input to this layer
        w (np.ndarray) : Matrix of the weights of the current layer

    Returns:
        dc_dw (np.ndarray) : Gradient of the weights to update
        dc_da (np.ndarray) : Derivative of this layer to pass onto the prev layer
        delta (np.ndarray) : Gradient of the bias to update
    """

    da_dz = np.where(z > 0, 1, 0)  # Relu derivative
    delta = da_dz * dc_da
    dc_dw = np.outer(delta, output_prev)

    trans_w = w.transpose()
    dc_da = []  # a(L-1)
    for row in trans_w:
        dc_da.append(np.dot(row, delta))  # a[k][L-1]

    new_dc_da = np.array(dc_da)
    return (dc_dw, new_dc_da, delta)


def check(prediction: np.ndarray, hot: np.ndarray) -> bool:
    """Validates our prediction - was it predict correctly (true) or not (false)"""
    target = prediction * hot
    return max(target) >= 0.5


if __name__ == "__main__":
    LEARNING_RATE = 0.01
    dogs: list = read("dog.csv", np.array([1, 0]))  # A list of Data Objects
    cats: list = read("cat.csv", np.array([0, 1]))

    cats = cats[:5050]
    dogs = dogs[:5050]

    SIZE = len(cats) + len(dogs)

    arr: list = dogs + cats
    shuffle(arr)

    test_arr: list = arr[: SIZE - 100]
    validate_arr = arr[SIZE - 100 :]  # Unseen dataset

    hidden_layer = Layer(np.random.rand(WIDTH * HEIGHT), 512)
    hidden_layer2 = Layer(np.random.rand(512), 256)
    output_layer = Layer(np.random.rand(256), 2)

    start = time.time()
    epoch = 10
    for _ in range(epoch):
        for index, data in enumerate(test_arr):
            s = time.time()
            target: np.ndarray = data.get_target()
            input: np.ndarray = data.get_data()

            hidden_layer.set_input_vector(input)
            z1 = hidden_layer.forward()
            a1 = ReLu(z1)

            hidden_layer2.set_input_vector(a1)
            z2 = hidden_layer2.forward()
            a2 = ReLu(z2)

            output_layer.set_input_vector(a2)
            z3 = output_layer.forward()
            prediction = Softmax(z3)

            loss = cross_entropy_loss(prediction, target)
            print(f"The loss is: {loss}")

            ## Backprop
            dc_dw1, dc_da1, b1 = softmax_backpropagation(prediction, target, a2, output_layer._weights)  # OUTPUT LAYER
            dc_dw2, dc_da2, b2 = relu_backpropagation(dc_da1, z2, hidden_layer2._weights, a1)  ## HIDDEN LAYER 2
            dc_dw3, dc_da3, b3 = relu_backpropagation(dc_da2, z1, hidden_layer._weights, input)  ## HIDDEN LAYER 1

            # Updated the gradients
            output_layer.update_bias(b1)
            output_layer.update_weights(dc_dw1)
            hidden_layer2.update_bias(b2)
            hidden_layer2.update_weights(dc_dw2)
            hidden_layer.update_bias(b3)
            hidden_layer.update_weights(dc_dw3)
            e = time.time()
            print(f"Current round {index}")
            print(f"TIme elapsed for this example is: {e - s}")

    end = time.time()

    print(f"Time to to train the network {end - start}")

    # # Save the weights and the bias - need to test this
    # hidden_layer.save()
    # hidden_layer2.save()
    # output_layer.save()

    ### TESTING AGAINST THE UNSEEN DATA
    correct = 0
    for data in validate_arr:
        target: np.ndarray = data.get_target()
        input: np.ndarray = data.get_data()

        hidden_layer.set_input_vector(input)
        z1 = hidden_layer.forward()
        a1 = ReLu(z1)

        hidden_layer2.set_input_vector(a1)
        z2 = hidden_layer2.forward()
        a2 = ReLu(z2)

        output_layer.set_input_vector(a2)
        z3 = output_layer.forward()
        prediction = Softmax(z3)

        if check(prediction, target):
            correct += 1

    print(f"The accuracy of the network is: {correct}%")
