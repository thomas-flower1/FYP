import time

import numpy as np
import scipy as sp

"""
This file was to test and verify the maths for the actual NN implementation. 
This basic model includes the following
    - Basic forwarding
    - ReLu activation function
    - Softmax activation function
    - Derivative of softmax with cross-entropy loss
    - Derivative of the cost with respect to Relu 

The implementation is stochastic gradient descent, so it doesn't support batches.

TO TEST
- Look into the loops, can we optimize
- Test if the activation functions blow up
- Type cast to float32
- Batch processing
"""


## FORWARD FUNCTIONS
def forward(a, w, b):
    return np.dot(w, a) + b


def relu(arr):
    return np.maximum(arr, 0)


def softmax(arr):
    return sp.special.softmax(arr)


def loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Since log(0) is undefined
    return -np.sum(y_true * np.log(y_pred))


## BACKPROPOGATION
def softmax_back(y_hat, y, a_prev, w) -> tuple:
    """
    Args:
        y_hat: The prediction / output of the NN,
        y: The one-hot real value,
        a_prev: The output of the previous layer A(L-1),
        w: The weights of this current layer w(L),

    Returns:
        dc_dw: The gradients of this layers weights
        dc_da: To be passed to the next layer
        delta: The gradients of this layers bias
    """
    delta = (
        y_hat - y
    )  # this is da / dz * dc / da, this is the: remember da / db is just 1
    dc_dw = np.outer(
        delta, a_prev
    )  # gradients for w, with respect to the cost function

    transpose_w = (
        w.transpose()
    )  # we want all the weights a neuron influences - each row is now that

    # TODO : OPTIMIZE THIS?
    dc_da = []  # a(L-1)
    for row in transpose_w:
        dc_da.append(np.dot(row, delta))  # a[k][L-1]

    dc_da = np.array(dc_da)
    return (dc_dw, dc_da, delta)


def relu_back(dc_da_prev, z, w, a_prev) -> tuple:
    """
    Args:
        dc_da_prev: The value of dc_da from the previous backpropagation step
        z: The output of this layer before the activation function
        w: The weight matrix of this layer
        a_prev: The output of the previous layer that acted as input to this layer

    Returns:
        dc_dw: The gradients of this layers weights
        dc_da: To be passed to the next layer
        delta: The gradients of this layers bias
    """

    da_dz = np.where(z > 0, 1, 0)  # Derivative of Relu
    delta = da_dz * dc_da_prev  # da_dz * dc_da
    dc_dw = np.outer(delta, a_prev)

    transpose_w = (
        w.transpose()
    )  # we want all the weights a neuron influences - each row is now that

    # TODO : OPTIMIZE THIS
    dc_da = []  # a(L-1)
    for row in transpose_w:
        dc_da.append(np.dot(row, delta))
    dc_da = np.array(dc_da)

    return (dc_dw, dc_da, delta)


def update_layer(gradients, vector, learning_rate) -> np.ndarray:
    """Given a vector subtracts a small amount"""

    vector -= gradients * learning_rate
    return vector


if __name__ == "__main__":
    ### TESTING NN WITH A LARGER HIDDEN LAYER TEST  ###
    # INPUT (8,) HIDDEN LAYER (16, ) OUTPUT LAYER (2,)

    # INPUT TO HIDDEN LAYER 1
    input = np.array(
        [0.33, 0.43, 0.31, 0.25, 0.34, 0.11, 0.09, 0.17], dtype=np.float32
    )  # (8,)
    w1 = np.random.uniform(-1, 1, size=(16, 8))
    b1 = np.random.uniform(-1, 1, size=(16,))

    ## OUTPUT LAYER ###
    w2 = np.random.uniform(-1, 1, size=(2, 16))
    b2 = np.random.uniform(-1, 1, size=(2,))

    y_true = np.array([1, 0], dtype=np.float32)
    LEARNING_RATE = 0.01

    forward_start = time.time()
    z1 = forward(input, w1, b1)
    a1 = relu(z1)
    z2 = forward(a1, w2, b2)
    a2 = softmax(z2)
    c = loss(y_true, a2)
    forward_end = time.time()
    print(
        f"Time elapsed for a single forwad pass is: {forward_end - forward_start} seconds"
    )
    print(f"Current loss is: {c}")
    print(f"Hidden Layer Weights: {w1}")
    print(f"Hidden Layer Weights: {b1}")
    print(f"Output Layer Weights: {w2}")
    print(f"Output Layer Weights: {b2}")

    ## BACKPROP ##
    back_start = time.time()
    dc_dw, dc_da, dc_db = softmax_back(a2, y_true, a1, w2)

    ## UPDATE OUTPUT LAYER ##
    w2 = update_layer(dc_dw, w2, LEARNING_RATE)
    b2 = update_layer(dc_db, b2, LEARNING_RATE)

    # UPDATE HIDDEN LAYER
    dc_dw, dc_da, dc_db = relu_back(dc_da, z1, w1, input)
    w1 = update_layer(dc_dw, w1, LEARNING_RATE)
    b1 = update_layer(dc_db, b1, LEARNING_RATE)

    z1 = forward(input, w1, b1)
    a1 = relu(z1)
    z2 = forward(a1, w2, b2)
    a2 = softmax(z2)
    c = loss(y_true, a2)

    back_end = time.time()

    print(f"Backpropagation time elapsed: {back_end - back_start}")
    print(f"Current loss is: {c}")
