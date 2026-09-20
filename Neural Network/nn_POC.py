import numpy as np
import scipy as sp


'''
This file was to test and verify the maths for the acual NN implementation. In partciular
some of the calaculus for the backpropogation algorithm proved to be tricky. This file is a small
implementation of a NN which was used as a template for the final NN implementation.



TO TEST
- Time a single training example
- Look into the loops, can we optimize
- Test if the activation functions blow up
- Type cast to float32
'''


## FORWARD FUNCTIONS
def forward(a, w, b):
    return np.dot(w, a) + b

def relu(arr):
    return np.maximum(arr, 0)

def softmax(arr):
    return sp.special.softmax(arr)


def loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Since log(0) is undefined
    return -np.sum(y_true * np.log(y_pred))


## BACKPROPOGATION 
def softmax_back(y_hat, y, a_prev, w):
    '''
    Args:
    y_hat: The prediction / output of the NN, (n,)
    y: The one-hot real value, (n,)
    a_prev: The output of the previous layer A(L-1), (n,)
    w: The weights of this current layer w(L), (n,)

    Retuns:
    dc_dw: The gradients of this layers weights,(n, n )
    dc_da: To be passed to the next layer, (n,)
    delta: The gradients of this layers bias, (n,)
    '''
    delta = (y_hat - y) # this is da / dz * dc / da, this is the: remember da / db is just 1
    dc_dw = np.outer(delta, a_prev) # gradients for w, with respect to the cost function

    transpose_w = w.transpose() # we want all the weights a neuron influences - each row is now that

    ### TODO OPTIMIZE THIS
    dc_da = [] # a(L-1)
    for row in transpose_w:
        dc_da.append(np.dot(row, delta)) # a[k][L-1]

    dc_da = np.array(dc_da)
    return (dc_dw, dc_da, delta)


def relu_back(dc_da_prev, z, w, a_prev):
    '''
    Args:
    dc_da_prev: The value of dc_da from the previous backpropogation step
    z: The output of this layer before the activation function
    w: The weight matrix of this layer
    a_prev: The output of the previous layer that acted as input to this layer

    Returns:
    '''

    da_dz = np.where(z > 0, 1, 0) # Derivative of Relu
    delta = da_dz * dc_da_prev # da_dz * dc_da

    dc_dw = np.outer(delta, a_prev)

    transpose_w = w.transpose() # we want all the weights a neuron influences - each row is now that

    ### TODO OPTIMIZE THIS
    dc_da = [] # a(L-1)
    for row in transpose_w:
        dc_da.append(np.dot(row, delta)) # a[k][L-1] 
    dc_da = np.array(dc_da)

    return (dc_dw, dc_da, delta)

def update_layer(gradients, vector, learning_rate) -> np.ndarray:
    '''
    Given a vector adds or subtracts a small amount
    '''

    vector -= gradients * learning_rate
    return vector


### SAMPLE NEURAL NETWORK ###
### INPUT LAYER SIZE 2, HIDDEN LAYER SIZE 1, OUTPUT LAYER SIZE 2 ###

## INPUT LAYER
input = np.array([0.3, 0.5], dtype=np.float32)

## HIDDEN LAYER
w1 = np.array([[0.2, -0.9]], dtype=np.float32)
b1 = np.array([0.78], dtype=np.float32)

## OUTPUT LAYER
w2 = np.array([[-0.3], [0.9]], dtype=np.float32)
b2 = np.array([0.6, 0.4], dtype=np.float32)

y_true = np.array([1, 0], dtype=np.float32)
LEARNING_RATE = 0.01

z1 = forward(input, w1, b1)
a1 = relu(z1)
z2 = forward(a1, w2, b2)
a2 = softmax(z2)
c = loss(y_true, a2)
print(f"Current loss is: {c}")
print(f"Hidden Layer Weights: {w1}")
print(f"Hidden Layer Weights: {b1}")
print(f"Output Layer Weights: {w2}")
print(f"Output Layer Weights: {b2}")


# Backpropogation step
dc_dw, dc_da, dc_db = softmax_back(a2, y_true, a1, w2)

## Updating the weights and bias for the output layer ##
w2 = update_layer(dc_dw, w2, LEARNING_RATE)
b2 = update_layer(dc_db, b2, LEARNING_RATE)

print(f"Updated Output layer weights: {w2}")
print(f"Updated Output layer bias: {b2}")

dc_dw, dc_da, dc_db = relu_back(dc_da, z1, w1, input)
w1 = update_layer(dc_dw, w1, LEARNING_RATE)
b1 = update_layer(dc_db, b1, LEARNING_RATE)

print(f"Updated Output layer weights: {w1}")
print(f"Updated Output layer bias: {b1}")

z1 = forward(input, w1, b1)
a1 = relu(z1)
z2 = forward(a1, w2, b2)
a2 = softmax(z2)
c = loss(y_true, a2)
print(f"Current loss is: {c}")



# ### RUNNING BACKPROPOGATION
# ### OUTPUT LAYER TO HIDDEN LAYER
# dc_dw1, dc_da1, delta1 = softmax_back(a2, y_true, a1, w2)
# dc_dw2, dc_da2, delta2 = relu_back(dc_da1, z1, w1, input)



