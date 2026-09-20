import numpy as np
import scipy as sp


'''
This file was to test and verify the maths for the acual NN implementation. In partciular
some of the calaculus for the backpropogation algorithm proved to be tricky. This file is a small
implementation of a NN which was used as a template for the final NN implementation.
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
    dc_da = [] # a(L-1)
    for row in transpose_w:
        dc_da.append(np.dot(row, delta)) # a[k][L-1]
    dc_da = np.array(dc_da)

    return (dc_dw, dc_da, delta)

def update_layer(gradients, w, b, learning_rate) -> np.ndarray:
    '''
    
    
    
    '''
    pass


### SAMPLE NEURAL NETWORK ###
### INPUT LAYER SIZE 2, HIDDEN LAYER SIZE 1, OUTPUT LAYER SIZE 2 ###

## INPUT LAYER
input = np.array([3, 5])

## HIDDEN LAYER
w1 = np.array([[2, 9]])
b1 = np.array([5])

## OUTPUT LAYER
w2 = np.array([[8], [9]])
b2 = np.array([6])

y_true = np.array([1, 0])

z1 = forward(input, w1, b1)
assert z1 == np.array([56]) # may want to do a type conversion of this later

a1 = relu(z1)
assert z1 == np.array([56])

z2 = forward(a1, w2, b2)
assert np.array_equal(np.array([454, 510]), z2)

a2 = softmax(z2)







# z2 = forward(a1, w2, b2)
# a2 = softmax(z2)
# c = loss(y_true, a2)

# ### RUNNING BACKPROPOGATION
# ### OUTPUT LAYER TO HIDDEN LAYER
# dc_dw1, dc_da1, delta1 = softmax_back(a2, y_true, a1, w2)
# dc_dw2, dc_da2, delta2 = relu_back(dc_da1, z1, w1, input)







