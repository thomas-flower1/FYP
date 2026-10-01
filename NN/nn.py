import numpy as np
import time
from img_process import read
from random import shuffle


"""
TODO: Unit tests - in the other file, docstrings, additional cost and activation functions


### Try a different learning rate 

"""

class Layer:
    '''
    XXX
    Class that keeps track of the Input Neurons and the weights and bias' associated with it
    Forward pass and backpropagation functions are handled here
    Activation functions and their respective backpropagation is handled elsewhere
    XXX
    '''
    def __init__(self, input_vector: np.ndarray, current_layer_size: int): # need to validate the next layer size too
        self._input_vector: np.ndarray = self._valdiate_input_vector(input_vector) # could be unsafe if this is not of shape(1, n)
        self._weights: np.ndarray = np.random.randn(current_layer_size, input_vector.size) * np.sqrt(2 / input_vector.size)
        self._bias: np.ndarray = np.random.uniform(-1, 1, (current_layer_size,))

    #FIXME HMMMMMMM
    def _valdiate_input_vector(self, input_vector: np.ndarray) -> np.ndarray:
        '''Make sure that the shape of the input vectore is (1, n)'''
        if type(input_vector) != np.ndarray:
            raise Exception("Input vector must be a numpy ndarray")

        try:
            rows, _ = input_vector.shape
        except ValueError:
            return input_vector
        else:
            raise Exception("Input vector should have shape of (1, n)")


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

    def save(self, weights_filename: str, bias_filename: str) -> None:
        np.save(weights_filename, self._weights)
        np.save(bias_filename, self._bias)
        
    def load(self, weights_filename: str, bias_filename: str) -> None:
        self._weights = np.load(weights_filename)
        self._boas = np.load(bias_filename)
      



def ReLu(vector: np.ndarray) -> np.ndarray:
    return np.maximum(vector, 0)

def Softmax(vector: np.ndarray):
    shifted = vector - np.max(vector)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def cross_entropy_loss(prediction: np.ndarray, target: np.ndarray) -> int:
    eps = 1e-12
    prediction = np.clip(prediction, eps, 1 - eps)
    return np.sum(-(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))) / prediction.shape[0]
    

### BACKPROPAGATION ###
def softmax_backpropagation(prediction: np.ndarray, target: np.ndarray, output_prev: np.ndarray, w: np.ndarray) -> tuple:
    """Function that computes the gradients of the output layer, with respect to softmax and cross-entropy loss

    Args:
        prediction (np.ndarray) : Vector output of the output layer (n, )
        target (np.ndarray) : one-hot encoded vector (n, )
        output_prev (np.ndarray) : Vector output of the prev layer that is input to this layer (m,)
        w (np.ndarray) : Matrix of the weights of the current layer (n,m)

    Returns:
        dc_dw (np.ndarray) :
        dc_da (np.ndarray) :
        delta (np.ndarray) : 

    """

    delta = (prediction - target) # da / dz * dc / da
    dc_dw = np.outer(delta, output_prev) # Gradients for W 

    ### DC/DA - TO PASS ONTO THE NEXT LAYER ###
    ### FIXME : get rid of the loop?
    trans_w = w.transpose() # each row now corresponds to an output neuron
    dc_da = [] # a(L-1)
    for row in trans_w:
        dc_da.append(np.dot(row, delta)) 

    dc_da = np.array(dc_da)
    return (dc_dw, dc_da, delta)



#TODO: DOCSTRING
def relu_backpropagation(dc_da: np.ndarray, z: np.ndarray, w: np.ndarray, output_prev: np.ndarray) -> tuple:
    """
        dc_da_prev: The value of dc_da from the previous backpropogation step
        z: The output of this layer before the activation function
        w: The weight matrix of this layer
        a_prev: The output of the previous layer that acted as input to this layer

        Returns:
            dc_dw (np.ndarray) : The gradients of the weights of this layer, ()
            dc_da (np.ndarray) : Vector of how much the prev layer impacts this layers neurons (n,)
            dc_db 
        
        Raises:

    """

    da_dz = np.where(z > 0, 1, 0) # Relu derivative
    delta = da_dz * dc_da
    dc_dw = np.outer(delta, output_prev) 

    trans_w = w.transpose()
    dc_da = [] # a(L-1)
    for row in trans_w:
        dc_da.append(np.dot(row, delta)) # a[k][L-1] 

    new_dc_da = np.array(dc_da)
    return (dc_dw, new_dc_da, delta)

    


if __name__ == "__main__":
    ### FUll NEURAL NETWORK TEST ###
    # INPUT (10,) HIDDEN (20,) OUTPUT (2,)

    # target = np.array([0, 1])
    # LEARNING_RATE = 0.01

    # start = time.time()
    # i = np.random.rand(64 * 64,)
    # hidden_layer = Layer(i, 512)
    # z1 = hidden_layer.forward()
    # a1 = ReLu(z1)
    # output_layer = Layer(a1, 2)
    # z2 = output_layer.forward()
    # prediction = Softmax(z2)
    # end = time.time()

    # loss = cross_entropy_loss(prediction, target)
    # print(f"The loss is: {loss}")

    # ## backprop
    # dc_dw, dc_da, b1 = softmax_backpropagation(prediction, target, a1, output_layer._weights) # OUTPUT LAYER
    # dc_dw2, dc_da2, b2 = relu_backpropagation(dc_da, z1, hidden_layer._weights, i) ## HIDDEN LAYER

    # output_layer.update_bias(b1)
    # output_layer.update_weights(dc_dw)
    # hidden_layer.update_bias(b2)
    # hidden_layer.update_weights(dc_dw2)

    # hidden_layer = Layer(i, 512)
    # z1 = hidden_layer.forward()
    # a1 = ReLu(z1)
    # output_layer = Layer(a1, 2)
    # z2 = output_layer.forward()
    # prediction = Softmax(z2)
    # loss = cross_entropy_loss(prediction, target)
    # print(f"The loss is: {loss}")

    ### ACTUAL TEST ###
    dogs: list = read("dog.csv", np.array([1, 0]))
    cats: list = read("cat.csv", np.array([0, 1]))
    LEARNING_RATE = 0.01
   
    arr = dogs + cats
    shuffle(arr)


    hidden_layer = Layer(np.random.rand(256 * 256,), 4096)
    hidden_layer2 = Layer(np.random.rand(4096,), 256)
    output_layer = Layer(np.random.rand(256,), 2)
   
    for i in range(900):
        target: np.ndarray = arr[i].get_target()
        input: np.ndarray = arr[i].get_data()

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
        dc_dw1, dc_da1, b1 = softmax_backpropagation(prediction, target, a2, output_layer._weights) # OUTPUT LAYER
        dc_dw2, dc_da2, b2 = relu_backpropagation(dc_da1, z2, hidden_layer2._weights, a1) ## HIDDEN LAYER 2
        dc_dw3, dc_da3, b3 = relu_backpropagation(dc_da2, z1, hidden_layer._weights, input) ## HIDDEN LAYER 1

        # Updated the gradients
        output_layer.update_bias(b1)
        output_layer.update_weights(dc_dw1)
        hidden_layer2.update_bias(b2)
        hidden_layer2.update_weights(dc_dw2)
        hidden_layer.update_bias(b3)
        hidden_layer.update_weights(dc_dw3)

    # Save the weights and the bias 
    hidden_layer.save()
    hidden_layer2.save()
    output_layer.save()

    target: np.ndarray = arr[950].get_target()
    input: np.ndarray = arr[950].get_data()

    hidden_layer.set_input_vector(input)
    z1 = hidden_layer.forward()
    a1 = ReLu(z1)

    hidden_layer2.set_input_vector(a1)
    z2 = hidden_layer2.forward()
    a2 = ReLu(z2)

    output_layer.set_input_vector(a2)
    z3 = output_layer.forward()
    prediction = Softmax(z3)

    print(f"Netowrk Prediction: {prediction * 100}%")
    print(f"Actual: {target}")
        







   



   






