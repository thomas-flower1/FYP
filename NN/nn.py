import numpy as np
import pytest


"""
TODO: Unit tests - in the other file, docstrings, additional cost and activation functions

"""

class Layer:
    '''
    XXX
    Class that keeps track of the Input Neurons and the weights and bias' associated with it
    Forward pass and backpropogation functions are handled here
    Activation functions and their respective backpropogation is handled elsewhere
    XXX
    '''
    def __init__(self, input_vector: np.ndarray, current_layer_size: int): # need to validate the next layer size too
        self._input_vector: np.ndarray = self._valdiate_input_vector(input_vector) # could be unsafe if this is not of shape(1, n)
        self._weights: np.ndarray = np.random.uniform(-1, 1, (current_layer_size, input_vector.size)) # this is a matrix
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

    def save(self):
        pass
        

    def load(self, filename: str) -> None:
        ## use the np.save
        pass



def ReLu(vector: np.ndarray) -> np.ndarray:
    return np.maximum(vector, 0)

def Softmax(vector: np.ndarray):
    d = np.exp(vector)
    n = np.sum(np.exp(vector))
    sol = [d[i] / n for i in range(vector.shape[0])]
    return np.round(np.array(sol), 3)

def cross_entropy_loss(prediction: np.ndarray, target: np.ndarray) -> int:
    return sum(-(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))) / prediction.shape
   

### BACKPROPOGATION ###
def softmax_backpropogation(prediction: np.ndarray, target: np.ndarray, output_prev: np.ndarray, w: np.ndarray) -> tuple:
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
    dc_dw = np.outer(output_prev, delta) # Gradients for W 

    ### DC/DA - TO PASS ONTO THE NEXT LAYER ###
    ### FIXME : get rid of the loop?
    trans_w = w.transpose() # each row now corresponds to an output neuron
    dc_da = [] # a(L-1)
    for row in trans_w:
        dc_da.append(np.dot(row, delta)) 

    dc_da = np.array(dc_da)
    return (dc_dw, dc_da, delta)

  


#FIXME
def relu_backpropogation(dc_da: np.ndarray, z: np.ndarray, w: np.ndarray, output_prev: np.ndarray) -> tuple:
    pass
    


if __name__ == "__main__":
   ### TESTING THE COST FUNCTION
   softmax_backpropogation()





