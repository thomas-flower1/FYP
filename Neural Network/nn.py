import numpy as np
import pytest


class Layer:
    '''
    Class that keeps track of the Input Neurons and the weights and bias' associated with it
    Forward pass and backpropogation functions are handled here
    Activation functions and their respective backpropogation is handled elsewhere
    '''
    def __init__(self, input_vector: np.ndarray, next_layer_size: int): # need to validate the next layer size too
        self._input_vector: np.ndarray = self._valdiate_input_vector(input_vector) # could be unsafe if this is not of shape(1, n)
        self._weights: np.ndarray = np.random.uniform(-1, 1, (next_layer_size, input_vector.size)) # this is a matrix
        self._bias: np.ndarray = np.random.uniform(-1, 1, (next_layer_size,))


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
        # Used to reasign during back propogation
        self._input_vector = input_vector
          

    def get_input_vector(self) -> np.ndarray:
        return self._input_vector

    def set_weights(self, weights: np.ndarray) -> None:
        # TODO: validate shape
        self._weights = weights

    def get_weights(self) -> np.ndarray:
        return self._weights

    def set_bias(self, bias: np.ndarray) -> None:
        # TODO: validate shape
        self._bias = bias

    def get_bias(self) -> np.ndarray:
        return self._bias


    def forward(self) -> np.ndarray:
        ### forward pass and returns a np.array of shape (1, next_layer_size) a vector of neuron values
        ### Needs to have the bias vector, the weights matrix and the current layer neurons
        dp = np.dot(self._weights, self._input_vector)
        return np.add(dp, self._bias)

        

    def backwards(self):
        pass


    def save(self):
        '''Saves the weights and bias for the current layer to csv file'''
        pass
        

    def load(self, filename: str) -> None:
        '''Given a filename we load the weights and biases'''
        pass



def Relu():
    pass

def Softmax():
    pass


### UNIT TESTS ###
def test_input_vector():

    ### Testing the happy path
    arr1 = np.array([1, 2, 3])
    layer1 = Layer(arr1, 10)
    assert np.array_equal(layer1.get_input_vector(), arr1)

    ### Testing wrong type
    arr2 = [1, 2, 3]
    with pytest.raises(Exception) as e:
        layer2 = Layer(arr2, 10)

    ### Test wrong shape 
    arr3 = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    with pytest.raises(Exception) as e:
        layer3 = Layer(arr3, 10)




def test_forward_pass():
    pass
 




### CREATION OF THE NN
layer = Layer(np.array([1, 2, 3]), 10)
print(f"Random Weights: \n {layer.get_weights()}")
print(f"Random Bias: \n {layer.get_bias()}")
print(f"Forward Vector: {layer.forward()}")



### This Neural Network for this example will have 2 hidden layers of 128 neurons
### The activation function for the hidden layers will be ReLu
### The final layer will use Softmax as the activation function
### The cost fucntion is the Mean Squared Error


