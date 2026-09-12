import numpy as np


class Layer:
    '''
    Class that keeps track of the Input Neurons and the weights and bias' associated with it
    Forward pass and backpropogation functions are handled here
    Activation functions and their respective backpropogation is handled elsewhere
    '''
    def __init__(self, input_vector: np.array = None, next_layer_size: int = None):
        self._input_vector: np.array = self.set_input_vector(input_vector, None)
        self._weights: np.array = self._set_weights(next_layer_size) # this is a matrix
        self._bias: np.array = self._set_bias(next_layer_size)


    def set_input_vector(self, input_vector: np.array, size: int | None) -> np.array:

        # Called duing the construction of the class
        if type(input_vector) == np.array:
            return input_vector

       # Called every other time - the neurons should never have random values 
       # Neuron values always have the previous layer to read off of
       # This code for creating a random neuron layer is redundant
        if not size:
            return Exception("Must provide a size for the input neuron vector")
        
        self._input_vector = np.random.rand(1, size)

    def get_input_vector(self) -> np.array:
        return self._input_vector

    
    def _set_weights(self, next_layer_size: int):
       # the width is the number of the input neurons
       # the height is the number of neurons in the second layer

       # when the class is first constructed, create a random matrix
       # TODO: Allow for the assignment of the weights - will need this for the loading feature down the line
       rows, cols = self._input_vector.shape
       if rows > 1:
           raise Exception("Neuron layer should be a vector not a matrix")
       return np.random.rand(next_layer_size, cols)


    def _get_weights(self) -> np.array:
        return self._weights

    def _set_bias(self, next_layer_size: int) -> np.array:
        # More simple since we just need on bias associated with each of the neurons on the next layer
        return np.random.rand(1, next_layer_size)

    def _get_bias(self) -> np.array:
        return self._bias

    def forward(self) -> np.array:
        ### forward pass and returns a np.array of shape (1, next_layer_size) a vector of neuron values
        ### Needs to have the bias vector, the weights matrix and the current layer neurons
        dp = np.dot(self._weights, self._input_vector)
        return np.add(dp, self._bias)

        

    def backwards(self):
        pass


    def save():
        '''Saves the weights and bias for the current layer to csv file'''

    def load(self, filename: str) -> None:
        '''Given a filename we load the weights and biases'''



def Relu():
    pass

def Softmax():
    pass


### UNIT TESTS ###
def test_input_vector():
    # arr = np.array([1,2, 3])
    # test_layer: Layer = Layer(arr)
   

    test_layer2 = Layer()
    test_layer2.set_input_vector(None, 10)
    print(test_layer2.get_input_vector())

 




### CREATION OF THE NN
### This Neural Network for this example will have 2 hidden layers of 128 neurons
### The activation function for the hidden layers will be ReLu
### The final layer will use Softmax as the activation function
### The cost fucntion is the Mean Squared Error


tmp = [
    [1, 2, 3],
    [4, 5, 6]
]
arr = np.array(tmp)

print(arr.shape)