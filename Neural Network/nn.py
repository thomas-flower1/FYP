import numpy as np


class Layer:


    def __init__(self, neuron_vector: np.array = None, next_layer_size: int = None):
        self._neuron_vector: np.array = self.set_neuron_vector(neuron_vector, None)
        self._neuron_weights: np.array = self.set_neuron_weights(next_layer_size)
        self._bias: np.array = None # list of the biases


    def set_neuron_vector(self, neuron_vector: np.array, size: int | None) -> np.array | None:

        # Called duing the construction of the class
        if type(neuron_vector) == np.array:
            return neuron_vector

       # Called every other time
        if not size:
            return Exception("Must provide a size for the input neuron vector")
        
        self._neuron_vector = np.random.rand(1, size)

    def get_neuron_vector(self) -> np.array:
        return self._neuron_vector

    
    def set_neuron_weights(self):
        pass
    

    

        

        


    def forward(self):
        pass

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
def test_neuron_vector():
    # arr = np.array([1,2, 3])
    # test_layer: Layer = Layer(arr)
    # assert np.array_equal(arr, test_layer.get_neuron_vector())

    test_layer2 = Layer()
    test_layer2.set_neuron_vector(None, 10)
    print(test_layer2.get_neuron_vector())

 



def main():
    test_neuron_vector()



### TRAINING ###


if __name__ == "__main__":
    main()