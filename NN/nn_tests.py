import numpy as np
from nn import *


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
 

def test_ReLu():
    arr = np.array([-1, 2, 3])
    assert np.array_equal(ReLu(arr), np.array([0, 2, 3]))


def test_softmax():
    # happy path
    arr = np.array([2, 1, 0.1])
    assert np.array_equal(Softmax(arr), np.array([0.659, 0.242, 0.099]))