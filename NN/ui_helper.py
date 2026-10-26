from random import shuffle

import numpy as np
import streamlit as st
from img import HEIGHT, WIDTH, process_single, read
from nn import Layer, ReLu, Softmax, cross_entropy_loss, relu_backpropagation, softmax_backpropagation


@st.cache_data
def load_nn(dir) -> tuple:
    """WIP function that loads the cats and dogs trained dataset
    Note: Current implementation means that file read are absolute paths that are hard coded.
    Fix this on later implementation
    """

    # initialize the neural network
    hidden_layer = Layer(np.random.rand(WIDTH * HEIGHT), 512)
    hidden_layer2 = Layer(np.random.rand(512), 256)
    output_layer = Layer(np.random.rand(256), 2)

    # Loading the weights and bias
    hidden_layer.load(f"{dir}/TRAINED_CATS_DOGS/layer1_weights.npy", f"{dir}/TRAINED_CATS_DOGS/layer1_bias.npy")
    hidden_layer2.load(f"{dir}/TRAINED_CATS_DOGS/layer2_weights.npy", f"{dir}/TRAINED_CATS_DOGS/layer2_bias.npy")
    output_layer.load(f"{dir}/TRAINED_CATS_DOGS/output_weights.npy", f"{dir}/TRAINED_CATS_DOGS/output_bias.npy")

    # Then return the layers for running the nn
    return (hidden_layer, hidden_layer2, output_layer)


def forward(path, h1, h2, o) -> np.ndarray:
    input: np.ndarray = process_single(path)

    h1.set_input_vector(input)
    z1 = h1.forward()
    a1 = ReLu(z1)

    h2.set_input_vector(a1)
    z2 = h2.forward()
    a2 = ReLu(z2)

    o.set_input_vector(a2)
    z3 = o.forward()
    prediction = Softmax(z3)

    return prediction


# Make sure that the entries are not null upon submission
def validate_csv_data(csv_data: list, tab) -> bool:
    error = False
    for csv, target in csv_data:
        if not csv:
            tab.error("Each entry must contain csv data")
            error = True
        if not target:
            tab.error("Each csv entry must have a corresponding target")
            error = True
    return not error


@st.cache_data
def train(nn_dict: dict, pathnames: dict, cwd: str, _message, learning_rate: float, number_of_epochs=10) -> None:
    """Helper function that creates and trains a NN from user input
    nn_dict: dict(index, (size, activation)) : The shape of each of the NN layers
    pathnames: dict(pathname, target) : The uploaded CSV data to train on
    cwd: str : The dir that you want to save the trained networks weights and biases
    _message: : An empty streamlit box to write to
    learning_rate: float : Learning rate of the NN
    number_of_epochs: int : The number of epochs for training the NN

    Note: The name of the saved files are hardcoded at the bottom of this function
    """

    # 1. DATA PREP
    loaded_data = []
    targets = {}  # key: [0, 0,1], value: "Cat"
    for index, (path, target) in enumerate(pathnames.items()):
        # one hot encoded
        zeros = [0] * len(pathnames)
        zeros[index] = 1

        # reading a full csv file
        loaded_data = read(path, np.array(zeros)) + loaded_data

        # keeping track of the one hot encoded and the string value
        targets[target] = zeros

    shuffle(loaded_data)

    # 2. NN CREATION
    layers = []  # (layer, activation)
    prev_layer_size = len(loaded_data[0].get_data())
    for value in nn_dict.values():
        size, activation = value
        layer = Layer(np.random.rand(int(prev_layer_size)), size)
        prev_layer_size = size
        layers.append((layer, activation))

    rev_layers = list(reversed(layers))

    # 3. FORWARDING
    for i in range(number_of_epochs):
        for data in loaded_data:
            # getting a single training example
            input_vector = data.get_data()  # Note: At the end of the iterations this becomes the output of the network
            target = data.get_target()

            for value in layers:
                layer, activation = value
                layer.set_input_vector(input_vector)
                current_z = layer.forward()

                # Add additional activation functions here later
                match activation:
                    case "ReLu":
                        current_a = ReLu(current_z)
                    case "Softmax":
                        current_a = Softmax(current_z)

                input_vector = current_a

            # 4. BACKPROP TODO
            prediction = input_vector
            dc_da = None

            dc_dw = []  # All weight gradients to update
            b = []  # All bias gradients to update

            for value in rev_layers:
                layer, activation = value

                match activation:
                    case "Softmax":
                        current_dc_dw, dc_da, current_b = softmax_backpropagation(prediction, target, layer.get_input_vector(), layer.get_weights())
                        dc_dw.append(current_dc_dw)
                        b.append(current_b)
                    case "ReLu":
                        current_dc_dw, dc_da, current_b = relu_backpropagation(
                            dc_da, layer.get_output(), layer.get_weights(), layer.get_input_vector()
                        )
                        dc_dw.append(current_dc_dw)
                        b.append(current_b)

            # Since we did backprop the order is reversed for the gradients, fix this
            dc_dw = list(reversed(dc_dw))
            b = list(reversed(b))

            # 5.  Update the gradients
            for index, tup in enumerate(layers):
                layer, activation = tup
                layer.update_weights(dc_dw[index], learning_rate)
                layer.update_bias(b[index], learning_rate)

            _message.write(f"EPOCH {i + 1} out of {number_of_epochs}. Current Loss: {cross_entropy_loss(input_vector, target)}")

    # 6. Saving the data
    for index, (layer, _) in enumerate(layers):
        layer.save(f"{cwd}layer{index}_weights_maybe.npy", f"{cwd}layer{index}_bias_maybe.npy")
