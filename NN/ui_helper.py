from nn import Layer, ReLu, Softmax
import numpy as np
from img import WIDTH, HEIGHT, process_single
import streamlit as st


@st.cache_data
def load_nn(dir) -> tuple:

    # initialize the neural network
    hidden_layer = Layer(np.random.rand(WIDTH * HEIGHT), 512)
    hidden_layer2 = Layer(np.random.rand(512), 256)
    output_layer = Layer(np.random.rand(256), 2)
  
    # Loading the weights and bias
    hidden_layer.load(f"{dir}/layer1_weights.npy", f"{dir}/layer1_bias.npy")
    hidden_layer2.load(f"{dir}/layer2_weights.npy", f"{dir}/layer2_bias.npy")
    output_layer.load(f"{dir}/output_weights.npy", f"{dir}/output_bias.npy")

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
