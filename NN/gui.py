import streamlit as st
from nn import Layer, ReLu, Softmax, softmax_backpropagation, relu_backpropagation
from img import read, WIDTH, HEIGHT
import numpy as np
from img import process_single
import os
import random

from ui_helper import load_nn, forward

st.title("Neural Network with NumPy")
st.markdown("*A visual representation of custom **neural network***")

tab1, tab2, tab3 = st.tabs(["Create" , "Run", "About"])


### INITIALIZE STATE VARIABLES
if "counter" not in st.session_state:
    st.session_state.counter = 0

if "pathname" not in st.session_state:
    st.session_state.pathname = ""

if "upload_hash" not in st.session_state:
    st.session_state.upload_hash = random.getrandbits(128)


# LOADING THE NN
cwd = "/home/thomas/UCC/FYP/NN/" 
h1, h2, o = load_nn(cwd)







container = tab1.container(border=True)
container.write("This UI allows the creation of customer neural networks. Users can specify what to train, how many hidden layers and even specify what the activation functions are. Under the hood we have Numpy and modular Python code to train and test the network. Try it for yourself below!")

#TODO: Append the code each time we add more
tab1.markdown("*A look into the code*")
code = """
target: np.ndarray = data.get_target()
input: np.ndarray = data.get_data()

"""


tab1.code(code, language="python")


### CREATING THE NN
MAX = 256

tab1.subheader("Create")
tab1.number_input("", value=None, placeholder="Input Layer Size", key=10000, max_value=MAX)



activation_functions = ("ReLu", "Softmax")
col1, col2 = tab1.columns([1, 1]) 






if tab1.button("Add Layer"):
    st.session_state.counter += 1



for i in range(st.session_state.counter):
    col1.number_input("", value=None, placeholder="Hidden Layer Size", key=i)
    col2.selectbox("", activation_functions, key=i/2)




   
### RUN ###

# intro
tab2.markdown("#### Overview")
container2 = tab2.container(border=True)
intro_message = """
This page allows you to upload an image to test the currently trained Neural Network.
This network is composed of 2 hidden layers both with the ReLu activation function. The first layer has
512 neurons and the second layer has 256 neurons.

The output layer uses Softmax as it's activation function and is composed of two neurons, one mapping to cat and one mapping
to dog.

This model was trained against 5000 dog images and 5000 cat images for 5 epochs. The total time to train was 5 hours and
the model achieved > 85\% \accuracy on unseen data.



"""
container2.write(intro_message)

nn_code = """
hidden_layer = Layer(np.random.rand(WIDTH * HEIGHT), 512)
hidden_layer2 = Layer(np.random.rand(512), 256)
output_layer = Layer(np.random.rand(256), 2)


target: np.ndarray = data.get_target()
input: np.ndarray = data.get_data()

hidden_layer.set_input_vector(input)
z1 = hidden_layer.forward()
a1 = ReLu(z1)

hidden_layer2.set_input_vector(a1)  
z2 = hidden_layer2.forward()
a2 = ReLu(z2)

output_layer.set_input_vector(a2)
z3 = output_layer.forward()
prediction = Softmax(z3)

"""

tab2.markdown("#### Code Snippet")
tab2.code(nn_code)

# TODO Add a diagram / animation video of teh neural network here:

# File upload and save
tab2.markdown("#### Upload")
classification_image = tab2.file_uploader("", accept_multiple_files=False,  label_visibility="collapsed", key=st.session_state.upload_hash)

# Saving the image locally
if classification_image:
    st.session_state.pathname = os.path.join(cwd, classification_image.name)
    with open(st.session_state.pathname, "wb") as f:
        f.write(classification_image.getbuffer())
        
results_container = None

# Creating the buttons
with tab2.container(horizontal=True):
    run_model_button = st.button("Run model", type="primary")
    reset_button = st.button("Reset", type="primary")


# RUN NN
if run_model_button and st.session_state.pathname: 
    prediction = forward(st.session_state.pathname, h1, h2, o)
    if prediction[0] >= 0.5:
        message = f"#### The prediction of the model is dog with **{prediction[0] * 100}**% confidence"
    else:
        message = f"#### The prediction of the model is cat with **{prediction[1] * 100}**% confidence"

    results_container = tab2.container(border=True)
    results_container.markdown(message)
    results_container.image(st.session_state.pathname)


# RESET
if reset_button and st.session_state.pathname:
    if results_container:
        results_container.empty()

    st.session_state.pathname = ""
    st.session_state.upload_hash = random.getrandbits(128)
    st.rerun() # Run the script again
    


### ABOUT ###
container3 = tab3.container(border=True)
container3.write("This interactive mode is a part of the Final Year Project for Student 123425562")