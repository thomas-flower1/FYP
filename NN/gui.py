import os
import random

import streamlit as st
from ui_helper import forward, load_nn, train, validate_csv_data

st.title("Neural Network with NumPy")
st.markdown("*A visual representation of custom **neural network***")

tab1, tab2, tab3 = st.tabs(["Create", "Run", "About"])


### INITIALIZE STATE VARIABLES
if "counter" not in st.session_state:
    st.session_state.counter = 0

if "pathname" not in st.session_state:
    st.session_state.pathname = ""

if "upload_hash" not in st.session_state:
    st.session_state.upload_hash = random.getrandbits(128)  # TODO REMOVE THIS AND DO THE ALTERNATIVE

if "nn_dict" not in st.session_state:
    st.session_state.nn_dict = {}  # key: layer_index, value: (size, activation)

if "csv_pathnames" not in st.session_state:
    st.session_state.csv_pathnames = {}  # Key: Absolute Path, Value: Target

if "csv_count" not in st.session_state:
    st.session_state.csv_count = 1

# LOADING THE NN
cwd = "/home/thomas/UCC/FYP/NN/"
h1, h2, o = load_nn(cwd)


### CREATE

tab1.markdown("#### Create")
container = tab1.container(border=True)
container.write(
    "This UI allows the creation of customer neural networks. Users can specify what to train, how many hidden layers and even specify what the activation functions are. Under the hood we have Numpy and modular Python code to train and test the network. Try it for yourself below!"
)


tab1.markdown("#### Add CSV Image Data")
csv_container = tab1.container(border=True)
csv_container.write("In this section you can upload you greyscale image data for a specific class. Match each label with the data then press submit!")

# data uploading
left, middle, right = tab1.columns([1, 1, 1])
csv_data = []  # (UploadedFile, Target)

# Rendering the upload boxes
for i in range(st.session_state.csv_count):
    if i % 3 == 1:
        position = middle
    elif i % 3 == 2:
        position = right
    else:
        position = left

    image_data = position.file_uploader("", accept_multiple_files=False, type="csv", label_visibility="collapsed", key=f"image_data{i}")
    target = position.text_input("", placeholder="Enter label", key=f"targe{i}", label_visibility="collapsed")
    csv_data.append((image_data, target))


# Adding CSV DATA
add_csv_button = tab1.button("Add new CSV Data")

if add_csv_button:
    st.session_state.csv_count += 1
    st.rerun()


# NN Layers
activation_functions = ("ReLu", "Softmax")  # Once we have more activation functions this will be expaned
tab1.markdown("#### Add Network Layers")

with tab1.container(border=True):
    st.write("Add layers to the network and configure the activation functions")

# UI for assigning the layer size and the activation function per layer
col1, col2 = tab1.columns([1, 1])
for i in range(st.session_state.counter):
    value = col1.number_input("", value=0, min_value=0, max_value=512, placeholder="Hidden Layer Size", key=f"Layer_size{i}")
    activation = col2.selectbox("", activation_functions, key=f"Activation{i}")
    st.session_state.nn_dict[st.session_state.counter] = (value, activation)


add_layer = tab1.button("Add Layer")


if add_layer:
    st.session_state.counter += 1
    st.rerun()


# NN Last Layer
tab1.markdown("### Configure the last layer")
with tab1.container(border=True):
    st.write(
        "The last layer has been configure for you. It has gathered the number of classes and is now running that through the specified function"
    )

col1, col2 = tab1.columns([1, 1])
last_layer_size = col1.number_input("CSV Count", value=st.session_state.csv_count, disabled=True, label_visibility="collapsed")
last_layer_activation = col2.selectbox("", "Softmax", label_visibility="collapsed")


# LEARNING RATE AND EPOCHS
tab1.markdown("#### Final Adjustments")
with tab1.container(border=True):
    st.write("This is the final section with some miscilanoues settings for the Neural Net")

# Getting last layer data
col1, col2 = tab1.columns([1, 1])
LEARNING_RATE: int = col1.number_input("Learning rate: ", min_value=0.01, max_value=1.0, value=0.01)
EPOCH_COUNT: int = col2.number_input("Number of epochs: ", min_value=1, max_value=100, value=1)


train_button = tab1.button("Train", type="primary")

### Pressing the train button will:
# 1. Validate csv data and save it locally
# 2. Add the last layer data to the nn_dict
# 3. Call train function

message = tab1.empty()

if train_button and validate_csv_data(csv_data, tab1):
    # Note: this is added at the end since we want the layer layer vars to be at the end of the dict
    st.session_state.nn_dict[st.session_state.counter + 1] = (last_layer_size, last_layer_activation)

    # TODO CLEAR THE EXISTING PATHNAMES FROM SESSION AND LOCALLY

    # Creating a dict with key: pathname and value: target for the csv uploads
    for csv in csv_data:
        uploaded, target = csv
        absolute_path = os.path.join(cwd, uploaded.name)
        st.session_state.csv_pathnames[absolute_path] = target

        # saving the files locally
        with open(absolute_path, "wb") as f:
            f.write(uploaded.getbuffer())

    # Calling the train helper function
    train(st.session_state.nn_dict, st.session_state.csv_pathnames, cwd, message, LEARNING_RATE, EPOCH_COUNT)


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
the model achieved > 85\\% \accuracy on unseen data.



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
classification_image = tab2.file_uploader("", accept_multiple_files=False, label_visibility="collapsed", key=st.session_state.upload_hash)

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
    st.rerun()  # Run the script again


### ABOUT ###
container3 = tab3.container(border=True)
container3.write("This interactive mode is a part of the Final Year Project for Student 123425562")
