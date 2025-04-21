from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

print(dataset[:10])

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)


index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])



# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """

    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(m, Tx, n_s)(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    # For testing purposes, please list 'a' first and 's_prev' second, in this order.
    concat = concatenator(a, s_prev)
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a", in this order, to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor(alphas)

    return context



n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64 # number of units for the post-attention LSTM's hidden state "s"

# Please note, this is the post attention LSTM cell.
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)


def modelf(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size=None):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- integer, optional, size of the python dictionary "machine_vocab"
                          This is not being used

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx, human_vocab_size)
    # Define s0 (initial hidden state) and c0 (initial cell state)
    # for the decoder LSTM with shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    # initial hidden state
    s0 = Input(shape=(n_s,), name='s0')
    # initial cell state
    c0 = Input(shape=(n_s,), name='c0')
    # hidden state
    s = s0
    # cell state
    c = c0

    # Initialize empty list of outputs
    outputs = []

    ### START CODE HERE ###

    # Step 1: Define your pre-attention Bi-LSTM. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_state = True))(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector. (≈ 1 line)
        # Don't forget to pass: initial_state = [hidden state, cell state]
        # Remember: s = hidden state, c = cell state
        _, s, c = post_activation_LSTM_cell(inputs=context, initial_state=[s0, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c], outputs=out)

    ### END CODE HERE ###

    return model


# UNIT TEST 1
from test_utils import *


def modelf_test(target):
    Tx = 30
    n_a = 32
    n_s = 64
    len_human_vocab = 37

    model = target(Tx, Ty, n_a, n_s, len_human_vocab)

    print(summary(model))

    expected_summary = [['InputLayer', 0],
                        ['InputLayer',  0],
                        ['Bidirectional',  17920],
                        ['RepeatVector',  0, 30],
                        ['Concatenate',  0],
                        ['Dense',  1290, 'tanh'],
                        ['Dense',  11, 'relu'],
                        ['Activation', 0],
                        ['Dot',  0],
                        ['InputLayer',  0],
                        ['LSTM',  33024,
                         'tanh'],
                        ['Dense', 715, 'softmax']]

    assert len(model.outputs) == 10, f"Wrong output shape. Expected 10 != {len(model.outputs)}"

    comparator(summary(model), expected_summary)


modelf_test(modelf)

# UNIT TEST 2

def modelf_states_test(target):
    Tx = 30
    n_a = 32
    n_s = 64
    len_human_vocab = 37

    model = target(Tx, Ty, n_a, n_s, len_human_vocab)

    # Create test inputs
    X_test = np.random.rand(1, Tx, len_human_vocab)
    s0_test = np.zeros((1, n_s))
    c0_test = np.zeros((1, n_s))

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, Tx, len_human_vocab], dtype=tf.float32),
        tf.TensorSpec(shape=[None, n_s], dtype=tf.float32),
        tf.TensorSpec(shape=[None, n_s], dtype=tf.float32)
    ])
    def predict_function(X, s0, c0):
        # Call the model directly with input tensors
        return model([X, s0, c0])

    # Get the outputs of the model for the first five time steps
    outputs = predict_function(X_test, s0_test, c0_test)

    # Extract the hidden states (s) from the LSTM outputs for each time step
    hidden_states = [np.array(output) for output in outputs]

    # Check if consecutive hidden states are different
    for i in range(len(hidden_states) - 1):
        assert not np.allclose(hidden_states[i], hidden_states[i + 1]), (
            "Consecutive hidden states should be different.\n"
            "Check if the LSTM cell is using the correct previous states.\n"
            "Make sure you are using s and c, and NOT using s0 and c0 in Step 2.B."
        )

    print("\033[32mAll tests passed!\033[0m")

modelf_states_test(modelf)

model = modelf(Tx, Ty, n_a, n_s, len(human_vocab))
model.summary()

# ## TODO - Exercise 3
# opt = Adam( Use given parameters in problem description) # Adam(...)
# model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'] * Ty)
#
# # UNIT TESTS
# assert opt.learning_rate == 0.005, "Set the lr parameter to 0.005"
# assert opt.beta_1 == 0.9, "Set the beta_1 parameter to 0.9"
# assert opt.beta_2 == 0.999, "Set the beta_2 parameter to 0.999"
# assert opt.weight_decay == 0.01, "Set the decay parameter to 0.01"
# assert model.loss == "categorical_crossentropy", "Wrong loss. Use 'categorical_crossentropy'"
# assert model.optimizer == opt, "Use the optimizer that you have instantiated"
#
# print("\033[92mAll tests passed!")
#
# s0 = np.zeros((m, n_s))
# c0 = np.zeros((m, n_s))
# outputs = list(Yoh.swapaxes(0,1))
#
# model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
# model.load_weights('models/model.h5')
# EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
# s00 = np.zeros((1, n_s))
# c00 = np.zeros((1, n_s))
# for example in EXAMPLES:
#     source = string_to_int(example, Tx, human_vocab)
#     #print(source)
#     source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
#     source = np.swapaxes(source, 0, 1)
#     source = np.expand_dims(source, axis=0)
#     prediction = model.predict([source, s00, c00])
#     prediction = np.argmax(prediction, axis = -1)
#     output = [inv_machine_vocab[int(i)] for i in prediction]
#     print("source:", example)
#     print("output:", ''.join(output),"\n")
