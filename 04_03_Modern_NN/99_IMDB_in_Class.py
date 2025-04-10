import tensorflow as tf
from keras.src.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l


def get_original_text(i, x_test):
    word_to_id = imdb.get_word_index()
    word_to_id = {k:(v+3) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value:key for key,value in word_to_id.items()}
    return ' '.join(id_to_word[id] for id in x_test[i])



SentimentDict={1:'positive', 0:'negative'}
def display_test_sentiment(i, predict_classes, x_test):
    print(get_original_text(i, x_test))
    print('label: ', SentimentDict[y_test[i]], ', prediction: ', SentimentDict[predict_classes[i]])


class GRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens)

def build_model(apply_clipping=False, use_GRU=False):
    # model architecture
    if use_GRU:
        model = Sequential([
            Embedding(input_dim=10000, output_dim=32, input_length=100),
            LSTM(100),
            Dense(1, activation='sigmoid')
        ])
    else:
        gru = GRU(num_inputs=10000, num_hiddens=32)
        model = Sequential([
            Embedding(input_dim=10000, output_dim=32, input_length=100),
            gru,
            Dense(1, activation='sigmoid')
        ])


    # configure optimizer with gradient clipping
    if apply_clipping:
        optimizer = tf.keras.optimizers.Adam(clipvalue=1.0)
    else:
        optimizer = 'adam'

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_history(histories, key='loss'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title() + 'Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),label=name.title() + 'Train')
        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_', ' ').title())
        plt.legend()
        plt.xlim([0, max(history.epoch)])


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)

model_without_clipping = build_model(apply_clipping=False)
history_without_clipping = model_without_clipping.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.1)
plot_history([('Without Clipping', history_without_clipping)], key='accuracy')
plt.show()
