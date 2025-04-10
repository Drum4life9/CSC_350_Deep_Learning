import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt

def build_model(apply_clipping=False):
    # model architecture
    model = Sequential([
        Embedding(input_dim=10000, output_dim=32, input_length=100),
        LSTM(100),
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

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)

    model_without_clipping = build_model(apply_clipping=True)
    history_without_clipping = model_without_clipping.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
    plot_history([('Without Clipping', history_without_clipping)], key='loss')
    plt.show()
