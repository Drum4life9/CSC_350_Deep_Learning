import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define AR(3) model parameters
a1, a2, a3 = 0.6, -0.5, -0.2
T = 200  # Length of each sequence
num_samples = 1000  # Number of sequences


# Function to generate AR(3) sequences
def generate_ar3_data(num_samples, T):
    data = np.zeros((num_samples, T))

    for i in range(num_samples):
        X = np.zeros(T)
        for t in range(3, T):
            U_t = np.random.uniform(0, 0.1)  # White noise
            X[t] = a1 * X[t - 1] + a2 * X[t - 2] + a3 * X[t - 3] + U_t
        data[i] = X
    return data


# Generate data
ar_data = generate_ar3_data(num_samples, T)

# Plot one sample
plt.plot(ar_data[0])
plt.title("Example AR(3) Generated Time Series")
plt.show()



# Define sequence length for training
seq_length = 10

# Prepare input-output pairs
def create_dataset(data, seq_length):
    X, Y = [], []
    for sample in data:
        for i in range(len(sample) - seq_length):
            X.append(sample[i:i+seq_length])
            Y.append(sample[i+seq_length])
    return np.array(X), np.array(Y)

if __name__ == '__main__':
    X, Y = create_dataset(ar_data, seq_length)

    # Split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Reshape for RNN (samples, timesteps, features)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")




    # Define the RNN model
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, activation='relu'),
        Dense(1)  # Output a single value
    ])

    model.compile(optimizer='adam', loss='mse')

    # Display model summary
    model.summary()

    # Train the model
    history = model.fit(X_train, Y_train, epochs=1, batch_size=4, validation_data=(X_test, Y_test))



    # Make predictions
    Y_pred = model.predict(X_test)

    # Compute error
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"Test MSE: {mse:.5f}")

    # Plot actual vs predicted
    plt.plot(Y_test[:100], label="Actual")
    plt.plot(Y_pred[:100], label="Predicted", linestyle="dashed")
    plt.legend()
    plt.title("Actual vs Predicted Time Series")
    plt.show()
