import pandas as pd
import numpy as np

from model import create_model


def create_sequences(data, sequence_length=5):
    X = []
    y = []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    return np.array(X), np.array(y)


# Load data
df = pd.read_csv("data/temperature.csv")

temperatures = df["temperature"].values.astype("float32")

# Create sequences
X, y = create_sequences(temperatures, sequence_length=5)

# Reshape for RNN
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X shape:", X.shape)
print("y shape:", y.shape)

# Create model
model = create_model()

# Train
history = model.fit(
    X,
    y,
    epochs=200,
    verbose=1
)

# Save model
model.save("models/temperature_rnn.keras")

print("\nModel saved successfully!")