import numpy as np
import pandas as pd


def create_sequences(data, sequence_length=5):
    X = []
    y = []

    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])

    return np.array(X), np.array(y)


# Load dataset
df = pd.read_csv("data/temperature.csv")

print("Original Data:")
print(df.head())

# Extract temperature values
temperatures = df["temperature"].values

print("\nTemperature values:")
print(temperatures)

# Create sequences
X, y = create_sequences(temperatures, sequence_length=5)
# Reshape for RNN
X = X.reshape((X.shape[0], X.shape[1], 1))

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst 5 sequences:")

for i in range(5):
    print("X:", X[i], "y:", y[i])
