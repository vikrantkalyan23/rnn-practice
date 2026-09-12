import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


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

# Reshape
X = X.reshape((X.shape[0], X.shape[1], 1))


# Load model
model = load_model("models/temperature_rnn.keras")


# Predictions
predictions = model.predict(X, verbose=0).flatten()


# Print results
for actual, predicted in zip(y, predictions):
    print(
        f"Actual: {actual:.2f} | "
        f"Predicted: {predicted:.2f}"
    )


# Plot
plt.figure(figsize=(10, 5))

plt.plot(y, label="Actual")
plt.plot(predictions, label="Predicted")

plt.xlabel("Sequence")
plt.ylabel("Temperature")

plt.title("RNN Temperature Prediction")

plt.legend()
plt.show()