import numpy as np
from tensorflow.keras.models import load_model


# Load trained model
model = load_model("models/temperature_rnn.keras")


# New temperature sequence
sequence = np.array([45, 46, 47, 48, 49], dtype="float32")


# Reshape for RNN
sequence = sequence.reshape(1, 5, 1)


# Make prediction
prediction = model.predict(sequence, verbose=0)


print("Input sequence:", [45, 46, 47, 48, 49])
print("Predicted next temperature:", prediction[0][0])