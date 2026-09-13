import numpy as np
import tensorflow as tf


class TemperaturePredictor:
    def __init__(self):

        # Load trained RNN model
        self.model = tf.keras.models.load_model("models/temperature_rnn.keras")

        # Load scaler information
        scaler = np.load("models/temperature_scaler.npz")

        self.mean = float(scaler["mean"])
        self.standard_deviation = float(scaler["standard_deviation"])

        self.sequence_length = int(scaler["sequence_length"])

        print("Model loaded successfully")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Mean: {self.mean}")
        print(f"Standard deviation: {self.standard_deviation}")

    def predict(self, temperatures):

        # Convert input to NumPy array
        temperatures = np.array(temperatures, dtype=np.float32)

        # Validate number of temperatures
        if len(temperatures) != self.sequence_length:
            raise ValueError(
                f"Expected {self.sequence_length} "
                f"temperatures, but received "
                f"{len(temperatures)}"
            )

        # Scale input using training parameters
        scaled = (temperatures - self.mean) / self.standard_deviation

        # RNN expects:
        # (batch_size, time_steps, features)
        X = scaled.reshape(1, self.sequence_length, 1)

        # Make prediction
        prediction = self.model.predict(X, verbose=0)

        # Convert prediction back to original
        # temperature scale
        prediction = prediction[0][0] * self.standard_deviation + self.mean

        return float(prediction)
