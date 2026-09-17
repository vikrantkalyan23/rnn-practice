import os
from pathlib import Path

import numpy as np
import pandas as pd


# Main project folders and files.
# pathlib helps us build file paths that work on macOS, Windows, and Linux.
PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_FOLDER / "data" / "temperature.csv"
MODEL_FILE = PROJECT_FOLDER / "models" / "temperature_rnn.keras"
SCALER_FILE = PROJECT_FOLDER / "models" / "temperature_scaler.npz"
CACHE_FOLDER = PROJECT_FOLDER / ".cache" / "matplotlib"


def setup_runtime():
    # Matplotlib sometimes wants to write cache files in the user home folder.
    # This keeps those cache files inside this project instead.
    CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_FOLDER))


def load_temperatures():
    # Read the CSV file.
    data_frame = pd.read_csv(DATA_FILE)

    # A small safety check makes errors easier to understand.
    if "temperature" not in data_frame.columns:
        raise ValueError("The CSV file must contain a 'temperature' column.")

    # The model only needs the temperature numbers.
    temperatures = data_frame["temperature"].to_numpy(dtype=np.float32)
    return temperatures


def make_sequences(temperatures, sequence_length):
    # This function converts one long list of temperatures into many examples.
    #
    # Example:
    # [20, 21, 22, 23, 24] becomes the input
    # 25 becomes the answer the model should learn to predict
    X = []
    y = []

    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0.")

    if len(temperatures) <= sequence_length:
        raise ValueError("There is not enough data to make training sequences.")

    for i in range(len(temperatures) - sequence_length):
        input_sequence = temperatures[i : i + sequence_length]
        next_temperature = temperatures[i + sequence_length]

        X.append(input_sequence)
        y.append(next_temperature)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # RNN input must have this shape:
    # number of examples, number of time steps, number of features
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y


def get_scaling_numbers(temperatures):
    # Scaling means changing values so they are easier for the model to learn.
    mean = np.float32(temperatures.mean())
    standard_deviation = np.float32(temperatures.std())

    if standard_deviation == 0:
        raise ValueError("All temperatures are the same, so scaling is not possible.")

    return mean, standard_deviation


def scale_temperatures(temperatures, mean, standard_deviation):
    # Formula: scaled value = (value - mean) / standard deviation
    return (np.asarray(temperatures, dtype=np.float32) - mean) / standard_deviation


def unscale_temperatures(scaled_temperatures, mean, standard_deviation):
    # This reverses the scaling formula.
    return (
        np.asarray(scaled_temperatures, dtype=np.float32) * standard_deviation
    ) + mean


def save_scaling_numbers(mean, standard_deviation, sequence_length):
    # Prediction needs the same scaling numbers that training used.
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        SCALER_FILE,
        mean=mean,
        standard_deviation=standard_deviation,
        sequence_length=sequence_length,
    )


def load_scaling_numbers():
    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            "Scaling file not found. Please run this first: python src/train.py"
        )

    saved_data = np.load(SCALER_FILE)

    mean = np.float32(saved_data["mean"])

    if "standard_deviation" in saved_data:
        standard_deviation = np.float32(saved_data["standard_deviation"])
    else:
        # Older versions of this app saved the same value as "std".
        standard_deviation = np.float32(saved_data["std"])

    sequence_length = int(saved_data["sequence_length"])

    return mean, standard_deviation, sequence_length


def split_temperatures(temperatures, test_size=0.2):
    """
    Split time-series data chronologically.

    The first part is training data.
    The last part is test data.

    We do NOT shuffle because time order matters.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(temperatures) * (1 - test_size))

    train = temperatures[:split_index]
    test = temperatures[split_index:]

    return train, test
