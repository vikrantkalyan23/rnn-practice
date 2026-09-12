import os
from pathlib import Path

import numpy as np
import pandas as pd


# These paths are built from this file's location, so scripts work even when
# you run them from a different folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "temperature.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "temperature_rnn.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "temperature_scaler.npz"
MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"


def configure_runtime():
    """Use a project-local Matplotlib cache to avoid home-folder warnings."""
    MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))


def load_temperatures(data_path=DATA_PATH):
    """Read the CSV file and return only the temperature column."""
    df = pd.read_csv(data_path)
    if "temperature" not in df.columns:
        raise ValueError("Expected a 'temperature' column in the dataset.")
    return df["temperature"].to_numpy(dtype=np.float32)


def create_sequences(data, sequence_length=5):
    """Turn a list of temperatures into input/output examples.

    Example with sequence_length=5:
    input  = [20, 21, 22, 23, 24]
    target = 25

    The RNN learns many examples like this, then predicts the next value.
    """
    data = np.asarray(data, dtype=np.float32)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero.")
    if len(data) <= sequence_length:
        raise ValueError("Not enough data points to create sequences.")

    inputs = []
    targets = []

    for start_index in range(len(data) - sequence_length):
        end_index = start_index + sequence_length
        inputs.append(data[start_index:end_index])
        targets.append(data[end_index])

    X = np.array(inputs, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    # Keras RNNs expect 3 dimensions:
    # (number_of_examples, sequence_length, number_of_features)
    return X.reshape((X.shape[0], X.shape[1], 1)), y


def fit_standardizer(values):
    """Calculate the mean and standard deviation used for scaling."""
    values = np.asarray(values, dtype=np.float32)
    mean = np.float32(values.mean())
    std = np.float32(values.std())
    if std == 0:
        raise ValueError("Temperature values have zero variance.")
    return mean, std


def transform(values, mean, std):
    """Scale values so the model trains more smoothly."""
    return (np.asarray(values, dtype=np.float32) - mean) / std


def inverse_transform(values, mean, std):
    """Convert scaled model output back to real temperatures."""
    return (np.asarray(values, dtype=np.float32) * std) + mean


def save_standardizer(mean, std, sequence_length, scaler_path=SCALER_PATH):
    """Save scaling settings next to the trained model."""
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(scaler_path, mean=mean, std=std, sequence_length=sequence_length)


def load_standardizer(scaler_path=SCALER_PATH):
    """Load the scaling settings that were saved during training."""
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler metadata not found at {scaler_path}. Run src/train.py first."
        )

    data = np.load(scaler_path)
    return (
        np.float32(data["mean"]),
        np.float32(data["std"]),
        int(data["sequence_length"]),
    )
