import argparse

import numpy as np

from data_utils import (
    MODEL_PATH,
    configure_runtime,
    inverse_transform,
    load_standardizer,
    transform,
)

configure_runtime()

from tensorflow.keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the next temperature.")
    parser.add_argument(
        "temperatures",
        nargs="*",
        type=float,
        default=[45, 46, 47, 48, 49],
        help="Recent temperatures, for example: 45 46 47 48 49",
    )
    return parser.parse_args()


def main():
    """Predict the next temperature from a short sequence of recent values."""
    args = parse_args()

    # Load the trained model and the scaling settings saved by train.py.
    mean, std, sequence_length = load_standardizer()
    model = load_model(MODEL_PATH)

    sequence = np.array(args.temperatures, dtype=np.float32)
    if len(sequence) != sequence_length:
        raise ValueError(
            f"Expected {sequence_length} values, got {len(sequence)} values."
        )

    # Scale the input before prediction because the model was trained on
    # scaled values. Then convert the prediction back to a normal temperature.
    scaled_sequence = transform(sequence, mean, std).reshape(1, sequence_length, 1)
    prediction_scaled = model.predict(scaled_sequence, verbose=0).flatten()
    prediction = inverse_transform(prediction_scaled, mean, std)[0]

    print("Input sequence:", sequence.astype(int).tolist())
    print(f"Predicted next temperature: {prediction:.2f}")


if __name__ == "__main__":
    main()
