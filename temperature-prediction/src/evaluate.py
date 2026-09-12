import argparse

from data_utils import (
    MODEL_PATH,
    configure_runtime,
    create_sequences,
    inverse_transform,
    load_standardizer,
    load_temperatures,
    transform,
)

configure_runtime()

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the temperature RNN.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the chart display.")
    return parser.parse_args()


def main():
    """Compare model predictions with the real temperatures in the CSV file."""
    args = parse_args()

    # Load the scaling settings saved by train.py.
    mean, std, sequence_length = load_standardizer()

    # Prepare the same kind of examples used during training.
    temperatures = load_temperatures()
    scaled_temperatures = transform(temperatures, mean, std)
    X, y_scaled = create_sequences(scaled_temperatures, sequence_length)

    # Predict scaled values, then convert them back to real temperatures.
    model = load_model(MODEL_PATH)
    predictions_scaled = model.predict(X, verbose=0).flatten()
    y = inverse_transform(y_scaled, mean, std)
    predictions = inverse_transform(predictions_scaled, mean, std)

    # MAE means "mean absolute error": the average prediction mistake.
    mae = abs(y - predictions).mean()
    print(f"Mean absolute error: {mae:.2f}")

    for actual, predicted in zip(y, predictions):
        print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f}")

    if not args.no_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(y, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.xlabel("Sequence")
        plt.ylabel("Temperature")
        plt.title("RNN Temperature Prediction")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
