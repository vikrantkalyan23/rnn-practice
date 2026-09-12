import argparse

from data_utils import (
    MODEL_FILE,
    load_scaling_numbers,
    load_temperatures,
    make_sequences,
    scale_temperatures,
    setup_runtime,
    unscale_temperatures,
)

setup_runtime()

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def read_command_line_options():
    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    options = read_command_line_options()

    print("Loading model and scaling numbers...")
    model = load_model(MODEL_FILE)
    mean, standard_deviation, sequence_length = load_scaling_numbers()

    print("Preparing evaluation data...")
    temperatures = load_temperatures()
    scaled_temperatures = scale_temperatures(
        temperatures,
        mean,
        standard_deviation,
    )
    X, y = make_sequences(scaled_temperatures, sequence_length)

    print("Making predictions...")
    scaled_predictions = model.predict(X, verbose=0).flatten()

    actual_temperatures = unscale_temperatures(y, mean, standard_deviation)
    predicted_temperatures = unscale_temperatures(
        scaled_predictions,
        mean,
        standard_deviation,
    )

    errors = abs(actual_temperatures - predicted_temperatures)
    mean_absolute_error = errors.mean()

    print(f"\nMean absolute error: {mean_absolute_error:.2f}")
    print("\nActual vs predicted:")

    for actual, predicted in zip(actual_temperatures, predicted_temperatures):
        print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f}")

    if options.no_plot:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(actual_temperatures, label="Actual")
    plt.plot(predicted_temperatures, label="Predicted")
    plt.xlabel("Sequence")
    plt.ylabel("Temperature")
    plt.title("RNN Temperature Prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
