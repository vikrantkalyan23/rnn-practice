import argparse

import numpy as np

from data_utils import (
    MODEL_FILE,
    load_scaling_numbers,
    scale_temperatures,
    setup_runtime,
    unscale_temperatures,
)

setup_runtime()

from tensorflow.keras.models import load_model


def read_command_line_options():
    parser = argparse.ArgumentParser(description="Predict the next temperature.")

    parser.add_argument(
        "temperatures",
        nargs="*",
        type=float,
        default=[45, 46, 47, 48, 49],
        help="Example: python src/predict.py 45 46 47 48 49",
    )

    return parser.parse_args()


def main():
    options = read_command_line_options()

    print("Loading model and scaling numbers...")
    model = load_model(MODEL_FILE)
    mean, standard_deviation, sequence_length = load_scaling_numbers()

    input_temperatures = np.array(options.temperatures, dtype=np.float32)

    if len(input_temperatures) != sequence_length:
        raise ValueError(
            f"Please enter exactly {sequence_length} temperatures. "
            f"You entered {len(input_temperatures)}."
        )

    scaled_input = scale_temperatures(
        input_temperatures,
        mean,
        standard_deviation,
    )

    # The model expects shape: 1 example, sequence_length time steps, 1 feature.
    model_input = scaled_input.reshape(1, sequence_length, 1)

    scaled_prediction = model.predict(model_input, verbose=0).flatten()[0]
    prediction = unscale_temperatures(
        scaled_prediction,
        mean,
        standard_deviation,
    )

    print("Input temperatures:", input_temperatures.astype(int).tolist())
    print(f"Predicted next temperature: {prediction:.2f}")


if __name__ == "__main__":
    main()
