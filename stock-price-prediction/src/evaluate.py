import os
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

from preprocessing import prepare_data


MODEL_PATH = "models/stock_rnn.keras"
SCALER_PATH = "models/scaler.pkl"
HISTORY_PATH = "models/training_history.json"

SEQUENCE_LENGTH = 60

OUTPUT_DIR = "outputs"


def create_output_directory():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )


def load_trained_model():

    print("Loading trained model...")

    model = load_model(MODEL_PATH)

    print("Model loaded successfully.")

    return model


def load_trained_scaler():

    print("Loading scaler...")

    scaler = joblib.load(SCALER_PATH)

    print("Scaler loaded successfully.")

    return scaler


def load_training_history():

    print("Loading training history...")

    with open(
        HISTORY_PATH,
        "r",
    ) as file:
        history = json.load(file)

    return history


def read_command_line_options():
    parser = argparse.ArgumentParser(description="Evaluate the stock price model.")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def get_history_values(history):
    if "history" in history:
        return history["history"]

    return history


def plot_actual_vs_predicted(
    actual_prices,
    predicted_prices,
    show_plot=True,
):

    plt.figure(figsize=(12, 6))

    plt.plot(
        actual_prices,
        label="Actual",
    )

    plt.plot(
        predicted_prices,
        label="Predicted",
    )

    plt.title("RNN Stock Price Prediction - Test Data Only")

    plt.xlabel("Sequence")

    plt.ylabel("Stock Price")

    plt.legend()

    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/actual_vs_predicted.png"

    plt.savefig(
        output_path,
        dpi=150,
    )

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Graph saved to: {output_path}")


def plot_training_history(history, show_plot=True):
    history_values = get_history_values(history)

    plt.figure(figsize=(12, 6))

    plt.plot(
        history_values["loss"],
        label="Training Loss",
    )

    plt.plot(
        history_values["val_loss"],
        label="Validation Loss",
    )

    plt.title("Training vs Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/training_vs_validation_loss.png"

    plt.savefig(
        output_path,
        dpi=150,
    )

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Graph saved to: {output_path}")


def evaluate_model():
    options = read_command_line_options()

    print("=" * 60)
    print("RNN MODEL EVALUATION")
    print("=" * 60)

    create_output_directory()

    # --------------------------------
    # 1. Prepare data
    # --------------------------------

    print("\n[1] Preparing data...")

    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
    ) = prepare_data(
        file_path="data/stock_data.csv",
        sequence_length=SEQUENCE_LENGTH,
        train_ratio=0.8,
    )

    # --------------------------------
    # 2. Load model
    # --------------------------------

    print("\n[2] Loading model...")

    model = load_trained_model()

    # --------------------------------
    # 3. Predict test data
    # --------------------------------

    print("\n[3] Predicting test data...")

    predictions = model.predict(
        X_test,
        verbose=0,
    )

    # --------------------------------
    # 4. Convert to original prices
    # --------------------------------

    predicted_prices = scaler.inverse_transform(predictions)

    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    errors = actual_prices - predicted_prices
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / actual_prices)) * 100

    print("\nEvaluation metrics:")
    print(f"MAE  : ${mae:.2f}")
    print(f"RMSE : ${rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    # --------------------------------
    # 5. Actual vs Predicted
    # --------------------------------

    print("\n[4] Creating Actual vs Predicted graph...")

    plot_actual_vs_predicted(
        actual_prices,
        predicted_prices,
        show_plot=not options.no_plot,
    )

    # --------------------------------
    # 6. Training vs Validation Loss
    # --------------------------------

    print("\n[5] Creating Training vs Validation graph...")

    history = load_training_history()

    plot_training_history(
        history,
        show_plot=not options.no_plot,
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()
