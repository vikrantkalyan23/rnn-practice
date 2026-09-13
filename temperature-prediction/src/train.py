import argparse

from data_utils import (
    MODEL_FILE,
    get_scaling_numbers,
    load_temperatures,
    make_sequences,
    save_scaling_numbers,
    scale_temperatures,
    setup_runtime,
    split_temperatures,
)

setup_runtime()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import create_model
import matplotlib.pyplot as plt


def read_command_line_options():
    parser = argparse.ArgumentParser(description="Train the temperature RNN model.")

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--rnn-units",
        type=int,
        default=32,
    )

    return parser.parse_args()


def main():
    options = read_command_line_options()

    print("Loading temperature data...")

    temperatures = load_temperatures()

    # ---------------------------------------
    # 1. Chronological train/test split
    # ---------------------------------------

    train_temperatures, test_temperatures = split_temperatures(
        temperatures,
        test_size=0.2,
    )

    print("\nData split:")
    print("Total:", len(temperatures))
    print("Train:", len(train_temperatures))
    print("Test :", len(test_temperatures))

    # ---------------------------------------
    # 2. Calculate scaling ONLY from training
    # ---------------------------------------

    mean, standard_deviation = get_scaling_numbers(train_temperatures)

    print("\nScaling numbers:")
    print("Mean:", mean)
    print("Standard deviation:", standard_deviation)

    # ---------------------------------------
    # 3. Scale train and test using
    #    training statistics
    # ---------------------------------------

    scaled_train = scale_temperatures(
        train_temperatures,
        mean,
        standard_deviation,
    )

    scaled_test = scale_temperatures(
        test_temperatures,
        mean,
        standard_deviation,
    )

    # ---------------------------------------
    # 4. Create training sequences
    # ---------------------------------------

    X_train, y_train = make_sequences(
        scaled_train,
        options.sequence_length,
    )

    print("\nTraining shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    # ---------------------------------------
    # 5. Create model
    # ---------------------------------------

    model = create_model(
        sequence_length=options.sequence_length,
        rnn_units=options.rnn_units,
    )

    # ---------------------------------------
    # 6. Callbacks
    # ---------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
        verbose=1,
    )

    save_best_model = ModelCheckpoint(
        MODEL_FILE,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # ---------------------------------------
    # 7. Train
    # ---------------------------------------

    history = model.fit(
        X_train,
        y_train,
        epochs=options.epochs,
        batch_size=options.batch_size,
        validation_split=0.2,
        # IMPORTANT for time-series
        shuffle=False,
        callbacks=[
            early_stopping,
            save_best_model,
        ],
        verbose=1,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # 8. Save scaling numbers
    # ---------------------------------------

    save_scaling_numbers(
        mean,
        standard_deviation,
        options.sequence_length,
    )

    # ---------------------------------------
    # 9. Print training information
    # ---------------------------------------

    best_epoch = history.history["val_loss"].index(min(history.history["val_loss"])) + 1

    print("\nTraining complete!")

    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")

    print(f"Best epoch: {best_epoch}")

    print(f"Model saved at: {MODEL_FILE}")


if __name__ == "__main__":
    main()
