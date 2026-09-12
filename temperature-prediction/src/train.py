import argparse

from data_utils import (
    MODEL_FILE,
    get_scaling_numbers,
    load_temperatures,
    make_sequences,
    save_scaling_numbers,
    scale_temperatures,
    setup_runtime,
)

setup_runtime()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import create_model


def read_command_line_options():
    parser = argparse.ArgumentParser(description="Train the temperature RNN model.")

    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rnn-units", type=int, default=32)

    return parser.parse_args()


def main():
    options = read_command_line_options()

    print("Loading temperature data...")
    temperatures = load_temperatures()

    print("Scaling temperatures...")
    mean, standard_deviation = get_scaling_numbers(temperatures)
    scaled_temperatures = scale_temperatures(
        temperatures,
        mean,
        standard_deviation,
    )

    print("Creating training examples...")
    X, y = make_sequences(scaled_temperatures, options.sequence_length)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("Creating the RNN model...")
    model = create_model(
        sequence_length=options.sequence_length,
        rnn_units=options.rnn_units,
    )

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    # EarlyStopping stops training when validation loss stops improving.
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
        verbose=1,
    )

    # ModelCheckpoint saves the best model during training.
    save_best_model = ModelCheckpoint(
        MODEL_FILE,
        monitor="val_loss",
        save_best_only=True,
        verbose=0,
    )

    print("Training started...")
    history = model.fit(
        X,
        y,
        epochs=options.epochs,
        batch_size=options.batch_size,
        validation_split=0.2,
        shuffle=False,
        callbacks=[early_stopping, save_best_model],
        verbose=1,
    )

    print("Saving model and scaling numbers...")
    model.save(MODEL_FILE)
    save_scaling_numbers(mean, standard_deviation, options.sequence_length)

    final_training_loss = history.history["loss"][-1]
    best_validation_loss = min(history.history["val_loss"])

    print("\nTraining complete!")
    print(f"Final training loss: {final_training_loss:.6f}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Model saved at: {MODEL_FILE}")


if __name__ == "__main__":
    main()
