import argparse

from data_utils import (
    MODEL_PATH,
    configure_runtime,
    create_sequences,
    fit_standardizer,
    load_temperatures,
    save_standardizer,
    transform,
)

configure_runtime()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the temperature RNN.")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="How many previous days the model uses to predict the next day.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Maximum number of training passes through the data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="How many examples the model sees before updating its weights.",
    )
    parser.add_argument(
        "--units",
        type=int,
        default=32,
        help="Number of memory units in the SimpleRNN layer.",
    )
    return parser.parse_args()


def main():
    """Train the model and save everything needed for later prediction."""
    args = parse_args()

    # Step 1: Load temperatures from the CSV file.
    temperatures = load_temperatures()

    # Step 2: Scale temperatures before training.
    # Neural networks usually learn faster with values near zero.
    mean, std = fit_standardizer(temperatures)
    scaled_temperatures = transform(temperatures, mean, std)

    # Step 3: Create examples like:
    # [20, 21, 22, 23, 24] -> 25
    X, y = create_sequences(scaled_temperatures, args.sequence_length)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Step 4: Build and train the model.
    model = create_model(sequence_length=args.sequence_length, units=args.units)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # shuffle=False matters for time-series data because order has meaning.
    history = model.fit(
        X,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    # Step 5: Save the model and the scaling settings.
    model.save(MODEL_PATH)
    save_standardizer(mean, std, args.sequence_length)

    print(f"\nFinal training loss: {history.history['loss'][-1]:.6f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
