from pathlib import Path

import joblib

from preprocessing import prepare_data
from model import build_rnn_model


# -----------------------------
# Configuration
# -----------------------------

SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.8

EPOCHS = 20
BATCH_SIZE = 32

MODEL_PATH = "models/stock_rnn.keras"
SCALER_PATH = "models/scaler.pkl"


# -----------------------------
# Train model
# -----------------------------


def train_model():

    print("=" * 50)
    print("STOCK PRICE RNN - TRAINING")
    print("=" * 50)

    # 1. Prepare data
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
        train_ratio=TRAIN_RATIO,
    )

    # 2. Build model
    print("\n[2] Building RNN model...")

    model = build_rnn_model(
        sequence_length=SEQUENCE_LENGTH,
        number_of_features=1,
    )

    print("\nModel summary:")
    model.summary()

    # 3. Train model
    print("\n[3] Training model...")

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    # 4. Evaluate model
    print("\n[4] Evaluating model...")

    test_loss = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    print(f"Test Loss: {test_loss:.6f}")

    # 5. Create models directory
    Path("models").mkdir(
        parents=True,
        exist_ok=True,
    )

    # 6. Save model
    print("\n[5] Saving model...")

    model.save(MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")

    # 7. Save scaler
    print("\n[6] Saving scaler...")

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    print(f"Scaler saved to: {SCALER_PATH}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)

    return model, history, scaler


if __name__ == "__main__":
    train_model()
