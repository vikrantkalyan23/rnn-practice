from pathlib import Path
import json

import joblib

from preprocessing import prepare_data
from model import build_rnn_model


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.8

EPOCHS = 20
BATCH_SIZE = 32

DATA_PATH = "data/stock_data.csv"

MODEL_PATH = "models/stock_rnn.keras"
SCALER_PATH = "models/scaler.pkl"
HISTORY_PATH = "models/training_history.json"


# ============================================================
# Train Model
# ============================================================


def train_model():

    print("=" * 60)
    print("STOCK PRICE RNN - TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Prepare Data
    # --------------------------------------------------------

    print("\n[1] Preparing data...")

    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
    ) = prepare_data(
        file_path=DATA_PATH,
        sequence_length=SEQUENCE_LENGTH,
        train_ratio=TRAIN_RATIO,
    )

    print("\nData preparation completed.")

    print(f"X_train shape: {X_train.shape}")

    print(f"y_train shape: {y_train.shape}")

    print(f"X_test shape : {X_test.shape}")

    print(f"y_test shape : {y_test.shape}")

    # --------------------------------------------------------
    # 2. Build RNN Model
    # --------------------------------------------------------

    print("\n[2] Building RNN model...")

    model = build_rnn_model(
        sequence_length=SEQUENCE_LENGTH,
        number_of_features=1,
    )

    print("\nModel summary:")

    model.summary()

    # --------------------------------------------------------
    # 3. Train Model
    # --------------------------------------------------------

    print("\n[3] Training model...")

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    print("\nTraining completed.")

    # --------------------------------------------------------
    # 4. Evaluate Model
    # --------------------------------------------------------

    print("\n[4] Evaluating model on test data...")

    test_loss = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    print(f"Test Loss: {test_loss:.6f}")

    # --------------------------------------------------------
    # 5. Create Models Directory
    # --------------------------------------------------------

    print("\n[5] Creating models directory...")

    Path("models").mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 6. Save Trained Model
    # --------------------------------------------------------

    print("\n[6] Saving trained model...")

    model.save(MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")

    # --------------------------------------------------------
    # 7. Save Scaler
    # --------------------------------------------------------

    print("\n[7] Saving scaler...")

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    print(f"Scaler saved to: {SCALER_PATH}")

    # --------------------------------------------------------
    # 8. Save Training History
    # --------------------------------------------------------

    print("\n[8] Saving training history...")

    with open(
        HISTORY_PATH,
        "w",
    ) as file:
        json.dump(
            history.history,
            file,
            indent=4,
        )

    print(f"Training history saved to: {HISTORY_PATH}")

    # --------------------------------------------------------
    # 9. Display Final Training Results
    # --------------------------------------------------------

    print("\n[9] Final training results...")

    final_training_loss = history.history["loss"][-1]

    final_validation_loss = history.history["val_loss"][-1]

    print(f"Final Training Loss   : {final_training_loss:.6f}")

    print(f"Final Validation Loss : {final_validation_loss:.6f}")

    # --------------------------------------------------------
    # 10. Training Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated files:")

    print(f"Model   : {MODEL_PATH}")

    print(f"Scaler  : {SCALER_PATH}")

    print(f"History : {HISTORY_PATH}")

    print("=" * 60)

    return (
        model,
        history,
        scaler,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train_model()
