import json
import os
import random
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache" / "matplotlib"))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: E402

from app.config import (  # noqa: E402
    BATCH_SIZE,
    DATA_PATH,
    DROPOUT,
    EMBEDDING_DIM,
    EPOCHS,
    L2_STRENGTH,
    LR_REFINEMENT_PATH,
    LSTM_UNITS,
    MAX_VOCAB_SIZE,
    SEQUENCE_LENGTH,
    TEST_RATIO,
    TEST_SPLIT_SEED,
    VALIDATION_RATIO,
    VALIDATION_SEEDS,
)
from app.data import (  # noqa: E402
    create_sequences,
    create_tokenizer,
    get_vocabulary_size,
    split_text_train_validation_test,
)
from app.network import build_model  # noqa: E402


LEARNING_RATES = (0.00125, 0.0015, 0.00175)
MEANINGFUL_ACCURACY_GAIN = 0.02
REFERENCE_ACCURACY = 0.35856543978055316


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def evaluate_learning_rate(learning_rate, text):
    folds = []

    for seed in VALIDATION_SEEDS:
        tf.keras.backend.clear_session()
        set_random_seed(seed)

        train_text, validation_text, _ = split_text_train_validation_test(
            text,
            validation_ratio=VALIDATION_RATIO,
            test_ratio=TEST_RATIO,
            test_seed=TEST_SPLIT_SEED,
            validation_seed=seed,
        )
        tokenizer = create_tokenizer(train_text, MAX_VOCAB_SIZE)
        X_train, y_train = create_sequences(train_text, tokenizer, SEQUENCE_LENGTH)
        X_validation, y_validation = create_sequences(
            validation_text, tokenizer, SEQUENCE_LENGTH
        )
        model = build_model(
            vocab_size=get_vocabulary_size(tokenizer),
            sequence_length=SEQUENCE_LENGTH,
            embedding_dim=EMBEDDING_DIM,
            lstm_units=LSTM_UNITS,
            dropout=DROPOUT,
            learning_rate=learning_rate,
            l2_strength=L2_STRENGTH,
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_validation, y_validation),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            verbose=0,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=15,
                    min_delta=0.001,
                    restore_best_weights=True,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=6,
                    min_lr=0.00001,
                ),
            ],
        )
        best_index = int(np.argmin(history.history["val_loss"]))
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        validation_loss, validation_accuracy = model.evaluate(
            X_validation,
            y_validation,
            verbose=0,
        )
        folds.append(
            {
                "seed": seed,
                "best_epoch": best_index + 1,
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "loss_gap": float(validation_loss - train_loss),
                "train_accuracy": float(train_accuracy),
                "validation_accuracy": float(validation_accuracy),
                "accuracy_gap": float(train_accuracy - validation_accuracy),
            }
        )

    result = {"learning_rate": learning_rate, "folds": folds}
    for metric in (
        "best_epoch",
        "train_loss",
        "validation_loss",
        "loss_gap",
        "train_accuracy",
        "validation_accuracy",
        "accuracy_gap",
    ):
        values = [fold[metric] for fold in folds]
        result[f"mean_{metric}"] = float(np.mean(values))
        result[f"std_{metric}"] = float(np.std(values))

    return result


def main():
    text = DATA_PATH.read_text(encoding="utf-8")
    results = []

    for index, learning_rate in enumerate(LEARNING_RATES, start=1):
        print(f"[{index}/{len(LEARNING_RATES)}] Testing lr={learning_rate}...")
        result = evaluate_learning_rate(learning_rate, text)
        results.append(result)
        print(
            f"  val_loss={result['mean_validation_loss']:.4f}, "
            f"val_accuracy={result['mean_validation_accuracy']:.2%}"
        )

    results.sort(key=lambda result: result["mean_validation_loss"])
    best = results[0]
    accuracy_gain = best["mean_validation_accuracy"] - REFERENCE_ACCURACY
    report = {
        "selection_metric": "lowest mean validation loss across three splits",
        "tuning_stage": "final learning-rate-only refinement",
        "validation_seeds": list(VALIDATION_SEEDS),
        "test_split_seed": TEST_SPLIT_SEED,
        "test_partition_used_for_tuning": False,
        "learning_rates": list(LEARNING_RATES),
        "reference_accuracy": REFERENCE_ACCURACY,
        "meaningful_accuracy_gain": MEANINGFUL_ACCURACY_GAIN,
        "best_learning_rate": best["learning_rate"],
        "best_mean_validation_accuracy": best["mean_validation_accuracy"],
        "best_accuracy_gain_over_reference": accuracy_gain,
        "decision": (
            "stop_lstm_tuning"
            if accuracy_gain < MEANINGFUL_ACCURACY_GAIN
            else "consider_adopting_refined_learning_rate"
        ),
        "results": results,
    }

    LR_REFINEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LR_REFINEMENT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nBest learning-rate result:")
    print(json.dumps({key: value for key, value in best.items() if key != "folds"}, indent=2))
    print(f"\nDecision: {report['decision']}")
    print(f"Report saved to: {LR_REFINEMENT_PATH}")


if __name__ == "__main__":
    main()
