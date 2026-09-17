import json
import os
import random
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: E402

from app.config import DATA_PATH, MAX_VOCAB_SIZE, OUTPUT_DIR, VALIDATION_RATIO  # noqa: E402
from app.data import get_vocabulary_size, prepare_datasets  # noqa: E402
from app.network import build_model  # noqa: E402


# A small, deliberate search is easier to understand than a large blind grid.
CANDIDATES = [
    {
        "name": "current",
        "embedding_dim": 16,
        "lstm_units": 12,
        "dropout": 0.40,
        "learning_rate": 0.001,
        "l2_strength": 0.001,
    },
    {
        "name": "smaller_embedding",
        "embedding_dim": 12,
        "lstm_units": 12,
        "dropout": 0.40,
        "learning_rate": 0.001,
        "l2_strength": 0.001,
    },
    {
        "name": "smaller_lstm",
        "embedding_dim": 16,
        "lstm_units": 8,
        "dropout": 0.40,
        "learning_rate": 0.001,
        "l2_strength": 0.001,
    },
    {
        "name": "more_dropout",
        "embedding_dim": 16,
        "lstm_units": 12,
        "dropout": 0.50,
        "learning_rate": 0.001,
        "l2_strength": 0.001,
    },
    {
        "name": "stronger_l2",
        "embedding_dim": 16,
        "lstm_units": 12,
        "dropout": 0.40,
        "learning_rate": 0.001,
        "l2_strength": 0.002,
    },
    {
        "name": "faster_learning",
        "embedding_dim": 16,
        "lstm_units": 12,
        "dropout": 0.40,
        "learning_rate": 0.002,
        "l2_strength": 0.001,
    },
]

VALIDATION_SEEDS = (7, 11, 19)
SEQUENCE_LENGTH = 5
EPOCHS = 150
BATCH_SIZE = 8


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def evaluate_candidate(candidate, text):
    fold_results = []

    for seed in VALIDATION_SEEDS:
        tf.keras.backend.clear_session()
        set_random_seed(seed)

        X_train, y_train, X_validation, y_validation, tokenizer = prepare_datasets(
            text,
            sequence_length=SEQUENCE_LENGTH,
            validation_ratio=VALIDATION_RATIO,
            random_seed=seed,
            max_vocab_size=MAX_VOCAB_SIZE,
        )
        model = build_model(
            vocab_size=get_vocabulary_size(tokenizer),
            sequence_length=SEQUENCE_LENGTH,
            embedding_dim=candidate["embedding_dim"],
            lstm_units=candidate["lstm_units"],
            dropout=candidate["dropout"],
            learning_rate=candidate["learning_rate"],
            l2_strength=candidate["l2_strength"],
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
        fold_results.append(
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

    result = dict(candidate)
    result["folds"] = fold_results
    for metric in (
        "best_epoch",
        "train_loss",
        "validation_loss",
        "loss_gap",
        "train_accuracy",
        "validation_accuracy",
        "accuracy_gap",
    ):
        values = [fold[metric] for fold in fold_results]
        result[f"mean_{metric}"] = float(np.mean(values))
        result[f"std_{metric}"] = float(np.std(values))

    return result


def plot_results(results):
    names = [result["name"] for result in results]
    losses = [result["mean_validation_loss"] for result in results]
    loss_errors = [result["std_validation_loss"] for result in results]
    accuracies = [result["mean_validation_accuracy"] for result in results]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(names, losses, yerr=loss_errors, color="#2878B5", capsize=4)
    axes[0].set_title("Mean Validation Loss Across Three Splits")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(names, accuracies, color="#59A14F")
    axes[1].set_title("Mean Validation Accuracy Across Three Splits")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, max(accuracies) * 1.25)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "hyperparameter_tuning.png", dpi=160)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text = DATA_PATH.read_text(encoding="utf-8")
    results = []

    for index, candidate in enumerate(CANDIDATES, start=1):
        print(f"[{index}/{len(CANDIDATES)}] Testing {candidate['name']}...")
        result = evaluate_candidate(candidate, text)
        results.append(result)
        print(
            f"  val_loss={result['mean_validation_loss']:.4f}, "
            f"val_accuracy={result['mean_validation_accuracy']:.2%}, "
            f"accuracy_gap={result['mean_accuracy_gap']:.2%}"
        )

    results.sort(key=lambda result: result["mean_validation_loss"])
    report = {
        "selection_metric": "lowest mean validation loss across three splits",
        "validation_seeds": list(VALIDATION_SEEDS),
        "best_candidate": results[0]["name"],
        "results": results,
    }
    (OUTPUT_DIR / "hyperparameter_tuning.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    plot_results(results)

    print("\nBest candidate:")
    print(json.dumps({key: value for key, value in results[0].items() if key != "folds"}, indent=2))
    print(f"\nReport saved to: {OUTPUT_DIR / 'hyperparameter_tuning.json'}")


if __name__ == "__main__":
    main()
