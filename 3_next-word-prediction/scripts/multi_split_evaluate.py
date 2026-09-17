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
    LEARNING_RATE,
    L2_STRENGTH,
    LSTM_UNITS,
    MAX_VOCAB_SIZE,
    MULTI_SPLIT_METRICS_PATH,
    SEQUENCE_LENGTH,
    TEST_RATIO,
    TEST_SPLIT_SEED,
    VALIDATION_RATIO,
    VALIDATION_SEEDS,
)
from app.data import (  # noqa: E402
    count_possible_targets,
    create_sequences,
    create_tokenizer,
    get_vocabulary_size,
    split_text_train_validation_test,
)
from app.network import build_model  # noqa: E402


def top_k_accuracy(probabilities, targets, k):
    k = min(k, probabilities.shape[1])
    top_ids = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_ids == targets[:, None], axis=1)))


def evaluate_split(text, validation_seed):
    random.seed(validation_seed)
    np.random.seed(validation_seed)
    tf.random.set_seed(validation_seed)
    tf.keras.backend.clear_session()

    train_text, validation_text, test_text = split_text_train_validation_test(
        text,
        validation_ratio=VALIDATION_RATIO,
        test_ratio=TEST_RATIO,
        test_seed=TEST_SPLIT_SEED,
        validation_seed=validation_seed,
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
        learning_rate=LEARNING_RATE,
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
                monitor="val_loss", factor=0.5, patience=6, min_lr=0.00001
            ),
        ],
    )
    loss, accuracy = model.evaluate(X_validation, y_validation, verbose=0)
    probabilities = model.predict(X_validation, verbose=0)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    return {
        "validation_seed": validation_seed,
        "best_epoch": best_epoch,
        "train_lines": len(train_text.splitlines()),
        "validation_lines": len(validation_text.splitlines()),
        "fixed_test_lines": len(test_text.splitlines()),
        "validation_examples": len(y_validation),
        "validation_target_coverage": len(y_validation)
        / count_possible_targets(validation_text),
        "validation_loss": float(loss),
        "validation_perplexity": float(np.exp(min(loss, 50))),
        "top_1_accuracy": float(accuracy),
        "top_3_accuracy": top_k_accuracy(probabilities, y_validation, 3),
        "top_5_accuracy": top_k_accuracy(probabilities, y_validation, 5),
    }


def main():
    text = DATA_PATH.read_text(encoding="utf-8")
    splits = []
    for index, seed in enumerate(VALIDATION_SEEDS, start=1):
        print(f"[{index}/{len(VALIDATION_SEEDS)}] Validation seed {seed}")
        splits.append(evaluate_split(text, seed))

    metric_names = (
        "validation_loss",
        "validation_perplexity",
        "top_1_accuracy",
        "top_3_accuracy",
        "top_5_accuracy",
        "validation_target_coverage",
    )
    aggregate = {}
    for metric in metric_names:
        values = [split[metric] for split in splits]
        aggregate[f"mean_{metric}"] = float(np.mean(values))
        aggregate[f"std_{metric}"] = float(np.std(values))

    report = {
        "protocol": "fixed-test-multi-validation-v1",
        "test_split_seed": TEST_SPLIT_SEED,
        "validation_seeds": list(VALIDATION_SEEDS),
        "test_partition_used_for_training_or_selection": False,
        "splits": splits,
        "aggregate": aggregate,
    }
    MULTI_SPLIT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MULTI_SPLIT_METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nMulti-split validation")
    print("-" * 35)
    print(
        f"Top-1      : {aggregate['mean_top_1_accuracy']:.2%} "
        f"+/- {aggregate['std_top_1_accuracy']:.2%}"
    )
    print(
        f"Top-3      : {aggregate['mean_top_3_accuracy']:.2%} "
        f"+/- {aggregate['std_top_3_accuracy']:.2%}"
    )
    print(
        f"Top-5      : {aggregate['mean_top_5_accuracy']:.2%} "
        f"+/- {aggregate['std_top_5_accuracy']:.2%}"
    )
    print(
        f"Perplexity : {aggregate['mean_validation_perplexity']:.2f} "
        f"+/- {aggregate['std_validation_perplexity']:.2f}"
    )
    print(f"Report saved to: {MULTI_SPLIT_METRICS_PATH}")


if __name__ == "__main__":
    main()
