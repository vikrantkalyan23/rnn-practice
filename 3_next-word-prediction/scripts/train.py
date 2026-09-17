import argparse
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
from tensorflow.keras.callbacks import (  # noqa: E402
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from app.config import (  # noqa: E402
    BATCH_SIZE,
    DATA_PATH,
    DROPOUT,
    EMBEDDING_DIM,
    EPOCHS,
    HISTORY_PATH,
    LEARNING_RATE,
    L2_STRENGTH,
    LSTM_UNITS,
    MAX_VOCAB_SIZE,
    MODEL_CONFIG_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TOKENIZER_PATH,
    TEST_RATIO,
    TEST_SPLIT_SEED,
    TRAINING_TEXT_PATH,
    VALIDATION_RATIO,
)
from app.data import (  # noqa: E402
    create_sequences,
    create_tokenizer,
    get_vocabulary_size,
    save_tokenizer,
    split_text_train_validation_test,
)
from app.network import build_model  # noqa: E402


def read_options():
    parser = argparse.ArgumentParser(description="Train the next-word LSTM.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    options = read_options()
    random.seed(options.seed)
    np.random.seed(options.seed)
    tf.random.set_seed(options.seed)
    tf.keras.backend.clear_session()

    print("Loading and preparing the corpus...")
    text = DATA_PATH.read_text(encoding="utf-8")
    train_text, validation_text, test_text = split_text_train_validation_test(
        text,
        validation_ratio=VALIDATION_RATIO,
        test_ratio=TEST_RATIO,
        test_seed=TEST_SPLIT_SEED,
        validation_seed=options.seed,
    )
    tokenizer = create_tokenizer(train_text, MAX_VOCAB_SIZE)
    X_train, y_train = create_sequences(
        train_text,
        tokenizer,
        options.sequence_length,
    )
    X_validation, y_validation = create_sequences(
        validation_text,
        tokenizer,
        options.sequence_length,
    )
    test_lines = len(test_text.splitlines())
    vocab_size = get_vocabulary_size(tokenizer)

    print(f"Training examples  : {len(X_train)}")
    print(f"Validation examples: {len(X_validation)}")
    print(f"Held-out test lines: {test_lines}")
    print(f"Vocabulary size    : {vocab_size}")
    print(f"Input shape        : {X_train.shape}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, TOKENIZER_PATH)
    TRAINING_TEXT_PATH.write_text(train_text, encoding="utf-8")

    model = build_model(
        vocab_size=vocab_size,
        sequence_length=options.sequence_length,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
        dropout=DROPOUT,
        learning_rate=options.learning_rate,
        l2_strength=L2_STRENGTH,
    )
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=0 if options.quiet else 1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=0.00001,
            verbose=0 if options.quiet else 1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=0 if options.quiet else 1,
        ),
    ]

    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=options.epochs,
        batch_size=options.batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=0 if options.quiet else 1,
    )

    best_epoch_index = int(np.argmin(history.history["val_loss"]))
    best_epoch = best_epoch_index + 1

    history_data = {
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "best_epoch": best_epoch,
        "sequence_length": options.sequence_length,
        "batch_size": options.batch_size,
        "random_seed": options.seed,
        "validation_seed": options.seed,
        "test_split_seed": TEST_SPLIT_SEED,
        "evaluation_protocol": "fixed-test-multi-validation-v1",
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "held_out_test": True,
    }
    HISTORY_PATH.write_text(json.dumps(history_data, indent=2), encoding="utf-8")

    model_config = {
        "sequence_length": options.sequence_length,
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "dropout": DROPOUT,
        "max_vocab_size": MAX_VOCAB_SIZE,
        "l2_strength": L2_STRENGTH,
        "learning_rate": options.learning_rate,
        "batch_size": options.batch_size,
    }
    MODEL_CONFIG_PATH.write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    print(f"Best epoch          : {best_epoch}")
    print(f"Best validation loss: {history.history['val_loss'][best_epoch_index]:.4f}")
    print(f"Model saved to      : {MODEL_PATH}")
    print("Run `uv run python scripts/evaluate.py` to create evaluation graphs.")


if __name__ == "__main__":
    main()
