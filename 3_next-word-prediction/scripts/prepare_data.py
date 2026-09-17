import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.config import (  # noqa: E402
    DATA_PATH,
    PROCESSED_DATA_PATH,
    MAX_VOCAB_SIZE,
    SEQUENCE_LENGTH,
    RANDOM_SEED,
    VALIDATION_RATIO,
)
from app.data import clean_text, get_vocabulary_size, prepare_datasets  # noqa: E402


def main():
    raw_text = DATA_PATH.read_text(encoding="utf-8")
    cleaned_text = clean_text(raw_text)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_PATH.write_text(cleaned_text + "\n", encoding="utf-8")

    X_train, y_train, X_validation, y_validation, tokenizer = prepare_datasets(
        cleaned_text,
        sequence_length=SEQUENCE_LENGTH,
        validation_ratio=VALIDATION_RATIO,
        random_seed=RANDOM_SEED,
        max_vocab_size=MAX_VOCAB_SIZE,
    )

    print(f"Cleaned corpus saved to: {PROCESSED_DATA_PATH}")
    print(f"Vocabulary size         : {get_vocabulary_size(tokenizer)}")
    print(f"Training X / y          : {X_train.shape} / {y_train.shape}")
    print(f"Validation X / y        : {X_validation.shape} / {y_validation.shape}")
    print("\nFirst training example:")
    print(f"X token IDs: {X_train[0].tolist()}")
    print(f"y token ID : {int(y_train[0])}")


if __name__ == "__main__":
    main()
