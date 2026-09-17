from pathlib import Path

from app.preprocessing.cleaner import clean_text
from app.preprocessing.tokenizer import (
    create_tokenizer,
    save_tokenizer,
)
from app.preprocessing.sequences import (
    create_training_sequences,
)
from app.model.architecture import build_model


DATA_PATH = Path("data/raw/corpus.txt")
MODEL_PATH = Path("models/next_word_model.keras")
TOKENIZER_PATH = Path("models/tokenizer.json")


def main():

    print("Loading dataset...")

    text = DATA_PATH.read_text(encoding="utf-8")

    print("Cleaning text...")

    text = clean_text(text)

    print("Creating tokenizer...")

    tokenizer = create_tokenizer(text)

    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")

    print("Creating sequences...")

    sequence_length = 5

    X, y = create_training_sequences(
        text,
        tokenizer,
        sequence_length=sequence_length,
    )

    vocab_size = len(tokenizer.word_index) + 1

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Sequence length: {sequence_length}")

    print("Building model...")

    model = build_model(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        embedding_dim=128,
        lstm_units=128,
    )

    model.summary()

    print("Training model...")

    model.fit(
        X,
        y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
    )

    print("Saving model...")

    model.save(MODEL_PATH)

    print("Saving tokenizer...")

    save_tokenizer(
        tokenizer,
        TOKENIZER_PATH,
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
