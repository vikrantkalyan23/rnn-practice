import re
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json


def clean_line(line: str) -> str:
    """Lowercase one line and remove punctuation and repeated spaces."""
    line = line.lower()
    line = re.sub(r"[^a-z0-9\s]", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def clean_text(text: str) -> str:
    """Clean non-empty lines while preserving sentence boundaries."""
    cleaned_lines = [clean_line(line) for line in text.splitlines()]
    return "\n".join(line for line in cleaned_lines if line)


def split_text_by_line(
    text: str,
    validation_ratio: float = 0.2,
    random_seed: int = 11,
):
    """Split independent sentences before creating overlapping sequences."""
    lines = [line for line in clean_text(text).splitlines() if line]

    if len(lines) < 2:
        raise ValueError("The corpus needs at least two non-empty lines.")

    random_generator = np.random.default_rng(random_seed)
    random_generator.shuffle(lines)

    validation_size = max(1, int(round(len(lines) * validation_ratio)))
    validation_size = min(validation_size, len(lines) - 1)
    split_index = len(lines) - validation_size

    return "\n".join(lines[:split_index]), "\n".join(lines[split_index:])


def create_tokenizer(text: str, max_vocab_size: int | None = None) -> Tokenizer:
    """Fit a tokenizer on training text only."""
    tokenizer = Tokenizer(
        num_words=max_vocab_size,
        oov_token="<OOV>",
    )
    tokenizer.fit_on_texts([text])
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tokenizer.to_json(), encoding="utf-8")


def load_tokenizer(path: Path) -> Tokenizer:
    return tokenizer_from_json(path.read_text(encoding="utf-8"))


def create_sequences(text: str, tokenizer: Tokenizer, sequence_length: int = 5):
    """Create contexts for targets contained in the training vocabulary."""
    examples = []

    for line in text.splitlines():
        token_ids = tokenizer.texts_to_sequences([line])[0]

        for target_index in range(1, len(token_ids)):
            # Token 1 is <OOV>. Unknown words are useful context, but they
            # cannot be meaningful next-word classes in this small model.
            if token_ids[target_index] == tokenizer.word_index["<OOV>"]:
                continue

            start = max(0, target_index - sequence_length)
            context_and_target = token_ids[start : target_index + 1]
            examples.append(context_and_target)

    if not examples:
        raise ValueError("No sequences were created from the supplied text.")

    padded = pad_sequences(
        examples,
        maxlen=sequence_length + 1,
        padding="pre",
    )

    X = np.asarray(padded[:, :-1], dtype=np.int32)
    y = np.asarray(padded[:, -1], dtype=np.int32)
    return X, y


def prepare_datasets(
    text: str,
    sequence_length: int,
    validation_ratio: float,
    random_seed: int = 11,
    max_vocab_size: int | None = None,
):
    """Create leakage-safe train and validation datasets."""
    train_text, validation_text = split_text_by_line(
        text,
        validation_ratio,
        random_seed,
    )
    tokenizer = create_tokenizer(train_text, max_vocab_size)

    X_train, y_train = create_sequences(train_text, tokenizer, sequence_length)
    X_validation, y_validation = create_sequences(
        validation_text,
        tokenizer,
        sequence_length,
    )

    return X_train, y_train, X_validation, y_validation, tokenizer


def get_vocabulary_size(tokenizer: Tokenizer) -> int:
    """Return the number of IDs the model can receive and predict."""
    full_size = len(tokenizer.word_index) + 1
    return min(tokenizer.num_words or full_size, full_size)


def count_possible_targets(text: str) -> int:
    """Count all next-word examples before rare targets are excluded."""
    return sum(max(0, len(line.split()) - 1) for line in clean_text(text).splitlines())
