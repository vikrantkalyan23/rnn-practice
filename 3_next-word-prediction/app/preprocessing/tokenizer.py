from tensorflow.keras.preprocessing.text import (
    Tokenizer,
    tokenizer_from_json,
)


def create_tokenizer(text: str) -> Tokenizer:
    tokenizer = Tokenizer(oov_token="<OOV>")

    tokenizer.fit_on_texts([text])

    return tokenizer


def save_tokenizer(tokenizer, path):
    tokenizer_json = tokenizer.to_json()

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(tokenizer_json)


def load_tokenizer(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        tokenizer_json = file.read()

    return tokenizer_from_json(tokenizer_json)
