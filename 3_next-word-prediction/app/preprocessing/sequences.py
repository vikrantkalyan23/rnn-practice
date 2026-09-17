from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_training_sequences(
    text: str,
    tokenizer,
    sequence_length: int = 5,
):
    input_sequences = []

    for sentence in text.splitlines():

        token_list = tokenizer.texts_to_sequences(
            [sentence]
        )[0]

        for i in range(1, len(token_list)):

            start = max(
                0,
                i - sequence_length
            )

            sequence = token_list[
                start:i + 1
            ]

            input_sequences.append(sequence)

    if not input_sequences:
        raise ValueError(
            "No training sequences were created."
        )

    input_sequences = pad_sequences(
        input_sequences,
        maxlen=sequence_length + 1,
        padding="pre"
    )

    X = input_sequences[:, :-1]

    y = input_sequences[:, -1]

    return X, y