from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dropout,
    Dense,
)


def build_model(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int = 128,
    lstm_units: int = 128,
):
    model = Sequential(
        [
            Input(shape=(sequence_length,)),
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
            ),
            LSTM(lstm_units),
            Dropout(0.2),
            Dense(
                vocab_size,
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
