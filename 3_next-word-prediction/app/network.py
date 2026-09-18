from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Input,
    LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_model(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int = 96,
    lstm_units: int = 96,
    dropout: float = 0.20,
    learning_rate: float = 0.001,
    l2_strength: float = 0.0001,
):

    model = Sequential(
        [
            Input(shape=(sequence_length,)),
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                mask_zero=True,
            ),
            LSTM(
                lstm_units,
                kernel_regularizer=l2(l2_strength),
                recurrent_regularizer=l2(l2_strength),
                dropout=0.05,
                recurrent_dropout=0.05,
            ),
            Dropout(dropout),
            Dense(
                vocab_size,
                activation="softmax",
                kernel_regularizer=l2(l2_strength),
            ),
        ]
    )

    optimizer = Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
