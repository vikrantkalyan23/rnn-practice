from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_model(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int = 32,
    lstm_units: int = 32,
    dropout: float = 0.40,
    learning_rate: float = 0.0015,
    l2_strength: float = 0.001,
):
    """Build an LSTM sized for the educational corpus."""
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
            ),
            Dropout(dropout),
            Dense(
                vocab_size,
                activation="softmax",
                kernel_regularizer=l2(l2_strength),
            ),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
