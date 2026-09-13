import os

os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

from tensorflow.keras.layers import Dense, Input, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_rnn_model(
    sequence_length=60,
    number_of_features=1,
):
    """
    Build and return an optimized recurrent model.

    Input:
        60 previous stock prices

    Output:
        1 predicted stock price
    """

    model = Sequential(
        [
            Input(
                shape=(
                    sequence_length,
                    number_of_features,
                )
            ),
            SimpleRNN(
                64,
                activation="relu",
            ),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )

    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["mae"],
    )

    return model


if __name__ == "__main__":
    model = build_rnn_model()

    print("\nRNN model created successfully!\n")

    model.summary()
