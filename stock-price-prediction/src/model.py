import os

os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

from tensorflow.keras.layers import Add, Dense, Input, SimpleRNN
from tensorflow.keras.models import Model
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

    inputs = Input(
        shape=(
            sequence_length,
            number_of_features,
        )
    )

    rnn_output = SimpleRNN(
        24,
        activation="relu",
    )(inputs)

    hidden_output = Dense(
        12,
        activation="relu",
    )(rnn_output)

    predicted_change = Dense(1)(hidden_output)

    last_known_price = inputs[:, -1, :]

    predicted_price = Add()(
        [
            last_known_price,
            predicted_change,
        ]
    )

    model = Model(
        inputs=inputs,
        outputs=predicted_price,
    )

    optimizer = Adam(
        learning_rate=0.0003,
        clipnorm=1.0,
    )

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
