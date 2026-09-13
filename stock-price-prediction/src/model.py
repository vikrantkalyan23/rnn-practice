from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense


def build_rnn_model(
    sequence_length=60,
    number_of_features=1,
):
    """
    Build and return a Simple RNN model.

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
                50,
                activation="tanh",
            ),
            Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
    )

    return model


if __name__ == "__main__":
    model = build_rnn_model()

    print("\nRNN model created successfully!\n")

    model.summary()
