from data_utils import configure_runtime

configure_runtime()

from tensorflow.keras.layers import Dense, Input, SimpleRNN
from tensorflow.keras.models import Sequential


def create_model(sequence_length=5, units=32):
    """Create and compile the RNN model.

    sequence_length tells the model how many past temperatures it sees.
    units controls the size of the SimpleRNN layer.
    """
    model = Sequential(
        [
            Input(shape=(sequence_length, 1)),
            SimpleRNN(units),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    return model


if __name__ == "__main__":
    model = create_model()
    model.summary()
