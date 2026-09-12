from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential


def create_model():
    model = Sequential([SimpleRNN(32, input_shape=(5, 1)), Dense(1)])

    model.compile(optimizer="adam", loss="mse")

    return model


if __name__ == "__main__":
    model = create_model()

    model.summary()
