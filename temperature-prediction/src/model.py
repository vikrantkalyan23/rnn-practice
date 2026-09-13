from data_utils import setup_runtime

setup_runtime()

from tensorflow.keras.layers import Dense, Input, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_model(sequence_length=5, rnn_units=32):
    # This model reads a short temperature sequence and predicts one number.
    model = Sequential()

    # Input shape means:
    # sequence_length = how many past temperatures we give the model
    # 1 = each time step has one feature, the temperature
    model.add(Input(shape=(sequence_length, 1)))

    # SimpleRNN is a beginner-friendly recurrent neural network layer.
    # ReLU helps this tiny trend dataset extrapolate upward better than tanh.
    model.add(SimpleRNN(rnn_units, activation="relu"))

    # Dense(1) gives one final answer: the next temperature.
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


if __name__ == "__main__":
    demo_model = create_model()
    demo_model.summary()
