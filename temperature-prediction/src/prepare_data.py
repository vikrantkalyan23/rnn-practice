import pandas as pd

from data_utils import DATA_FILE, load_temperatures, make_sequences


SEQUENCE_LENGTH = 5


def main():
    # This file is only for understanding the data preparation step.
    data_frame = pd.read_csv(DATA_FILE)
    temperatures = load_temperatures()

    X, y = make_sequences(temperatures, SEQUENCE_LENGTH)

    print("First rows from the CSV file:")
    print(data_frame.head())

    print("\nAll temperature values:")
    print(temperatures.astype(int))

    print("\nShape of X:", X.shape)
    print("Shape of y:", y.shape)

    print("\nFirst 5 training examples:")
    for i in range(5):
        input_sequence = X[i].flatten()
        answer = y[i]

        print("Input:", input_sequence.astype(int).tolist(), "Answer:", int(answer))


if __name__ == "__main__":
    main()
