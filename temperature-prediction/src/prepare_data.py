import pandas as pd

from data_utils import DATA_PATH, create_sequences, load_temperatures


def main():
    """Show how raw temperatures become training examples."""
    df = pd.read_csv(DATA_PATH)
    temperatures = load_temperatures()
    X, y = create_sequences(temperatures, sequence_length=5)

    print("Original Data:")
    print(df.head())

    print("\nTemperature values:")
    print(temperatures.astype(int))

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    print("\nFirst 5 sequences:")
    for sequence, target in zip(X[:5], y[:5]):
        print("X:", sequence, "y:", target)


if __name__ == "__main__":
    main()
