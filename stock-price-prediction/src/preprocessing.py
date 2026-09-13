import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_stock_data(file_path="data/stock_data.csv"):
    """
    Load stock data from CSV.
    """

    data = pd.read_csv(
        file_path,
        header=[0, 1],
        index_col=0,
        parse_dates=True,
    )

    print("Stock data loaded successfully.")
    print(f"Shape: {data.shape}")

    return data


def get_close_prices(data):
    """
    Extract the Close price column.
    """

    close_prices = data["Close"]

    # yfinance may return a DataFrame with one ticker column.
    # Convert it to a simple Series.
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    close_prices = close_prices.dropna()

    return close_prices


def split_data(close_prices, train_ratio=0.8):
    """
    Split stock prices into training and testing data.

    Time-series data should NOT be randomly shuffled.
    """

    train_size = int(len(close_prices) * train_ratio)

    train_data = close_prices.iloc[:train_size]
    test_data = close_prices.iloc[train_size:]

    print("\nData split:")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples : {len(test_data)}")

    return train_data, test_data


def scale_data(train_data, test_data):
    """
    Scale prices between 0 and 1.

    The scaler is fitted ONLY on training data.
    """

    scaler = MinMaxScaler(feature_range=(0, 1))

    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))

    test_scaled = scaler.transform(test_data.values.reshape(-1, 1))

    return train_scaled, test_scaled, scaler


def create_sequences(data, sequence_length=60):
    """
    Create sequences for RNN.

    Example:

    Previous 60 days -> Next day
    """

    X = []
    y = []

    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length : i])

        y.append(data[i])

    X = np.array(X)
    y = np.array(y)

    return X, y


def prepare_data(
    file_path="data/stock_data.csv",
    sequence_length=60,
    train_ratio=0.8,
):
    """
    Complete preprocessing pipeline.
    """

    # 1. Load data
    data = load_stock_data(file_path)

    # 2. Extract closing prices
    close_prices = get_close_prices(data)

    print("\nClosing price data:")
    print(close_prices.head())

    print(f"\nTotal closing prices: {len(close_prices)}")

    # 3. Train/test split
    train_data, test_data = split_data(
        close_prices,
        train_ratio,
    )

    # 4. Scale data
    train_scaled, test_scaled, scaler = scale_data(
        train_data,
        test_data,
    )

    # 5. Create sequences
    X_train, y_train = create_sequences(
        train_scaled,
        sequence_length,
    )

    X_test, y_test = create_sequences(
        test_scaled,
        sequence_length,
    )

    print("\nSequence information:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_test shape : {y_test.shape}")

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
    )


if __name__ == "__main__":
    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
    ) = prepare_data()

    print("\nPreprocessing completed successfully!")
