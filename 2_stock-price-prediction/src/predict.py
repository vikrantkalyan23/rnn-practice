import joblib
import numpy as np
import os
import pandas as pd
import yfinance as yf

os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

from tensorflow.keras.models import load_model


# --------------------------------
# Configuration
# --------------------------------

SYMBOL = "AAPL"

SEQUENCE_LENGTH = 60

MODEL_PATH = "models/stock_rnn.keras"
SCALER_PATH = "models/scaler.pkl"
LOCAL_DATA_PATH = "data/stock_data.csv"


# --------------------------------
# Load model
# --------------------------------


def load_prediction_model():

    print("Loading trained model...")

    model = load_model(MODEL_PATH)

    print("Model loaded successfully.")

    return model


# --------------------------------
# Load scaler
# --------------------------------


def load_scaler():

    print("Loading scaler...")

    scaler = joblib.load(SCALER_PATH)

    print("Scaler loaded successfully.")

    return scaler


# --------------------------------
# Download latest stock data
# --------------------------------


def get_latest_stock_data(
    symbol,
    sequence_length=60,
):

    print(f"\nDownloading latest data for {symbol}...")

    data = yf.download(
        symbol,
        period="6mo",
        auto_adjust=False,
    )

    if data.empty:
        print("Online download failed. Using local CSV data instead.")
        data = pd.read_csv(
            LOCAL_DATA_PATH,
            header=[0, 1],
            index_col=0,
            parse_dates=True,
        )

    close_prices = data["Close"]

    # yfinance can return a DataFrame
    # when downloading a single ticker.
    if hasattr(close_prices, "columns"):
        close_prices = close_prices.iloc[:, 0]

    close_prices = close_prices.dropna()

    if len(close_prices) < sequence_length:
        raise ValueError(
            f"Not enough data. "
            f"Required: {sequence_length}, "
            f"Available: {len(close_prices)}"
        )

    return close_prices


# --------------------------------
# Predict next day's price
# --------------------------------


def predict_next_price(
    symbol=SYMBOL,
):

    # 1. Load model
    model = load_prediction_model()

    # 2. Load scaler
    scaler = load_scaler()

    # 3. Get latest stock data
    close_prices = get_latest_stock_data(
        symbol,
        SEQUENCE_LENGTH,
    )

    # 4. Get last 60 closing prices
    last_prices = close_prices.tail(SEQUENCE_LENGTH)

    # 5. Convert prices to numpy array
    last_prices = last_prices.values.reshape(
        -1,
        1,
    )

    # 6. Scale prices
    scaled_prices = scaler.transform(last_prices)

    # 7. Create RNN input
    X = scaled_prices.reshape(
        1,
        SEQUENCE_LENGTH,
        1,
    )

    print(f"\nInput shape: {X.shape}")

    # 8. Predict
    prediction = model.predict(
        X,
        verbose=0,
    )

    # 9. Convert prediction back
    #    to original price scale
    predicted_price = scaler.inverse_transform(prediction)

    predicted_price = float(predicted_price[0][0])

    current_price = float(last_prices[-1][0])

    return current_price, predicted_price


# --------------------------------
# Main
# --------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("STOCK PRICE PREDICTION")
    print("=" * 50)

    current_price, predicted_price = predict_next_price(SYMBOL)

    print("\nPrediction Result")
    print("-" * 30)

    print(f"Stock Symbol      : {SYMBOL}")

    print(f"Current Price     : ${current_price:.2f}")

    print(f"Predicted Price   : ${predicted_price:.2f}")

    change = predicted_price - current_price

    change_percent = (change / current_price) * 100

    print(f"Expected Change   : ${change:.2f}")

    print(f"Expected Change % : {change_percent:.2f}%")

    print("=" * 50)
