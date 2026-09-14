from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "stock_rnn.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
LOCAL_DATA_PATH = BASE_DIR / "data" / "stock_data.csv"

SEQUENCE_LENGTH = 60

SUPPORTED_SYMBOL = "AAPL"


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Stock Price Prediction API",
    description="RNN-based stock price prediction API",
    version="1.0.0",
)


# ============================================================
# Load model and scaler
# ============================================================

print("Loading RNN model...")

model = load_model(MODEL_PATH)

print("RNN model loaded successfully.")


print("Loading scaler...")

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully.")


# ============================================================
# Response schema
# ============================================================


class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    expected_change: float
    expected_change_percent: float
    direction: str


# ============================================================
# Root endpoint
# ============================================================


@app.get("/")
def root():
    return {
        "message": "Stock Price Prediction API",
        "status": "running",
        "model": "SimpleRNN",
        "sequence_length": SEQUENCE_LENGTH,
        "supported_symbol": SUPPORTED_SYMBOL,
    }


# ============================================================
# Health check
# ============================================================


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


# ============================================================
# Get latest stock data
# ============================================================


def get_latest_stock_data(symbol: str):

    try:
        print(f"Downloading latest data for {symbol}...")

        data = yf.download(
            symbol,
            period="6mo",
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            raise ValueError("No online data received.")

        close_prices = data["Close"]

        # yfinance may return a DataFrame
        # for a single ticker.
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        close_prices = close_prices.dropna()

        if len(close_prices) < SEQUENCE_LENGTH:
            raise ValueError(
                f"Not enough stock data. "
                f"Required: {SEQUENCE_LENGTH}, "
                f"Available: {len(close_prices)}"
            )

        return close_prices

    except Exception as online_error:
        print(f"Online download failed: {online_error}")

        print("Using local CSV data...")

        try:
            data = pd.read_csv(
                LOCAL_DATA_PATH,
                header=[0, 1],
                index_col=0,
                parse_dates=True,
            )

            close_prices = data["Close"]

            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.iloc[:, 0]

            close_prices = close_prices.dropna()

            if len(close_prices) < SEQUENCE_LENGTH:
                raise ValueError("Local CSV does not contain enough data.")

            return close_prices

        except Exception as local_error:
            raise ValueError(
                f"Could not obtain stock data. "
                f"Online error: {online_error}. "
                f"Local error: {local_error}"
            )


# ============================================================
# Prediction function
# ============================================================


def predict_stock_price(symbol: str):

    # --------------------------------------------------------
    # Get stock data
    # --------------------------------------------------------

    close_prices = get_latest_stock_data(symbol)

    # --------------------------------------------------------
    # Last 60 closing prices
    # --------------------------------------------------------

    last_prices = close_prices.tail(SEQUENCE_LENGTH)

    last_prices_array = last_prices.values.reshape(
        -1,
        1,
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    scaled_prices = scaler.transform(last_prices_array)

    # --------------------------------------------------------
    # Create RNN input
    #
    # Shape:
    # (batch_size, sequence_length, features)
    #
    # (1, 60, 1)
    # --------------------------------------------------------

    X = scaled_prices.reshape(
        1,
        SEQUENCE_LENGTH,
        1,
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction = model.predict(
        X,
        verbose=0,
    )

    # --------------------------------------------------------
    # Convert prediction back to original price
    # --------------------------------------------------------

    predicted_price_array = scaler.inverse_transform(prediction)

    predicted_price = float(predicted_price_array[0][0])

    current_price = float(last_prices_array[-1][0])

    # --------------------------------------------------------
    # Calculate change
    # --------------------------------------------------------

    expected_change = predicted_price - current_price

    expected_change_percent = (expected_change / current_price) * 100

    # --------------------------------------------------------
    # Determine direction
    # --------------------------------------------------------

    if expected_change > 0:
        direction = "UP"

    elif expected_change < 0:
        direction = "DOWN"

    else:
        direction = "UNCHANGED"

    return (
        current_price,
        predicted_price,
        expected_change,
        expected_change_percent,
        direction,
    )


# ============================================================
# Prediction endpoint
# ============================================================


@app.get(
    "/predict/{symbol}",
    response_model=PredictionResponse,
)
def predict(symbol: str):

    symbol = symbol.upper()

    # --------------------------------------------------------
    # Current model is trained for AAPL.
    # --------------------------------------------------------

    if symbol != SUPPORTED_SYMBOL:
        raise HTTPException(
            status_code=400,
            detail=(f"This model currently supports {SUPPORTED_SYMBOL} only."),
        )

    try:
        (
            current_price,
            predicted_price,
            expected_change,
            expected_change_percent,
            direction,
        ) = predict_stock_price(symbol)

        return PredictionResponse(
            symbol=symbol,
            current_price=round(
                current_price,
                2,
            ),
            predicted_price=round(
                predicted_price,
                2,
            ),
            expected_change=round(
                expected_change,
                2,
            ),
            expected_change_percent=round(
                expected_change_percent,
                2,
            ),
            direction=direction,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
