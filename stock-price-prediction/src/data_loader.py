import yfinance as yf
from pathlib import Path


def download_stock_data(
    symbol="AAPL",
    start="2015-01-01",
    end="2026-01-01",
):
    """
    Download historical stock data from Yahoo Finance.
    """

    print(f"Downloading data for {symbol}...")

    data = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError(f"No data found for stock symbol: {symbol}")

    return data


def save_stock_data(data, file_path="data/stock_data.csv"):
    """
    Save stock data to CSV.
    """

    path = Path(file_path)

    # Create parent directory if it doesn't exist
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(path)

    print(f"Stock data saved to: {path}")


if __name__ == "__main__":
    data = download_stock_data()

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nData shape:")
    print(data.shape)

    save_stock_data(data)
