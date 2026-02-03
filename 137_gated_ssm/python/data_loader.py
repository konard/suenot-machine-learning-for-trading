"""
Data loading and feature engineering for Gated SSM trading.

Provides:
- TradingDataset: PyTorch dataset for financial time series with sliding window
- fetch_bybit_data: Fetch kline data from Bybit API
- fetch_stock_data: Fetch stock data via yfinance
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingDataset(Dataset):
    """
    Dataset for financial time series with sliding window.

    Features computed per window:
    - Log returns
    - Rolling volatility (20-period)
    - RSI (14-period)
    - Normalized volume

    Target: next-period return direction (1=up, 0=down).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        window: int = 60,
        horizon: int = 1,
    ):
        self.window = window
        self.horizon = horizon
        self.features, self.targets = self._prepare(prices)

    def _prepare(self, prices: pd.DataFrame):
        close = prices["close"].values.astype(float)
        volume = (
            prices["volume"].values.astype(float)
            if "volume" in prices
            else np.ones_like(close)
        )

        # Log returns
        log_ret = np.diff(np.log(close + 1e-10))

        # Rolling volatility (20-period std of returns)
        vol_20 = pd.Series(log_ret).rolling(20).std().values

        # RSI (14-period)
        gains = np.where(log_ret > 0, log_ret, 0)
        losses = np.where(log_ret < 0, -log_ret, 0)
        avg_gain = pd.Series(gains).rolling(14).mean().values
        avg_loss = pd.Series(losses).rolling(14).mean().values
        rsi = avg_gain / (avg_gain + avg_loss + 1e-10)

        # Normalized volume
        vol_series = pd.Series(volume[1:])
        vol_norm = (vol_series - vol_series.rolling(20).mean()) / (
            vol_series.rolling(20).std() + 1e-10
        )
        vol_norm = vol_norm.values

        # Stack features
        raw = np.column_stack([log_ret, vol_20, rsi, vol_norm])

        # Drop NaN rows (after rolling window stabilizes)
        valid_start = 20
        raw = raw[valid_start:]
        close_valid = close[valid_start + 1:]

        # Replace remaining NaN with 0
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sliding windows
        features, targets = [], []
        for i in range(len(raw) - self.window - self.horizon + 1):
            feat = raw[i : i + self.window]
            future_ret = np.log(
                (close_valid[i + self.window + self.horizon - 1] + 1e-10)
                / (close_valid[i + self.window - 1] + 1e-10)
            )
            target = 1.0 if future_ret > 0 else 0.0
            features.append(feat)
            targets.append(target)

        logger.info(f"Created dataset with {len(features)} samples, window={self.window}")

        return (
            torch.tensor(np.array(features), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def fetch_bybit_data(
    symbol: str = "BTCUSDT",
    interval: str = "D",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch kline data from Bybit public API.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
        interval: Kline interval ("1", "5", "15", "60", "D", "W")
        limit: Number of klines to fetch (max 1000)

    Returns:
        DataFrame with columns: open_time, open, high, low, close, volume
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = data["result"]["list"]
    df = pd.DataFrame(
        records,
        columns=["open_time", "open", "high", "low", "close", "volume", "turnover"],
    )
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms")
    df = df.sort_values("open_time").reset_index(drop=True)
    logger.info(f"Fetched {len(df)} klines for {symbol} from Bybit")
    return df


def fetch_stock_data(
    ticker: str = "AAPL",
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch stock data via yfinance.

    Args:
        ticker: Stock ticker symbol
        start: Start date
        end: End date (defaults to today)

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    data = yf.download(ticker, start=start, end=end, progress=False)
    data.columns = [c.lower() for c in data.columns]
    logger.info(f"Fetched {len(data)} rows for {ticker} from Yahoo Finance")
    return data


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n)
    prices = pd.DataFrame(
        {
            "close": 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02)),
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=dates,
    )

    dataset = TradingDataset(prices, window=60)
    logger.info(f"Dataset size: {len(dataset)}")
    features, target = dataset[0]
    logger.info(f"Feature shape: {features.shape}, Target: {target.item()}")
