"""Data loading utilities for stock and cryptocurrency markets.

Supports:
- Stock market data via yfinance
- Cryptocurrency data via Bybit public API
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader, Dataset


def fetch_stock_data(
    ticker: str = "SPY",
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch stock market data using yfinance.

    Args:
        ticker: Stock symbol
        period: Time period (e.g., '1y', '2y', '5y')
        interval: Data interval (e.g., '1d', '1h')

    Returns:
        DataFrame with OHLCV data
    """
    import yfinance as yf

    data = yf.download(ticker, period=period, interval=interval, progress=False)
    data = data.dropna()
    return data


def fetch_bybit_data(
    symbol: str = "BTCUSDT",
    interval: str = "D",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch cryptocurrency data from Bybit public API.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
        interval: Kline interval ('1', '5', '15', '60', 'D', 'W')
        limit: Number of candles to fetch (max 1000)

    Returns:
        DataFrame with OHLCV data
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    result = response.json()

    if result["retCode"] != 0:
        raise ValueError(f"Bybit API error: {result['retMsg']}")

    rows = result["result"]["list"]
    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df = df.set_index("timestamp").sort_index()
    df = df.drop(columns=["turnover"])
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicator features from OHLCV data.

    Features:
        - Log returns
        - Realized volatility (20-period)
        - RSI (14-period)
        - MACD signal
        - Volume ratio (relative to 20-period average)
        - High-Low range
        - Close-Open range
        - Price position within range

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns

    Returns:
        DataFrame with computed features (NaN rows dropped)
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    features = pd.DataFrame(index=df.index)

    # Log returns
    features["log_return"] = np.log(close / close.shift(1))

    # Realized volatility (20-period rolling std of returns)
    features["volatility_20"] = features["log_return"].rolling(20).std()

    # RSI (14-period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    features["macd_signal"] = macd - signal_line

    # Volume ratio
    avg_vol = volume.rolling(20).mean()
    features["volume_ratio"] = volume / avg_vol.replace(0, np.nan)

    # Range features
    price_range = high - low
    features["hl_range"] = price_range / close
    features["co_range"] = (close - open_) / close

    # Price position within range
    features["price_position"] = (close - low) / price_range.replace(0, np.nan)

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features


def prepare_targets(features: pd.DataFrame) -> pd.DataFrame:
    """Prepare prediction targets from feature DataFrame.

    Targets:
        - future_return: Next-period log return
        - direction: 1 if positive return, 0 otherwise
        - future_volatility: Forward-looking realized volatility

    Args:
        features: DataFrame with 'log_return' and 'volatility_20' columns

    Returns:
        DataFrame with target columns (shifted for forward-looking)
    """
    targets = pd.DataFrame(index=features.index)
    targets["future_return"] = features["log_return"].shift(-1)
    targets["direction"] = (targets["future_return"] > 0).astype(float)
    targets["future_volatility"] = features["volatility_20"].shift(-1)
    return targets.dropna()


class FinancialDataset(Dataset):
    """PyTorch Dataset for windowed financial time series.

    Creates sliding windows of features and corresponding targets
    for sequence-to-one prediction.
    """

    def __init__(
        self,
        features: np.ndarray,
        target_return: np.ndarray,
        target_direction: np.ndarray,
        target_volatility: np.ndarray,
        window_size: int = 64,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target_return = torch.tensor(target_return, dtype=torch.float32)
        self.target_direction = torch.tensor(target_direction, dtype=torch.float32)
        self.target_volatility = torch.tensor(target_volatility, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.features) - self.window_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        x = self.features[idx : idx + self.window_size]
        target_idx = idx + self.window_size - 1
        return (
            x,
            self.target_return[target_idx],
            self.target_direction[target_idx],
            self.target_volatility[target_idx],
        )


def create_dataloaders(
    source: str = "stock",
    ticker: str = "SPY",
    window_size: int = 64,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from financial data.

    Args:
        source: Data source ('stock' or 'bybit')
        ticker: Ticker symbol
        window_size: Lookback window for sequences
        batch_size: Batch size for DataLoaders
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if source == "bybit":
        df = fetch_bybit_data(symbol=ticker)
    else:
        df = fetch_stock_data(ticker=ticker)

    features = compute_features(df)
    targets = prepare_targets(features)

    # Align features and targets
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]

    # Normalize features
    feature_vals = features.values
    mean = feature_vals.mean(axis=0)
    std = feature_vals.std(axis=0) + 1e-8
    feature_vals = (feature_vals - mean) / std

    target_ret = targets["future_return"].values
    target_dir = targets["direction"].values
    target_vol = targets["future_volatility"].values

    # Chronological split
    n = len(feature_vals)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_ds = FinancialDataset(
        feature_vals[:train_end], target_ret[:train_end],
        target_dir[:train_end], target_vol[:train_end], window_size
    )
    val_ds = FinancialDataset(
        feature_vals[train_end:val_end], target_ret[train_end:val_end],
        target_dir[train_end:val_end], target_vol[train_end:val_end], window_size
    )
    test_ds = FinancialDataset(
        feature_vals[val_end:], target_ret[val_end:],
        target_dir[val_end:], target_vol[val_end:], window_size
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
