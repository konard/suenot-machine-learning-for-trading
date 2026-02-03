"""
Data Loading and Preprocessing for LRP Trading Models

This module provides utilities for loading market data from various sources
(Bybit, Yahoo Finance) and preprocessing it for LRP-enabled models.

Features:
    - Multi-source data fetching (Bybit, Yahoo Finance)
    - Technical indicator calculation (RSI, MACD, Bollinger Bands, ATR)
    - Sequence creation for time series models
    - PyTorch Dataset implementation

Example:
    >>> data = prepare_trading_data(['BTCUSDT'], lookback=60)
    >>> dataset = TradingDataset(data['X'], data['y'])
    >>> loader = DataLoader(dataset, batch_size=32)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import requests
from datetime import datetime, timedelta

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    lookback: int = 60
    horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    normalize: bool = True
    target_type: str = "direction"  # "direction" or "return"


def fetch_bybit_data(
    symbol: str,
    interval: str = "60",
    limit: int = 1000,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Bybit exchange.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval in minutes ('1', '5', '15', '60', '240', 'D')
        limit: Number of candles to fetch (max 1000)
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    base_url = "https://api.bybit.com/v5/market/kline"

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    if start_time:
        params["start"] = start_time
    if end_time:
        params["end"] = end_time

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0:
            raise ValueError(f"API error: {data.get('retMsg')}")

        # Parse kline data
        klines = data["result"]["list"]
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # Sort by timestamp (API returns newest first)
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    except requests.RequestException as e:
        print(f"Error fetching Bybit data: {e}")
        return pd.DataFrame()


def fetch_yahoo_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Args:
        symbol: Stock/crypto ticker (e.g., 'AAPL', 'BTC-USD')
        period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        interval: Data interval ('1d', '1wk', '1mo')

    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    except ImportError:
        print("yfinance not installed. Using sample data.")
        return _generate_sample_data()
    except Exception as e:
        print(f"Error fetching Yahoo data: {e}")
        return _generate_sample_data()


def _generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq="H")
    price = 100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.02))

    df = pd.DataFrame({
        "timestamp": dates,
        "open": price * (1 + np.random.randn(n_samples) * 0.01),
        "high": price * (1 + np.abs(np.random.randn(n_samples) * 0.02)),
        "low": price * (1 - np.abs(np.random.randn(n_samples) * 0.02)),
        "close": price,
        "volume": np.random.exponential(1000, n_samples),
    })

    return df


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    Args:
        prices: Price series (typically close prices)
        period: RSI lookback period (default: 14)

    Returns:
        RSI values (0-100 scale)
    """
    delta = prices.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD (Moving Average Convergence Divergence).

    Args:
        prices: Price series
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        Tuple of (MACD line, signal line, histogram)
    """
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_bollinger_position(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> pd.Series:
    """
    Compute position within Bollinger Bands (0 to 1 scale).

    Args:
        prices: Price series
        period: Moving average period
        num_std: Number of standard deviations for bands

    Returns:
        Position value (0 = at lower band, 1 = at upper band)
    """
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = ma + num_std * std
    lower = ma - num_std * std

    position = (prices - lower) / (upper - lower + 1e-10)

    return position.clip(0, 1)


def compute_atr(
    df: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """
    Compute Average True Range (ATR).

    Args:
        df: DataFrame with high, low, close columns
        period: ATR period (default: 14)

    Returns:
        ATR values
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical features from OHLCV data.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()

    # Returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)
    df["return_20"] = df["close"].pct_change(20)

    # Volume features
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["volume_change"] = df["volume"].pct_change()

    # Volatility
    df["volatility_5"] = df["log_return"].rolling(5).std()
    df["volatility_20"] = df["log_return"].rolling(20).std()
    df["volatility_ratio"] = df["volatility_5"] / (df["volatility_20"] + 1e-10)

    # RSI
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["rsi_7"] = compute_rsi(df["close"], 7)

    # MACD
    macd, signal, hist = compute_macd(df["close"])
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    # Bollinger Bands
    df["bb_position"] = compute_bollinger_position(df["close"])

    # ATR
    df["atr_14"] = compute_atr(df, 14)
    df["atr_ratio"] = df["atr_14"] / df["close"]

    # Moving averages
    df["ma_10"] = df["close"].rolling(10).mean() / df["close"]
    df["ma_20"] = df["close"].rolling(20).mean() / df["close"]
    df["ma_50"] = df["close"].rolling(50).mean() / df["close"]

    # Price position
    df["high_low_ratio"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)

    return df


def prepare_trading_data(
    symbols: List[str],
    source: str = "bybit",
    lookback: int = 60,
    horizon: int = 1,
    features: Optional[List[str]] = None,
    config: Optional[DataConfig] = None
) -> Dict:
    """
    Prepare complete trading dataset for LRP model.

    Args:
        symbols: List of trading symbols
        source: Data source ('bybit', 'yahoo', 'sample')
        lookback: Historical window size
        horizon: Prediction horizon
        features: List of feature names to use (default: all)
        config: Data configuration

    Returns:
        Dictionary with:
            - X: Feature array [samples, lookback, features]
            - y: Target array [samples]
            - feature_names: List of feature names
            - train_idx, val_idx, test_idx: Split indices
    """
    config = config or DataConfig(lookback=lookback, horizon=horizon)

    if features is None:
        features = [
            "log_return", "return_5", "return_10",
            "volume_ratio", "volume_change",
            "volatility_5", "volatility_20", "volatility_ratio",
            "rsi_14", "rsi_7",
            "macd", "macd_signal", "macd_hist",
            "bb_position",
            "atr_ratio",
            "ma_10", "ma_20",
            "high_low_ratio"
        ]

    all_data = []

    for symbol in symbols:
        # Fetch data
        if source == "bybit":
            df = fetch_bybit_data(symbol)
        elif source == "yahoo":
            df = fetch_yahoo_data(symbol)
        else:
            df = _generate_sample_data()

        # Compute features
        df = compute_features(df)

        # Create target
        if config.target_type == "direction":
            df["target"] = (df["log_return"].shift(-horizon) > 0).astype(int)
        else:
            df["target"] = df["log_return"].shift(-horizon)

        # Select features and drop NaN
        df_features = df[features + ["target"]].dropna()
        all_data.append(df_features)

    # Combine all symbols
    combined = pd.concat(all_data, ignore_index=True)

    # Create sequences
    X, y = [], []
    n_samples = len(combined) - lookback - horizon + 1

    for i in range(n_samples):
        X.append(combined[features].iloc[i:i + lookback].values)
        y.append(combined["target"].iloc[i + lookback - 1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # Normalize features
    if config.normalize:
        mean = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True) + 1e-8
        X = (X - mean) / std

    # Train/val/test split
    n = len(X)
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))

    return {
        "X": X,
        "y": y,
        "feature_names": features,
        "lookback": lookback,
        "horizon": horizon,
        "train_idx": np.arange(0, train_end),
        "val_idx": np.arange(train_end, val_end),
        "test_idx": np.arange(val_end, n),
    }


class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading data.

    Args:
        X: Feature array [samples, lookback, features]
        y: Target array [samples]

    Example:
        >>> dataset = TradingDataset(data['X'], data['y'])
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long if y.dtype in [np.int32, np.int64] else torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_dataloaders(
    data: Dict,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        data: Output from prepare_trading_data
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    X, y = data["X"], data["y"]

    train_dataset = TradingDataset(
        X[data["train_idx"]], y[data["train_idx"]]
    )
    val_dataset = TradingDataset(
        X[data["val_idx"]], y[data["val_idx"]]
    )
    test_dataset = TradingDataset(
        X[data["test_idx"]], y[data["test_idx"]]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
