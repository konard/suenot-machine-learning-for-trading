"""
Data loading and preprocessing for Multi-Scale Attention.

This module provides utilities for fetching financial data from Bybit,
creating multi-scale features, and preparing data for training.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from torch.utils.data import Dataset
import torch


def fetch_bybit_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1",
    limit: int = 1000,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Bybit API.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval ('1', '5', '15', '60', '240', 'D', 'W')
        limit: Number of candles to fetch (max 1000)
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds

    Returns:
        DataFrame with OHLCV data indexed by timestamp
    """
    url = "https://api.bybit.com/v5/market/kline"

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }

    if start_time is not None:
        params["start"] = start_time
    if end_time is not None:
        params["end"] = end_time

    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    if data["retCode"] != 0:
        raise ValueError(f"Bybit API error: {data['retMsg']}")

    klines = data["result"]["list"]

    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(
        klines,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)

    df = df.set_index("timestamp").sort_index()

    return df


def fetch_bybit_klines_extended(
    symbol: str = "BTCUSDT",
    interval: str = "1",
    days: int = 7,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Fetch extended OHLCV data by making multiple API calls.

    Args:
        symbol: Trading pair
        interval: Candle interval
        days: Number of days of data to fetch
        verbose: Whether to print progress

    Returns:
        DataFrame with OHLCV data
    """
    # Calculate interval in milliseconds
    interval_ms = {
        "1": 60 * 1000,
        "3": 3 * 60 * 1000,
        "5": 5 * 60 * 1000,
        "15": 15 * 60 * 1000,
        "30": 30 * 60 * 1000,
        "60": 60 * 60 * 1000,
        "120": 2 * 60 * 60 * 1000,
        "240": 4 * 60 * 60 * 1000,
        "360": 6 * 60 * 60 * 1000,
        "720": 12 * 60 * 60 * 1000,
        "D": 24 * 60 * 60 * 1000,
        "W": 7 * 24 * 60 * 60 * 1000,
    }.get(interval, 60 * 1000)

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    all_data = []
    current_end = end_time

    while current_end > start_time:
        if verbose:
            print(
                f"Fetching data ending at {datetime.fromtimestamp(current_end/1000)}"
            )

        df = fetch_bybit_klines(
            symbol=symbol,
            interval=interval,
            limit=1000,
            end_time=current_end,
        )

        if df.empty:
            break

        all_data.append(df)
        current_end = int(df.index.min().timestamp() * 1000) - interval_ms

        # Rate limiting
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data).sort_index()
    result = result[~result.index.duplicated(keep="first")]

    return result


def create_multi_scale_features(
    df: pd.DataFrame,
    scales: List[str] = ["1min", "5min", "15min", "1H", "4H"],
) -> Dict[str, pd.DataFrame]:
    """
    Create feature DataFrames for multiple time scales.

    Args:
        df: DataFrame with 1-minute OHLCV data
        scales: List of target time scales

    Returns:
        Dictionary mapping scale name to feature DataFrame
    """
    result = {}

    for scale in scales:
        # Resample if needed
        if scale == "1min":
            resampled = df.copy()
        else:
            resampled = (
                df.resample(scale)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )

        if len(resampled) < 30:
            print(f"Warning: Scale {scale} has only {len(resampled)} samples")
            continue

        # Calculate features
        features = pd.DataFrame(index=resampled.index)

        # Returns
        features["log_return"] = np.log(
            resampled["close"] / resampled["close"].shift(1)
        )
        features["return_1"] = resampled["close"].pct_change(1)
        features["return_5"] = resampled["close"].pct_change(5)
        features["return_10"] = resampled["close"].pct_change(10)

        # Volatility
        features["volatility_5"] = features["log_return"].rolling(5).std()
        features["volatility_20"] = features["log_return"].rolling(20).std()
        features["volatility_ratio"] = features["volatility_5"] / (
            features["volatility_20"] + 1e-8
        )

        # Volume features
        vol_ma = resampled["volume"].rolling(20).mean()
        features["volume_ratio"] = resampled["volume"] / (vol_ma + 1e-8)
        features["volume_change"] = resampled["volume"].pct_change()

        # Price features
        features["high_low_ratio"] = (resampled["high"] - resampled["low"]) / (
            resampled["close"] + 1e-8
        )
        features["close_position"] = (resampled["close"] - resampled["low"]) / (
            resampled["high"] - resampled["low"] + 1e-8
        )

        # Moving averages
        for period in [5, 10, 20, 50]:
            if len(resampled) > period:
                ma = resampled["close"].rolling(period).mean()
                features[f"ma_{period}_ratio"] = resampled["close"] / ma - 1

        # RSI
        delta = resampled["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        features["rsi"] = 100 - (100 / (1 + rs))
        features["rsi_normalized"] = (features["rsi"] - 50) / 50

        # MACD
        if len(resampled) > 26:
            ema12 = resampled["close"].ewm(span=12, adjust=False).mean()
            ema26 = resampled["close"].ewm(span=26, adjust=False).mean()
            features["macd"] = (ema12 - ema26) / (resampled["close"] + 1e-8)
            features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
            features["macd_hist"] = features["macd"] - features["macd_signal"]

        # Bollinger Bands
        if len(resampled) > 20:
            sma20 = resampled["close"].rolling(20).mean()
            std20 = resampled["close"].rolling(20).std()
            features["bb_position"] = (resampled["close"] - sma20) / (
                2 * std20 + 1e-8
            )

        # Drop NaN and clip extreme values
        features = features.dropna()
        for col in features.columns:
            features[col] = features[col].clip(-10, 10)

        result[scale] = features

    return result


def create_sequences(
    multi_scale_features: Dict[str, pd.DataFrame],
    lookback_periods: Dict[str, int],
    prediction_horizon: int = 1,
    target_scale: str = "1H",
    target_column: str = "log_return",
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Create sequences for multi-scale training.

    Args:
        multi_scale_features: Dictionary of feature DataFrames by scale
        lookback_periods: Number of periods to look back for each scale
        prediction_horizon: How many periods ahead to predict
        target_scale: Scale for the prediction target
        target_column: Column to predict

    Returns:
        X: Dictionary of feature arrays by scale
        y: Target array
        timestamps: Array of target timestamps
    """
    if target_scale not in multi_scale_features:
        raise ValueError(f"Target scale {target_scale} not in features")

    target_df = multi_scale_features[target_scale]

    X = {scale: [] for scale in multi_scale_features.keys()}
    y = []
    timestamps = []

    max_lookback = max(lookback_periods.values())

    for i in range(max_lookback, len(target_df) - prediction_horizon):
        target_time = target_df.index[i]

        valid = True
        scale_sequences = {}

        for scale, features in multi_scale_features.items():
            lookback = lookback_periods.get(scale, 24)

            # Find the index closest to target_time
            if scale == target_scale:
                idx = i
            else:
                idx = features.index.get_indexer([target_time], method="ffill")[0]

            if idx < lookback or idx < 0:
                valid = False
                break

            # Extract sequence
            seq = features.iloc[idx - lookback : idx].values
            if len(seq) != lookback:
                valid = False
                break

            scale_sequences[scale] = seq

        if valid:
            for scale, seq in scale_sequences.items():
                X[scale].append(seq)

            # Target: future return
            target_return = target_df[target_column].iloc[i + prediction_horizon]
            y.append(target_return)
            timestamps.append(target_time)

    # Convert to numpy arrays
    X = {scale: np.array(seqs, dtype=np.float32) for scale, seqs in X.items()}
    y = np.array(y, dtype=np.float32)
    timestamps = np.array(timestamps)

    return X, y, timestamps


class MultiScaleDataset(Dataset):
    """PyTorch Dataset for multi-scale time series."""

    def __init__(
        self,
        X: Dict[str, np.ndarray],
        y: np.ndarray,
        scale_names: List[str],
    ):
        """
        Args:
            X: Dictionary of feature arrays by scale
            y: Target array
            scale_names: List of scale names in order
        """
        self.X = {k: torch.FloatTensor(v) for k, v in X.items()}
        self.y = torch.FloatTensor(y).unsqueeze(-1)
        self.scale_names = scale_names
        self.n_samples = len(y)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        x = {scale: self.X[scale][idx] for scale in self.scale_names}
        return x, self.y[idx]


def prepare_train_val_test_split(
    X: Dict[str, np.ndarray],
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[
    Tuple[Dict[str, np.ndarray], np.ndarray],
    Tuple[Dict[str, np.ndarray], np.ndarray],
    Tuple[Dict[str, np.ndarray], np.ndarray],
]:
    """
    Split data into train/validation/test sets.

    Uses time-based split (no shuffling) to avoid data leakage.

    Args:
        X: Feature dictionary
        y: Target array
        train_ratio: Fraction for training
        val_ratio: Fraction for validation

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n_samples = len(y)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_X = {k: v[:train_end] for k, v in X.items()}
    train_y = y[:train_end]

    val_X = {k: v[train_end:val_end] for k, v in X.items()}
    val_y = y[train_end:val_end]

    test_X = {k: v[val_end:] for k, v in X.items()}
    test_y = y[val_end:]

    return (train_X, train_y), (val_X, val_y), (test_X, test_y)


def generate_synthetic_data(
    n_samples: int = 1000,
    scales: List[str] = ["1min", "5min", "1H"],
    n_features: int = 10,
    lookbacks: Dict[str, int] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Generate synthetic multi-scale data for testing.

    Args:
        n_samples: Number of samples
        scales: List of scale names
        n_features: Number of features per scale
        lookbacks: Lookback period for each scale

    Returns:
        X: Dictionary of synthetic features
        y: Synthetic targets
    """
    if lookbacks is None:
        lookbacks = {scale: 24 for scale in scales}

    X = {}
    for scale in scales:
        lookback = lookbacks[scale]
        X[scale] = np.random.randn(n_samples, lookback, n_features).astype(np.float32)

    # Generate targets correlated with features
    y = np.zeros(n_samples, dtype=np.float32)
    for scale in scales:
        y += X[scale][:, -1, 0] * np.random.randn() * 0.1

    y += np.random.randn(n_samples) * 0.01

    return X, y
