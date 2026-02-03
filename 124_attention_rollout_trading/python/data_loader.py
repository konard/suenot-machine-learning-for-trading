"""
data_loader.py - Data loading utilities for stock and crypto markets

This module provides functions to load market data from various sources
including Yahoo Finance for stocks and Bybit for cryptocurrency.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import requests
from datetime import datetime, timedelta


def load_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Load stock data using yfinance.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval (1d, 1h, etc.)

    Returns:
        DataFrame with OHLCV data
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    df = df.reset_index()
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    return df


def load_bybit_data(
    symbol: str = "BTCUSDT",
    interval: str = "D",
    limit: int = 200
) -> pd.DataFrame:
    """
    Load cryptocurrency data from Bybit exchange.

    Args:
        symbol: Trading pair (e.g., BTCUSDT, ETHUSDT)
        interval: Candle interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        limit: Number of candles to fetch (max 200)

    Returns:
        DataFrame with OHLCV data
    """
    url = "https://api.bybit.com/v5/market/kline"

    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["retCode"] != 0:
        raise ValueError(f"Bybit API error: {data['retMsg']}")

    candles = data["result"]["list"]

    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def prepare_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    lookback: int = 20,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features for transformer model.

    Args:
        df: DataFrame with OHLCV data
        feature_columns: Columns to use as features
        lookback: Number of time steps to look back
        normalize: Whether to normalize features

    Returns:
        X: Feature sequences (n_samples, lookback, n_features)
        y: Target labels (n_samples,)
    """
    if feature_columns is None:
        feature_columns = ["open", "high", "low", "close", "volume"]

    # Add technical indicators
    df = add_technical_indicators(df)

    # Forward returns for labels
    df["returns"] = df["close"].pct_change().shift(-1)

    # Create labels: 0=sell, 1=hold, 2=buy
    df["label"] = 1  # hold
    df.loc[df["returns"] > 0.01, "label"] = 2  # buy
    df.loc[df["returns"] < -0.01, "label"] = 0  # sell

    # Normalize features
    if normalize:
        for col in feature_columns:
            if col in df.columns:
                df[col] = (df[col] - df[col].rolling(lookback).mean()) / (
                    df[col].rolling(lookback).std() + 1e-8
                )

    df = df.dropna()

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(df[feature_columns].iloc[i-lookback:i].values)
        y.append(df["label"].iloc[i])

    return np.array(X), np.array(y)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added indicators
    """
    # Moving averages
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std

    # Volatility
    df["volatility"] = df["close"].pct_change().rolling(20).std()

    return df


class TradingDataset:
    """Dataset class for trading data with batch iteration."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32
    ):
        """
        Initialize TradingDataset.

        Args:
            X: Feature sequences
            y: Target labels
            batch_size: Batch size for iteration
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_samples = len(X)

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.random.permutation(self.n_samples)
        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]
