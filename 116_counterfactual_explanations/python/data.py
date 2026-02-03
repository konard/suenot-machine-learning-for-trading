"""
Data loading and feature engineering for trading models.

This module provides utilities for fetching market data from exchanges
and preparing features for trading classifiers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from datetime import datetime, timedelta


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class BybitClient:
    """
    Client for fetching data from Bybit exchange.

    Provides methods to fetch historical OHLCV data for cryptocurrency pairs.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "60",
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Candle]:
        """
        Fetch kline (candlestick) data from Bybit.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candle interval in minutes ("1", "5", "15", "60", "240", "D")
            limit: Number of candles to fetch (max 200)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            List of Candle objects
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 200)
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                raise ValueError(f"API error: {data.get('retMsg')}")

            candles = []
            for item in data.get("result", {}).get("list", []):
                candles.append(Candle(
                    timestamp=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5])
                ))

            # API returns newest first, reverse to chronological order
            return list(reversed(candles))

        except requests.RequestException as e:
            print(f"Failed to fetch data: {e}")
            return []

    def get_dataframe(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "60",
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Fetch kline data as a pandas DataFrame.

        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            limit: Number of candles

        Returns:
            DataFrame with OHLCV data
        """
        candles = self.get_klines(symbol, interval, limit)

        if not candles:
            return pd.DataFrame()

        data = {
            'timestamp': [c.timestamp for c in candles],
            'open': [c.open for c in candles],
            'high': [c.high for c in candles],
            'low': [c.low for c in candles],
            'close': [c.close for c in candles],
            'volume': [c.volume for c in candles]
        }

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    Args:
        prices: Series of prices
        period: RSI period

    Returns:
        RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
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
        prices: Series of prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        Tuple of (MACD line, signal line, histogram)
    """
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands.

    Args:
        prices: Series of prices
        period: Moving average period
        num_std: Number of standard deviations

    Returns:
        Tuple of (middle band, upper band, lower band)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return middle, upper, lower


def prepare_features(
    df: pd.DataFrame,
    lookback: int = 20
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare features for trading model.

    Args:
        df: DataFrame with OHLCV data
        lookback: Period for technical indicators

    Returns:
        Tuple of (feature matrix, feature names)
    """
    features = {}

    # Price features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close']).diff()

    # RSI
    features['rsi'] = compute_rsi(df['close'], 14)

    # Normalize RSI to [-1, 1] range
    features['rsi_normalized'] = (features['rsi'] - 50) / 50

    # MACD
    macd, signal, hist = compute_macd(df['close'])
    # Normalize MACD by price
    features['macd'] = macd / df['close']
    features['macd_signal'] = signal / df['close']
    features['macd_hist'] = hist / df['close']

    # Bollinger Bands
    middle, upper, lower = compute_bollinger_bands(df['close'])
    # Position within bands (-1 to 1)
    bb_width = upper - lower
    features['bb_position'] = (df['close'] - middle) / (bb_width / 2)
    features['bb_width'] = bb_width / middle

    # Volume features
    volume_ma = df['volume'].rolling(window=lookback).mean()
    features['volume_ratio'] = df['volume'] / volume_ma

    # Volatility
    features['volatility'] = features['returns'].rolling(window=lookback).std()

    # Momentum
    features['momentum'] = df['close'] / df['close'].shift(lookback) - 1

    # Trend strength (price vs moving average)
    ma = df['close'].rolling(window=lookback).mean()
    features['trend_strength'] = (df['close'] - ma) / ma

    # Combine into DataFrame and drop NaN
    feature_df = pd.DataFrame(features)
    feature_df = feature_df.dropna()

    feature_names = list(features.keys())

    return feature_df.values, feature_names


def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series model.

    Args:
        features: Feature matrix (num_samples, num_features)
        labels: Label array
        sequence_length: Length of each sequence

    Returns:
        Tuple of (sequences, sequence_labels)
    """
    sequences = []
    sequence_labels = []

    for i in range(sequence_length, len(features)):
        sequences.append(features[i - sequence_length:i])
        sequence_labels.append(labels[i])

    return np.array(sequences), np.array(sequence_labels)


def create_labels(
    df: pd.DataFrame,
    threshold: float = 0.005,
    forward_period: int = 1
) -> np.ndarray:
    """
    Create trading labels from price data.

    Args:
        df: DataFrame with 'close' column
        threshold: Return threshold for buy/sell signals
        forward_period: Periods to look ahead

    Returns:
        Array of labels (0=SELL, 1=HOLD, 2=BUY)
    """
    forward_returns = df['close'].pct_change(forward_period).shift(-forward_period)

    labels = np.ones(len(df), dtype=np.int64)  # Default HOLD
    labels[forward_returns > threshold] = 2    # BUY
    labels[forward_returns < -threshold] = 0   # SELL

    return labels


def get_sample_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Get sample data for testing.

    Returns:
        Tuple of (features, labels, feature_names)
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500

    # Create correlated features
    rsi = np.random.normal(50, 15, n_samples)
    rsi = np.clip(rsi, 0, 100)

    macd = np.random.normal(0, 0.01, n_samples)
    volume_ratio = np.random.lognormal(0, 0.3, n_samples)
    volatility = np.random.exponential(0.02, n_samples)
    bb_position = np.random.normal(0, 0.5, n_samples)
    momentum = np.random.normal(0, 0.05, n_samples)

    features = np.column_stack([
        (rsi - 50) / 50,  # Normalized RSI
        macd,
        volume_ratio - 1,
        volatility,
        bb_position,
        momentum
    ])

    feature_names = [
        'rsi_normalized',
        'macd',
        'volume_ratio',
        'volatility',
        'bb_position',
        'momentum'
    ]

    # Create labels based on features
    # Simple rule: buy if RSI low and momentum positive, sell if RSI high and momentum negative
    labels = np.ones(n_samples, dtype=np.int64)
    labels[(features[:, 0] < -0.4) & (features[:, 5] > 0)] = 2  # BUY
    labels[(features[:, 0] > 0.4) & (features[:, 5] < 0)] = 0   # SELL

    return features, labels, feature_names
