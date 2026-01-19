"""
Data Loading and Preprocessing for Time Series

This module provides data loading utilities for financial time series,
with support for Bybit cryptocurrency data and general stock market data.
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KlineData:
    """Container for OHLCV kline data."""
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    symbol: str
    interval: str


class BybitDataLoader:
    """
    Data loader for Bybit cryptocurrency exchange.

    Fetches historical kline (candlestick) data via Bybit's public API.
    No authentication required for public market data.

    Args:
        base_url: Bybit API base URL

    Example:
        >>> loader = BybitDataLoader()
        >>> data = loader.load_klines('BTCUSDT', interval='1h', limit=1000)
        >>> print(data.close.shape)
    """

    def __init__(self, base_url: str = "https://api.bybit.com"):
        self.base_url = base_url
        self.session = requests.Session()

    def load_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 1000,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> KlineData:
        """
        Load kline data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles (max 1000 per request)
            start_time: Start datetime (optional)
            end_time: End datetime (optional)

        Returns:
            KlineData containing OHLCV data
        """
        # Convert interval to Bybit format
        interval_map = {
            '1m': '1', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '4h': '240', '1d': 'D', '1w': 'W'
        }
        bybit_interval = interval_map.get(interval, interval)

        # Build request parameters
        params = {
            'symbol': symbol,
            'interval': bybit_interval,
            'limit': min(limit, 1000)
        }

        if start_time:
            params['start'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['end'] = int(end_time.timestamp() * 1000)

        # Make API request
        url = f"{self.base_url}/v5/market/kline"

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('retCode') != 0:
                raise ValueError(f"Bybit API error: {data.get('retMsg')}")

            klines = data.get('result', {}).get('list', [])

            if not klines:
                raise ValueError(f"No data returned for {symbol}")

            # Parse klines (Bybit returns newest first, so reverse)
            klines = list(reversed(klines))

            timestamps = np.array([int(k[0]) for k in klines])
            opens = np.array([float(k[1]) for k in klines])
            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            closes = np.array([float(k[4]) for k in klines])
            volumes = np.array([float(k[5]) for k in klines])

            return KlineData(
                timestamp=timestamps,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                volume=volumes,
                symbol=symbol,
                interval=interval
            )

        except requests.RequestException as e:
            print(f"Warning: API request failed for {symbol}: {e}")
            # Return synthetic data for testing
            return self._generate_synthetic_data(symbol, interval, limit)

    def _generate_synthetic_data(
        self,
        symbol: str,
        interval: str,
        limit: int
    ) -> KlineData:
        """Generate synthetic data for testing when API is unavailable."""
        print(f"Generating synthetic data for {symbol}...")

        # Generate timestamps
        now = datetime.now()
        interval_seconds = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        seconds = interval_seconds.get(interval, 3600)

        timestamps = np.array([
            int((now - timedelta(seconds=seconds * (limit - i - 1))).timestamp() * 1000)
            for i in range(limit)
        ])

        # Generate realistic-looking price data
        np.random.seed(42)  # For reproducibility

        # Start price (approximate current BTC price)
        base_price = 45000 if 'BTC' in symbol else 2500 if 'ETH' in symbol else 100

        # Random walk with mean reversion
        returns = np.random.normal(0, 0.01, limit)  # 1% daily volatility
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC from close prices
        opens = prices * (1 + np.random.uniform(-0.005, 0.005, limit))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.003, limit)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.003, limit)))
        closes = prices
        volumes = np.random.exponential(1000, limit) * base_price / 1000

        return KlineData(
            timestamp=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            volume=volumes,
            symbol=symbol,
            interval=interval
        )

    def load_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "1h",
        limit: int = 1000
    ) -> Dict[str, KlineData]:
        """
        Load kline data for multiple symbols.

        Args:
            symbols: List of trading pairs
            interval: Timeframe
            limit: Number of candles per symbol

        Returns:
            Dictionary mapping symbol to KlineData
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.load_klines(symbol, interval, limit)
                print(f"Loaded {len(data[symbol].close)} candles for {symbol}")
            except Exception as e:
                print(f"Failed to load {symbol}: {e}")

        return data


def prepare_features(
    kline_data: KlineData,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Prepare features from raw kline data.

    Computes technical features:
    - Log returns
    - Volatility (rolling std)
    - Volume change
    - RSI
    - Price momentum

    Args:
        kline_data: Raw OHLCV data
        lookback: Lookback period for rolling calculations

    Returns:
        DataFrame with computed features
    """
    df = pd.DataFrame({
        'timestamp': kline_data.timestamp,
        'open': kline_data.open,
        'high': kline_data.high,
        'low': kline_data.low,
        'close': kline_data.close,
        'volume': kline_data.volume
    })

    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility
    df['volatility'] = df['log_return'].rolling(lookback).std()

    # Volume change
    df['volume_change'] = df['volume'] / df['volume'].rolling(lookback).mean()

    # Price momentum
    df['momentum'] = df['close'] / df['close'].shift(lookback) - 1

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(lookback).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(lookback).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # High-low range
    df['range'] = (df['high'] - df['low']) / df['close']

    # VWAP deviation
    df['vwap'] = (df['close'] * df['volume']).rolling(lookback).sum() / df['volume'].rolling(lookback).sum()
    df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

    # Calendar features
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['quarter'] = df['datetime'].dt.quarter - 1  # 0-indexed

    # Drop NaN rows
    df = df.dropna()

    return df


def create_sequences(
    features: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'log_return',
    seq_len: int = 168,
    horizon: int = 24,
    include_timestamps: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    Create sequences for time series modeling.

    Args:
        features: DataFrame with features
        feature_cols: Columns to use as input features
        target_col: Column to predict
        seq_len: Input sequence length
        horizon: Prediction horizon
        include_timestamps: Whether to include timestamp info

    Returns:
        Tuple of (X, y, timestamps_dict)
        X: [n_samples, seq_len, n_features]
        y: [n_samples, horizon] or [n_samples] if horizon=1
        timestamps_dict: Dict with calendar features if include_timestamps=True
    """
    data = features[feature_cols].values
    target = features[target_col].values

    n_samples = len(features) - seq_len - horizon + 1

    if n_samples <= 0:
        raise ValueError(f"Not enough data: {len(features)} rows, need {seq_len + horizon}")

    X = np.zeros((n_samples, seq_len, len(feature_cols)))
    y = np.zeros((n_samples, horizon)) if horizon > 1 else np.zeros(n_samples)

    for i in range(n_samples):
        X[i] = data[i:i + seq_len]
        if horizon > 1:
            y[i] = target[i + seq_len:i + seq_len + horizon]
        else:
            y[i] = target[i + seq_len]

    timestamps = None
    if include_timestamps:
        timestamps = {}
        for col in ['hour', 'dayofweek', 'day', 'month', 'quarter']:
            if col in features.columns:
                ts_data = features[col].values
                timestamps[col] = np.array([
                    ts_data[i:i + seq_len]
                    for i in range(n_samples)
                ])

    return X, y, timestamps


def train_test_split_time_series(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: Optional[Dict[str, np.ndarray]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple:
    """
    Split time series data into train/val/test sets.

    Uses chronological split (no shuffling) to prevent data leakage.

    Args:
        X: Input sequences
        y: Target values
        timestamps: Optional timestamp dictionary
        train_ratio: Fraction for training
        val_ratio: Fraction for validation

    Returns:
        Tuple of train/val/test splits
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    if timestamps is not None:
        ts_train = {k: v[:train_end] for k, v in timestamps.items()}
        ts_val = {k: v[train_end:val_end] for k, v in timestamps.items()}
        ts_test = {k: v[val_end:] for k, v in timestamps.items()}
        return (X_train, X_val, X_test,
                y_train, y_val, y_test,
                ts_train, ts_val, ts_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Test data loading
    print("Testing Data Loading...")
    print("=" * 60)

    # Test Bybit loader
    loader = BybitDataLoader()

    # Try to load real data, fall back to synthetic
    print("\n1. Loading BTCUSDT data...")
    btc_data = loader.load_klines('BTCUSDT', interval='1h', limit=500)
    print(f"   Loaded {len(btc_data.close)} candles")
    print(f"   Price range: ${btc_data.close.min():.2f} - ${btc_data.close.max():.2f}")

    # Test feature preparation
    print("\n2. Preparing features...")
    features = prepare_features(btc_data)
    print(f"   Features shape: {features.shape}")
    print(f"   Columns: {list(features.columns)}")

    # Test sequence creation
    print("\n3. Creating sequences...")
    feature_cols = ['log_return', 'volatility', 'volume_change', 'momentum', 'rsi', 'range']
    X, y, timestamps = create_sequences(
        features,
        feature_cols,
        seq_len=168,
        horizon=24
    )
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    if timestamps:
        print(f"   Timestamp keys: {list(timestamps.keys())}")

    # Test train/test split
    print("\n4. Train/test split...")
    splits = train_test_split_time_series(X, y, timestamps)
    X_train, X_val, X_test, y_train, y_val, y_test, ts_train, ts_val, ts_test = splits
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("\n" + "=" * 60)
    print("Data loading tests completed!")
