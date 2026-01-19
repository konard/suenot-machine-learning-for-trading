"""
Data Loading and Feature Engineering for FNet

This module provides data loading utilities for:
- Bybit (cryptocurrency data)
- Yahoo Finance (stock market data)

Also includes feature engineering functions optimized for FNet.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import requests
import time


class BybitDataLoader:
    """
    Data loader for Bybit cryptocurrency exchange.

    Provides methods to fetch OHLCV (Open, High, Low, Close, Volume) data
    for various trading pairs.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self, timeout: int = 30):
        """
        Args:
            timeout: Request timeout in seconds
        """
        self.session = requests.Session()
        self.timeout = timeout

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Candle interval in minutes ('1', '5', '15', '60', '240', 'D', 'W')
            limit: Number of candles to fetch (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data and timestamp
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        try:
            response = self.session.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if data["retCode"] != 0:
                raise ValueError(f"API error: {data['retMsg']}")

            # Parse kline data
            klines = data["result"]["list"]
            if not klines:
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = df[col].astype(float)

            # Sort by timestamp ascending
            df = df.sort_values("timestamp").reset_index(drop=True)

            return df

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return pd.DataFrame()

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "60",
        limit: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of trading pairs
            interval: Candle interval
            limit: Number of candles per symbol

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            df = self.fetch_klines(symbol, interval, limit)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.1)  # Rate limiting
        return data


class YahooDataLoader:
    """
    Data loader for Yahoo Finance stock data.

    Uses the public Yahoo Finance API to fetch historical stock data.
    """

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(self, timeout: int = 30):
        self.session = requests.Session()
        self.timeout = timeout
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_historical(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance.

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'GOOGL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d', '1wk')

        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.BASE_URL}/{symbol}"
        params = {
            "range": period,
            "interval": interval,
            "includePrePost": "false",
            "events": "div,splits"
        }

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if "chart" not in data or "result" not in data["chart"]:
                raise ValueError(f"Invalid response for {symbol}")

            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]

            df = pd.DataFrame({
                "timestamp": pd.to_datetime(timestamps, unit="s"),
                "open": quotes["open"],
                "high": quotes["high"],
                "low": quotes["low"],
                "close": quotes["close"],
                "volume": quotes["volume"]
            })

            # Remove rows with NaN values
            df = df.dropna()

            return df

        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
            return pd.DataFrame()

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stock symbols.
        """
        data = {}
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            df = self.fetch_historical(symbol, period, interval)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.2)  # Rate limiting
        return data


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical features for FNet model.

    Features computed:
    - Log returns
    - Volatility (rolling std)
    - Volume ratio
    - Price momentum (multiple horizons)
    - RSI
    - Bollinger Band position
    - MACD-related features

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()

    # Log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Volatility (20-period rolling std)
    df["volatility"] = df["log_return"].rolling(20).std()

    # Volume analysis
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma"] + 1e-8)

    # Price momentum at different horizons
    for period in [5, 10, 20]:
        df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1

    # RSI (14-period)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_normalized"] = (df["rsi"] - 50) / 50  # Normalize to [-1, 1]

    # Bollinger Bands position
    sma_20 = df["close"].rolling(20).mean()
    std_20 = df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - sma_20) / (2 * std_20 + 1e-8)

    # MACD-related
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_diff"] = df["macd"] - df["macd_signal"]

    # Price range
    df["high_low_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)

    # Day of week (for daily data) - cyclical encoding
    if "timestamp" in df.columns:
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int = 168,
    horizon: int = 24,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for FNet training.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        target_col: Target column name
        seq_len: Input sequence length
        horizon: Prediction horizon (steps ahead)
        stride: Step size between sequences

    Returns:
        X: Input sequences [n_samples, seq_len, n_features]
        y: Target values [n_samples]
    """
    features = df[feature_cols].values
    target = df[target_col].values

    X, y = [], []
    for i in range(seq_len, len(df) - horizon + 1, stride):
        X.append(features[i - seq_len:i])
        y.append(target[i + horizon - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def create_multi_asset_sequences(
    data: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    target_col: str,
    seq_len: int = 168,
    horizon: int = 24
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create sequences for multi-asset prediction.

    Args:
        data: Dictionary mapping symbol to DataFrame
        feature_cols: Feature columns to use
        target_col: Target column
        seq_len: Input sequence length
        horizon: Prediction horizon

    Returns:
        X: Input sequences [n_samples, seq_len, n_assets, n_features]
        y: Target values [n_samples, n_assets]
        symbols: List of symbols in order
    """
    symbols = list(data.keys())

    # Find common timestamps
    common_timestamps = None
    for symbol, df in data.items():
        ts_set = set(df["timestamp"])
        if common_timestamps is None:
            common_timestamps = ts_set
        else:
            common_timestamps = common_timestamps.intersection(ts_set)

    common_timestamps = sorted(common_timestamps)

    # Align all DataFrames
    aligned_data = {}
    for symbol, df in data.items():
        df_aligned = df[df["timestamp"].isin(common_timestamps)].copy()
        df_aligned = df_aligned.sort_values("timestamp").reset_index(drop=True)
        aligned_data[symbol] = df_aligned

    n_samples = len(common_timestamps)
    n_assets = len(symbols)
    n_features = len(feature_cols)

    # Create combined array
    all_features = np.zeros((n_samples, n_assets, n_features))
    all_targets = np.zeros((n_samples, n_assets))

    for i, symbol in enumerate(symbols):
        df = aligned_data[symbol]
        all_features[:, i, :] = df[feature_cols].values
        all_targets[:, i] = df[target_col].values

    # Create sequences
    X, y = [], []
    for i in range(seq_len, n_samples - horizon + 1):
        X.append(all_features[i - seq_len:i])
        y.append(all_targets[i + horizon - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), symbols


def normalize_features(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.

    Args:
        X: Input array [n_samples, seq_len, n_features]
        mean: Pre-computed mean (for test data)
        std: Pre-computed std (for test data)

    Returns:
        X_normalized: Normalized input
        mean: Feature means
        std: Feature standard deviations
    """
    if mean is None:
        mean = X.mean(axis=(0, 1))
    if std is None:
        std = X.std(axis=(0, 1))

    X_normalized = (X - mean) / (std + 1e-8)
    return X_normalized, mean, std


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple:
    """
    Split data into train, validation, and test sets.

    Uses sequential split to prevent data leakage in time series.

    Args:
        X: Input features
        y: Target values
        train_ratio: Proportion for training
        val_ratio: Proportion for validation

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Test data loading
    print("Testing Bybit data loader...")

    loader = BybitDataLoader()
    df = loader.fetch_klines("BTCUSDT", interval="60", limit=500)

    if not df.empty:
        print(f"Loaded {len(df)} candles for BTCUSDT")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Calculate features
        df = calculate_features(df)
        print(f"\nFeatures calculated. Columns: {list(df.columns)}")

        # Create sequences
        feature_cols = [
            "log_return", "volatility", "volume_ratio",
            "momentum_5", "momentum_10", "momentum_20",
            "rsi_normalized", "bb_position"
        ]

        df_clean = df.dropna()
        X, y = create_sequences(df_clean, feature_cols, "log_return", seq_len=168, horizon=24)
        print(f"\nSequences created: X={X.shape}, y={y.shape}")

        # Normalize
        X_norm, mean, std = normalize_features(X)
        print(f"Normalized: mean={mean.shape}, std={std.shape}")

        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X_norm, y)
        print(f"\nSplit sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    else:
        print("Failed to load data from Bybit API")

    print("\nAll data tests completed!")
