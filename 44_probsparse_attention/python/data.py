"""
Data Loading and Preprocessing for Informer Model

Provides:
- BybitDataLoader: Load cryptocurrency data from Bybit API
- StockDataLoader: Load stock data from yfinance
- TimeSeriesDataset: PyTorch Dataset for time series
- Feature engineering functions
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime, timedelta
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KlineData:
    """Container for OHLCV data"""
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return pd.DataFrame({
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        })


class BybitDataLoader:
    """
    Load cryptocurrency data from Bybit API

    Supports:
    - Historical kline (candlestick) data
    - Multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)
    - Rate limiting and error handling

    Example:
        loader = BybitDataLoader()
        df = loader.get_klines("BTCUSDT", "1h", limit=1000)
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self, testnet: bool = False):
        """
        Args:
            testnet: Use testnet API if True
        """
        if testnet:
            self.BASE_URL = "https://api-testnet.bybit.com"

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical kline data from Bybit

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            limit: Number of candles (max 1000 per request)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        # Map interval strings to Bybit format
        interval_map = {
            "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
            "1d": "D", "1w": "W", "1M": "M"
        }
        bybit_interval = interval_map.get(interval, interval)

        endpoint = f"{self.BASE_URL}/v5/market/kline"

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": min(limit, 1000)
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        all_data = []
        remaining = limit

        while remaining > 0:
            params["limit"] = min(remaining, 1000)

            try:
                response = requests.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data.get("retCode") != 0:
                    logger.error(f"Bybit API error: {data.get('retMsg')}")
                    break

                klines = data.get("result", {}).get("list", [])
                if not klines:
                    break

                all_data.extend(klines)
                remaining -= len(klines)

                # Update end_time for pagination (Bybit returns newest first)
                oldest_time = int(klines[-1][0])
                params["end"] = oldest_time - 1

                # Rate limiting
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        # Parse kline data: [timestamp, open, high, low, close, volume, turnover]
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Sort by time ascending
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def get_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "1h",
        limit: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols

        Args:
            symbols: List of trading pairs
            interval: Timeframe
            limit: Number of candles per symbol

        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        data = {}
        for symbol in symbols:
            logger.info(f"Fetching {symbol}...")
            df = self.get_klines(symbol, interval, limit)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.2)  # Rate limiting between symbols

        return data


class StockDataLoader:
    """
    Load stock data using yfinance

    Example:
        loader = StockDataLoader()
        df = loader.get_ohlcv("AAPL", "1h", period="1y")
    """

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            self.yf = None

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        period: str = "1y",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            interval: Timeframe (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        if self.yf is None:
            logger.error("yfinance not available")
            return pd.DataFrame()

        try:
            ticker = self.yf.Ticker(symbol)

            if start and end:
                df = ticker.history(start=start, end=end, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return df

            # Rename columns to lowercase
            df.columns = df.columns.str.lower()
            df = df.reset_index()
            df = df.rename(columns={'date': 'timestamp', 'datetime': 'timestamp'})

            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return pd.DataFrame()


def compute_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute technical features for time series

    Args:
        df: DataFrame with OHLCV data
        lookback: Window size for rolling calculations

    Returns:
        DataFrame with added features
    """
    df = df.copy()

    # Returns
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return'] = df['returns']

    # Volatility
    df['volatility'] = df['returns'].rolling(lookback).std()

    # Volume features
    df['volume_ma'] = df['volume'].rolling(lookback).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Price features
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_open_range'] = (df['close'] - df['open']) / df['open']

    # Moving averages
    df['ma_fast'] = df['close'].rolling(lookback // 2).mean()
    df['ma_slow'] = df['close'].rolling(lookback).mean()
    df['ma_ratio'] = df['ma_fast'] / df['ma_slow']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(lookback).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(lookback).mean()
    rs = gain / (loss + 1e-10)  # Add epsilon to prevent division by zero
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands position
    bb_mid = df['close'].rolling(lookback).mean()
    bb_std = df['close'].rolling(lookback).std()
    df['bb_position'] = (df['close'] - bb_mid) / (2 * bb_std)

    return df


def normalize_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    method: str = 'zscore',
    window: int = 100
) -> pd.DataFrame:
    """
    Normalize features using rolling statistics

    Args:
        df: DataFrame with features
        feature_cols: Columns to normalize
        method: 'zscore' or 'minmax'
        window: Rolling window size

    Returns:
        DataFrame with normalized features
    """
    df = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            continue

        if method == 'zscore':
            mean = df[col].rolling(window, min_periods=1).mean()
            std = df[col].rolling(window, min_periods=1).std()
            df[f'{col}_norm'] = (df[col] - mean) / (std + 1e-8)

        elif method == 'minmax':
            min_val = df[col].rolling(window, min_periods=1).min()
            max_val = df[col].rolling(window, min_periods=1).max()
            df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val + 1e-8)

    return df


def prepare_informer_data(
    df: pd.DataFrame,
    seq_len: int = 96,
    pred_len: int = 24,
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'returns'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for Informer model training

    Args:
        df: DataFrame with OHLCV data
        seq_len: Input sequence length
        pred_len: Prediction horizon
        feature_cols: Feature columns to use (default: OHLCV normalized)
        target_col: Target column for prediction

    Returns:
        X: Input sequences [n_samples, seq_len, n_features]
        y: Target sequences [n_samples, pred_len]
    """
    # Compute features
    df = compute_features(df)

    # Default feature columns
    if feature_cols is None:
        feature_cols = ['close', 'volume', 'high', 'low', 'open', 'volatility']

    # Normalize features
    df = normalize_features(df, feature_cols)

    # Use normalized columns
    norm_cols = [f'{col}_norm' for col in feature_cols if f'{col}_norm' in df.columns]

    # Drop NaN values
    df = df.dropna(subset=norm_cols + [target_col])

    if len(df) < seq_len + pred_len:
        raise ValueError(f"Not enough data: {len(df)} < {seq_len + pred_len}")

    data = df[norm_cols].values
    targets = df[target_col].values

    # Create sequences
    X, y = [], []
    for i in range(seq_len, len(data) - pred_len + 1):
        X.append(data[i-seq_len:i])
        y.append(targets[i:i+pred_len])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data

    Args:
        X: Input sequences [n_samples, seq_len, n_features]
        y: Target sequences [n_samples, pred_len]
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders

    Args:
        X: Input sequences
        y: Target sequences
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        batch_size: Batch size
        shuffle_train: Whether to shuffle training data

    Returns:
        train_loader, val_loader, test_loader
    """
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    # Time series split (no shuffling for val/test)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")

    # Test Bybit loader
    print("\n1. Testing Bybit loader:")
    bybit = BybitDataLoader()
    df = bybit.get_klines("BTCUSDT", "1h", limit=100)
    if not df.empty:
        print(f"   Loaded {len(df)} candles for BTCUSDT")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Columns: {list(df.columns)}")
    else:
        print("   Failed to load data (API may be unavailable)")

    # Test feature computation
    print("\n2. Testing feature computation:")
    if not df.empty:
        df_features = compute_features(df)
        print(f"   Original columns: {len(df.columns)}")
        print(f"   After features: {len(df_features.columns)}")
        print(f"   New features: {[c for c in df_features.columns if c not in df.columns]}")

    # Test data preparation
    print("\n3. Testing data preparation:")
    if not df.empty and len(df) >= 50:
        try:
            X, y = prepare_informer_data(df, seq_len=24, pred_len=6)
            print(f"   X shape: {X.shape}")
            print(f"   y shape: {y.shape}")
        except ValueError as e:
            print(f"   Not enough data: {e}")

    # Test DataLoader creation
    print("\n4. Testing DataLoader creation:")
    if not df.empty and len(df) >= 50:
        try:
            X, y = prepare_informer_data(df, seq_len=24, pred_len=6)
            train_loader, val_loader, test_loader = create_dataloaders(
                X, y, batch_size=8
            )
            print(f"   Train batches: {len(train_loader)}")
            print(f"   Val batches: {len(val_loader)}")
            print(f"   Test batches: {len(test_loader)}")

            # Test batch
            batch_x, batch_y = next(iter(train_loader))
            print(f"   Batch X shape: {batch_x.shape}")
            print(f"   Batch y shape: {batch_y.shape}")
        except Exception as e:
            print(f"   Error: {e}")

    print("\nAll tests completed!")
