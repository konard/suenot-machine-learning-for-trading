"""
Data Loading and Preprocessing for DCT Model

Provides:
- Stock data loading from Yahoo Finance
- Crypto data loading from Bybit API
- Feature engineering with technical indicators
- Dataset creation for training
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import time


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    lookback: int = 30  # Look-back window
    horizon: int = 1    # Prediction horizon
    movement_threshold: float = 0.005  # Movement threshold (0.5%)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class BybitClient:
    """
    Bybit API client for fetching crypto market data.

    Fetches OHLCV data from Bybit's public API.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str,
        interval: str = "D",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Candle interval ('1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Maximum number of candles (max 1000)

        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
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
                print(f"Bybit API error: {data.get('retMsg')}")
                return pd.DataFrame()

            klines = data.get("result", {}).get("list", [])
            if not klines:
                return pd.DataFrame()

            # Parse klines: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = df[col].astype(float)

            df = df.set_index('timestamp').sort_index()
            return df

        except Exception as e:
            print(f"Error fetching data from Bybit: {e}")
            return pd.DataFrame()

    def get_historical_data(
        self,
        symbol: str,
        interval: str = "D",
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical data with pagination.

        Args:
            symbol: Trading pair
            interval: Candle interval
            days: Number of days of history

        Returns:
            DataFrame with OHLCV data
        """
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        all_data = []
        current_end = end_time

        while current_end > start_time:
            df = self.get_klines(
                symbol=symbol,
                interval=interval,
                end_time=current_end,
                limit=1000
            )

            if df.empty:
                break

            all_data.append(df)
            current_end = int(df.index.min().timestamp() * 1000) - 1

            # Rate limiting
            time.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data).sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]

        return combined


def load_stock_data(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load stock data using yfinance.

    Args:
        symbol: Stock ticker (e.g., 'AAPL', 'MSFT')
        start_date: Start date string
        end_date: End date string (defaults to today)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df

    except ImportError:
        print("yfinance not installed. Install with: pip install yfinance")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading stock data: {e}")
        return pd.DataFrame()


def load_crypto_data(
    symbol: str,
    interval: str = "D",
    days: int = 365
) -> pd.DataFrame:
    """
    Load crypto data from Bybit.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval
        days: Number of days of history

    Returns:
        DataFrame with OHLCV data
    """
    client = BybitClient()
    return client.get_historical_data(symbol, interval, days)


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for DCT model.

    Features:
    - Price ratios (HL, OC)
    - Moving averages and ratios
    - Volatility
    - RSI
    - MACD
    - Volume features

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added technical indicators
    """
    result = df.copy()

    # Price ratios
    result['hl_ratio'] = (df['high'] - df['low']) / df['close']
    result['oc_ratio'] = (df['close'] - df['open']) / df['open']

    # Log returns
    result['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Moving averages
    for window in [5, 10, 20]:
        result[f'ma_{window}'] = df['close'].rolling(window).mean()
        result[f'ma_ratio_{window}'] = df['close'] / result[f'ma_{window}']

    # Exponential moving averages
    for span in [12, 26]:
        result[f'ema_{span}'] = df['close'].ewm(span=span).mean()

    # Volatility
    result['volatility_5'] = df['close'].pct_change().rolling(5).std()
    result['volatility_20'] = df['close'].pct_change().rolling(20).std()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    result['rsi'] = 100 - (100 / (1 + rs))
    result['rsi_normalized'] = result['rsi'] / 100  # Normalize to [0, 1]

    # MACD
    result['macd'] = result['ema_12'] - result['ema_26']
    result['macd_signal'] = result['macd'].ewm(span=9).mean()
    result['macd_histogram'] = result['macd'] - result['macd_signal']

    # Volume features
    result['volume_ma'] = df['volume'].rolling(20).mean()
    result['volume_ratio'] = df['volume'] / (result['volume_ma'] + 1e-10)
    result['volume_log'] = np.log1p(df['volume'])

    # Bollinger Bands
    result['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    result['bb_upper'] = result['bb_middle'] + 2 * bb_std
    result['bb_lower'] = result['bb_middle'] - 2 * bb_std
    result['bb_position'] = (df['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-10)

    return result


def create_movement_labels(
    prices: pd.Series,
    threshold: float = 0.005,
    horizon: int = 1
) -> pd.Series:
    """
    Create movement labels based on future returns.

    Args:
        prices: Close prices
        threshold: Movement threshold (0.5% default)
        horizon: Prediction horizon in periods

    Returns:
        Series with labels: 0=Up, 1=Down, 2=Stable
    """
    future_return = prices.pct_change(horizon).shift(-horizon)

    labels = pd.Series(index=prices.index, dtype=int)
    labels[future_return > threshold] = 0   # Up
    labels[future_return < -threshold] = 1  # Down
    labels[(future_return >= -threshold) & (future_return <= threshold)] = 2  # Stable

    return labels


def prepare_dataset(
    df: pd.DataFrame,
    config: DataConfig,
    feature_columns: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Prepare dataset for training DCT model.

    Args:
        df: DataFrame with OHLCV and technical indicators
        config: Data configuration
        feature_columns: List of feature columns to use

    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Default feature columns
    if feature_columns is None:
        feature_columns = [
            'log_return', 'hl_ratio', 'oc_ratio',
            'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20',
            'volatility_5', 'volatility_20',
            'rsi_normalized',
            'macd', 'macd_signal',
            'volume_ratio',
            'bb_position'
        ]

    # Filter available columns
    available_columns = [c for c in feature_columns if c in df.columns]
    if len(available_columns) < len(feature_columns):
        missing = set(feature_columns) - set(available_columns)
        print(f"Warning: Missing columns: {missing}")

    # Create labels
    df = df.copy()
    df['label'] = create_movement_labels(
        df['close'],
        threshold=config.movement_threshold,
        horizon=config.horizon
    )

    # Drop NaN rows
    df = df.dropna()

    # Extract features
    features = df[available_columns].values
    labels = df['label'].values

    # Normalize features
    mean = np.nanmean(features, axis=0, keepdims=True)
    std = np.nanstd(features, axis=0, keepdims=True) + 1e-10
    features = (features - mean) / std

    # Create sequences
    X, y = [], []
    for i in range(config.lookback, len(features) - config.horizon):
        X.append(features[i - config.lookback:i])
        y.append(labels[i])

    X = np.array(X)
    y = np.array(y)

    # Split data
    n_samples = len(X)
    train_end = int(n_samples * config.train_ratio)
    val_end = int(n_samples * (config.train_ratio + config.val_ratio))

    return {
        'X_train': X[:train_end],
        'y_train': y[:train_end],
        'X_val': X[train_end:val_end],
        'y_val': y[train_end:val_end],
        'X_test': X[val_end:],
        'y_test': y[val_end:],
        'feature_columns': available_columns,
        'normalization': {'mean': mean, 'std': std}
    }


class DCTDataset:
    """Dataset class for DCT model training."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device: str = 'cpu'
    ):
        """
        Initialize dataset.

        Args:
            X: Feature array [n_samples, seq_len, n_features]
            y: Label array [n_samples]
            device: Device to store tensors
        """
        import torch

        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.LongTensor(y).to(device)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple:
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")

    # Test Bybit client
    print("\n1. Testing Bybit API...")
    client = BybitClient()
    btc_data = client.get_klines("BTCUSDT", interval="D", limit=100)
    if not btc_data.empty:
        print(f"Loaded {len(btc_data)} BTC candles")
        print(f"Date range: {btc_data.index.min()} to {btc_data.index.max()}")
        print(f"Columns: {btc_data.columns.tolist()}")
    else:
        print("Failed to load Bybit data")

    # Test technical indicators
    print("\n2. Testing technical indicators...")
    if not btc_data.empty:
        btc_features = compute_technical_indicators(btc_data)
        print(f"Feature columns: {btc_features.columns.tolist()}")
        print(f"Shape: {btc_features.shape}")

    # Test dataset preparation
    print("\n3. Testing dataset preparation...")
    if not btc_data.empty:
        btc_features = compute_technical_indicators(btc_data)
        config = DataConfig(lookback=30, horizon=1)
        dataset = prepare_dataset(btc_features, config)
        print(f"Train shapes: X={dataset['X_train'].shape}, y={dataset['y_train'].shape}")
        print(f"Val shapes: X={dataset['X_val'].shape}, y={dataset['y_val'].shape}")
        print(f"Test shapes: X={dataset['X_test'].shape}, y={dataset['y_test'].shape}")
        print(f"Label distribution (train): {np.bincount(dataset['y_train'])}")
