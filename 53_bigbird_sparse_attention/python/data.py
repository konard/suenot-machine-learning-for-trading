"""
Data Loading and Preprocessing for BigBird Trading Model

Provides:
- fetch_bybit_data: Fetch cryptocurrency data from Bybit
- fetch_stock_data: Fetch stock data from Yahoo Finance
- prepare_features: Feature engineering for trading
- create_sequences: Create input sequences for model
- TradingDataset: PyTorch dataset for training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

# Optional imports with fallbacks
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    warnings.warn("ccxt not installed. Bybit data fetching will not work.")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    warnings.warn("yfinance not installed. Stock data fetching will not work.")


def fetch_bybit_data(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h',
    limit: int = 1000,
    since: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Bybit exchange.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
        timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
        limit: Number of candles to fetch (max ~1000 per request)
        since: Start datetime (optional)

    Returns:
        DataFrame with columns: open, high, low, close, volume

    Example:
        df = fetch_bybit_data('BTCUSDT', '1h', limit=500)
        print(df.head())
    """
    if not HAS_CCXT:
        raise ImportError("ccxt is required for Bybit data. Install with: pip install ccxt")

    exchange = ccxt.bybit({
        'enableRateLimit': True,
    })

    # Convert since to timestamp if provided
    since_ms = None
    if since:
        since_ms = int(since.timestamp() * 1000)

    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(
        symbol,
        timeframe,
        since=since_ms,
        limit=limit
    )

    # Convert to DataFrame
    df = pd.DataFrame(
        ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df


def fetch_stock_data(
    symbol: str = 'AAPL',
    period: str = '1y',
    interval: str = '1h',
    start: Optional[str] = None,
    end: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.

    Args:
        symbol: Stock ticker (e.g., 'AAPL', 'GOOGL', 'SPY')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume

    Example:
        df = fetch_stock_data('AAPL', period='6mo', interval='1h')
        print(df.head())
    """
    if not HAS_YFINANCE:
        raise ImportError("yfinance is required for stock data. Install with: pip install yfinance")

    ticker = yf.Ticker(symbol)

    if start and end:
        df = ticker.history(start=start, end=end, interval=interval)
    else:
        df = ticker.history(period=period, interval=interval)

    # Rename columns to lowercase for consistency
    df.columns = [c.lower() for c in df.columns]

    # Keep only OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df


def prepare_features(
    df: pd.DataFrame,
    include_technicals: bool = True,
    include_returns: bool = True,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Prepare features for BigBird model.

    Args:
        df: DataFrame with OHLCV data
        include_technicals: Include technical indicators
        include_returns: Include return-based features
        normalize: Normalize features

    Returns:
        DataFrame with additional feature columns

    Features included:
        - log_return: Log returns
        - volatility_20, volatility_50: Rolling volatility
        - rsi: Relative Strength Index (14-period)
        - volume_ratio: Volume relative to 20-period MA
        - range: High-Low range relative to close
        - ma_ratio_20, ma_ratio_50: Price relative to moving averages
        - bb_position: Bollinger Band position
    """
    df = df.copy()

    # Log returns
    if include_returns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Volatility (rolling std of returns)
        df['volatility_20'] = df['log_return'].rolling(20).std()
        df['volatility_50'] = df['log_return'].rolling(50).std()

    if include_technicals:
        # RSI (14-period)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Volume features
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)

        # Price range
        df['range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)

        # Moving average ratios
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        df['ma_ratio_20'] = df['close'] / (df['ma_20'] + 1e-10) - 1
        df['ma_ratio_50'] = df['close'] / (df['ma_50'] + 1e-10) - 1

        # Bollinger Bands position
        bb_std = df['close'].rolling(20).std()
        bb_upper = df['ma_20'] + 2 * bb_std
        bb_lower = df['ma_20'] - 2 * bb_std
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # Drop intermediate columns
    drop_cols = ['volume_ma_20', 'ma_20', 'ma_50']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop NaN rows
    df = df.dropna()

    # Normalize features
    if normalize:
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        for col in feature_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std

    return df


def create_sequences(
    df: pd.DataFrame,
    seq_len: int = 256,
    pred_len: int = 1,
    features: Optional[List[str]] = None,
    target_col: str = 'log_return',
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences for model training.

    Args:
        df: DataFrame with features
        seq_len: Input sequence length
        pred_len: Prediction horizon
        features: List of feature columns (None = use all except OHLCV)
        target_col: Target column name
        step: Step size between sequences

    Returns:
        X: Input sequences [n_samples, seq_len, n_features]
        y: Target values [n_samples, pred_len] or [n_samples] if pred_len=1

    Example:
        X, y = create_sequences(df, seq_len=256, pred_len=1)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
    """
    if features is None:
        # Use all columns except raw OHLCV
        features = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    data = df[features].values
    target = df[target_col].values

    X, y = [], []
    for i in range(seq_len, len(data) - pred_len + 1, step):
        X.append(data[i-seq_len:i])
        if pred_len == 1:
            y.append(target[i])
        else:
            y.append(target[i:i+pred_len])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading data.

    Args:
        X: Input sequences [n_samples, seq_len, n_features]
        y: Target values [n_samples, ...]
        transform: Optional transform to apply to inputs

    Example:
        dataset = TradingDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        transform=None
    ):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def prepare_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare train, validation, and test data loaders.

    Args:
        X: Input sequences
        y: Target values
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        batch_size: Batch size
        shuffle_train: Whether to shuffle training data

    Returns:
        train_loader, val_loader, test_loader
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split data (temporal split, no shuffling to avoid lookahead)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Create datasets
    train_dataset = TradingDataset(X_train, y_train)
    val_dataset = TradingDataset(X_val, y_val)
    test_dataset = TradingDataset(X_test, y_test)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def generate_synthetic_data(
    n_samples: int = 10000,
    seq_len: int = 256,
    n_features: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for testing.

    Creates data with simple autoregressive patterns that BigBird should learn.

    Args:
        n_samples: Number of samples
        seq_len: Sequence length
        n_features: Number of features

    Returns:
        X: Input sequences
        y: Target values
    """
    np.random.seed(42)

    # Generate base series with trend and seasonality
    t = np.arange(n_samples + seq_len)
    base = (
        0.001 * t +                           # Trend
        0.1 * np.sin(2 * np.pi * t / 24) +   # Daily seasonality
        0.05 * np.sin(2 * np.pi * t / 168) + # Weekly seasonality
        0.1 * np.random.randn(len(t))        # Noise
    )

    # Create features
    features = np.zeros((len(t), n_features))
    features[:, 0] = base  # Main feature
    features[:, 1] = np.roll(base, 1)  # Lag 1
    features[:, 2] = np.roll(base, 2)  # Lag 2
    features[:, 3] = pd.Series(base).rolling(5).std().fillna(0).values  # Volatility
    features[:, 4] = np.random.randn(len(t)) * 0.1  # Random feature
    features[:, 5] = np.sin(2 * np.pi * t / 24)  # Time feature

    # Create sequences
    X, y = [], []
    for i in range(seq_len, len(t)):
        X.append(features[i-seq_len:i])
        y.append(base[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if __name__ == "__main__":
    # Test data utilities
    print("Testing data utilities...")

    # Test synthetic data
    print("\n1. Testing synthetic data generation...")
    X, y = generate_synthetic_data(n_samples=1000, seq_len=64, n_features=6)
    print(f"   X shape: {X.shape}, y shape: {y.shape}")

    # Test data loaders
    print("\n2. Testing data loaders...")
    train_loader, val_loader, test_loader = prepare_data_loaders(
        X, y, batch_size=32
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Test a batch
    batch_x, batch_y = next(iter(train_loader))
    print(f"   Batch X shape: {batch_x.shape}")
    print(f"   Batch y shape: {batch_y.shape}")

    # Test Bybit data if available
    if HAS_CCXT:
        print("\n3. Testing Bybit data fetching...")
        try:
            df = fetch_bybit_data('BTCUSDT', '1h', limit=100)
            print(f"   Fetched {len(df)} rows")
            print(f"   Columns: {list(df.columns)}")

            # Test feature preparation
            df_features = prepare_features(df)
            print(f"   After feature prep: {len(df_features)} rows, {len(df_features.columns)} columns")

            # Test sequence creation
            X, y = create_sequences(df_features, seq_len=48, pred_len=1)
            print(f"   Sequences: X shape {X.shape}, y shape {y.shape}")
        except Exception as e:
            print(f"   Bybit test failed: {e}")

    # Test stock data if available
    if HAS_YFINANCE:
        print("\n4. Testing stock data fetching...")
        try:
            df = fetch_stock_data('AAPL', period='3mo', interval='1h')
            print(f"   Fetched {len(df)} rows")
            print(f"   Columns: {list(df.columns)}")
        except Exception as e:
            print(f"   Stock test failed: {e}")

    print("\nAll tests completed!")
