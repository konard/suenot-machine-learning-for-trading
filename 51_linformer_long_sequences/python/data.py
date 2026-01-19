"""
Data loading and preprocessing utilities for Linformer.

This module provides functions to:
- Load historical data from Bybit exchange
- Calculate technical indicators and features
- Prepare data for Linformer training

Supports both stock market (via Yahoo Finance) and cryptocurrency
(via Bybit API) data sources.
"""

import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def load_bybit_data(
    symbol: str,
    interval: str = '1h',
    limit: int = 5000,
    category: str = 'linear'
) -> pd.DataFrame:
    """
    Load historical kline (candlestick) data from Bybit exchange.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval ('1', '5', '15', '30', '60', '240', 'D', 'W', 'M')
        limit: Number of klines to fetch (max 1000 per request)
        category: Market category ('linear', 'inverse', 'spot')

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, turnover

    Example:
        >>> df = load_bybit_data('BTCUSDT', interval='1h', limit=2000)
        >>> df.head()
                           timestamp     open     high      low    close     volume
        0 2024-01-01 00:00:00  42000.0  42100.0  41900.0  42050.0  1234.56
    """
    # Convert interval to Bybit format
    interval_map = {
        '1m': '1', '5m': '5', '15m': '15', '30m': '30',
        '1h': '60', '2h': '120', '4h': '240',
        '6h': '360', '12h': '720', '1d': 'D', '1w': 'W', '1M': 'M'
    }
    bybit_interval = interval_map.get(interval, interval)

    url = 'https://api.bybit.com/v5/market/kline'

    all_data = []
    end_time = None

    # Fetch data in batches (max 1000 per request)
    while len(all_data) < limit:
        params = {
            'category': category,
            'symbol': symbol,
            'interval': bybit_interval,
            'limit': min(1000, limit - len(all_data))
        }
        if end_time:
            params['end'] = end_time

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if data['retCode'] != 0:
                logger.error(f"Bybit API error: {data['retMsg']}")
                break

            result = data['result']['list']
            if not result:
                break

            all_data.extend(result)

            # Get end time for next batch (oldest timestamp in current batch)
            end_time = int(result[-1][0]) - 1

        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            break

    if not all_data:
        raise ValueError(f"No data received for {symbol}")

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    # Sort by timestamp (oldest first)
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def load_yahoo_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1h'
) -> pd.DataFrame:
    """
    Load historical data from Yahoo Finance.

    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'SPY')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1h', '15m', '5m', '1m')

    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    ticker = yf.Ticker(symbol)

    if start_date and end_date:
        df = ticker.history(start=start_date, end=end_date, interval=interval)
    else:
        # Default to last year
        df = ticker.history(period='1y', interval=interval)

    df = df.reset_index()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'splits']
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators and features for trading.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional feature columns

    Features computed:
        - log_return: Log returns
        - volatility: Rolling volatility (20 periods)
        - volume_change: Volume relative to moving average
        - rsi: Relative Strength Index (14 periods)
        - momentum: Price momentum (20 periods)
        - ma_ratio: Price relative to moving average
        - high_low_range: High-low range normalized
        - close_position: Close position within high-low range
    """
    df = df.copy()

    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Rolling volatility (20 periods)
    df['volatility'] = df['log_return'].rolling(window=20).std()

    # Volume change relative to 20-period MA
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_change'] = df['volume'] / df['volume_ma']

    # RSI (14 periods)
    df['rsi'] = calculate_rsi(df['close'], period=14)

    # Price momentum (20 periods)
    df['momentum'] = df['close'] / df['close'].shift(20) - 1

    # Price relative to 50-period MA
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['ma_ratio'] = df['close'] / df['ma_50']

    # High-low range normalized by close
    df['high_low_range'] = (df['high'] - df['low']) / df['close']

    # Close position within high-low range
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = (ema_12 - ema_26) / df['close']

    # Bollinger Bands position
    bb_ma = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_position'] = (df['close'] - bb_ma) / (2 * bb_std + 1e-8)

    # Drop intermediate columns
    df = df.drop(columns=['volume_ma', 'ma_50'], errors='ignore')

    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: Price series
        period: RSI period (default 14)

    Returns:
        RSI series (0-100)
    """
    delta = prices.diff()

    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def prepare_long_sequence_data(
    symbols: List[str],
    lookback: int = 2048,
    horizon: int = 24,
    data_source: str = 'bybit',
    interval: str = '1h',
    feature_columns: Optional[List[str]] = None
) -> Dict:
    """
    Prepare data for Linformer training with long sequences.

    This function:
    1. Loads data for multiple symbols
    2. Calculates features
    3. Aligns all symbols on timestamp
    4. Creates training sequences

    Args:
        symbols: List of trading symbols
        lookback: Number of historical timesteps (sequence length)
        horizon: Prediction horizon
        data_source: Data source ('bybit' or 'yahoo')
        interval: Data interval
        feature_columns: Optional list of feature columns to use

    Returns:
        Dictionary with:
            - X: Feature array [n_samples, lookback, n_features]
            - y: Target array [n_samples, n_outputs]
            - symbols: List of symbols
            - feature_names: List of feature names
            - timestamps: Array of timestamps

    Example:
        >>> data = prepare_long_sequence_data(
        ...     symbols=['BTCUSDT', 'ETHUSDT'],
        ...     lookback=2048,
        ...     horizon=24
        ... )
        >>> data['X'].shape
        (1000, 2048, 20)
    """
    all_data = []

    for symbol in symbols:
        logger.info(f"Loading data for {symbol}...")

        # Load data
        if data_source == 'bybit':
            df = load_bybit_data(
                symbol,
                interval=interval,
                limit=lookback + horizon + 1000
            )
        elif data_source == 'yahoo':
            df = load_yahoo_data(symbol, interval=interval)
        else:
            raise ValueError(f"Unknown data source: {data_source}")

        # Calculate features
        df = calculate_features(df)

        # Set timestamp as index
        df = df.set_index('timestamp')

        all_data.append(df)

    # Align all dataframes on timestamp
    logger.info("Aligning data on timestamps...")

    # Get common timestamps
    common_index = all_data[0].index
    for df in all_data[1:]:
        common_index = common_index.intersection(df.index)

    # Reindex all dataframes
    aligned_data = []
    for df in all_data:
        aligned_data.append(df.loc[common_index])

    # Combine features from all symbols
    combined = pd.concat(aligned_data, axis=1, keys=symbols)
    combined = combined.dropna()

    # Select feature columns
    if feature_columns is None:
        feature_columns = [
            'log_return', 'volatility', 'volume_change', 'rsi',
            'momentum', 'ma_ratio', 'high_low_range', 'close_position',
            'macd', 'bb_position'
        ]

    # Get feature indices
    feature_data = []
    for symbol in symbols:
        for col in feature_columns:
            if (symbol, col) in combined.columns:
                feature_data.append(combined[(symbol, col)])

    features = pd.concat(feature_data, axis=1)
    features = features.dropna()

    # Create sequences
    logger.info(f"Creating sequences with lookback={lookback}, horizon={horizon}...")

    X, y, timestamps = [], [], []
    for i in range(lookback, len(features) - horizon):
        X.append(features.iloc[i-lookback:i].values)
        # Target: log return of first symbol at horizon
        target_col = f"{symbols[0]}_log_return" if symbols else 'log_return'
        y.append(features.iloc[i+horizon].values[0])  # First column is usually log_return
        timestamps.append(features.index[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    timestamps = np.array(timestamps)

    # Replace any remaining NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

    logger.info(f"Created {len(X)} samples with shape {X.shape}")

    return {
        'X': X,
        'y': y,
        'symbols': symbols,
        'feature_names': list(features.columns),
        'timestamps': timestamps,
        'lookback': lookback,
        'horizon': horizon
    }


def train_val_test_split(
    data: Dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[Dict, Dict, Dict]:
    """
    Split data into train, validation, and test sets.

    Uses chronological split (no shuffling) to prevent data leakage.

    Args:
        data: Data dictionary from prepare_long_sequence_data
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing

    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n_samples = len(data['X'])
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_data = {
        'X': data['X'][:train_end],
        'y': data['y'][:train_end],
        'timestamps': data['timestamps'][:train_end]
    }

    val_data = {
        'X': data['X'][train_end:val_end],
        'y': data['y'][train_end:val_end],
        'timestamps': data['timestamps'][train_end:val_end]
    }

    test_data = {
        'X': data['X'][val_end:],
        'y': data['y'][val_end:],
        'timestamps': data['timestamps'][val_end:]
    }

    return train_data, val_data, test_data


def create_data_loaders(
    train_data: Dict,
    val_data: Dict,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple:
    """
    Create PyTorch DataLoaders for training.

    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    train_dataset = TensorDataset(
        torch.FloatTensor(train_data['X']),
        torch.FloatTensor(train_data['y'])
    )

    val_dataset = TensorDataset(
        torch.FloatTensor(val_data['X']),
        torch.FloatTensor(val_data['y'])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader
