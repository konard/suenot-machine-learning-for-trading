"""
Data Loading and Preprocessing for Reformer

Provides:
- BybitDataLoader: Load historical data from Bybit API
- prepare_long_sequence_data: Prepare data for Reformer training
- Feature engineering utilities
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading"""
    symbols: List[str]
    interval: str = '1h'  # 1m, 5m, 15m, 30m, 1h, 4h, 1d
    lookback: int = 4096  # Long sequence for Reformer
    horizon: int = 24
    train_ratio: float = 0.8
    features: List[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = [
                'log_return', 'volatility', 'volume_zscore',
                'price_zscore', 'high_low_range', 'trend'
            ]


class BybitDataLoader:
    """
    Load historical kline data from Bybit API

    Bybit API documentation:
    https://bybit-exchange.github.io/docs/v5/market/kline
    """

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    INTERVAL_MAP = {
        '1m': '1',
        '3m': '3',
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '1h': '60',
        '2h': '120',
        '4h': '240',
        '6h': '360',
        '12h': '720',
        '1d': 'D',
        '1w': 'W',
        '1M': 'M'
    }

    def __init__(self, category: str = 'linear'):
        """
        Initialize Bybit data loader.

        Args:
            category: Product category ('linear', 'inverse', 'spot')
        """
        self.category = category
        self.session = requests.Session()

    def fetch_klines(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch kline data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Time interval
            limit: Number of klines to fetch (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        params = {
            'category': self.category,
            'symbol': symbol,
            'interval': self.INTERVAL_MAP.get(interval, interval),
            'limit': min(limit, 1000)
        }

        if start_time:
            params['start'] = start_time
        if end_time:
            params['end'] = end_time

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data['retCode'] != 0:
                logger.warning(f"API error: {data['retMsg']}")
                return pd.DataFrame()

            klines = data['result']['list']

            if not klines:
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = df[col].astype(float)

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_historical(
        self,
        symbol: str,
        interval: str = '1h',
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch extended historical data by paginating through API.

        Args:
            symbol: Trading pair
            interval: Time interval
            days: Number of days of history to fetch

        Returns:
            DataFrame with complete historical data
        """
        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        logger.info(f"Fetching {days} days of {interval} data for {symbol}")

        while end_time > start_time:
            df = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                end_time=end_time
            )

            if df.empty:
                break

            all_data.append(df)

            # Move end_time to before the earliest fetched timestamp
            end_time = int(df['timestamp'].min().timestamp() * 1000) - 1

            # Rate limiting
            time.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        # Combine and deduplicate
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset='timestamp')
        result = result.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Fetched {len(result)} rows for {symbol}")

        return result


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical features for model input.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added features
    """
    df = df.copy()

    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility (rolling standard deviation of returns)
    df['volatility'] = df['log_return'].rolling(20).std()

    # Volume z-score (normalized volume)
    vol_mean = df['volume'].rolling(100).mean()
    vol_std = df['volume'].rolling(100).std()
    df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)

    # Price z-score (deviation from moving average)
    price_mean = df['close'].rolling(100).mean()
    price_std = df['close'].rolling(100).std()
    df['price_zscore'] = (df['close'] - price_mean) / (price_std + 1e-8)

    # High-low range (normalized by close)
    df['high_low_range'] = (df['high'] - df['low']) / df['close']

    # Trend indicator (short MA / long MA - 1)
    df['trend'] = df['close'].rolling(50).mean() / df['close'].rolling(200).mean() - 1

    # RSI
    df['rsi'] = calculate_rsi(df['close'], 14)

    # MACD components
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = (ema_12 - ema_26) / df['close']

    # Bollinger Band position
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - bb_middle) / (2 * bb_std + 1e-8)

    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    # Normalize to [-1, 1]
    return (rsi - 50) / 50


def prepare_long_sequence_data(
    symbols: List[str],
    lookback: int = 4096,
    horizon: int = 24,
    interval: str = '1h',
    days: int = 365
) -> Dict:
    """
    Prepare long sequence data for Reformer training.

    Reformer excels at processing long sequences,
    so we can use 4096+ timesteps (weeks of hourly data).

    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        lookback: Historical timesteps (can be much longer than standard)
        horizon: Prediction horizon
        interval: Data interval
        days: Days of historical data to fetch

    Returns:
        Dictionary with X, y arrays for training
    """
    loader = BybitDataLoader()
    all_data = []

    for symbol in symbols:
        logger.info(f"Processing {symbol}")

        # Fetch historical data
        df = loader.fetch_historical(symbol, interval=interval, days=days)

        if df.empty:
            logger.warning(f"No data for {symbol}, skipping")
            continue

        # Calculate features
        df = calculate_features(df)

        # Drop rows with NaN (from rolling calculations)
        df = df.dropna()

        all_data.append(df)

    if not all_data:
        raise ValueError("No data loaded for any symbol")

    # Find common timestamps across all symbols
    common_timestamps = set(all_data[0]['timestamp'])
    for df in all_data[1:]:
        common_timestamps &= set(df['timestamp'])

    common_timestamps = sorted(common_timestamps)
    logger.info(f"Common timestamps: {len(common_timestamps)}")

    # Filter to common timestamps and stack
    feature_cols = [
        'log_return', 'volatility', 'volume_zscore',
        'price_zscore', 'high_low_range', 'trend'
    ]

    stacked = []
    for df in all_data:
        df_filtered = df[df['timestamp'].isin(common_timestamps)].sort_values('timestamp')
        stacked.append(df_filtered[feature_cols].values)

    # Shape: [time, n_symbols, features]
    stacked = np.stack(stacked, axis=1)

    logger.info(f"Stacked data shape: {stacked.shape}")

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(stacked) - horizon):
        X.append(stacked[i-lookback:i])  # [lookback, n_symbols, features]
        # Target: next period returns
        y.append(stacked[i+horizon-1, :, 0])  # log_return for each symbol

    X = np.array(X)  # [n_samples, lookback, n_symbols, features]
    y = np.array(y)  # [n_samples, n_symbols]

    # Transpose to [n_samples, n_symbols, lookback, features] for model
    X = X.transpose(0, 2, 1, 3)

    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    return {
        'X': X,
        'y': y,
        'symbols': symbols,
        'feature_names': feature_cols,
        'timestamps': common_timestamps[-len(y)-horizon:]
    }


def load_bybit_data(
    symbol: str,
    interval: str = '1h',
    days: int = 30
) -> pd.DataFrame:
    """
    Convenience function to load data for a single symbol.

    Args:
        symbol: Trading pair
        interval: Time interval
        days: Days of history

    Returns:
        DataFrame with OHLCV and calculated features
    """
    loader = BybitDataLoader()
    df = loader.fetch_historical(symbol, interval=interval, days=days)

    if df.empty:
        return df

    return calculate_features(df)


def create_train_val_split(
    data: Dict,
    train_ratio: float = 0.8
) -> Tuple[Dict, Dict]:
    """
    Split data into training and validation sets.

    Uses time-based split to prevent data leakage.

    Args:
        data: Dictionary with X, y arrays
        train_ratio: Fraction of data for training

    Returns:
        train_data, val_data dictionaries
    """
    n_samples = len(data['X'])
    split_idx = int(n_samples * train_ratio)

    train_data = {
        'X': data['X'][:split_idx],
        'y': data['y'][:split_idx],
        'symbols': data['symbols'],
        'feature_names': data['feature_names']
    }

    val_data = {
        'X': data['X'][split_idx:],
        'y': data['y'][split_idx:],
        'symbols': data['symbols'],
        'feature_names': data['feature_names']
    }

    return train_data, val_data


if __name__ == "__main__":
    # Test data loading
    print("Testing Bybit data loader...")

    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

    # Test single symbol
    df = load_bybit_data('BTCUSDT', interval='1h', days=7)
    print(f"Single symbol shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Test multi-symbol preparation
    print("\nPreparing multi-symbol data...")
    data = prepare_long_sequence_data(
        symbols=symbols,
        lookback=168,  # 1 week
        horizon=24,
        interval='1h',
        days=30
    )

    print(f"X shape: {data['X'].shape}")
    print(f"y shape: {data['y'].shape}")

    # Test train/val split
    train_data, val_data = create_train_val_split(data)
    print(f"Train X: {train_data['X'].shape}, Val X: {val_data['X'].shape}")

    print("\nAll tests passed!")
