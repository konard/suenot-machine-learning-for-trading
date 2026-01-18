"""
Data Loading and Feature Engineering for Cross-Attention Multi-Asset Model

Provides:
- BybitDataLoader: Load cryptocurrency data from Bybit exchange
- StockDataLoader: Load stock data from Yahoo Finance
- Feature engineering utilities
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import requests
from datetime import datetime, timedelta
import time


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    lookback: int = 168  # 7 days of hourly data
    horizon: int = 24    # Predict 24 hours ahead
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    # test_ratio is 1 - train_ratio - val_ratio


class BybitDataLoader:
    """
    Load cryptocurrency data from Bybit exchange.

    Example:
        loader = BybitDataLoader()
        data = loader.fetch_multi_asset(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    """

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CrossAttention-MultiAsset/1.0'
        })

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 1000,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            limit: Number of candles (max 1000)
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if end_time is not None:
            params['end'] = end_time

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data['retCode'] != 0:
                raise Exception(f"API Error: {data['retMsg']}")

            klines = data['result']['list']

            if not klines:
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = df[col].astype(float)

            return df.sort_values('timestamp').reset_index(drop=True)

        except requests.exceptions.RequestException as e:
            print(f"Request error for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_historical(
        self,
        symbol: str,
        days: int = 30,
        interval: str = "60"
    ) -> pd.DataFrame:
        """
        Fetch historical data by paginating through multiple requests.

        Args:
            symbol: Trading pair
            days: Number of days of history
            interval: Kline interval

        Returns:
            DataFrame with historical OHLCV data
        """
        all_data = []
        end_time = None

        # Calculate required number of candles
        interval_minutes = self._parse_interval(interval)
        total_candles = (days * 24 * 60) // interval_minutes

        while len(all_data) < total_candles:
            df = self.fetch_klines(symbol, interval, limit=1000, end_time=end_time)

            if df.empty:
                break

            all_data.append(df)

            # Update end_time for next request
            end_time = int(df['timestamp'].min().timestamp() * 1000) - 1

            # Rate limiting
            time.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset='timestamp').sort_values('timestamp')
        return result.reset_index(drop=True)

    def fetch_multi_asset(
        self,
        symbols: List[str],
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple assets.

        Args:
            symbols: List of trading pairs
            **kwargs: Arguments passed to fetch_klines or fetch_historical

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            if 'days' in kwargs:
                df = self.fetch_historical(symbol, **kwargs)
            else:
                df = self.fetch_klines(symbol, **kwargs)
            data[symbol] = df
            time.sleep(0.2)  # Rate limiting
        return data

    @staticmethod
    def _parse_interval(interval: str) -> int:
        """Parse interval string to minutes."""
        if interval == 'D':
            return 24 * 60
        elif interval == 'W':
            return 7 * 24 * 60
        elif interval == 'M':
            return 30 * 24 * 60
        else:
            return int(interval)


class StockDataLoader:
    """
    Load stock data from Yahoo Finance.

    Example:
        loader = StockDataLoader()
        data = loader.fetch_multi_asset(['AAPL', 'GOOGL', 'MSFT'])
    """

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("Please install yfinance: pip install yfinance")

    def fetch_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch stock data for a single symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval

        Returns:
            DataFrame with OHLCV data
        """
        ticker = self.yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        df.columns = df.columns.str.lower()
        df = df.reset_index()
        df.columns = ['timestamp'] + list(df.columns[1:])
        return df

    def fetch_multi_asset(
        self,
        symbols: List[str],
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks."""
        data = {}
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            data[symbol] = self.fetch_data(symbol, **kwargs)
        return data


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features from OHLCV data.

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume

    Returns:
        DataFrame with computed features
    """
    features = pd.DataFrame(index=df.index)

    # Price-based features
    features['log_return'] = np.log(df['close'] / df['close'].shift(1))
    features['intraday_return'] = (df['close'] - df['open']) / df['open']
    features['high_low_range'] = (df['high'] - df['low']) / df['close']
    features['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)

    # Volume features
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['volume_change'] = df['volume'].pct_change()

    # Volatility features
    features['volatility_20'] = features['log_return'].rolling(20).std()
    features['volatility_5'] = features['log_return'].rolling(5).std()

    # Momentum features
    features['momentum_5'] = df['close'].pct_change(5)
    features['momentum_20'] = df['close'].pct_change(20)

    # RSI
    features['rsi_14'] = compute_rsi(df['close'], 14)

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    features['macd'] = (ema12 - ema26) / df['close']

    # Moving average distances
    features['ma_5_dist'] = (df['close'] - df['close'].rolling(5).mean()) / df['close']
    features['ma_20_dist'] = (df['close'] - df['close'].rolling(20).mean()) / df['close']

    return features


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def prepare_cross_attention_data(
    asset_data: Dict[str, pd.DataFrame],
    config: DataConfig = None,
    feature_cols: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
    """
    Prepare data for cross-attention multi-asset model.

    Args:
        asset_data: Dictionary mapping symbol to DataFrame with OHLCV
        config: DataConfig with lookback, horizon settings
        feature_cols: List of feature columns to use

    Returns:
        X: [n_samples, n_assets, lookback, n_features]
        y: [n_samples, n_assets] - Future returns
        symbols: List of asset symbols
        timestamps: DatetimeIndex for each sample
    """
    if config is None:
        config = DataConfig()

    if feature_cols is None:
        feature_cols = [
            'log_return', 'volume_ratio', 'volatility_20',
            'rsi_14', 'macd', 'momentum_5'
        ]

    symbols = list(asset_data.keys())
    n_assets = len(symbols)

    # Compute features for each asset
    processed = {}
    for symbol, df in asset_data.items():
        features = compute_features(df)
        features['timestamp'] = df['timestamp']
        processed[symbol] = features

    # Find common timestamps
    common_idx = processed[symbols[0]]['timestamp'].values
    for symbol in symbols[1:]:
        common_idx = np.intersect1d(
            common_idx,
            processed[symbol]['timestamp'].values
        )

    # Filter and align data
    aligned = {}
    for symbol in symbols:
        df = processed[symbol]
        mask = df['timestamp'].isin(common_idx)
        aligned[symbol] = df[mask].sort_values('timestamp').reset_index(drop=True)

    # Create sequences
    X, y = [], []
    timestamps = []

    n_samples = len(common_idx) - config.lookback - config.horizon
    for i in range(config.lookback, len(common_idx) - config.horizon):
        x_sample = []
        y_sample = []

        for symbol in symbols:
            df = aligned[symbol]

            # Features
            feat = df.iloc[i - config.lookback:i][feature_cols].values
            x_sample.append(feat)

            # Target: future return
            future_return = df.iloc[i + config.horizon]['log_return']
            y_sample.append(future_return)

        X.append(np.stack(x_sample, axis=0))
        y.append(np.array(y_sample))
        timestamps.append(pd.Timestamp(common_idx[i]))

    X = np.array(X)  # [n_samples, n_assets, lookback, n_features]
    y = np.array(y)  # [n_samples, n_assets]

    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    return X, y, symbols, pd.DatetimeIndex(timestamps)


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    config: DataConfig = None
) -> Dict:
    """
    Split data into train, validation, and test sets.

    Uses temporal split to avoid look-ahead bias.

    Returns:
        Dictionary with train, val, test data
    """
    if config is None:
        config = DataConfig()

    n_samples = len(X)
    train_end = int(n_samples * config.train_ratio)
    val_end = int(n_samples * (config.train_ratio + config.val_ratio))

    return {
        'train': {
            'X': X[:train_end],
            'y': y[:train_end],
            'timestamps': timestamps[:train_end]
        },
        'val': {
            'X': X[train_end:val_end],
            'y': y[train_end:val_end],
            'timestamps': timestamps[train_end:val_end]
        },
        'test': {
            'X': X[val_end:],
            'y': y[val_end:],
            'timestamps': timestamps[val_end:]
        }
    }


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
):
    """
    Create PyTorch DataLoader from numpy arrays.

    Returns:
        DataLoader for training/evaluation
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")

    # Test Bybit loader
    loader = BybitDataLoader()

    print("\nFetching sample crypto data from Bybit...")
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    data = loader.fetch_multi_asset(symbols, interval='60', limit=500)

    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} candles")
        if not df.empty:
            print(f"  Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Test feature computation
    print("\nComputing features...")
    if data and not data['BTCUSDT'].empty:
        features = compute_features(data['BTCUSDT'])
        print(f"Features computed: {list(features.columns)}")

    # Test data preparation
    print("\nPreparing cross-attention data...")
    X, y, symbols, timestamps = prepare_cross_attention_data(data)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Symbols: {symbols}")

    # Test splitting
    splits = train_val_test_split(X, y, timestamps)
    print(f"\nTrain size: {len(splits['train']['X'])}")
    print(f"Val size: {len(splits['val']['X'])}")
    print(f"Test size: {len(splits['test']['X'])}")

    print("\nAll data tests passed!")
