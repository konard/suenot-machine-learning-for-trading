"""
Data loading and feature engineering for meta-gradient optimization.

Supports both stock market data (via yfinance) and cryptocurrency
data from the Bybit exchange.
"""

import numpy as np
import pandas as pd
import requests
import torch
from typing import Tuple, Generator, Dict


def create_trading_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Create technical features for trading.

    Args:
        prices: Price series
        window: Lookback window for indicators

    Returns:
        DataFrame with 11 technical features
    """
    features = pd.DataFrame(index=prices.index)

    # Returns at different horizons
    features['return_1d'] = prices.pct_change(1)
    features['return_5d'] = prices.pct_change(5)
    features['return_10d'] = prices.pct_change(10)

    # Moving average ratios
    features['sma_ratio'] = prices / prices.rolling(window).mean()
    features['ema_ratio'] = prices / prices.ewm(span=window).mean()

    # Volatility
    features['volatility'] = prices.pct_change().rolling(window).std()

    # Momentum
    features['momentum'] = prices / prices.shift(window) - 1

    # RSI
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    features['macd'] = (ema12 - ema26) / prices

    # Bollinger Band position
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    features['bb_position'] = (prices - sma) / (2 * std + 1e-10)

    return features.dropna()


def create_meta_gradient_tasks(
    prices: pd.Series,
    features: pd.DataFrame,
    train_size: int = 30,
    val_size: int = 10,
    target_horizon: int = 5,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create train/validation split for meta-gradient optimization.

    Args:
        prices: Price series
        features: Feature DataFrame
        train_size: Number of training samples
        val_size: Number of validation samples
        target_horizon: Prediction horizon in periods

    Returns:
        ((train_features, train_targets), (val_features, val_targets))
    """
    target = prices.pct_change(target_horizon).shift(-target_horizon)
    aligned = features.join(target.rename('target')).dropna()

    total_needed = train_size + val_size
    if len(aligned) < total_needed:
        raise ValueError(f"Not enough data: {len(aligned)} < {total_needed}")

    start_idx = np.random.randint(0, len(aligned) - total_needed)
    feature_cols = [c for c in aligned.columns if c != 'target']

    train_df = aligned.iloc[start_idx:start_idx + train_size]
    val_df = aligned.iloc[start_idx + train_size:start_idx + total_needed]

    train_features = torch.FloatTensor(train_df[feature_cols].values)
    train_targets = torch.FloatTensor(train_df['target'].values).unsqueeze(1)
    val_features = torch.FloatTensor(val_df[feature_cols].values)
    val_targets = torch.FloatTensor(val_df['target'].values).unsqueeze(1)

    return (train_features, train_targets), (val_features, val_targets)


def task_generator(
    asset_data: Dict[str, Tuple[pd.Series, pd.DataFrame]],
    batch_size: int = 4,
) -> Generator:
    """
    Generate batches of tasks from multiple assets.

    Args:
        asset_data: Dict of {asset_name: (prices, features)}
        batch_size: Number of tasks per batch

    Yields:
        List of (train_data, val_data) task tuples
    """
    asset_names = list(asset_data.keys())

    while True:
        tasks = []
        for _ in range(batch_size):
            asset = np.random.choice(asset_names)
            prices, features = asset_data[asset]
            try:
                train_data, val_data = create_meta_gradient_tasks(prices, features)
                tasks.append((train_data, val_data))
            except ValueError:
                continue
        if tasks:
            yield tasks


def fetch_bybit_klines(
    symbol: str,
    interval: str = '60',
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch historical klines from Bybit exchange.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '1', '5', '15', '60', 'D')
        limit: Number of klines to fetch (max 1000)

    Returns:
        DataFrame with OHLCV data
    """
    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }
    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.set_index('timestamp').sort_index()

    return df


if __name__ == "__main__":
    print("Data Loader for Meta-Gradient Optimization")
    print("=" * 50)

    # Generate synthetic price data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    prices = pd.Series(
        100 * np.exp(np.cumsum(np.random.randn(500) * 0.02)),
        index=dates,
        name='Close',
    )

    features = create_trading_features(prices)
    print(f"Generated {len(features)} feature vectors with {len(features.columns)} features")
    print(f"Features: {list(features.columns)}")

    # Create a task
    train_data, val_data = create_meta_gradient_tasks(prices, features)
    train_f, train_t = train_data
    val_f, val_t = val_data
    print(f"\nTrain features shape: {train_f.shape}")
    print(f"Train targets shape: {train_t.shape}")
    print(f"Val features shape: {val_f.shape}")
    print(f"Val targets shape: {val_t.shape}")
