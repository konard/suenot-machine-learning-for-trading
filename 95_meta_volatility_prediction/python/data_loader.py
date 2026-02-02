"""
Data loading utilities for meta-volatility prediction.

Supports loading data from:
- Yahoo Finance (stock market)
- Bybit API (cryptocurrency market)
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load stock data using yfinance.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL').
        start_date: Start date string (e.g., '2020-01-01').
        end_date: End date string (e.g., '2024-01-01').

    Returns:
        DataFrame with OHLCV data and log returns.
    """
    import yfinance as yf

    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()
    logger.info("Loaded %d rows for %s", len(df), symbol)
    return df


def load_bybit_data(
    symbol: str,
    interval: str = '60',
    limit: int = 1000,
) -> pd.DataFrame:
    """Load cryptocurrency data from Bybit API.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT').
        interval: Kline interval in minutes (e.g., '60' for 1h).
        limit: Number of candles to fetch (max 1000).

    Returns:
        DataFrame with OHLCV data and log returns.
    """
    import requests

    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover',
    ])
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna()
    logger.info("Loaded %d rows for %s from Bybit", len(df), symbol)
    return df


def compute_features(
    df: pd.DataFrame,
    price_col: str = 'close',
) -> pd.DataFrame:
    """Compute volatility prediction features.

    Features:
        - log_return, abs_return, squared_return
        - rv_5, rv_10, rv_20 (realized volatility at different horizons)
        - volume_ratio (relative to 20-period average)
        - rsi (14-period Relative Strength Index)
        - bb_width (Bollinger Band width)
        - spread (placeholder, 0.0 when not available)

    Args:
        df: DataFrame with 'log_return' and price column.
        price_col: Name of the price column.

    Returns:
        DataFrame with added feature columns (NaN rows dropped).
    """
    df = df.copy()
    r = df['log_return']

    df['abs_return'] = r.abs()
    df['squared_return'] = r ** 2
    df['rv_5'] = r.rolling(5).std()
    df['rv_10'] = r.rolling(10).std()
    df['rv_20'] = r.rolling(20).std()

    if 'volume' in df.columns:
        vol_mean = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (vol_mean + 1e-10)
    else:
        df['volume_ratio'] = 1.0

    # RSI (14-period)
    if price_col in df.columns:
        delta = df[price_col].diff()
    else:
        delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss_val = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss_val + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Band width
    if price_col in df.columns:
        price = df[price_col]
    else:
        price = df['Close']
    sma = price.rolling(20).mean()
    std = price.rolling(20).std()
    df['bb_width'] = (2 * std) / (sma + 1e-10)

    # Spread placeholder
    if 'spread' not in df.columns:
        df['spread'] = 0.0

    return df.dropna()


FEATURE_COLS = [
    'log_return', 'abs_return', 'squared_return',
    'rv_5', 'rv_10', 'rv_20', 'volume_ratio',
    'spread', 'rsi', 'bb_width',
]


def create_tasks(
    df: pd.DataFrame,
    window_size: int = 60,
    forecast_horizon: int = 5,
    support_size: int = 10,
    query_size: int = 5,
    feature_cols: Optional[List[str]] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Create meta-learning tasks from a single asset's data.

    Each task consists of:
        - support_x, support_y: K-shot examples for adaptation
        - query_x, query_y: examples for meta-evaluation

    Args:
        df: DataFrame with computed features.
        window_size: Lookback window for feature context.
        forecast_horizon: Number of periods to forecast volatility over.
        support_size: Number of support examples per task.
        query_size: Number of query examples per task.
        feature_cols: List of feature column names.

    Returns:
        List of (support_x, support_y, query_x, query_y) numpy arrays.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    total = support_size + query_size
    tasks = []

    for start in range(
        0,
        len(df) - window_size - forecast_horizon - total,
        total,
    ):
        chunk = df.iloc[start:start + window_size + forecast_horizon + total]

        x_all, y_all = [], []
        for i in range(window_size, window_size + total):
            features = chunk[feature_cols].iloc[i - 1].values
            target = chunk['log_return'].iloc[i:i + forecast_horizon].std()
            x_all.append(features)
            y_all.append(target)

        x_all = np.array(x_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.float32)

        # Replace NaN/inf with 0
        x_all = np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0)
        y_all = np.nan_to_num(y_all, nan=0.0, posinf=0.0, neginf=0.0)

        tasks.append((
            x_all[:support_size],
            y_all[:support_size],
            x_all[support_size:],
            y_all[support_size:],
        ))

    logger.info("Created %d tasks from %d rows", len(tasks), len(df))
    return tasks
