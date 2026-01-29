"""
Data loading and feature engineering for multi-task trading.

Supports:
- Technical feature generation from price series
- Multi-task target construction (return, volatility, direction, volume)
- Bybit API integration for cryptocurrency data
- Dataset preparation utilities
"""

import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_mtl_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Create technical features for multi-task learning.

    Args:
        prices: Price series (e.g., close prices)
        window: Lookback window for rolling features

    Returns:
        DataFrame with computed features (NaN rows dropped)
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


def create_mtl_targets(prices: pd.Series, horizon: int = 5) -> pd.DataFrame:
    """
    Create multi-task learning targets.

    Args:
        prices: Price series
        horizon: Prediction horizon in periods

    Returns:
        DataFrame with columns: return, direction, volatility, volume
    """
    targets = pd.DataFrame(index=prices.index)

    future_return = prices.pct_change(horizon).shift(-horizon)
    targets['return'] = future_return
    targets['direction'] = (future_return > 0).astype(float)
    targets['volatility'] = prices.pct_change().rolling(horizon).std().shift(-horizon)
    # Volume placeholder (use actual volume data when available)
    targets['volume'] = 0.0

    return targets


def create_mtl_targets_with_volume(
    prices: pd.Series,
    volume: pd.Series,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Create multi-task targets including actual volume data.

    Args:
        prices: Price series
        volume: Volume series
        horizon: Prediction horizon

    Returns:
        DataFrame with return, direction, volatility, volume targets
    """
    targets = create_mtl_targets(prices, horizon)

    # Normalize volume relative to rolling mean
    vol_mean = volume.rolling(20).mean()
    targets['volume'] = (volume.shift(-horizon) / (vol_mean + 1e-10))

    return targets


def fetch_bybit_klines(
    symbol: str,
    interval: str = '60',
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch historical klines (candlestick data) from the Bybit exchange.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
        interval: Candle interval ('1', '5', '15', '60', '240', 'D')
        limit: Number of candles to fetch (max 1000)

    Returns:
        DataFrame with columns: open, high, low, close, volume, turnover
    """
    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.set_index('timestamp').sort_index()

    return df


def prepare_mtl_dataset(
    prices: pd.Series,
    volume: Optional[pd.Series] = None,
    feature_window: int = 20,
    target_horizon: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare aligned features and targets for MTL training.

    Args:
        prices: Price series
        volume: Optional volume series
        feature_window: Lookback window for features
        target_horizon: Prediction horizon for targets

    Returns:
        (features_df, targets_df) with aligned indices, NaN-free
    """
    features = create_mtl_features(prices, window=feature_window)

    if volume is not None:
        targets = create_mtl_targets_with_volume(prices, volume, horizon=target_horizon)
    else:
        targets = create_mtl_targets(prices, horizon=target_horizon)

    aligned = features.join(targets, how='inner', lsuffix='_feat').dropna()
    feat_cols = [c for c in features.columns if c in aligned.columns or c + '_feat' in aligned.columns]
    tgt_cols = [c for c in targets.columns if c in aligned.columns]

    # Handle column name collisions
    final_feat_cols = []
    for c in features.columns:
        if c + '_feat' in aligned.columns:
            final_feat_cols.append(c + '_feat')
        elif c in aligned.columns:
            final_feat_cols.append(c)
    final_tgt_cols = [c for c in targets.columns if c in aligned.columns]

    return aligned[final_feat_cols], aligned[final_tgt_cols]


def load_multi_asset_data(
    symbols: List[str],
    source: str = "bybit",
    interval: str = '60',
    limit: int = 1000,
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple assets.

    Args:
        symbols: List of trading symbols
        source: Data source ('bybit')
        interval: Candle interval
        limit: Number of candles

    Returns:
        Dict mapping symbol -> DataFrame with OHLCV data
    """
    data = {}
    for symbol in symbols:
        try:
            if source == "bybit":
                df = fetch_bybit_klines(symbol, interval, limit)
                data[symbol] = df
                logger.info(f"Loaded {len(df)} candles for {symbol}")
            else:
                logger.warning(f"Unknown source: {source}")
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")

    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo with synthetic data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='h')
    prices = pd.Series(
        100 * np.exp(np.cumsum(np.random.randn(500) * 0.01)),
        index=dates,
        name='close',
    )

    features, targets = prepare_mtl_dataset(prices)
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Targets shape: {targets.shape}")
    logger.info(f"Feature columns: {list(features.columns)}")
    logger.info(f"Target columns: {list(targets.columns)}")
