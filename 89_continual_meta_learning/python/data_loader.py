"""
Data Loader for Continual Meta-Learning Trading

This module provides data loading and preprocessing utilities for
continual meta-learning in algorithmic trading.

Supports:
- Stock data from Yahoo Finance (yfinance)
- Cryptocurrency data from Bybit API
- Regime detection and task generation
"""

import pandas as pd
import numpy as np
import torch
from typing import Tuple, Generator, Dict, List, Optional
import requests
from datetime import datetime, timedelta


def fetch_bybit_klines(
    symbol: str,
    interval: str = '60',
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch historical klines (candlestick data) from Bybit.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval in minutes ('1', '5', '15', '60', '240', 'D', 'W')
        limit: Number of klines to fetch (max 1000)

    Returns:
        DataFrame with OHLCV data
    """
    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data['retCode'] != 0:
            raise ValueError(f"Bybit API error: {data['retMsg']}")

        klines = data['result']['list']

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df = df.set_index('timestamp').sort_index()

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Bybit: {e}")
        return pd.DataFrame()


def create_trading_features(
    prices: pd.Series,
    window: int = 20
) -> pd.DataFrame:
    """
    Create technical features for trading.

    Args:
        prices: Price series
        window: Lookback window for features

    Returns:
        DataFrame with trading features
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
    rs = gain / (loss + 1e-8)
    features['rsi'] = (100 - (100 / (1 + rs)) - 50) / 50  # Normalized to [-1, 1]

    return features.dropna()


def detect_market_regime(
    prices: pd.Series,
    window: int = 20,
    vol_threshold_high: float = 1.5,
    vol_threshold_low: float = 0.5
) -> pd.Series:
    """
    Detect market regime based on price dynamics.

    Classifies market into four regimes:
    - 'bull': Positive returns, normal volatility
    - 'bear': Negative returns, normal volatility
    - 'high_vol': High volatility (any direction)
    - 'low_vol': Low volatility (any direction)

    Args:
        prices: Price series
        window: Lookback window for calculations
        vol_threshold_high: Multiplier for high volatility threshold
        vol_threshold_low: Multiplier for low volatility threshold

    Returns:
        Series with regime labels
    """
    returns = prices.pct_change()
    rolling_return = returns.rolling(window).mean()
    rolling_vol = returns.rolling(window).std()

    vol_median = rolling_vol.median()

    regimes = pd.Series(index=prices.index, dtype=str)

    for i in range(window, len(prices)):
        ret = rolling_return.iloc[i]
        vol = rolling_vol.iloc[i]

        if pd.isna(ret) or pd.isna(vol):
            continue

        if vol > vol_median * vol_threshold_high:
            regimes.iloc[i] = 'high_vol'
        elif vol < vol_median * vol_threshold_low:
            regimes.iloc[i] = 'low_vol'
        elif ret > 0:
            regimes.iloc[i] = 'bull'
        else:
            regimes.iloc[i] = 'bear'

    return regimes


def create_task_data(
    prices: pd.Series,
    features: pd.DataFrame,
    support_size: int = 20,
    query_size: int = 10,
    target_horizon: int = 5
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create support and query sets for a trading task.

    Args:
        prices: Price series
        features: Feature DataFrame
        support_size: Number of samples for adaptation
        query_size: Number of samples for evaluation
        target_horizon: Prediction horizon for returns

    Returns:
        (support_data, query_data) tuples
    """
    # Create target (future returns)
    target = prices.pct_change(target_horizon).shift(-target_horizon)

    # Align and drop NaN
    aligned = features.join(target.rename('target')).dropna()

    total_needed = support_size + query_size
    if len(aligned) < total_needed:
        raise ValueError(f"Not enough data: {len(aligned)} < {total_needed}")

    # Random split point
    start_idx = np.random.randint(0, len(aligned) - total_needed)

    # Split into support and query
    support_df = aligned.iloc[start_idx:start_idx + support_size]
    query_df = aligned.iloc[start_idx + support_size:start_idx + total_needed]

    # Get feature columns
    feature_cols = [c for c in aligned.columns if c != 'target']

    # Convert to tensors
    support_features = torch.FloatTensor(support_df[feature_cols].values)
    support_labels = torch.FloatTensor(support_df['target'].values).unsqueeze(1)

    query_features = torch.FloatTensor(query_df[feature_cols].values)
    query_labels = torch.FloatTensor(query_df['target'].values).unsqueeze(1)

    return (support_features, support_labels), (query_features, query_labels)


def create_regime_tasks(
    prices: pd.Series,
    features: pd.DataFrame,
    regimes: pd.Series,
    support_size: int = 20,
    query_size: int = 10,
    target_horizon: int = 5
) -> Generator:
    """
    Generate tasks organized by market regime.

    Yields tasks that can be used for continual meta-learning,
    with regime labels for tracking.

    Args:
        prices: Price series
        features: Feature DataFrame
        regimes: Regime labels
        support_size: Number of samples for adaptation
        query_size: Number of samples for evaluation
        target_horizon: Prediction horizon

    Yields:
        ((support_data, query_data), regime) tuples
    """
    # Create target
    target = prices.pct_change(target_horizon).shift(-target_horizon)
    aligned = features.join(target.rename('target')).join(regimes.rename('regime')).dropna()

    feature_cols = [c for c in aligned.columns if c not in ['target', 'regime']]

    while True:
        # Sample a regime
        valid_regimes = aligned['regime'].dropna().unique()
        if len(valid_regimes) == 0:
            continue

        regime = np.random.choice(valid_regimes)
        regime_data = aligned[aligned['regime'] == regime]

        total_needed = support_size + query_size
        if len(regime_data) < total_needed:
            continue

        # Sample contiguous window from this regime
        start_idx = np.random.randint(0, len(regime_data) - total_needed)
        window = regime_data.iloc[start_idx:start_idx + total_needed]

        support_df = window.iloc[:support_size]
        query_df = window.iloc[support_size:]

        support_features = torch.FloatTensor(support_df[feature_cols].values)
        support_labels = torch.FloatTensor(support_df['target'].values).unsqueeze(1)

        query_features = torch.FloatTensor(query_df[feature_cols].values)
        query_labels = torch.FloatTensor(query_df['target'].values).unsqueeze(1)

        yield ((support_features, support_labels),
               (query_features, query_labels)), regime


def multi_asset_task_generator(
    asset_data: Dict[str, Tuple[pd.Series, pd.DataFrame, pd.Series]],
    support_size: int = 20,
    query_size: int = 10,
    target_horizon: int = 5
) -> Generator:
    """
    Generate tasks from multiple assets and regimes.

    This provides maximum diversity for continual meta-learning.

    Args:
        asset_data: Dict of {asset_name: (prices, features, regimes)}
        support_size: Number of samples for adaptation
        query_size: Number of samples for evaluation
        target_horizon: Prediction horizon

    Yields:
        ((support_data, query_data), regime_label) tuples
    """
    # Create task generators for each asset
    generators = {}
    for asset_name, (prices, features, regimes) in asset_data.items():
        generators[asset_name] = create_regime_tasks(
            prices, features, regimes, support_size, query_size, target_horizon
        )

    asset_names = list(asset_data.keys())

    while True:
        # Sample random asset
        asset = np.random.choice(asset_names)
        task, regime = next(generators[asset])

        # Include asset name in regime label for tracking
        regime_label = f"{asset}_{regime}"

        yield task, regime_label


class DataLoader:
    """
    Unified data loader for continual meta-learning trading.

    Supports loading data from multiple sources and generating
    tasks for meta-learning.
    """

    def __init__(self, window: int = 20, target_horizon: int = 5):
        """
        Initialize DataLoader.

        Args:
            window: Feature calculation window
            target_horizon: Target prediction horizon
        """
        self.window = window
        self.target_horizon = target_horizon
        self.asset_data: Dict[str, Tuple[pd.Series, pd.DataFrame, pd.Series]] = {}

    def add_crypto(self, symbol: str, interval: str = '60', limit: int = 1000):
        """
        Add cryptocurrency data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval
            limit: Number of klines
        """
        df = fetch_bybit_klines(symbol, interval, limit)
        if df.empty:
            print(f"Warning: Could not fetch data for {symbol}")
            return

        prices = df['close']
        features = create_trading_features(prices, self.window)
        regimes = detect_market_regime(prices, self.window)

        self.asset_data[symbol] = (prices, features, regimes)
        print(f"Added {symbol}: {len(prices)} bars, {len(features)} features")

    def add_stock(self, symbol: str, period: str = '2y'):
        """
        Add stock data from Yahoo Finance.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            period: Data period ('1y', '2y', '5y', etc.)
        """
        try:
            import yfinance as yf
            df = yf.download(symbol, period=period, progress=False)

            if df.empty:
                print(f"Warning: Could not fetch data for {symbol}")
                return

            prices = df['Close']
            features = create_trading_features(prices, self.window)
            regimes = detect_market_regime(prices, self.window)

            self.asset_data[symbol] = (prices, features, regimes)
            print(f"Added {symbol}: {len(prices)} bars, {len(features)} features")

        except ImportError:
            print("yfinance not installed. Install with: pip install yfinance")

    def get_task_generator(
        self,
        support_size: int = 20,
        query_size: int = 10
    ) -> Generator:
        """
        Get a task generator for continual meta-learning.

        Args:
            support_size: Samples for adaptation
            query_size: Samples for evaluation

        Returns:
            Generator yielding (task, regime) tuples
        """
        if not self.asset_data:
            raise ValueError("No data loaded. Use add_crypto() or add_stock() first.")

        return multi_asset_task_generator(
            self.asset_data,
            support_size=support_size,
            query_size=query_size,
            target_horizon=self.target_horizon
        )

    def get_feature_size(self) -> int:
        """Get the number of features."""
        if not self.asset_data:
            return 8  # Default

        _, features, _ = list(self.asset_data.values())[0]
        return len(features.columns)


if __name__ == "__main__":
    # Example usage
    print("Data Loader for Continual Meta-Learning")
    print("=" * 50)

    # Test Bybit data fetching
    print("\nFetching BTC data from Bybit...")
    btc_data = fetch_bybit_klines('BTCUSDT', interval='60', limit=500)

    if not btc_data.empty:
        print(f"Fetched {len(btc_data)} bars")
        print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")

        # Create features
        prices = btc_data['close']
        features = create_trading_features(prices)
        regimes = detect_market_regime(prices)

        print(f"\nFeatures shape: {features.shape}")
        print(f"Feature columns: {list(features.columns)}")

        # Regime distribution
        regime_counts = regimes.value_counts()
        print(f"\nRegime distribution:")
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count}")

        # Test task generation
        print("\nGenerating sample tasks...")
        task_gen = create_regime_tasks(prices, features, regimes)

        for i in range(5):
            (support, query), regime = next(task_gen)
            print(f"  Task {i+1}: Regime={regime}, "
                  f"Support={support[0].shape}, Query={query[0].shape}")

    else:
        print("Could not fetch data (API may be unavailable)")

    # Test with DataLoader class
    print("\n" + "=" * 50)
    print("Testing DataLoader class...")

    loader = DataLoader(window=20, target_horizon=5)

    # Add crypto data
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        loader.add_crypto(symbol, limit=500)

    if loader.asset_data:
        print(f"\nFeature size: {loader.get_feature_size()}")

        # Get task generator
        task_gen = loader.get_task_generator(support_size=20, query_size=10)

        print("\nSample tasks from multi-asset generator:")
        for i in range(5):
            task, regime = next(task_gen)
            support, query = task
            print(f"  Task {i+1}: Regime={regime}")

    print("\nDone!")
