"""
Data Fetcher for Bybit via CCXT

This module provides functions to fetch cryptocurrency market data
from Bybit exchange using the CCXT library.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time


class BybitDataFetcher:
    """
    Fetches market data from Bybit exchange via CCXT.
    """

    def __init__(self, testnet: bool = False):
        """
        Initialize the Bybit data fetcher.

        Args:
            testnet: Whether to use Bybit testnet
        """
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 500,
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch (max 1000)
            since: Start timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading pair symbols
            timeframe: Candlestick timeframe
            limit: Number of candles per symbol

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}

        for symbol in symbols:
            print(f"Fetching {symbol}...")
            df = self.fetch_ohlcv(symbol, timeframe, limit)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.5)  # Rate limiting

        return data

    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data dictionary
        """
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            print(f"Error fetching ticker for {symbol}: {e}")
            return {}

    def fetch_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """
        Fetch order book data.

        Args:
            symbol: Trading pair symbol
            limit: Order book depth

        Returns:
            Order book dictionary with 'bids' and 'asks'
        """
        try:
            return self.exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            print(f"Error fetching orderbook for {symbol}: {e}")
            return {'bids': [], 'asks': []}


def prepare_features(df: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and targets for model training.

    Args:
        df: OHLCV DataFrame
        lookback: Number of periods to look back for features

    Returns:
        Tuple of (features, targets) arrays
    """
    # Calculate returns
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['returns_5'] = df['close'].pct_change(5)
    df['returns_15'] = df['close'].pct_change(15)

    # Calculate volatility
    df['volatility'] = df['returns'].rolling(20).std()

    # Calculate volume ratio
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Calculate target (future returns)
    df['target'] = df['close'].pct_change(5).shift(-5)

    # Drop NaN values
    df.dropna(inplace=True)

    # Select features
    feature_columns = [
        'returns', 'returns_5', 'returns_15',
        'volatility', 'volume_ratio',
        'rsi', 'macd_diff', 'bb_position'
    ]

    features = df[feature_columns].values
    targets = df['target'].values

    return features, targets


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series model.

    Args:
        features: Feature array
        targets: Target array
        sequence_length: Length of each sequence

    Returns:
        Tuple of (X, y) sequence arrays
    """
    X, y = [], []

    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])

    return np.array(X), np.array(y)


def generate_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 8,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for testing.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise_level: Standard deviation of noise

    Returns:
        Tuple of (features, targets)
    """
    np.random.seed(42)

    # Generate features
    features = np.random.randn(n_samples, n_features)

    # Generate targets with some pattern
    weights = np.random.randn(n_features)
    targets = features @ weights + noise_level * np.random.randn(n_samples)

    # Add some non-linearity
    targets = targets + 0.3 * np.sin(features[:, 0] * 2)

    return features.astype(np.float32), targets.astype(np.float32)


if __name__ == "__main__":
    # Example usage
    print("Deep Ensembles Trading - Data Fetcher")
    print("=" * 50)

    # Initialize fetcher
    fetcher = BybitDataFetcher()

    # Fetch data for major cryptocurrencies
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

    print("\nFetching data...")
    data = fetcher.fetch_multiple_symbols(symbols, timeframe='1h', limit=200)

    for symbol, df in data.items():
        print(f"\n{symbol}:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")

    # Prepare features for one symbol
    if 'BTC/USDT' in data:
        print("\nPreparing features for BTC/USDT...")
        features, targets = prepare_features(data['BTC/USDT'])
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Feature statistics:")
        print(f"  Mean: {np.mean(features, axis=0)}")
        print(f"  Std: {np.std(features, axis=0)}")
