"""
Data Loading Module

This module provides functions for fetching market data from:
- Bybit (cryptocurrency exchange)
- Yahoo Finance (stock market)

It also includes feature engineering utilities for computing
technical indicators used in rule extraction.
"""

import numpy as np
import pandas as pd
import requests
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timedelta


class BybitClient:
    """Client for fetching cryptocurrency data from Bybit API."""

    BASE_URL = "https://api.bybit.com/v5"

    def __init__(self):
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "60",
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (kline) data from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
            interval: Candlestick interval ("1", "5", "15", "60", "D")
            limit: Number of candles to fetch (max 200)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        url = f"{self.BASE_URL}/market/kline"
        params = {
            'category': 'spot',
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 200)
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if data['retCode'] != 0:
            raise ValueError(f"Bybit API error: {data['retMsg']}")

        # Parse response
        klines = data['result']['list']

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        # Sort by timestamp (oldest first)
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def get_ticker(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Fetch current ticker information.

        Args:
            symbol: Trading pair

        Returns:
            Dictionary with ticker info
        """
        url = f"{self.BASE_URL}/market/tickers"
        params = {
            'category': 'spot',
            'symbol': symbol
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if data['retCode'] != 0:
            raise ValueError(f"Bybit API error: {data['retMsg']}")

        if not data['result']['list']:
            raise ValueError(f"No ticker data found for {symbol}")

        ticker = data['result']['list'][0]

        return {
            'symbol': ticker['symbol'],
            'last_price': float(ticker['lastPrice']),
            'high_24h': float(ticker['highPrice24h']),
            'low_24h': float(ticker['lowPrice24h']),
            'volume_24h': float(ticker['volume24h']),
            'change_24h': float(ticker['price24hPcnt'])
        }


def get_stock_data(
    ticker: str = "SPY",
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch stock market data using yfinance.

    Args:
        ticker: Stock symbol
        period: Data period (e.g., "1mo", "1y", "2y")
        interval: Data interval (e.g., "1d", "1h")

    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Please install yfinance: pip install yfinance")

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    # Rename columns to match our format
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    df = df[['open', 'high', 'low', 'close', 'volume']].reset_index()
    df = df.rename(columns={'Date': 'timestamp', 'Datetime': 'timestamp'})

    return df


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    Args:
        prices: Series of closing prices
        period: RSI period (default 14)

    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()

    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def compute_sma(prices: pd.Series, period: int) -> pd.Series:
    """Compute Simple Moving Average."""
    return prices.rolling(window=period, min_periods=1).mean()


def compute_ema(prices: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average."""
    return prices.ewm(span=period, adjust=False).mean()


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD indicator.

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    fast_ema = compute_ema(prices, fast)
    slow_ema = compute_ema(prices, slow)

    macd_line = fast_ema - slow_ema
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
    """Compute rolling volatility (standard deviation of returns)."""
    return returns.rolling(window=period, min_periods=1).std()


def compute_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute all features from OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume

    Returns:
        Tuple of (features DataFrame, feature names list)
    """
    features = pd.DataFrame(index=df.index)

    # Price-based features
    features['returns'] = df['close'].pct_change()

    # RSI
    features['RSI'] = compute_rsi(df['close'], 14)

    # MACD
    macd, signal, hist = compute_macd(df['close'])
    features['MACD'] = macd

    # Moving Average Ratios
    sma_20 = compute_sma(df['close'], 20)
    sma_50 = compute_sma(df['close'], 50)
    features['SMA_Ratio'] = sma_20 / sma_50.replace(0, np.nan)

    # Volatility
    features['Volatility'] = compute_volatility(features['returns'], 20)

    # Volume Change
    features['Volume_Change'] = df['volume'].pct_change()

    # Fill NaN values
    features = features.fillna(method='ffill').fillna(method='bfill')
    features = features.fillna({
        'RSI': 50,
        'MACD': 0,
        'SMA_Ratio': 1,
        'Volatility': 0.01,
        'Volume_Change': 0,
        'returns': 0
    })

    feature_names = ['RSI', 'MACD', 'SMA_Ratio', 'Volatility', 'Volume_Change']

    return features[feature_names], feature_names


def generate_labels(df: pd.DataFrame, lookahead: int = 1) -> pd.Series:
    """
    Generate trading labels based on future price movement.

    Args:
        df: DataFrame with 'close' column
        lookahead: Number of periods to look ahead

    Returns:
        Series of labels: 1 (BUY/up), -1 (SELL/down)
    """
    future_returns = df['close'].shift(-lookahead) / df['close'] - 1
    labels = (future_returns > 0).astype(int) * 2 - 1
    labels = labels.fillna(0).astype(int)
    return labels


def prepare_dataset(
    source: str = "bybit",
    symbol: str = "BTCUSDT",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare a complete dataset for rule extraction.

    Args:
        source: Data source ("bybit" or "yfinance")
        symbol: Trading symbol
        **kwargs: Additional arguments for data fetching

    Returns:
        Tuple of (features, labels, prices, feature_names)
    """
    # Fetch data
    if source == "bybit":
        client = BybitClient()
        df = client.get_klines(symbol, **kwargs)
    elif source == "yfinance":
        df = get_stock_data(symbol, **kwargs)
    else:
        raise ValueError(f"Unknown source: {source}")

    # Compute features
    features_df, feature_names = compute_features(df)

    # Generate labels
    labels = generate_labels(df)

    # Convert to numpy
    X = features_df.values
    y = labels.values
    prices = df['close'].values

    return X, y, prices, feature_names


if __name__ == "__main__":
    # Example usage
    print("=== Data Loading Example ===\n")

    # Fetch Bybit data
    print("Fetching BTCUSDT data from Bybit...")
    client = BybitClient()

    try:
        df = client.get_klines("BTCUSDT", interval="60", limit=100)
        print(f"Fetched {len(df)} candles")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\nLatest candle:")
        print(df.tail(1).to_string())

        # Get ticker
        ticker = client.get_ticker("BTCUSDT")
        print(f"\nCurrent ticker:")
        print(f"  Price: ${ticker['last_price']:.2f}")
        print(f"  24h Change: {ticker['change_24h']*100:.2f}%")

        # Compute features
        features, feature_names = compute_features(df)
        print(f"\nComputed features: {feature_names}")
        print(f"\nLatest feature values:")
        print(features.tail(1).to_string())

        # Generate labels
        labels = generate_labels(df)
        buy_pct = (labels == 1).mean() * 100
        print(f"\nLabel distribution: {buy_pct:.1f}% BUY, {100-buy_pct:.1f}% SELL")

    except Exception as e:
        print(f"Error: {e}")
        print("\nUsing generated sample data instead...")

        # Generate sample data
        np.random.seed(42)
        n = 200

        # Simulated price data
        prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='H'),
            'open': prices * (1 + np.random.randn(n) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(n) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(n) * 0.01)),
            'close': prices,
            'volume': np.random.uniform(100000, 500000, n)
        })

        features, feature_names = compute_features(df)
        print(f"Generated sample data with {len(df)} rows")
        print(f"Features: {feature_names}")
