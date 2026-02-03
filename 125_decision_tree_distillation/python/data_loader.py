"""
Data Loader for Decision Tree Distillation

This module provides utilities for loading stock and cryptocurrency data
from various sources (Yahoo Finance, Bybit) and computing technical features.
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def load_stock_data(
    symbol: str,
    period: str = '2y',
    interval: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load stock data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        start_date: Start date string (YYYY-MM-DD), overrides period if set
        end_date: End date string (YYYY-MM-DD)

    Returns:
        data: DataFrame with OHLCV data and date column

    Example:
        >>> data = load_stock_data('AAPL', period='1y')
        >>> print(data.head())
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)

        if start_date:
            data = ticker.history(start=start_date, end=end_date, interval=interval)
        else:
            data = ticker.history(period=period, interval=interval)

        if data.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Standardize column names
        data = data.reset_index()
        data.columns = [c.lower().replace(' ', '_') for c in data.columns]

        # Ensure required columns exist
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if 'datetime' in data.columns:
            data = data.rename(columns={'datetime': 'date'})

        return data[required_cols].copy()

    except ImportError:
        print("yfinance not installed. Install with: pip install yfinance")
        return _generate_synthetic_stock_data(symbol, period)
    except Exception as e:
        print(f"Error loading data for {symbol}: {e}")
        return _generate_synthetic_stock_data(symbol, period)


def load_crypto_data(
    symbol: str = 'BTCUSDT',
    interval: str = '1h',
    limit: int = 1000,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load cryptocurrency data from Bybit API.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
        interval: Candle interval ('1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M')
        limit: Number of candles to fetch (max 1000)
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds

    Returns:
        data: DataFrame with OHLCV data and date column

    Example:
        >>> data = load_crypto_data('BTCUSDT', interval='1h', limit=500)
        >>> print(data.head())
    """
    try:
        import requests

        # Bybit API endpoint
        url = "https://api.bybit.com/v5/market/kline"

        params = {
            'category': 'spot',
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000),
        }

        if start_time:
            params['start'] = start_time
        if end_time:
            params['end'] = end_time

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        result = response.json()

        if result.get('retCode') != 0:
            raise ValueError(f"API error: {result.get('retMsg')}")

        klines = result.get('result', {}).get('list', [])
        if not klines:
            raise ValueError("No data returned from API")

        # Parse kline data
        # Format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        data = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # Convert types
        data['timestamp'] = pd.to_numeric(data['timestamp'])
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Sort by date ascending
        data = data.sort_values('date').reset_index(drop=True)

        return data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

    except ImportError:
        print("requests not installed. Install with: pip install requests")
        return _generate_synthetic_crypto_data(symbol, limit)
    except Exception as e:
        print(f"Error loading crypto data for {symbol}: {e}")
        return _generate_synthetic_crypto_data(symbol, limit)


def compute_features(
    data: pd.DataFrame,
    target_periods: int = 1,
    include_labels: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Compute technical indicators and trading features.

    Args:
        data: DataFrame with OHLCV data
        target_periods: Number of periods ahead for label calculation
        include_labels: Whether to compute labels (price direction)

    Returns:
        features: DataFrame with computed features
        labels: Array of labels (1=up, 0=down) or None if include_labels=False

    Example:
        >>> data = load_stock_data('AAPL', period='1y')
        >>> X, y = compute_features(data)
        >>> print(X.columns.tolist())
    """
    df = data.copy()

    # Price columns
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # RSI (Relative Strength Index)
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    for period in [20]:
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        df[f'bb_upper_{period}'] = sma + (std * 2)
        df[f'bb_lower_{period}'] = sma - (std * 2)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        df[f'bb_position_{period}'] = (close - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)

    # Moving Averages
    for period in [5, 10, 20, 50, 200]:
        df[f'sma_{period}'] = close.rolling(window=period).mean()
        df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()

    # Price relative to MAs
    df['price_sma_20_ratio'] = close / df['sma_20']
    df['price_sma_50_ratio'] = close / df['sma_50']

    # ATR (Average True Range)
    for period in [14]:
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)
        df[f'atr_{period}'] = tr.rolling(window=period).mean()

    # ADX (Average Directional Index)
    for period in [14]:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df[f'adx_{period}'] = dx.rolling(window=period).mean()
        df[f'plus_di_{period}'] = plus_di
        df[f'minus_di_{period}'] = minus_di

    # CCI (Commodity Channel Index)
    for period in [20]:
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad + 1e-10)

    # ROC (Rate of Change)
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = close.pct_change(periods=period) * 100

    # Volume features
    df['volume_sma_20'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / (df['volume_sma_20'] + 1e-10)

    # Volatility
    for period in [10, 20]:
        df[f'volatility_{period}'] = close.pct_change().rolling(window=period).std() * np.sqrt(252)

    # Returns
    df['returns_1d'] = close.pct_change()
    df['returns_5d'] = close.pct_change(periods=5)

    # Select feature columns
    feature_cols = [col for col in df.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 'timestamp', 'turnover']]

    # Compute labels
    labels = None
    if include_labels:
        future_returns = close.shift(-target_periods) / close - 1
        labels = (future_returns > 0).astype(int).values

    # Drop NaN rows
    features = df[feature_cols].copy()
    valid_mask = ~features.isna().any(axis=1)

    if include_labels and labels is not None:
        valid_mask &= ~pd.isna(pd.Series(labels))
        labels = labels[valid_mask]

    features = features[valid_mask].reset_index(drop=True)

    return features, labels


def create_train_test_split(
    features: pd.DataFrame,
    labels: np.ndarray,
    train_ratio: float = 0.8,
    shuffle: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.

    Args:
        features: Feature DataFrame
        labels: Label array
        train_ratio: Fraction of data for training
        shuffle: Whether to shuffle data (False for time series)

    Returns:
        X_train, X_test, y_train, y_test: Split data
    """
    n = len(features)
    split_idx = int(n * train_ratio)

    if shuffle:
        indices = np.random.permutation(n)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
    else:
        train_indices = np.arange(split_idx)
        test_indices = np.arange(split_idx, n)

    X_train = features.iloc[train_indices].reset_index(drop=True)
    X_test = features.iloc[test_indices].reset_index(drop=True)
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    return X_train, X_test, y_train, y_test


def _generate_synthetic_stock_data(symbol: str, period: str) -> pd.DataFrame:
    """Generate synthetic stock data when real data is unavailable."""
    np.random.seed(hash(symbol) % 2**32)

    # Determine number of days from period
    period_days = {
        '1d': 1, '5d': 5, '1mo': 22, '3mo': 66, '6mo': 126,
        '1y': 252, '2y': 504, '5y': 1260, '10y': 2520,
    }
    n_days = period_days.get(period, 252)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    # Generate random walk prices
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.cumprod(1 + returns)

    # Generate OHLC from close
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': prices * (1 + np.random.uniform(0, 0.02, n_days)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n_days)),
        'close': prices,
        'volume': np.random.uniform(1e6, 1e7, n_days).astype(int),
    })

    return data


def _generate_synthetic_crypto_data(symbol: str, limit: int) -> pd.DataFrame:
    """Generate synthetic crypto data when API is unavailable."""
    np.random.seed(hash(symbol) % 2**32)

    dates = pd.date_range(end=datetime.now(), periods=limit, freq='1H')

    # Crypto typically has higher volatility
    returns = np.random.normal(0.0001, 0.03, limit)

    # Set base price based on symbol
    base_price = {'BTCUSDT': 50000, 'ETHUSDT': 3000}.get(symbol, 1000)
    prices = base_price * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, limit)),
        'high': prices * (1 + np.random.uniform(0, 0.015, limit)),
        'low': prices * (1 - np.random.uniform(0, 0.015, limit)),
        'close': prices,
        'volume': np.random.uniform(100, 10000, limit),
    })

    return data


if __name__ == "__main__":
    # Example usage
    print("Data Loader Example")
    print("=" * 50)

    # Load stock data
    print("\nLoading stock data (AAPL)...")
    stock_data = load_stock_data('AAPL', period='1y')
    print(f"Loaded {len(stock_data)} rows")
    print(stock_data.head())

    # Load crypto data
    print("\nLoading crypto data (BTCUSDT)...")
    crypto_data = load_crypto_data('BTCUSDT', interval='60', limit=500)
    print(f"Loaded {len(crypto_data)} rows")
    print(crypto_data.head())

    # Compute features
    print("\nComputing features...")
    X, y = compute_features(stock_data)
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape if y is not None else 'N/A'}")
    print(f"\nFeature columns:\n{X.columns.tolist()}")

    # Create train/test split
    print("\nCreating train/test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
