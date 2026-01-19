"""
Data Loading Utilities for GQA Trading Model

This module provides utilities for loading financial data from:
- Bybit cryptocurrency exchange
- Yahoo Finance for stocks

All data is normalized and prepared for the GQA trading model.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict
from datetime import datetime, timedelta
import warnings


def load_bybit_data(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 1000,
    verbose: bool = True
) -> np.ndarray:
    """
    Load OHLCV data from Bybit exchange.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT", "ETHUSDT")
        interval: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d")
        limit: Number of candles to fetch (max 1000)
        verbose: Whether to print progress

    Returns:
        numpy array of shape (limit, 5) with OHLCV data

    Example:
        >>> data = load_bybit_data("BTCUSDT", "1h", limit=500)
        >>> data.shape
        (500, 5)
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests library required. Install with: pip install requests")

    if verbose:
        print(f"Loading {symbol} data from Bybit ({interval} interval)...")

    # Bybit API endpoint
    url = "https://api.bybit.com/v5/market/kline"

    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": _convert_interval(interval),
        "limit": min(limit, 1000)
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["retCode"] != 0:
            raise ValueError(f"Bybit API error: {data['retMsg']}")

        # Parse kline data
        klines = data["result"]["list"]

        # Bybit returns: [timestamp, open, high, low, close, volume, turnover]
        ohlcv = np.array([
            [float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
            for k in klines
        ], dtype=np.float32)

        # Reverse to chronological order (Bybit returns newest first)
        ohlcv = ohlcv[::-1]

        if verbose:
            print(f"  Loaded {len(ohlcv)} candles")
            print(f"  Price range: ${ohlcv[:, 3].min():.2f} - ${ohlcv[:, 1].max():.2f}")

        return ohlcv

    except requests.RequestException as e:
        warnings.warn(f"Failed to fetch from Bybit: {e}. Using synthetic data.")
        return _generate_synthetic_data(limit, symbol)


def _convert_interval(interval: str) -> str:
    """Convert common interval notation to Bybit format."""
    mapping = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "1d": "D",
        "1w": "W"
    }
    return mapping.get(interval, interval)


def load_yahoo_data(
    symbol: str = "AAPL",
    period: str = "1y",
    interval: str = "1d",
    verbose: bool = True
) -> np.ndarray:
    """
    Load OHLCV data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "GOOGL", "MSFT")
        period: Data period ("1mo", "3mo", "6mo", "1y", "2y", "5y")
        interval: Data interval ("1d", "1wk", "1mo")
        verbose: Whether to print progress

    Returns:
        numpy array of shape (n_samples, 5) with OHLCV data

    Example:
        >>> data = load_yahoo_data("AAPL", period="6mo")
        >>> data.shape
        (126, 5)
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance library required. Install with: pip install yfinance")

    if verbose:
        print(f"Loading {symbol} data from Yahoo Finance ({period}, {interval})...")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Extract OHLCV
        ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)

        if verbose:
            print(f"  Loaded {len(ohlcv)} candles")
            print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"  Price range: ${ohlcv[:, 2].min():.2f} - ${ohlcv[:, 1].max():.2f}")

        return ohlcv

    except Exception as e:
        warnings.warn(f"Failed to fetch from Yahoo Finance: {e}. Using synthetic data.")
        return _generate_synthetic_data(252 if period == "1y" else 126, symbol)


def _generate_synthetic_data(
    length: int,
    symbol: str = "SYNTHETIC",
    base_price: float = 100.0
) -> np.ndarray:
    """
    Generate synthetic OHLCV data for testing.

    Args:
        length: Number of candles to generate
        symbol: Symbol name (for logging)
        base_price: Starting price

    Returns:
        numpy array of shape (length, 5) with OHLCV data
    """
    print(f"  Generating synthetic data for {symbol}...")

    np.random.seed(42)

    # Generate price movement with trend and noise
    returns = np.random.randn(length) * 0.02  # 2% daily volatility
    trend = np.linspace(0, 0.5, length)  # Slight upward trend

    prices = base_price * np.exp(np.cumsum(returns) + trend)

    # Generate OHLCV
    ohlcv = np.zeros((length, 5), dtype=np.float32)

    for i in range(length):
        close = prices[i]
        open_price = close * (1 + np.random.randn() * 0.005)
        high = max(open_price, close) * (1 + abs(np.random.randn()) * 0.01)
        low = min(open_price, close) * (1 - abs(np.random.randn()) * 0.01)
        volume = np.random.exponential(1000000)

        ohlcv[i] = [open_price, high, low, close, volume]

    return ohlcv


def normalize_data(
    data: np.ndarray,
    method: str = "zscore"
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize OHLCV data for model input.

    Args:
        data: OHLCV data of shape (n_samples, 5)
        method: Normalization method ("zscore", "minmax", "returns")

    Returns:
        Tuple of (normalized_data, normalization_params)

    Example:
        >>> data = load_bybit_data("BTCUSDT", limit=100)
        >>> normalized, params = normalize_data(data, method="zscore")
        >>> normalized.shape
        (100, 5)
    """
    params = {"method": method}

    if method == "zscore":
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        normalized = (data - mean) / std
        params["mean"] = mean
        params["std"] = std

    elif method == "minmax":
        min_val = data.min(axis=0)
        max_val = data.max(axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        normalized = (data - min_val) / range_val
        params["min"] = min_val
        params["max"] = max_val

    elif method == "returns":
        # Calculate percentage returns
        normalized = np.zeros_like(data)
        normalized[1:] = (data[1:] - data[:-1]) / (data[:-1] + 1e-8)
        normalized[0] = 0
        params["first_row"] = data[0]

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.astype(np.float32), params


def denormalize_data(
    data: np.ndarray,
    params: Dict
) -> np.ndarray:
    """
    Reverse normalization to get original scale.

    Args:
        data: Normalized data
        params: Normalization parameters from normalize_data()

    Returns:
        Denormalized data
    """
    method = params["method"]

    if method == "zscore":
        return data * params["std"] + params["mean"]
    elif method == "minmax":
        return data * (params["max"] - params["min"]) + params["min"]
    elif method == "returns":
        # This is approximate - returns can't be perfectly reversed
        return data
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def prepare_sequences(
    data: np.ndarray,
    seq_len: int = 60,
    pred_horizon: int = 1,
    threshold: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare sequences for training the GQA model.

    Creates overlapping sequences and labels for price movement prediction.

    Args:
        data: OHLCV data of shape (n_samples, 5)
        seq_len: Length of input sequences
        pred_horizon: How many steps ahead to predict
        threshold: Threshold for classifying up/down (0 = any movement)

    Returns:
        Tuple of (sequences, labels) as PyTorch tensors
        - sequences: shape (n_sequences, seq_len, 5)
        - labels: shape (n_sequences,) with values 0 (down), 1 (neutral), 2 (up)

    Example:
        >>> data = load_bybit_data("BTCUSDT", limit=1000)
        >>> X, y = prepare_sequences(data, seq_len=60)
        >>> X.shape, y.shape
        (torch.Size([939, 60, 5]), torch.Size([939]))
    """
    n_samples = len(data)
    n_sequences = n_samples - seq_len - pred_horizon + 1

    if n_sequences <= 0:
        raise ValueError(
            f"Not enough data: {n_samples} samples for seq_len={seq_len}, "
            f"pred_horizon={pred_horizon}"
        )

    sequences = np.zeros((n_sequences, seq_len, 5), dtype=np.float32)
    labels = np.zeros(n_sequences, dtype=np.int64)

    for i in range(n_sequences):
        # Extract sequence
        sequences[i] = data[i:i + seq_len]

        # Calculate price change for label
        current_close = data[i + seq_len - 1, 3]  # Close price
        future_close = data[i + seq_len - 1 + pred_horizon, 3]

        pct_change = (future_close - current_close) / current_close

        # Classify movement
        if pct_change > threshold:
            labels[i] = 2  # Up
        elif pct_change < -threshold:
            labels[i] = 0  # Down
        else:
            labels[i] = 1  # Neutral

    # Convert to PyTorch tensors
    X = torch.from_numpy(sequences)
    y = torch.from_numpy(labels)

    return X, y


def create_data_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    train_split: float = 0.8
) -> Tuple:
    """
    Create training and validation data loaders.

    Args:
        X: Input sequences
        y: Labels
        batch_size: Batch size
        shuffle: Whether to shuffle training data
        train_split: Fraction of data for training

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import TensorDataset, DataLoader

    # Split data
    n_train = int(len(X) * train_split)

    train_dataset = TensorDataset(X[:n_train], y[:n_train])
    val_dataset = TensorDataset(X[n_train:], y[n_train:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loading
    print("Testing Data Loading Utilities...")
    print("=" * 50)

    # Test Bybit data
    print("\n1. Testing Bybit data loading:")
    try:
        btc_data = load_bybit_data("BTCUSDT", "1h", limit=100)
        print(f"   Shape: {btc_data.shape}")
        print(f"   Latest close: ${btc_data[-1, 3]:.2f}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test normalization
    print("\n2. Testing normalization:")
    normalized, params = normalize_data(btc_data, method="zscore")
    print(f"   Normalized mean: {normalized.mean():.4f}")
    print(f"   Normalized std: {normalized.std():.4f}")

    # Test sequence preparation
    print("\n3. Testing sequence preparation:")
    X, y = prepare_sequences(btc_data, seq_len=20)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Label distribution: {torch.bincount(y).tolist()}")

    # Test data loader
    print("\n4. Testing data loader:")
    train_loader, val_loader = create_data_loader(X, y, batch_size=16)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    print("\n" + "=" * 50)
    print("All data loading tests passed!")
