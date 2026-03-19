"""Data loading and preprocessing for Disentangled VAE Trading.

Provides utilities to fetch market data from the Bybit v5 API, compute
technical indicators, and package time-series windows into a PyTorch
Dataset suitable for training disentangled VAE models.

Functions:
    load_bybit_data -- fetch OHLCV candles from Bybit
    load_stock_data -- generate synthetic equity data for testing
    prepare_features -- compute technical indicators from OHLCV data
    create_synthetic_data -- produce realistic synthetic market data

Classes:
    DisentangledVAEDataset -- windowed PyTorch Dataset for VAE training
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Bybit API data loader
# ---------------------------------------------------------------------------

def load_bybit_data(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV candle data from the Bybit v5 public API.

    Args:
        symbol: Trading pair symbol (e.g. ``"BTCUSDT"``).
        interval: Candle interval in minutes. Common values: ``"1"``,
            ``"5"``, ``"15"``, ``"60"``, ``"240"``, ``"D"``.
        limit: Maximum number of candles to retrieve (up to 1000).

    Returns:
        DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``volume`` indexed by datetime.

    Raises:
        RuntimeError: If the API request fails.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required for Bybit API access") from exc

    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {data.get('retMsg', 'unknown')}")

    rows = data["result"]["list"]
    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df.drop(columns=["turnover"], inplace=True)
    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Synthetic / CSV stock data loader
# ---------------------------------------------------------------------------

def load_stock_data(
    symbol: str = "SYNTH",
    period: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily equity OHLCV data for testing.

    Uses geometric Brownian motion with stochastic volatility to produce
    data that exhibits realistic clustering and mean-reversion.

    Args:
        symbol: Label for the data (unused beyond metadata).
        period: Number of daily bars to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``volume`` and a datetime index.
    """
    rng = np.random.default_rng(seed)
    price = 100.0
    prices = []
    vol = 0.02

    for _ in range(period):
        # Stochastic volatility
        vol = max(0.005, vol + 0.001 * rng.standard_normal())
        ret = rng.normal(0.0002, vol)
        price *= np.exp(ret)

        open_ = price * (1 + rng.normal(0, 0.002))
        high = price * (1 + abs(rng.normal(0, 0.005)))
        low = price * (1 - abs(rng.normal(0, 0.005)))
        close = price
        volume = abs(rng.normal(1e6, 3e5))
        prices.append([open_, high, low, close, volume])

    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=period, freq="B")
    df = pd.DataFrame(prices, index=idx, columns=["open", "high", "low", "close", "volume"])
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)


def _macd(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Compute MACD line and signal line."""
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trading features from an OHLCV DataFrame.

    Features generated:
        - ``log_return``: log returns of the close price
        - ``volume_change``: percentage change in volume
        - ``volatility``: rolling 20-bar standard deviation of log returns
        - ``rsi``: 14-period Relative Strength Index (normalised to [0, 1])
        - ``macd``: MACD histogram (MACD line minus signal)
        - ``bb_position``: position within Bollinger Bands (0 = lower, 1 = upper)

    Args:
        df: DataFrame with at least ``close`` and ``volume`` columns.

    Returns:
        DataFrame containing only the computed feature columns, with NaN
        rows dropped.
    """
    features = pd.DataFrame(index=df.index)

    # Log returns
    features["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Volume change
    features["volume_change"] = df["volume"].pct_change()

    # Rolling volatility
    features["volatility"] = features["log_return"].rolling(20, min_periods=1).std()

    # RSI (normalised to 0-1)
    features["rsi"] = _rsi(df["close"]) / 100.0

    # MACD histogram
    macd_line, signal_line = _macd(df["close"])
    features["macd"] = macd_line - signal_line

    # Bollinger Band position
    sma20 = df["close"].rolling(20, min_periods=1).mean()
    std20 = df["close"].rolling(20, min_periods=1).std().replace(0, 1e-10)
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    features["bb_position"] = (df["close"] - lower) / (upper - lower)

    features.dropna(inplace=True)
    return features


# ---------------------------------------------------------------------------
# Synthetic data helper (for quick demos)
# ---------------------------------------------------------------------------

def create_synthetic_data(
    n_samples: int = 2000,
    n_features: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Create fully synthetic feature data with known latent factors.

    Generates data driven by a small number of independent latent factors
    mixed through a random linear map, which provides a ground truth for
    evaluating disentanglement quality.

    Args:
        n_samples: Number of time steps.
        n_features: Number of observed features.
        seed: Random seed.

    Returns:
        DataFrame of shape (n_samples, n_features).
    """
    rng = np.random.default_rng(seed)
    n_factors = 3

    # Independent latent factors
    t = np.linspace(0, 8 * np.pi, n_samples)
    factors = np.column_stack([
        np.sin(t) + 0.1 * rng.standard_normal(n_samples),          # trend
        0.5 * np.sin(3 * t) + 0.1 * rng.standard_normal(n_samples), # momentum
        np.abs(np.sin(0.5 * t)) + 0.1 * rng.standard_normal(n_samples),  # volatility
    ])

    # Mix to observed space
    mixing = rng.standard_normal((n_factors, n_features))
    observed = factors @ mixing + 0.05 * rng.standard_normal((n_samples, n_features))

    columns = [f"feature_{i}" for i in range(n_features)]
    return pd.DataFrame(observed, columns=columns)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class DisentangledVAEDataset(Dataset):
    """Sliding-window dataset for disentangled VAE training.

    Each sample is a window of ``seq_len`` consecutive feature vectors,
    returned as a tensor of shape ``(n_features, seq_len)`` (channels-first
    for Conv1d compatibility).

    Optionally provides a target tensor consisting of the ``prediction_horizon``
    log-returns immediately following each window, useful for supervised
    forecasting heads.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int = 60,
        prediction_horizon: int = 5,
        normalise: bool = True,
    ) -> None:
        """
        Args:
            data: Feature DataFrame (rows = time steps, columns = features).
            seq_len: Number of time steps per window.
            prediction_horizon: Steps ahead to include as targets.
            normalise: Whether to z-score normalise each feature.
        """
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon

        values = data.values.astype(np.float32)

        if normalise:
            self._mean = values.mean(axis=0)
            self._std = values.std(axis=0)
            self._std[self._std < 1e-8] = 1.0
            values = (values - self._mean) / self._std
        else:
            self._mean = np.zeros(values.shape[1], dtype=np.float32)
            self._std = np.ones(values.shape[1], dtype=np.float32)

        self.data = torch.from_numpy(values)
        self.n_features = self.data.shape[1]

        # Build valid index list
        max_start = len(self.data) - seq_len - prediction_horizon
        self.indices = list(range(max(max_start + 1, 0)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        window = self.data[start : start + self.seq_len].T  # (C, T)
        target_start = start + self.seq_len
        target = self.data[
            target_start : target_start + self.prediction_horizon, 0
        ]  # first feature as target
        return window, target
