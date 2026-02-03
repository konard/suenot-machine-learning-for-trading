"""
Data loading utilities for SSM-Transformer Hybrid trading model.

Supports data from:
- Bybit (cryptocurrency via public REST API)
- Yahoo Finance (stocks via yfinance)

Generates features: OHLCV, technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV).
Produces multi-task targets: direction, volatility, return magnitude.
"""

import numpy as np
import pandas as pd
import requests
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    sma_periods: Tuple[int, ...] = (5, 10, 20, 50)


def fetch_bybit_klines(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch kline data from Bybit public API.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
        interval: Candle interval in minutes ("1", "5", "15", "60", "240", "D")
        limit: Number of candles to fetch (max 1000)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0:
            logger.warning(f"Bybit API error: {data.get('retMsg')}")
            return _generate_synthetic_data(limit)

        rows = data["result"]["list"]
        df = pd.DataFrame(rows, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch Bybit data: {e}. Using synthetic data.")
        return _generate_synthetic_data(limit)


def fetch_yfinance_data(
    symbol: str = "AAPL",
    period: str = "2y",
) -> pd.DataFrame:
    """
    Fetch stock data using yfinance.

    Falls back to synthetic data if yfinance is not available.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except ImportError:
        logger.warning("yfinance not installed. Using synthetic data.")
        return _generate_synthetic_data(500)


def _generate_synthetic_data(n: int = 1000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    volumes = np.random.lognormal(mean=10, sigma=1, size=n)

    highs = prices * (1 + np.abs(np.random.randn(n) * 0.01))
    lows = prices * (1 - np.abs(np.random.randn(n) * 0.01))
    opens = lows + (highs - lows) * np.random.rand(n)

    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="h"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    })


def compute_features(df: pd.DataFrame, config: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """
    Compute technical indicator features from OHLCV data.

    Features generated:
    - Log returns
    - RSI
    - MACD (line, signal, histogram)
    - Bollinger Bands (upper, lower, width)
    - ATR
    - OBV
    - Volume ratio
    - SMA ratios
    """
    if config is None:
        config = FeatureConfig()

    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Log returns
    df["log_return"] = np.log(close / close.shift(1))

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(config.rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(config.rsi_period).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"] / 100.0  # normalize to [0, 1]

    # MACD
    ema_fast = close.ewm(span=config.macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=config.macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=config.macd_signal, adjust=False).mean()
    df["macd"] = macd_line / close  # normalize
    df["macd_signal"] = macd_signal / close
    df["macd_hist"] = (macd_line - macd_signal) / close

    # Bollinger Bands
    sma = close.rolling(config.bb_period).mean()
    std = close.rolling(config.bb_period).std()
    df["bb_upper"] = (sma + config.bb_std * std - close) / close
    df["bb_lower"] = (close - sma + config.bb_std * std) / close
    df["bb_width"] = (2 * config.bb_std * std) / close

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(config.atr_period).mean() / close

    # OBV (normalized)
    obv = (np.sign(close.diff()) * volume).cumsum()
    df["obv"] = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std() + 1e-10)

    # Volume ratio
    df["volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-10)

    # SMA ratios
    for period in config.sma_periods:
        sma_val = close.rolling(period).mean()
        df[f"sma_{period}_ratio"] = (close - sma_val) / (sma_val + 1e-10)

    return df


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multi-task targets.

    Targets:
    - direction: 1 if next close > current close, else 0
    - volatility: absolute log return of next bar
    - return_mag: log return of next bar
    """
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))

    df["target_direction"] = (df["close"].shift(-1) > df["close"]).astype(float)
    df["target_volatility"] = log_ret.shift(-1).abs()
    df["target_return_mag"] = log_ret.shift(-1)

    return df


class TradingDataset(Dataset):
    """
    PyTorch dataset for SSM-Transformer Hybrid model.

    Creates sliding windows of features and corresponding multi-task targets.
    """

    FEATURE_COLS = [
        "log_return", "rsi", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width", "atr", "obv",
        "volume_ratio", "sma_5_ratio", "sma_10_ratio",
        "sma_20_ratio", "sma_50_ratio",
    ]

    TARGET_COLS = ["target_direction", "target_volatility", "target_return_mag"]

    def __init__(self, df: pd.DataFrame, seq_length: int = 200):
        self.seq_length = seq_length

        # Extract features and targets, drop NaN rows
        feature_df = df[self.FEATURE_COLS + self.TARGET_COLS].dropna()
        feature_df = feature_df.replace([np.inf, -np.inf], 0.0)

        self.features = torch.tensor(
            feature_df[self.FEATURE_COLS].values, dtype=torch.float32
        )
        self.targets = {
            "direction": torch.tensor(
                feature_df["target_direction"].values, dtype=torch.float32
            ),
            "volatility": torch.tensor(
                feature_df["target_volatility"].values, dtype=torch.float32
            ),
            "return_mag": torch.tensor(
                feature_df["target_return_mag"].values, dtype=torch.float32
            ),
        }

        self.n_samples = len(feature_df) - seq_length
        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough data: {len(feature_df)} rows < seq_length {seq_length}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        end = idx + self.seq_length
        x = self.features[idx:end]
        targets = {
            "direction": self.targets["direction"][end],
            "volatility": self.targets["volatility"][end],
            "return_mag": self.targets["return_mag"][end],
        }
        return x, targets


def _collate_fn(batch):
    """Custom collate function for multi-task targets."""
    features = torch.stack([b[0] for b in batch])
    targets = {
        key: torch.stack([b[1][key] for b in batch])
        for key in batch[0][1].keys()
    }
    return features, targets


class SSMHybridDataLoader:
    """
    Data loader factory for SSM-Transformer Hybrid model.

    Supports fetching data from Bybit and Yahoo Finance,
    computing features and targets, and creating PyTorch DataLoaders.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        source: str = "bybit",
        seq_length: int = 200,
        feature_set: str = "full",
        interval: str = "60",
        limit: int = 1000,
    ):
        self.symbols = symbols or ["BTCUSDT"]
        self.source = source
        self.seq_length = seq_length
        self.feature_set = feature_set
        self.interval = interval
        self.limit = limit

    def load_data(self) -> pd.DataFrame:
        """Load and combine data for all symbols."""
        frames = []
        for symbol in self.symbols:
            if self.source == "bybit":
                df = fetch_bybit_klines(symbol, self.interval, self.limit)
            elif self.source == "yfinance":
                df = fetch_yfinance_data(symbol)
            else:
                df = _generate_synthetic_data(self.limit)

            df["symbol"] = symbol
            df = compute_features(df)
            df = compute_targets(df)
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def get_data_loaders(
        self,
        batch_size: int = 32,
        train_ratio: float = 0.8,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation data loaders.

        Args:
            batch_size: Batch size for DataLoader
            train_ratio: Fraction of data used for training

        Returns:
            Tuple of (train_loader, val_loader)
        """
        df = self.load_data()

        # Split by time (no future leakage)
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        train_dataset = TradingDataset(train_df, self.seq_length)
        val_dataset = TradingDataset(val_df, self.seq_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
        )

        return train_loader, val_loader
