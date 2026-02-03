"""Data loading utilities for stock and cryptocurrency data.

Provides data loaders for:
- Stock data via yfinance
- Cryptocurrency data via Bybit API
- Feature engineering for financial time series
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class FeatureEngineering:
    """Technical indicator computation for financial data."""

    @staticmethod
    def compute_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """Compute log returns."""
        return np.log(prices / prices.shift(periods))

    @staticmethod
    def compute_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """Compute rolling volatility."""
        return returns.rolling(window=window).std()

    @staticmethod
    def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def compute_macd(
        prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute MACD and signal line."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal

    @staticmethod
    def compute_bollinger_bands(
        prices: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return upper, sma, lower

    @staticmethod
    def normalize_features(df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """Normalize features using z-score or min-max."""
        if method == "zscore":
            return (df - df.mean()) / (df.std() + 1e-9)
        elif method == "minmax":
            return (df - df.min()) / (df.max() - df.min() + 1e-9)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class StockDataLoader:
    """Load and process stock data from Yahoo Finance."""

    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        """Initialize stock data loader.

        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        if not HAS_YFINANCE:
            raise ImportError("yfinance is required for stock data loading")

        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}

    def load_data(self) -> pd.DataFrame:
        """Load OHLCV data for all symbols."""
        all_data = []

        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=self.start_date, end=self.end_date)
            df["symbol"] = symbol
            all_data.append(df)
            self.data[symbol] = df

        return pd.concat(all_data)

    def prepare_features(
        self, symbol: str, seq_len: int = 60, pred_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets for a single symbol.

        Args:
            symbol: Stock symbol
            seq_len: Sequence length (lookback window)
            pred_horizon: Prediction horizon in periods

        Returns:
            Tuple of (features, returns, directions, volatilities)
        """
        if symbol not in self.data:
            self.load_data()

        df = self.data[symbol].copy()
        fe = FeatureEngineering()

        # Compute features
        df["returns"] = fe.compute_returns(df["Close"])
        df["volatility"] = fe.compute_volatility(df["returns"])
        df["rsi"] = fe.compute_rsi(df["Close"])
        macd, macd_signal = fe.compute_macd(df["Close"])
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        upper, middle, lower = fe.compute_bollinger_bands(df["Close"])
        df["bb_position"] = (df["Close"] - lower) / (upper - lower + 1e-9)

        # Select features
        feature_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "returns", "volatility", "rsi"
        ]
        df_features = df[feature_cols].dropna()

        # Normalize
        df_features = fe.normalize_features(df_features)

        # Compute targets
        future_returns = df["returns"].shift(-pred_horizon)
        future_direction = (future_returns > 0).astype(float)
        future_volatility = df["volatility"].shift(-pred_horizon)

        # Drop rows with NaN targets
        valid_idx = ~(
            future_returns.isna() |
            future_direction.isna() |
            future_volatility.isna() |
            df_features.isna().any(axis=1)
        )

        df_features = df_features[valid_idx]
        future_returns = future_returns[valid_idx]
        future_direction = future_direction[valid_idx]
        future_volatility = future_volatility[valid_idx]

        # Create sequences
        features, returns, directions, volatilities = [], [], [], []
        values = df_features.values
        ret_vals = future_returns.values
        dir_vals = future_direction.values
        vol_vals = future_volatility.values

        for i in range(len(values) - seq_len):
            features.append(values[i:i + seq_len])
            returns.append(ret_vals[i + seq_len - 1])
            directions.append(dir_vals[i + seq_len - 1])
            volatilities.append(vol_vals[i + seq_len - 1])

        return (
            np.array(features, dtype=np.float32),
            np.array(returns, dtype=np.float32),
            np.array(directions, dtype=np.float32),
            np.array(volatilities, dtype=np.float32),
        )


class BybitDataLoader:
    """Load and process cryptocurrency data from Bybit API."""

    BASE_URL = "https://api.bybit.com"

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "60"):
        """Initialize Bybit data loader.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval in minutes ('1', '5', '15', '60', '240', 'D')
        """
        if not HAS_REQUESTS:
            raise ImportError("requests is required for Bybit data loading")

        self.symbol = symbol
        self.interval = interval
        self.data = None

    def load_data(self, limit: int = 1000) -> pd.DataFrame:
        """Load kline (candlestick) data from Bybit.

        Args:
            limit: Number of candles to fetch (max 1000)

        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": limit,
        }

        response = requests.get(endpoint, params=params)
        response.raise_for_status()

        data = response.json()
        if data["retCode"] != 0:
            raise ValueError(f"API error: {data['retMsg']}")

        # Parse kline data
        klines = data["result"]["list"]
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)

        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })

        self.data = df
        return df

    def prepare_features(
        self, seq_len: int = 60, pred_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets.

        Args:
            seq_len: Sequence length (lookback window)
            pred_horizon: Prediction horizon in periods

        Returns:
            Tuple of (features, returns, directions, volatilities)
        """
        if self.data is None:
            self.load_data()

        df = self.data.copy()
        fe = FeatureEngineering()

        # Compute features
        df["returns"] = fe.compute_returns(df["Close"])
        df["volatility"] = fe.compute_volatility(df["returns"])
        df["rsi"] = fe.compute_rsi(df["Close"])
        macd, macd_signal = fe.compute_macd(df["Close"])
        df["macd"] = macd
        df["macd_signal"] = macd_signal

        # Select features
        feature_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "returns", "volatility", "rsi"
        ]
        df_features = df[feature_cols].dropna()

        # Normalize
        df_features = fe.normalize_features(df_features)

        # Compute targets
        future_returns = df["returns"].shift(-pred_horizon)
        future_direction = (future_returns > 0).astype(float)
        future_volatility = df["volatility"].shift(-pred_horizon)

        # Drop rows with NaN targets
        valid_idx = ~(
            future_returns.isna() |
            future_direction.isna() |
            future_volatility.isna() |
            df_features.isna().any(axis=1)
        )

        df_features = df_features[valid_idx]
        future_returns = future_returns[valid_idx]
        future_direction = future_direction[valid_idx]
        future_volatility = future_volatility[valid_idx]

        # Create sequences
        features, returns, directions, volatilities = [], [], [], []
        values = df_features.values
        ret_vals = future_returns.values
        dir_vals = future_direction.values
        vol_vals = future_volatility.values

        for i in range(len(values) - seq_len):
            features.append(values[i:i + seq_len])
            returns.append(ret_vals[i + seq_len - 1])
            directions.append(dir_vals[i + seq_len - 1])
            volatilities.append(vol_vals[i + seq_len - 1])

        return (
            np.array(features, dtype=np.float32),
            np.array(returns, dtype=np.float32),
            np.array(directions, dtype=np.float32),
            np.array(volatilities, dtype=np.float32),
        )


class TradingDataset(Dataset):
    """PyTorch Dataset for financial time series."""

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        directions: np.ndarray,
        volatilities: np.ndarray,
    ):
        """Initialize dataset.

        Args:
            features: Feature array of shape (n_samples, seq_len, n_features)
            returns: Return targets of shape (n_samples,)
            directions: Direction targets of shape (n_samples,)
            volatilities: Volatility targets of shape (n_samples,)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32)
        self.directions = torch.tensor(directions, dtype=torch.float32)
        self.volatilities = torch.tensor(volatilities, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.features[idx],
            self.returns[idx],
            self.directions[idx],
            self.volatilities[idx],
        )


def create_data_loaders(
    features: np.ndarray,
    returns: np.ndarray,
    directions: np.ndarray,
    volatilities: np.ndarray,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.

    Args:
        features: Feature array
        returns: Return targets
        directions: Direction targets
        volatilities: Volatility targets
        batch_size: Batch size
        train_ratio: Fraction for training
        val_ratio: Fraction for validation

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    n_samples = len(features)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # Time-series split (no shuffling)
    train_dataset = TradingDataset(
        features[:train_end],
        returns[:train_end],
        directions[:train_end],
        volatilities[:train_end],
    )
    val_dataset = TradingDataset(
        features[train_end:val_end],
        returns[train_end:val_end],
        directions[train_end:val_end],
        volatilities[train_end:val_end],
    )
    test_dataset = TradingDataset(
        features[val_end:],
        returns[val_end:],
        directions[val_end:],
        volatilities[val_end:],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
