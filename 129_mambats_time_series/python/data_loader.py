"""
Financial Data Loader for MambaTS

This module provides data loading and preprocessing utilities for
financial time series, supporting multiple data sources including
Bybit cryptocurrency exchange and Yahoo Finance.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict, Union
from datetime import datetime, timedelta
import requests
import time


class TechnicalIndicators:
    """
    Technical indicators for feature engineering.
    """

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        series: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def volatility(series: pd.Series, period: int = 20) -> pd.Series:
        """Historical volatility (rolling standard deviation of returns)"""
        returns = series.pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)


class BybitDataFetcher:
    """
    Fetches historical cryptocurrency data from Bybit exchange.
    """

    BASE_URL = "https://api.bybit.com/v5"
    RATE_LIMIT = 0.5  # seconds between requests

    INTERVAL_MAP = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
        "1M": "M",
    }

    def __init__(self):
        self.session = requests.Session()
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        self.last_request_time = time.time()

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """
        Fetch historical candlestick data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Time interval (e.g., "1h", "4h", "1d")
            start_time: Start datetime
            end_time: End datetime

        Returns:
            DataFrame with OHLCV data
        """
        if interval not in self.INTERVAL_MAP:
            raise ValueError(f"Invalid interval: {interval}")

        all_data = []
        current_start = start_time

        while current_start < end_time:
            self._rate_limit()

            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": self.INTERVAL_MAP[interval],
                "start": int(current_start.timestamp() * 1000),
                "end": int(end_time.timestamp() * 1000),
                "limit": 1000,
            }

            response = self.session.get(f"{self.BASE_URL}/market/kline", params=params)
            data = response.json()

            if data["retCode"] != 0:
                raise Exception(f"Bybit API error: {data['retMsg']}")

            if not data["result"]["list"]:
                break

            for candle in data["result"]["list"]:
                all_data.append(
                    {
                        "timestamp": pd.to_datetime(int(candle[0]), unit="ms"),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                        "turnover": float(candle[6]),
                    }
                )

            # Move to next batch
            if len(data["result"]["list"]) < 1000:
                break
            last_ts = int(data["result"]["list"][-1][0])
            current_start = pd.to_datetime(last_ts, unit="ms") + timedelta(milliseconds=1)

        df = pd.DataFrame(all_data)
        if not df.empty:
            df = df.sort_values("timestamp").drop_duplicates("timestamp")
            df = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]
        return df


class YahooFinanceDataFetcher:
    """
    Fetches historical stock data from Yahoo Finance using yfinance.
    """

    def get_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """
        Fetch historical stock data.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            start_time: Start datetime
            end_time: End datetime
            interval: Time interval

        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_time, end=end_time, interval=interval)
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            df = df.rename(columns={"date": "timestamp", "datetime": "timestamp"})
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except ImportError:
            raise ImportError("yfinance is required for Yahoo Finance data. Install with: pip install yfinance")


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting.
    """

    def __init__(
        self,
        data: np.ndarray,
        lookback: int,
        forecast_horizon: int,
        stride: int = 1,
        target_idx: Optional[int] = None,
    ):
        """
        Args:
            data: Array of shape (n_samples, n_features)
            lookback: Number of historical steps to use as input
            forecast_horizon: Number of steps to predict
            stride: Step size for creating samples
            target_idx: Index of target variable (None = all variables)
        """
        self.data = data
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.target_idx = target_idx

        # Calculate valid indices
        self.valid_indices = list(
            range(
                lookback,
                len(data) - forecast_horizon + 1,
                stride,
            )
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = self.valid_indices[idx]

        # Input: lookback window
        x = self.data[i - self.lookback : i]

        # Target: forecast horizon
        if self.target_idx is not None:
            y = self.data[i : i + self.forecast_horizon, self.target_idx]
        else:
            y = self.data[i : i + self.forecast_horizon]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class FinancialDataLoader:
    """
    High-level data loader for financial time series.

    Handles data fetching, preprocessing, feature engineering,
    and DataLoader creation.
    """

    def __init__(
        self,
        symbols: Union[str, List[str]],
        interval: str = "1h",
        lookback: int = 720,
        forecast_horizon: int = 24,
        source: str = "bybit",
        add_indicators: bool = True,
        normalize: bool = True,
    ):
        """
        Args:
            symbols: Symbol or list of symbols to fetch
            interval: Time interval
            lookback: Historical window size
            forecast_horizon: Prediction horizon
            source: Data source ('bybit' or 'yahoo')
            add_indicators: Whether to add technical indicators
            normalize: Whether to normalize the data
        """
        self.symbols = [symbols] if isinstance(symbols, str) else symbols
        self.interval = interval
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.source = source
        self.add_indicators = add_indicators
        self.normalize = normalize

        self.scalers: Dict[str, Tuple[float, float]] = {}
        self._data: Optional[pd.DataFrame] = None

        if source == "bybit":
            self.fetcher = BybitDataFetcher()
        elif source == "yahoo":
            self.fetcher = YahooFinanceDataFetcher()
        else:
            raise ValueError(f"Unknown source: {source}")

    def fetch_data(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """
        Fetch and preprocess data for all symbols.
        """
        all_dfs = []

        for symbol in self.symbols:
            print(f"Fetching {symbol}...")

            if self.source == "bybit":
                df = self.fetcher.get_klines(symbol, self.interval, start_time, end_time)
            else:
                df = self.fetcher.get_data(symbol, start_time, end_time, self.interval)

            if df.empty:
                print(f"Warning: No data for {symbol}")
                continue

            # Add symbol column
            df["symbol"] = symbol

            # Add technical indicators
            if self.add_indicators:
                df = self._add_indicators(df)

            all_dfs.append(df)

        if not all_dfs:
            raise ValueError("No data fetched for any symbol")

        # Combine all dataframes
        if len(all_dfs) == 1:
            self._data = all_dfs[0]
        else:
            # Merge on timestamp
            self._data = all_dfs[0]
            for df in all_dfs[1:]:
                suffix = f"_{df['symbol'].iloc[0]}"
                self._data = self._data.merge(
                    df.drop("symbol", axis=1),
                    on="timestamp",
                    suffixes=("", suffix),
                )

        return self._data

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        """
        close = df["close"]

        # Moving averages
        df["sma_10"] = TechnicalIndicators.sma(close, 10)
        df["sma_20"] = TechnicalIndicators.sma(close, 20)
        df["ema_10"] = TechnicalIndicators.ema(close, 10)
        df["ema_20"] = TechnicalIndicators.ema(close, 20)

        # RSI
        df["rsi"] = TechnicalIndicators.rsi(close, 14)

        # MACD
        macd, signal, hist = TechnicalIndicators.macd(close)
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist

        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(close)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        df["bb_width"] = (upper - lower) / middle

        # ATR
        df["atr"] = TechnicalIndicators.atr(df["high"], df["low"], close)

        # Volatility
        df["volatility"] = TechnicalIndicators.volatility(close)

        # Returns
        df["returns"] = close.pct_change()
        df["log_returns"] = np.log(close / close.shift(1))

        # Price position
        df["price_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        return df

    def prepare_features(
        self,
        feature_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Prepare feature matrix for training.

        Args:
            feature_cols: List of column names to use as features.
                         If None, uses default features.

        Returns:
            Numpy array of shape (n_samples, n_features)
        """
        if self._data is None:
            raise ValueError("No data loaded. Call fetch_data() first.")

        if feature_cols is None:
            # Default features
            feature_cols = [
                "open", "high", "low", "close", "volume",
                "returns", "rsi", "macd", "macd_signal",
                "bb_width", "atr", "volatility",
            ]

        # Filter to available columns
        available_cols = [c for c in feature_cols if c in self._data.columns]
        if not available_cols:
            raise ValueError(f"None of the specified features are available")

        # Extract features
        df = self._data[available_cols].copy()

        # Handle missing values
        df = df.dropna()

        # Normalize if requested
        if self.normalize:
            for col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                self.scalers[col] = (mean, std)
                df[col] = (df[col] - mean) / (std + 1e-8)

        return df.values

    def get_dataloaders(
        self,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        feature_cols: Optional[List[str]] = None,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test DataLoaders.

        Args:
            batch_size: Batch size
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            feature_cols: Feature columns to use
            num_workers: Number of workers for data loading

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        data = self.prepare_features(feature_cols)

        n_samples = len(data)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        train_dataset = TimeSeriesDataset(
            train_data, self.lookback, self.forecast_horizon
        )
        val_dataset = TimeSeriesDataset(
            val_data, self.lookback, self.forecast_horizon
        )
        test_dataset = TimeSeriesDataset(
            test_data, self.lookback, self.forecast_horizon
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    from datetime import datetime, timedelta

    # Create loader
    loader = FinancialDataLoader(
        symbols=["BTCUSDT"],
        interval="1h",
        lookback=168,  # 1 week
        forecast_horizon=24,  # 1 day
        source="bybit",
    )

    # Fetch data (last 30 days)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30)

    print("Fetching data...")
    try:
        df = loader.fetch_data(start_time, end_time)
        print(f"Fetched {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")

        # Prepare features
        features = loader.prepare_features()
        print(f"Feature matrix shape: {features.shape}")

        # Create dataloaders
        train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=32)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Check one batch
        for x, y in train_loader:
            print(f"Input shape: {x.shape}")
            print(f"Target shape: {y.shape}")
            break

    except Exception as e:
        print(f"Error: {e}")
        print("Note: This example requires network access to Bybit API")
