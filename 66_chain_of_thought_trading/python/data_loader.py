"""
Data loaders for Chain-of-Thought Trading.

Supports loading data from:
- Yahoo Finance (stocks, ETFs)
- Bybit (cryptocurrency)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class OHLCV:
    """OHLCV (Open, High, Low, Close, Volume) data point."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., "AAPL", "BTCUSDT")
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (e.g., "1d", "1h", "15m")

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass

    @abstractmethod
    def get_latest(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Get latest OHLCV data for a symbol.

        Args:
            symbol: Trading symbol
            limit: Number of data points to retrieve

        Returns:
            DataFrame with OHLCV data
        """
        pass


class YahooFinanceLoader(DataLoader):
    """
    Load data from Yahoo Finance.

    Requires: yfinance package
    """

    def __init__(self):
        """Initialize Yahoo Finance loader."""
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise ImportError(
                "yfinance is required for YahooFinanceLoader. "
                "Install with: pip install yfinance"
            )

    def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Load OHLCV data from Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., "AAPL", "MSFT", "SPY")
            start_date: Start date
            end_date: End date
            interval: Data interval ("1d", "1h", "5m", etc.)

        Returns:
            DataFrame with OHLCV data
        """
        ticker = self._yf.Ticker(symbol)

        # Yahoo Finance interval mapping
        interval_map = {
            "1d": "1d",
            "1h": "1h",
            "4h": "4h",
            "15m": "15m",
            "5m": "5m",
            "1m": "1m",
        }

        yf_interval = interval_map.get(interval, interval)

        df = ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=yf_interval,
        )

        if df.empty:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # Standardize column names
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # Select only needed columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        return df

    def get_latest(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Get latest OHLCV data from Yahoo Finance.

        Args:
            symbol: Stock symbol
            limit: Number of data points (approximate)

        Returns:
            DataFrame with OHLCV data
        """
        # Calculate start date based on limit (assuming daily data)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(limit * 1.5))

        df = self.load(symbol, start_date, end_date, "1d")

        # Return last 'limit' rows
        return df.tail(limit).reset_index(drop=True)

    def get_info(self, symbol: str) -> dict:
        """
        Get company/asset information.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with asset information
        """
        ticker = self._yf.Ticker(symbol)
        return ticker.info


class BybitLoader(DataLoader):
    """
    Load data from Bybit cryptocurrency exchange.

    Requires: pybit package (optional) or uses HTTP API directly
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Bybit loader.

        Args:
            api_key: Optional API key (only needed for private endpoints)
            api_secret: Optional API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret

        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError(
                "requests is required for BybitLoader. "
                "Install with: pip install requests"
            )

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes."""
        mapping = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "12h": 720,
            "1d": 1440,
            "1D": 1440,
            "1w": 10080,
            "1W": 10080,
        }
        return mapping.get(interval, 60)

    def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Load OHLCV data from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
            start_date: Start date
            end_date: End date
            interval: Data interval ("1m", "5m", "15m", "1h", "4h", "1d")

        Returns:
            DataFrame with OHLCV data
        """
        # Convert interval to Bybit format
        interval_minutes = self._interval_to_minutes(interval)

        # Bybit uses milliseconds for timestamps
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        all_data = []
        current_start = start_ms

        while current_start < end_ms:
            # Bybit API endpoint for klines
            url = f"{self.BASE_URL}/v5/market/kline"

            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": str(interval_minutes) if interval_minutes < 1440 else "D",
                "start": current_start,
                "end": min(current_start + 200 * interval_minutes * 60 * 1000, end_ms),
                "limit": 200,
            }

            response = self._requests.get(url, params=params)
            data = response.json()

            if data.get("retCode") != 0:
                print(f"Bybit API error: {data.get('retMsg')}")
                break

            klines = data.get("result", {}).get("list", [])

            if not klines:
                break

            # Bybit returns data in reverse order (newest first)
            for kline in reversed(klines):
                all_data.append({
                    "timestamp": datetime.fromtimestamp(int(kline[0]) / 1000),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                })

            # Move to next batch
            current_start = int(klines[0][0]) + interval_minutes * 60 * 1000

        if not all_data:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(all_data)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Filter to exact date range
        df = df[
            (df["timestamp"] >= start_date) &
            (df["timestamp"] <= end_date)
        ].reset_index(drop=True)

        return df

    def get_latest(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Get latest OHLCV data from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            limit: Number of data points

        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.BASE_URL}/v5/market/kline"

        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": "D",
            "limit": min(limit, 200),
        }

        response = self._requests.get(url, params=params)
        data = response.json()

        if data.get("retCode") != 0:
            print(f"Bybit API error: {data.get('retMsg')}")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        klines = data.get("result", {}).get("list", [])

        rows = []
        for kline in reversed(klines):
            rows.append({
                "timestamp": datetime.fromtimestamp(int(kline[0]) / 1000),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
            })

        return pd.DataFrame(rows)

    def get_ticker(self, symbol: str) -> dict:
        """
        Get current ticker information.

        Args:
            symbol: Trading pair

        Returns:
            Dictionary with ticker information
        """
        url = f"{self.BASE_URL}/v5/market/tickers"

        params = {
            "category": "spot",
            "symbol": symbol,
        }

        response = self._requests.get(url, params=params)
        data = response.json()

        if data.get("retCode") != 0:
            return {}

        tickers = data.get("result", {}).get("list", [])

        if not tickers:
            return {}

        ticker = tickers[0]
        return {
            "symbol": ticker.get("symbol"),
            "last_price": float(ticker.get("lastPrice", 0)),
            "high_24h": float(ticker.get("highPrice24h", 0)),
            "low_24h": float(ticker.get("lowPrice24h", 0)),
            "volume_24h": float(ticker.get("volume24h", 0)),
            "price_change_24h": float(ticker.get("price24hPcnt", 0)) * 100,
        }


class MockDataLoader(DataLoader):
    """
    Mock data loader for testing without API connections.

    Generates realistic-looking synthetic data.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize mock data loader.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Generate mock OHLCV data.

        Args:
            symbol: Symbol (used to seed different patterns)
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            DataFrame with mock OHLCV data
        """
        # Calculate number of periods
        delta = end_date - start_date

        interval_hours = {
            "1m": 1/60,
            "5m": 5/60,
            "15m": 0.25,
            "1h": 1,
            "4h": 4,
            "1d": 24,
            "1w": 168,
        }

        hours_per_period = interval_hours.get(interval, 24)
        num_periods = int(delta.total_seconds() / 3600 / hours_per_period)

        if num_periods <= 0:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # Generate base price series with trend and noise
        base_price = 100.0  # Starting price

        # Use symbol hash to create different patterns
        symbol_seed = sum(ord(c) for c in symbol)
        np.random.seed(self.seed + symbol_seed)

        # Random walk with drift
        drift = np.random.uniform(-0.001, 0.001)
        volatility = np.random.uniform(0.01, 0.03)

        returns = np.random.normal(drift, volatility, num_periods)
        prices = base_price * np.cumprod(1 + returns)

        # Generate OHLCV data
        rows = []
        current_time = start_date
        time_delta = timedelta(hours=hours_per_period)

        for i, close_price in enumerate(prices):
            # Generate OHLC from close
            daily_volatility = abs(np.random.normal(0, volatility * close_price))

            open_price = prices[i-1] if i > 0 else base_price
            high_price = max(open_price, close_price) + daily_volatility * np.random.random()
            low_price = min(open_price, close_price) - daily_volatility * np.random.random()

            # Ensure high >= max(open, close) and low <= min(open, close)
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)

            # Generate volume (correlated with price movement)
            base_volume = 1000000
            price_change = abs(close_price - open_price) / open_price
            volume = base_volume * (1 + price_change * 10) * np.random.uniform(0.5, 1.5)

            rows.append({
                "timestamp": current_time,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 0),
            })

            current_time += time_delta

        return pd.DataFrame(rows)

    def get_latest(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Get latest mock OHLCV data.

        Args:
            symbol: Symbol
            limit: Number of data points

        Returns:
            DataFrame with mock OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=limit + 10)

        df = self.load(symbol, start_date, end_date, "1d")

        return df.tail(limit).reset_index(drop=True)


def create_loader(source: str, **kwargs) -> DataLoader:
    """
    Factory function to create appropriate data loader.

    Args:
        source: Data source ("yahoo", "bybit", "mock")
        **kwargs: Additional arguments for the loader

    Returns:
        DataLoader instance
    """
    loaders = {
        "yahoo": YahooFinanceLoader,
        "bybit": BybitLoader,
        "mock": MockDataLoader,
    }

    loader_class = loaders.get(source.lower())

    if loader_class is None:
        raise ValueError(f"Unknown data source: {source}. Available: {list(loaders.keys())}")

    return loader_class(**kwargs)


# Utility functions for data preprocessing
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to OHLCV DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional indicator columns
    """
    df = df.copy()

    # Simple Moving Averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()

    # Exponential Moving Averages
    df["ema_12"] = df["close"].ewm(span=12).mean()
    df["ema_26"] = df["close"].ewm(span=26).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std

    # ATR (Average True Range)
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=14).mean()

    # Volume indicators
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # Price change
    df["price_change"] = df["close"].pct_change()
    df["price_change_5d"] = df["close"].pct_change(5)
    df["price_change_20d"] = df["close"].pct_change(20)

    return df


def prepare_for_analysis(
    df: pd.DataFrame,
    lookback: int = 100,
) -> dict:
    """
    Prepare data for Chain-of-Thought analysis.

    Args:
        df: DataFrame with OHLCV and indicators
        lookback: Number of periods to include

    Returns:
        Dictionary with prepared data for analysis
    """
    # Get recent data
    recent = df.tail(lookback).copy()
    latest = recent.iloc[-1]

    return {
        "current_price": latest["close"],
        "price_change_1d": latest.get("price_change", 0) * 100,
        "price_change_5d": latest.get("price_change_5d", 0) * 100,
        "price_change_20d": latest.get("price_change_20d", 0) * 100,
        "rsi": latest.get("rsi", 50),
        "macd": latest.get("macd", 0),
        "macd_signal": latest.get("macd_signal", 0),
        "sma_20": latest.get("sma_20", latest["close"]),
        "sma_50": latest.get("sma_50", latest["close"]),
        "sma_200": latest.get("sma_200", latest["close"]),
        "atr": latest.get("atr", 0),
        "volume_ratio": latest.get("volume_ratio", 1),
        "bb_upper": latest.get("bb_upper", latest["close"] * 1.02),
        "bb_lower": latest.get("bb_lower", latest["close"] * 0.98),
        "high_20d": recent["high"].max(),
        "low_20d": recent["low"].min(),
        "avg_volume": recent["volume"].mean(),
        "historical_data": recent.to_dict("records"),
    }
