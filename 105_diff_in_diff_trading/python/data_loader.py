"""Data loaders for stock market and cryptocurrency data.

Provides unified interfaces for loading financial data from:
- Yahoo Finance (stock market data)
- Bybit API (cryptocurrency data)
- CSV files (custom data sources)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class OHLCV:
    """OHLCV candlestick data structure."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical data for a symbol.

        Args:
            symbol: Ticker symbol
            start_date: Start of data range
            end_date: End of data range
            interval: Data interval (e.g., "1d", "1h", "5m")

        Returns:
            DataFrame with OHLCV data
        """
        pass

    @abstractmethod
    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        pass


class YFinanceDataLoader(DataLoader):
    """Data loader for Yahoo Finance (stock market data).

    Uses yfinance library to fetch historical stock data.
    Falls back to synthetic data if yfinance is not available.
    """

    def __init__(self):
        try:
            import yfinance as yf
            self._yf = yf
            self._available = True
        except ImportError:
            self._yf = None
            self._available = False

    def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical stock data from Yahoo Finance."""
        if not self._available:
            return self._generate_synthetic_data(symbol, start_date, end_date, interval)

        try:
            ticker = self._yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                return self._generate_synthetic_data(symbol, start_date, end_date, interval)

            # Normalize column names
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            elif "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})

            df["symbol"] = symbol
            df["return"] = df["close"].pct_change()

            return df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "return"]]

        except Exception:
            return self._generate_synthetic_data(symbol, start_date, end_date, interval)

    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stock symbols."""
        result = {}
        for symbol in symbols:
            result[symbol] = self.fetch(symbol, start_date, end_date, interval)
        return result

    def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """Generate synthetic stock data for testing."""
        # Determine number of periods
        if interval == "1d":
            delta = timedelta(days=1)
        elif interval == "1h":
            delta = timedelta(hours=1)
        elif interval == "5m":
            delta = timedelta(minutes=5)
        else:
            delta = timedelta(days=1)

        dates = []
        current = start_date
        while current <= end_date:
            if interval == "1d" and current.weekday() < 5:  # Skip weekends
                dates.append(current)
            elif interval != "1d":
                dates.append(current)
            current += delta

        n = len(dates)
        if n == 0:
            return pd.DataFrame()

        # Generate random walk prices
        np.random.seed(hash(symbol) % (2**32))
        returns = np.random.normal(0.0002, 0.02, n)
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate OHLCV
        data = {
            "timestamp": dates,
            "symbol": [symbol] * n,
            "open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            "high": prices * (1 + np.random.uniform(0, 0.02, n)),
            "low": prices * (1 - np.random.uniform(0, 0.02, n)),
            "close": prices,
            "volume": np.random.uniform(1e6, 1e8, n),
            "return": np.concatenate([[np.nan], returns[1:]]),
        }

        return pd.DataFrame(data)


class BybitDataLoader(DataLoader):
    """Data loader for Bybit cryptocurrency exchange.

    Uses Bybit public API to fetch historical kline (candlestick) data.
    Falls back to synthetic data if API is unavailable.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        try:
            import requests
            self._requests = requests
            self._available = True
        except ImportError:
            self._requests = None
            self._available = False

    def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical crypto data from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
            start_date: Start of data range
            end_date: End of data range
            interval: Kline interval ("1", "5", "15", "60", "D", "W")

        Returns:
            DataFrame with OHLCV data
        """
        if not self._available:
            return self._generate_synthetic_data(symbol, start_date, end_date, interval)

        # Convert interval to Bybit format
        interval_map = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "1h": "60",
            "4h": "240",
            "1d": "D",
            "1w": "W",
        }
        bybit_interval = interval_map.get(interval, "D")

        try:
            # Fetch kline data from Bybit API
            url = f"{self.BASE_URL}/v5/market/kline"

            all_candles = []
            end_ts = int(end_date.timestamp() * 1000)
            start_ts = int(start_date.timestamp() * 1000)

            while end_ts > start_ts:
                params = {
                    "category": "spot",
                    "symbol": symbol,
                    "interval": bybit_interval,
                    "limit": 1000,
                    "end": end_ts,
                }

                response = self._requests.get(url, params=params, timeout=10)
                data = response.json()

                if data.get("retCode") != 0:
                    break

                candles = data.get("result", {}).get("list", [])
                if not candles:
                    break

                all_candles.extend(candles)

                # Move to earlier data
                oldest_ts = int(candles[-1][0])
                if oldest_ts >= end_ts:
                    break
                end_ts = oldest_ts - 1

            if not all_candles:
                return self._generate_synthetic_data(symbol, start_date, end_date, interval)

            # Parse candle data
            records = []
            for candle in all_candles:
                if len(candle) >= 6:
                    ts_ms = int(candle[0])
                    records.append({
                        "timestamp": datetime.fromtimestamp(ts_ms / 1000),
                        "symbol": symbol,
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    })

            df = pd.DataFrame(records)
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Filter to requested date range
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

            # Compute returns
            df["return"] = df["close"].pct_change()

            return df

        except Exception:
            return self._generate_synthetic_data(symbol, start_date, end_date, interval)

    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple crypto symbols."""
        result = {}
        for symbol in symbols:
            result[symbol] = self.fetch(symbol, start_date, end_date, interval)
        return result

    def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """Generate synthetic crypto data for testing."""
        # Crypto trades 24/7
        if interval in ["1m", "1"]:
            delta = timedelta(minutes=1)
        elif interval in ["5m", "5"]:
            delta = timedelta(minutes=5)
        elif interval in ["15m", "15"]:
            delta = timedelta(minutes=15)
        elif interval in ["1h", "60"]:
            delta = timedelta(hours=1)
        elif interval in ["4h", "240"]:
            delta = timedelta(hours=4)
        elif interval in ["1d", "D"]:
            delta = timedelta(days=1)
        else:
            delta = timedelta(days=1)

        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += delta

        n = len(dates)
        if n == 0:
            return pd.DataFrame()

        # Set base price based on symbol
        if "BTC" in symbol:
            base_price = 40000
            volatility = 0.03
        elif "ETH" in symbol:
            base_price = 2500
            volatility = 0.04
        else:
            base_price = 100
            volatility = 0.05

        np.random.seed(hash(symbol) % (2**32))
        returns = np.random.normal(0.0001, volatility, n)
        prices = base_price * np.exp(np.cumsum(returns))

        data = {
            "timestamp": dates,
            "symbol": [symbol] * n,
            "open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            "high": prices * (1 + np.random.uniform(0, 0.02, n)),
            "low": prices * (1 - np.random.uniform(0, 0.02, n)),
            "close": prices,
            "volume": np.random.uniform(1e4, 1e7, n) * base_price / 100,
            "return": np.concatenate([[np.nan], returns[1:]]),
        }

        return pd.DataFrame(data)


def prepare_did_data(
    treated_data: pd.DataFrame,
    control_data: pd.DataFrame,
    event_date: datetime,
    pre_window: int = 30,
    post_window: int = 30,
) -> pd.DataFrame:
    """Prepare panel data for DiD analysis.

    Args:
        treated_data: DataFrame with treated group data
        control_data: DataFrame with control group data
        event_date: Date of the treatment event
        pre_window: Number of days before event
        post_window: Number of days after event

    Returns:
        Combined panel data ready for DiD estimation
    """
    # Define time window
    start_date = event_date - timedelta(days=pre_window)
    end_date = event_date + timedelta(days=post_window)

    # Filter and label treated data
    treated = treated_data[
        (treated_data["timestamp"] >= start_date) &
        (treated_data["timestamp"] <= end_date)
    ].copy()
    treated["treated"] = 1
    treated["post_treatment"] = (treated["timestamp"] >= event_date).astype(int)

    # Filter and label control data
    control = control_data[
        (control_data["timestamp"] >= start_date) &
        (control_data["timestamp"] <= end_date)
    ].copy()
    control["treated"] = 0
    control["post_treatment"] = (control["timestamp"] >= event_date).astype(int)

    # Combine
    panel = pd.concat([treated, control], ignore_index=True)
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Add event time (days relative to event)
    panel["event_time"] = (panel["timestamp"] - event_date).dt.days

    return panel


def compute_cumulative_returns(
    data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute cumulative returns from a base date.

    Args:
        data: DataFrame with price data
        start_date: Base date for cumulative returns
        end_date: End date
        price_col: Column containing prices

    Returns:
        DataFrame with cumulative returns added
    """
    df = data.copy()
    df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

    for symbol in df["symbol"].unique():
        mask = df["symbol"] == symbol
        symbol_data = df.loc[mask, price_col]
        base_price = symbol_data.iloc[0] if len(symbol_data) > 0 else 1.0
        df.loc[mask, "cumulative_return"] = symbol_data / base_price - 1

    return df
