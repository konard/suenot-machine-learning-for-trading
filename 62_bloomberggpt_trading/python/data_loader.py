"""
Data Loading Module

This module provides utilities for loading financial data from various sources
including Yahoo Finance and Bybit for cryptocurrency data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class MarketData:
    """Container for market data."""
    symbol: str
    ohlcv: pd.DataFrame
    source: str
    start_date: datetime
    end_date: datetime


class YahooFinanceLoader:
    """
    Load stock data from Yahoo Finance.

    Examples:
        >>> loader = YahooFinanceLoader()
        >>> data = loader.load("AAPL", period="1y")
        >>> print(data.ohlcv.head())
    """

    def __init__(self):
        self._yf = None

    def _ensure_yfinance(self):
        """Lazy import yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance required. Install with: pip install yfinance"
                )

    def load(
        self,
        symbol: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: Optional[str] = "1y",
        interval: str = "1d"
    ) -> MarketData:
        """
        Load stock data.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            start: Start date
            end: End date
            period: Period to load ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval ("1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo")

        Returns:
            MarketData with OHLCV DataFrame
        """
        self._ensure_yfinance()

        ticker = self._yf.Ticker(symbol)

        if start and end:
            df = ticker.history(start=start, end=end, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for {symbol}")

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]

        return MarketData(
            symbol=symbol,
            ohlcv=df,
            source="yahoo",
            start_date=df.index.min().to_pydatetime(),
            end_date=df.index.max().to_pydatetime()
        )

    def load_multiple(
        self,
        symbols: List[str],
        **kwargs
    ) -> Dict[str, MarketData]:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            **kwargs: Arguments passed to load()

        Returns:
            Dict mapping symbol to MarketData
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.load(symbol, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to load {symbol}: {e}")
        return result


class BybitLoader:
    """
    Load cryptocurrency data from Bybit.

    Examples:
        >>> loader = BybitLoader()
        >>> data = loader.load("BTCUSDT", days=30)
        >>> print(data.ohlcv.head())
    """

    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit loader.

        Args:
            testnet: Use testnet API
        """
        self.testnet = testnet
        self._client = None
        self.base_url = (
            "https://api-testnet.bybit.com" if testnet
            else "https://api.bybit.com"
        )

    def _ensure_client(self):
        """Initialize HTTP client."""
        if self._client is None:
            try:
                import requests
                self._client = requests.Session()
            except ImportError:
                raise ImportError("requests required. Install with: pip install requests")

    def load(
        self,
        symbol: str,
        interval: str = "60",  # Minutes
        days: int = 30,
        end_time: Optional[datetime] = None
    ) -> MarketData:
        """
        Load cryptocurrency data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval in minutes ("1", "5", "15", "30", "60", "240", "D", "W")
            days: Number of days of data to fetch
            end_time: End timestamp (default: now)

        Returns:
            MarketData with OHLCV DataFrame
        """
        self._ensure_client()

        end_time = end_time or datetime.now()
        start_time = end_time - timedelta(days=days)

        # Convert to milliseconds
        end_ts = int(end_time.timestamp() * 1000)
        start_ts = int(start_time.timestamp() * 1000)

        # Bybit API endpoint for spot market
        endpoint = f"{self.base_url}/v5/market/kline"

        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "start": start_ts,
            "end": end_ts,
            "limit": 1000
        }

        all_candles = []
        current_end = end_ts

        while current_end > start_ts:
            params["end"] = current_end

            response = self._client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                raise ValueError(f"Bybit API error: {data.get('retMsg')}")

            candles = data.get("result", {}).get("list", [])
            if not candles:
                break

            all_candles.extend(candles)

            # Update end time for pagination
            oldest_candle = min(candles, key=lambda x: int(x[0]))
            current_end = int(oldest_candle[0]) - 1

            if len(candles) < params["limit"]:
                break

        if not all_candles:
            raise ValueError(f"No data found for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)

        df = df.set_index("timestamp").sort_index()
        df = df[df.index >= start_time]

        return MarketData(
            symbol=symbol,
            ohlcv=df,
            source="bybit",
            start_date=df.index.min().to_pydatetime(),
            end_date=df.index.max().to_pydatetime()
        )

    def load_multiple(
        self,
        symbols: List[str],
        **kwargs
    ) -> Dict[str, MarketData]:
        """
        Load data for multiple symbols.
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.load(symbol, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to load {symbol}: {e}")
        return result


class DataLoader:
    """
    Unified data loader supporting multiple sources.

    Examples:
        >>> loader = DataLoader()
        >>> stock_data = loader.load("AAPL", source="yahoo")
        >>> crypto_data = loader.load("BTCUSDT", source="bybit")
    """

    def __init__(self):
        self._yahoo = None
        self._bybit = None

    def load(
        self,
        symbol: str,
        source: str = "auto",
        **kwargs
    ) -> MarketData:
        """
        Load market data from specified source.

        Args:
            symbol: Trading symbol
            source: Data source ("yahoo", "bybit", "auto")
            **kwargs: Source-specific arguments

        Returns:
            MarketData with OHLCV data
        """
        if source == "auto":
            # Guess source from symbol
            if symbol.endswith(("USDT", "USDC", "BTC", "ETH")):
                source = "bybit"
            else:
                source = "yahoo"

        if source == "yahoo":
            if self._yahoo is None:
                self._yahoo = YahooFinanceLoader()
            return self._yahoo.load(symbol, **kwargs)

        elif source == "bybit":
            if self._bybit is None:
                self._bybit = BybitLoader()
            return self._bybit.load(symbol, **kwargs)

        else:
            raise ValueError(f"Unknown source: {source}")

    def load_multiple(
        self,
        symbols: List[str],
        source: str = "auto",
        **kwargs
    ) -> Dict[str, MarketData]:
        """Load data for multiple symbols."""
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.load(symbol, source=source, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to load {symbol}: {e}")
        return result


def combine_prices(data: Dict[str, MarketData]) -> pd.DataFrame:
    """
    Combine multiple MarketData objects into a single price DataFrame.

    Args:
        data: Dict mapping symbol to MarketData

    Returns:
        DataFrame with symbols as columns and close prices
    """
    prices = {}
    for symbol, market_data in data.items():
        prices[symbol] = market_data.ohlcv["close"]

    df = pd.DataFrame(prices)
    df = df.dropna(how="all")

    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical features from OHLCV data.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with additional feature columns
    """
    result = df.copy()

    # Returns
    result["log_return"] = np.log(df["close"] / df["close"].shift(1))
    result["return"] = df["close"].pct_change()

    # Volatility
    result["volatility_20"] = result["log_return"].rolling(20).std()

    # Volume features
    result["volume_sma_20"] = df["volume"].rolling(20).mean()
    result["volume_ratio"] = df["volume"] / result["volume_sma_20"]

    # Price features
    result["sma_20"] = df["close"].rolling(20).mean()
    result["sma_50"] = df["close"].rolling(50).mean()
    result["price_sma_ratio"] = df["close"] / result["sma_20"]

    # Momentum
    result["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    result["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    # Range
    result["true_range"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )
    result["atr_14"] = result["true_range"].rolling(14).mean()

    return result


if __name__ == "__main__":
    print("Data Loader Demo\n" + "=" * 50)

    # Demo with mock data (avoids external API calls)
    print("\nGenerating synthetic stock data...")

    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")
    np.random.seed(42)

    mock_data = {}
    for symbol, initial_price in [("AAPL", 180), ("MSFT", 370), ("GOOGL", 140)]:
        returns = np.random.randn(len(dates)) * 0.02
        close = initial_price * (1 + returns).cumprod()

        df = pd.DataFrame({
            "open": close * (1 + np.random.randn(len(dates)) * 0.005),
            "high": close * (1 + abs(np.random.randn(len(dates)) * 0.01)),
            "low": close * (1 - abs(np.random.randn(len(dates)) * 0.01)),
            "close": close,
            "volume": np.random.randint(1e6, 1e8, len(dates))
        }, index=dates)

        mock_data[symbol] = MarketData(
            symbol=symbol,
            ohlcv=df,
            source="mock",
            start_date=dates[0].to_pydatetime(),
            end_date=dates[-1].to_pydatetime()
        )

        print(f"\n{symbol}:")
        print(f"  Period: {dates[0].date()} to {dates[-1].date()}")
        print(f"  Start price: ${df['close'].iloc[0]:.2f}")
        print(f"  End price: ${df['close'].iloc[-1]:.2f}")
        print(f"  Return: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1):.2%}")

    # Combine prices
    combined = combine_prices(mock_data)
    print(f"\nCombined price matrix shape: {combined.shape}")

    # Calculate features
    aapl_features = calculate_features(mock_data["AAPL"].ohlcv)
    print(f"\nFeatures calculated: {list(aapl_features.columns)}")
