"""
Data Loading Module for Multi-Agent LLM Trading

This module provides utilities for loading financial data from various sources
including Yahoo Finance for stocks and Bybit for cryptocurrency data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for market data."""
    symbol: str
    ohlcv: pd.DataFrame
    source: str
    start_date: datetime
    end_date: datetime
    metadata: Dict = field(default_factory=dict)

    @property
    def close(self) -> pd.Series:
        """Get close prices."""
        return self.ohlcv["close"]

    @property
    def returns(self) -> pd.Series:
        """Calculate returns."""
        return self.close.pct_change()

    @property
    def log_returns(self) -> pd.Series:
        """Calculate log returns."""
        return np.log(self.close / self.close.shift(1))


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
            period: Period to load if start/end not specified
            interval: Data interval

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

        # Get company info for metadata
        try:
            info = ticker.info
            metadata = {
                "name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "currency": info.get("currency", "USD"),
            }
        except Exception:
            metadata = {"name": symbol}

        return MarketData(
            symbol=symbol,
            ohlcv=df,
            source="yahoo",
            start_date=df.index.min().to_pydatetime(),
            end_date=df.index.max().to_pydatetime(),
            metadata=metadata
        )

    def load_multiple(
        self,
        symbols: List[str],
        **kwargs
    ) -> Dict[str, MarketData]:
        """Load data for multiple symbols."""
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.load(symbol, **kwargs)
                logger.info(f"Loaded {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
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
        interval: str = "60",
        days: int = 30,
        end_time: Optional[datetime] = None,
        category: str = "spot"
    ) -> MarketData:
        """
        Load cryptocurrency data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval in minutes ("1", "5", "15", "30", "60", "240", "D", "W")
            days: Number of days of data to fetch
            end_time: End timestamp (default: now)
            category: Market category ("spot", "linear", "inverse")

        Returns:
            MarketData with OHLCV DataFrame
        """
        self._ensure_client()

        end_time = end_time or datetime.now()
        start_time = end_time - timedelta(days=days)

        # Convert to milliseconds
        end_ts = int(end_time.timestamp() * 1000)
        start_ts = int(start_time.timestamp() * 1000)

        endpoint = f"{self.base_url}/v5/market/kline"

        params = {
            "category": category,
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
            end_date=df.index.max().to_pydatetime(),
            metadata={"category": category}
        )

    def load_multiple(
        self,
        symbols: List[str],
        **kwargs
    ) -> Dict[str, MarketData]:
        """Load data for multiple symbols."""
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.load(symbol, **kwargs)
                logger.info(f"Loaded {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
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
            if symbol.endswith(("USDT", "USDC", "BTC", "ETH", "PERP")):
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
                logger.warning(f"Failed to load {symbol}: {e}")
        return result


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical indicators from OHLCV data.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with additional indicator columns
    """
    result = df.copy()

    # Returns
    result["return"] = df["close"].pct_change()
    result["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Moving Averages
    result["sma_20"] = df["close"].rolling(20).mean()
    result["sma_50"] = df["close"].rolling(50).mean()
    result["sma_200"] = df["close"].rolling(200).mean()
    result["ema_12"] = df["close"].ewm(span=12).mean()
    result["ema_26"] = df["close"].ewm(span=26).mean()

    # MACD
    result["macd"] = result["ema_12"] - result["ema_26"]
    result["macd_signal"] = result["macd"].ewm(span=9).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]

    # Bollinger Bands
    result["bb_middle"] = result["sma_20"]
    bb_std = df["close"].rolling(20).std()
    result["bb_upper"] = result["bb_middle"] + 2 * bb_std
    result["bb_lower"] = result["bb_middle"] - 2 * bb_std
    result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    result["rsi"] = 100 - (100 / (1 + rs))

    # Volatility
    result["volatility_20"] = result["log_return"].rolling(20).std() * np.sqrt(252)

    # Volume indicators
    result["volume_sma_20"] = df["volume"].rolling(20).mean()
    result["volume_ratio"] = df["volume"] / result["volume_sma_20"]

    # Momentum
    result["momentum_10"] = df["close"] / df["close"].shift(10) - 1
    result["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    # ATR
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    result["atr_14"] = true_range.rolling(14).mean()

    return result


def create_mock_data(
    symbol: str = "MOCK",
    days: int = 252,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    seed: Optional[int] = None
) -> MarketData:
    """
    Create synthetic market data for testing.

    Args:
        symbol: Symbol name
        days: Number of trading days
        initial_price: Starting price
        volatility: Daily volatility
        seed: Random seed for reproducibility

    Returns:
        MarketData with synthetic OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start="2024-01-01", periods=days, freq="B")
    returns = np.random.randn(days) * volatility
    close = initial_price * (1 + returns).cumprod()

    df = pd.DataFrame({
        "open": close * (1 + np.random.randn(days) * 0.005),
        "high": close * (1 + abs(np.random.randn(days) * 0.01)),
        "low": close * (1 - abs(np.random.randn(days) * 0.01)),
        "close": close,
        "volume": np.random.randint(1e6, 1e8, days).astype(float)
    }, index=dates)

    # Ensure high >= close >= low
    df["high"] = df[["high", "close", "open"]].max(axis=1)
    df["low"] = df[["low", "close", "open"]].min(axis=1)

    return MarketData(
        symbol=symbol,
        ohlcv=df,
        source="mock",
        start_date=dates[0].to_pydatetime(),
        end_date=dates[-1].to_pydatetime(),
        metadata={"type": "synthetic"}
    )


if __name__ == "__main__":
    print("Data Loader Demo\n" + "=" * 50)

    # Demo with mock data
    print("\nGenerating synthetic data...")

    mock_data = create_mock_data("DEMO", days=252, initial_price=100, seed=42)
    print(f"\nSymbol: {mock_data.symbol}")
    print(f"Period: {mock_data.start_date.date()} to {mock_data.end_date.date()}")
    print(f"Start price: ${mock_data.close.iloc[0]:.2f}")
    print(f"End price: ${mock_data.close.iloc[-1]:.2f}")
    print(f"Total return: {(mock_data.close.iloc[-1] / mock_data.close.iloc[0] - 1):.2%}")

    # Calculate indicators
    with_indicators = calculate_technical_indicators(mock_data.ohlcv)
    print(f"\nTechnical indicators calculated: {len(with_indicators.columns)} columns")
    print(f"Latest RSI: {with_indicators['rsi'].iloc[-1]:.1f}")
    print(f"Latest 20-day volatility: {with_indicators['volatility_20'].iloc[-1]:.2%}")
