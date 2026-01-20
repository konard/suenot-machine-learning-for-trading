"""
Data loading utilities for financial anomaly detection.

Supports:
- Yahoo Finance (stocks, ETFs, indices)
- Bybit (cryptocurrency futures and spot)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """OHLCV candle data structure."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Load OHLCV data for a symbol."""
        pass

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute additional features for anomaly detection."""
        df = df.copy()

        # Price-based features
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["returns"].rolling(window=20).std()

        # Volume-based features
        df["volume_ma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # Price range
        df["range"] = (df["high"] - df["low"]) / df["close"]
        df["range_ma"] = df["range"].rolling(window=20).mean()
        df["range_ratio"] = df["range"] / df["range_ma"]

        # Technical indicators
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["price_sma_ratio"] = df["close"] / df["sma_20"]

        # Bollinger Bands
        df["bb_upper"] = df["sma_20"] + 2 * df["volatility"] * df["close"]
        df["bb_lower"] = df["sma_20"] - 2 * df["volatility"] * df["close"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Z-scores for anomaly detection
        df["returns_zscore"] = (df["returns"] - df["returns"].rolling(50).mean()) / df["returns"].rolling(50).std()
        df["volume_zscore"] = (df["volume"] - df["volume"].rolling(50).mean()) / df["volume"].rolling(50).std()

        return df


class YahooFinanceLoader(BaseDataLoader):
    """Load financial data from Yahoo Finance."""

    # Interval mapping from user-friendly to yfinance format
    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "1d": "1d",
        "1w": "1wk",
        "1M": "1mo",
    }

    def __init__(self):
        """Initialize Yahoo Finance loader."""
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError(
                "yfinance is required for YahooFinanceLoader. "
                "Install with: pip install yfinance"
            )

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g., "AAPL", "BTC-USD")
            interval: Time interval (1m, 5m, 15m, 30m, 1h, 1d, 1w, 1M)
            start: Start date (default: based on limit)
            end: End date (default: now)
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        yf_interval = self.INTERVAL_MAP.get(interval, interval)

        if end is None:
            end = datetime.now()

        if start is None:
            # Estimate start based on interval and limit
            interval_days = {
                "1m": 1/1440, "5m": 5/1440, "15m": 15/1440, "30m": 30/1440,
                "1h": 1/24, "1d": 1, "1wk": 7, "1mo": 30,
            }
            days = interval_days.get(yf_interval, 1) * limit * 1.5  # Add buffer
            start = end - timedelta(days=max(days, 1))

        ticker = self.yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=yf_interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"stock splits": "splits"})

        # Keep only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "timestamp"
        df = df.reset_index()

        # Limit rows
        if len(df) > limit:
            df = df.tail(limit)

        logger.info(f"Loaded {len(df)} candles for {symbol} from Yahoo Finance")
        return df

    def get_multiple(
        self,
        symbols: List[str],
        interval: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols."""
        data = {}
        for symbol in symbols:
            try:
                df = self.get_ohlcv(symbol, interval, start, end)
                if not df.empty:
                    data[symbol] = df
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
        return data


class BybitDataLoader(BaseDataLoader):
    """Load cryptocurrency data from Bybit exchange."""

    BASE_URL = "https://api.bybit.com"

    # Interval mapping
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

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Bybit loader.

        Args:
            api_key: Bybit API key (optional, for authenticated endpoints)
            api_secret: Bybit API secret (optional)
        """
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests is required. Install with: pip install requests")

        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
        category: str = "linear",
    ) -> pd.DataFrame:
        """
        Load OHLCV data from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Time interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
            start: Start timestamp
            end: End timestamp
            limit: Number of candles (max 1000)
            category: Market category (linear, inverse, spot)

        Returns:
            DataFrame with OHLCV data
        """
        bybit_interval = self.INTERVAL_MAP.get(interval, interval)

        # Build request parameters
        params = {
            "category": category,
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": min(limit, 1000),
        }

        if start:
            params["start"] = int(start.timestamp() * 1000)
        if end:
            params["end"] = int(end.timestamp() * 1000)

        # Make API request
        url = f"{self.BASE_URL}/v5/market/kline"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch data from Bybit: {e}")
            return pd.DataFrame()

        if data.get("retCode") != 0:
            logger.error(f"Bybit API error: {data.get('retMsg')}")
            return pd.DataFrame()

        # Parse response
        klines = data.get("result", {}).get("list", [])

        if not klines:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        # Bybit returns: [timestamp, open, high, low, close, volume, turnover]
        records = []
        for k in klines:
            records.append({
                "timestamp": pd.to_datetime(int(k[0]), unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Loaded {len(df)} candles for {symbol} from Bybit")
        return df

    def get_orderbook(
        self,
        symbol: str,
        limit: int = 50,
        category: str = "linear",
    ) -> Dict[str, Any]:
        """
        Get current orderbook snapshot.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            limit: Depth limit (1, 50, 200, 500)
            category: Market category

        Returns:
            Dictionary with bids and asks
        """
        url = f"{self.BASE_URL}/v5/market/orderbook"
        params = {
            "category": category,
            "symbol": symbol,
            "limit": limit,
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch orderbook: {e}")
            return {}

        if data.get("retCode") != 0:
            return {}

        result = data.get("result", {})

        return {
            "symbol": result.get("s"),
            "timestamp": pd.to_datetime(int(result.get("ts", 0)), unit="ms"),
            "bids": [(float(b[0]), float(b[1])) for b in result.get("b", [])],
            "asks": [(float(a[0]), float(a[1])) for a in result.get("a", [])],
        }

    def get_recent_trades(
        self,
        symbol: str,
        limit: int = 500,
        category: str = "linear",
    ) -> pd.DataFrame:
        """
        Get recent trades.

        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)
            category: Market category

        Returns:
            DataFrame with recent trades
        """
        url = f"{self.BASE_URL}/v5/market/recent-trade"
        params = {
            "category": category,
            "symbol": symbol,
            "limit": min(limit, 1000),
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return pd.DataFrame()

        if data.get("retCode") != 0:
            return pd.DataFrame()

        trades = data.get("result", {}).get("list", [])

        records = []
        for t in trades:
            records.append({
                "timestamp": pd.to_datetime(int(t.get("time", 0)), unit="ms"),
                "price": float(t.get("price", 0)),
                "size": float(t.get("size", 0)),
                "side": t.get("side", "").lower(),
            })

        return pd.DataFrame(records)

    def get_ticker(self, symbol: str, category: str = "linear") -> Dict[str, Any]:
        """
        Get 24hr ticker statistics.

        Args:
            symbol: Trading pair
            category: Market category

        Returns:
            Ticker data dictionary
        """
        url = f"{self.BASE_URL}/v5/market/tickers"
        params = {"category": category, "symbol": symbol}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch ticker: {e}")
            return {}

        if data.get("retCode") != 0:
            return {}

        tickers = data.get("result", {}).get("list", [])
        if not tickers:
            return {}

        t = tickers[0]
        return {
            "symbol": t.get("symbol"),
            "last_price": float(t.get("lastPrice", 0)),
            "bid": float(t.get("bid1Price", 0)),
            "ask": float(t.get("ask1Price", 0)),
            "high_24h": float(t.get("highPrice24h", 0)),
            "low_24h": float(t.get("lowPrice24h", 0)),
            "volume_24h": float(t.get("volume24h", 0)),
            "turnover_24h": float(t.get("turnover24h", 0)),
            "price_change_pct": float(t.get("price24hPcnt", 0)) * 100,
        }


def load_sample_data(source: str = "bybit") -> pd.DataFrame:
    """
    Load sample data for testing.

    Args:
        source: "bybit" or "yahoo"

    Returns:
        DataFrame with sample OHLCV data and features
    """
    if source == "bybit":
        loader = BybitDataLoader()
        df = loader.get_ohlcv("BTCUSDT", interval="1h", limit=500)
    else:
        loader = YahooFinanceLoader()
        df = loader.get_ohlcv("SPY", interval="1d", limit=500)

    if not df.empty:
        df = loader.compute_features(df)

    return df


if __name__ == "__main__":
    # Example usage
    print("Testing Yahoo Finance loader...")
    yf_loader = YahooFinanceLoader()
    spy_data = yf_loader.get_ohlcv("SPY", interval="1d", limit=100)
    if not spy_data.empty:
        spy_data = yf_loader.compute_features(spy_data)
        print(spy_data.tail())

    print("\nTesting Bybit loader...")
    bybit_loader = BybitDataLoader()
    btc_data = bybit_loader.get_ohlcv("BTCUSDT", interval="1h", limit=100)
    if not btc_data.empty:
        btc_data = bybit_loader.compute_features(btc_data)
        print(btc_data.tail())
