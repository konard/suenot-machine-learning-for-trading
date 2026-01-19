"""
Data loaders for financial data from Yahoo Finance and Bybit.

This module provides utilities for fetching market data from:
- Yahoo Finance (stocks, ETFs, indices)
- Bybit exchange (cryptocurrency perpetual and spot markets)
"""

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Optional imports with fallbacks
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class MarketData:
    """Container for market data."""
    symbol: str
    data: pd.DataFrame
    source: str
    last_updated: datetime
    metadata: Dict[str, Any]


class YahooFinanceLoader:
    """
    Load financial data from Yahoo Finance.

    Supports stocks, ETFs, indices, and other securities available
    on Yahoo Finance.

    Example:
        >>> loader = YahooFinanceLoader()
        >>> data = loader.get_daily("AAPL", start="2023-01-01", end="2024-01-01")
        >>> print(data.head())
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize Yahoo Finance loader.

        Args:
            cache_dir: Optional directory for caching data
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is required. Install with: pip install yfinance"
            )

        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, symbol: str, interval: str, start: str, end: str) -> str:
        """Generate cache file path."""
        key = f"{symbol}_{interval}_{start}_{end}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"{symbol}_{hash_key}.parquet")

    def get_daily(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data.

        Args:
            symbol: Ticker symbol (e.g., "AAPL", "SPY")
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)
            period: Period if start/end not specified (1d, 5d, 1mo, 3mo, 1y, etc.)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        """
        ticker = yf.Ticker(symbol)

        if start and end:
            data = ticker.history(start=start, end=end)
        else:
            data = ticker.history(period=period)

        if data.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Standardize column names
        data.columns = [c.lower().replace(" ", "_") for c in data.columns]

        return data

    def get_intraday(
        self,
        symbol: str,
        interval: str = "1h",
        period: str = "7d"
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV data.

        Args:
            symbol: Ticker symbol
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h)
            period: How far back to fetch (1d, 5d, 7d, 1mo, max)

        Returns:
            DataFrame with OHLCV data
        """
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)

        if data.empty:
            raise ValueError(f"No intraday data for {symbol}")

        data.columns = [c.lower().replace(" ", "_") for c in data.columns]
        return data

    def get_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get security information.

        Args:
            symbol: Ticker symbol

        Returns:
            Dictionary with security metadata
        """
        ticker = yf.Ticker(symbol)
        return ticker.info

    def get_news(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get recent news for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            List of news articles with title, link, publisher, etc.
        """
        ticker = yf.Ticker(symbol)
        return ticker.news

    def get_multiple(
        self,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start: Start date
            end: End date
            period: Period if dates not specified

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_daily(symbol, start, end, period)
            except Exception as e:
                print(f"Warning: Failed to fetch {symbol}: {e}")
        return results

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute common technical features from OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional feature columns
        """
        df = df.copy()

        # Returns
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)

        # Volatility
        df["volatility_20d"] = df["return_1d"].rolling(20).std() * np.sqrt(252)

        # Moving averages
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # Volume features
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # RSI
        df["rsi_14"] = self._compute_rsi(df["close"], 14)

        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = self._compute_macd(df["close"])

        return df

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute MACD indicator."""
        fast_ema = prices.ewm(span=fast).mean()
        slow_ema = prices.ewm(span=slow).mean()
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist


class BybitDataLoader:
    """
    Load cryptocurrency data from Bybit exchange.

    Supports perpetual futures, spot markets, and historical kline data.

    Example:
        >>> loader = BybitDataLoader()
        >>> data = loader.get_klines("BTCUSDT", interval="1h", limit=1000)
        >>> print(data.head())
    """

    BASE_URL = "https://api.bybit.com"

    # Interval mappings
    INTERVALS = {
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
        "1M": "M"
    }

    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit loader.

        Args:
            testnet: Whether to use testnet API
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is required. Install with: pip install requests"
            )

        if testnet:
            self.BASE_URL = "https://api-testnet.bybit.com"

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make API request to Bybit."""
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0:
            raise ValueError(f"Bybit API error: {data.get('retMsg')}")

        return data.get("result", {})

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None,
        category: str = "linear"
    ) -> pd.DataFrame:
        """
        Fetch kline (candlestick) data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
            limit: Number of klines (max 200)
            start: Start timestamp in milliseconds
            end: End timestamp in milliseconds
            category: Market category (linear, inverse, spot)

        Returns:
            DataFrame with OHLCV data
        """
        interval_code = self.INTERVALS.get(interval, interval)

        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval_code,
            "limit": min(limit, 200)
        }

        if start:
            params["start"] = start
        if end:
            params["end"] = end

        result = self._make_request("/v5/market/kline", params)
        klines = result.get("list", [])

        if not klines:
            raise ValueError(f"No kline data for {symbol}")

        # Convert to DataFrame
        # Bybit format: [startTime, open, high, low, close, volume, turnover]
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by time (Bybit returns newest first)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.set_index("timestamp")

        return df

    def get_historical_klines(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 30,
        category: str = "linear"
    ) -> pd.DataFrame:
        """
        Fetch extended historical kline data by making multiple requests.

        Args:
            symbol: Trading pair
            interval: Kline interval
            days: Number of days of history
            category: Market category

        Returns:
            DataFrame with extended historical data
        """
        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)

        # Calculate bars per request based on interval
        interval_minutes = self._interval_to_minutes(interval)
        bars_per_day = 24 * 60 / interval_minutes
        total_bars = int(days * bars_per_day)

        while len(all_data) < total_bars:
            df = self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=200,
                end=end_time,
                category=category
            )

            if df.empty:
                break

            all_data.append(df)

            # Update end time to fetch older data
            end_time = int(df.index[0].timestamp() * 1000) - 1

            # Rate limiting
            time.sleep(0.1)

        if not all_data:
            raise ValueError(f"No historical data for {symbol}")

        # Combine all data
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()

        return combined

    @staticmethod
    def _interval_to_minutes(interval: str) -> int:
        """Convert interval string to minutes."""
        multipliers = {"m": 1, "h": 60, "d": 1440, "w": 10080, "M": 43200}
        for suffix, mult in multipliers.items():
            if interval.endswith(suffix):
                return int(interval[:-1]) * mult
        return 60  # Default 1 hour

    def get_tickers(self, category: str = "linear") -> pd.DataFrame:
        """
        Fetch all ticker information.

        Args:
            category: Market category (linear, inverse, spot)

        Returns:
            DataFrame with ticker data
        """
        result = self._make_request("/v5/market/tickers", {"category": category})
        tickers = result.get("list", [])

        return pd.DataFrame(tickers)

    def get_orderbook(
        self,
        symbol: str,
        limit: int = 50,
        category: str = "linear"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch order book data.

        Args:
            symbol: Trading pair
            limit: Number of levels (max 200)
            category: Market category

        Returns:
            Dictionary with "bids" and "asks" DataFrames
        """
        result = self._make_request("/v5/market/orderbook", {
            "category": category,
            "symbol": symbol,
            "limit": limit
        })

        bids = pd.DataFrame(result.get("b", []), columns=["price", "size"])
        asks = pd.DataFrame(result.get("a", []), columns=["price", "size"])

        for df in [bids, asks]:
            df["price"] = pd.to_numeric(df["price"])
            df["size"] = pd.to_numeric(df["size"])

        return {"bids": bids, "asks": asks}

    def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100,
        category: str = "linear"
    ) -> pd.DataFrame:
        """
        Fetch recent trades.

        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)
            category: Market category

        Returns:
            DataFrame with recent trades
        """
        result = self._make_request("/v5/market/recent-trade", {
            "category": category,
            "symbol": symbol,
            "limit": limit
        })

        trades = result.get("list", [])
        df = pd.DataFrame(trades)

        if not df.empty:
            df["time"] = pd.to_datetime(df["time"].astype(int), unit="ms")
            df["price"] = pd.to_numeric(df["price"])
            df["size"] = pd.to_numeric(df["size"])

        return df

    def get_funding_rate(
        self,
        symbol: str,
        limit: int = 200,
        category: str = "linear"
    ) -> pd.DataFrame:
        """
        Fetch funding rate history.

        Args:
            symbol: Trading pair
            limit: Number of records
            category: Market category (linear or inverse)

        Returns:
            DataFrame with funding rate history
        """
        result = self._make_request("/v5/market/funding/history", {
            "category": category,
            "symbol": symbol,
            "limit": limit
        })

        rates = result.get("list", [])
        df = pd.DataFrame(rates)

        if not df.empty:
            df["fundingRateTimestamp"] = pd.to_datetime(
                df["fundingRateTimestamp"].astype(int), unit="ms"
            )
            df["fundingRate"] = pd.to_numeric(df["fundingRate"])

        return df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute common crypto trading features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional features
        """
        df = df.copy()

        # Returns
        df["return_1h"] = df["close"].pct_change(1)
        df["return_4h"] = df["close"].pct_change(4)
        df["return_24h"] = df["close"].pct_change(24)

        # Volatility
        df["volatility_24h"] = df["return_1h"].rolling(24).std() * np.sqrt(24 * 365)

        # Moving averages
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()

        # Volume profile
        df["volume_sma"] = df["volume"].rolling(24).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Price position
        df["high_24h"] = df["high"].rolling(24).max()
        df["low_24h"] = df["low"].rolling(24).min()
        df["price_position"] = (df["close"] - df["low_24h"]) / (df["high_24h"] - df["low_24h"])

        # Turnover (useful for crypto)
        df["turnover_ratio"] = df["turnover"] / df["turnover"].rolling(24).mean()

        return df


def load_financial_phrasebank(
    path: Optional[str] = None,
    agreement_level: str = "all"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Financial PhraseBank dataset for sentiment analysis.

    This is a simulated version - in practice, download from:
    https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10

    Args:
        path: Path to dataset (if None, returns sample data)
        agreement_level: Agreement level (50, 66, 75, 100, all)

    Returns:
        Tuple of (train_df, val_df) with columns: text, label
    """
    # Sample financial phrases for demonstration
    sample_data = [
        ("Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007.", 2),
        ("The company reported a net loss of $5.2 million for the quarter.", 0),
        ("Revenue remained flat compared to the previous year.", 1),
        ("Strong demand drove record sales in the automotive segment.", 2),
        ("Management expects challenging market conditions to persist.", 0),
        ("The acquisition is expected to be neutral to earnings in year one.", 1),
        ("Gross margins expanded by 200 basis points year-over-year.", 2),
        ("Rising costs and supply chain disruptions impacted profitability.", 0),
        ("The company maintained its full-year guidance.", 1),
        ("Dividend increased by 15% reflecting strong cash generation.", 2),
        ("Restructuring charges weighed on quarterly results.", 0),
        ("Organic growth was in line with market expectations.", 1),
        ("Market share gains accelerated in key regions.", 2),
        ("Inventory write-downs impacted the bottom line.", 0),
        ("Order backlog provides visibility into future revenue.", 1),
        ("Bitcoin surged past $50,000 on institutional buying.", 2),
        ("Crypto market sentiment remains cautious amid regulatory uncertainty.", 0),
        ("Bybit reports record trading volume for BTC perpetuals.", 1),
        ("Ethereum network upgrade drives renewed investor interest.", 2),
        ("Flash crash triggers mass liquidations on leveraged positions.", 0),
    ]

    df = pd.DataFrame(sample_data, columns=["text", "label"])

    # Simple train/val split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)

    return train_df, val_df


if __name__ == "__main__":
    print("Testing Yahoo Finance Loader...")
    try:
        yahoo = YahooFinanceLoader()
        aapl = yahoo.get_daily("AAPL", period="1mo")
        print(f"AAPL data shape: {aapl.shape}")
        print(aapl.tail())

        # Compute features
        aapl_features = yahoo.compute_features(aapl)
        print(f"\nFeatures computed: {list(aapl_features.columns)}")
    except Exception as e:
        print(f"Yahoo Finance test failed: {e}")

    print("\n" + "="*50)
    print("Testing Bybit Loader...")
    try:
        bybit = BybitDataLoader()
        btc = bybit.get_klines("BTCUSDT", interval="1h", limit=100)
        print(f"BTCUSDT data shape: {btc.shape}")
        print(btc.tail())

        # Compute features
        btc_features = bybit.compute_features(btc)
        print(f"\nFeatures computed: {list(btc_features.columns)}")
    except Exception as e:
        print(f"Bybit test failed: {e}")

    print("\n" + "="*50)
    print("Testing Financial PhraseBank...")
    train_df, val_df = load_financial_phrasebank()
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print("\nSample:")
    print(train_df.head())
