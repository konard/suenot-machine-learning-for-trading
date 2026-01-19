"""
Bybit Data Client Module

This module provides classes for fetching cryptocurrency market data
from the Bybit exchange API.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

import requests
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """OHLCV (Open, High, Low, Close, Volume) candlestick data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "datetime": self.datetime.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class BybitClient:
    """
    Bybit API client for fetching market data.

    Supports fetching klines (candlesticks), tickers, and order book data
    from Bybit's public API.
    """

    BASE_URL = "https://api.bybit.com"

    # Interval mapping
    INTERVALS = {
        "1": "1",      # 1 minute
        "3": "3",      # 3 minutes
        "5": "5",      # 5 minutes
        "15": "15",    # 15 minutes
        "30": "30",    # 30 minutes
        "60": "60",    # 1 hour
        "120": "120",  # 2 hours
        "240": "240",  # 4 hours
        "360": "360",  # 6 hours
        "720": "720",  # 12 hours
        "D": "D",      # 1 day
        "W": "W",      # 1 week
        "M": "M",      # 1 month
    }

    def __init__(self, timeout: int = 30):
        """
        Initialize Bybit client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })

    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request."""
        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                raise ValueError(f"API error: {data.get('retMsg', 'Unknown error')}")

            return data.get("result", {})

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 200,
    ) -> List[OHLCV]:
        """
        Fetch kline/candlestick data.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Kline interval (1, 5, 15, 30, 60, 240, D, W, M)
            limit: Number of candles (max 1000)

        Returns:
            List of OHLCV objects
        """
        if interval not in self.INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Valid: {list(self.INTERVALS.keys())}")

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        result = self._request("/v5/market/kline", params)
        klines = result.get("list", [])

        # Bybit returns data in reverse chronological order
        ohlcv_list = []
        for k in reversed(klines):
            ohlcv_list.append(OHLCV(
                timestamp=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
            ))

        return ohlcv_list

    def fetch_historical_klines(
        self,
        symbol: str,
        interval: str = "60",
        days: int = 7,
    ) -> List[OHLCV]:
        """
        Fetch historical kline data for a specified number of days.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            days: Number of days of history

        Returns:
            List of OHLCV objects
        """
        # Calculate how many candles we need
        interval_minutes = {
            "1": 1, "3": 3, "5": 5, "15": 15, "30": 30,
            "60": 60, "120": 120, "240": 240, "360": 360, "720": 720,
            "D": 1440, "W": 10080, "M": 43200,
        }

        minutes = interval_minutes.get(interval, 60)
        total_candles = (days * 24 * 60) // minutes

        # Fetch in batches if needed
        all_klines = []
        end_time = int(time.time() * 1000)

        while len(all_klines) < total_candles:
            batch_size = min(1000, total_candles - len(all_klines))

            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": batch_size,
                "end": end_time,
            }

            result = self._request("/v5/market/kline", params)
            klines = result.get("list", [])

            if not klines:
                break

            for k in reversed(klines):
                all_klines.append(OHLCV(
                    timestamp=int(k[0]),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                ))

            # Update end time for next batch
            end_time = int(klines[-1][0]) - 1

            # Rate limiting
            time.sleep(0.1)

        # Sort by timestamp and remove duplicates
        all_klines.sort(key=lambda x: x.timestamp)
        seen = set()
        unique_klines = []
        for k in all_klines:
            if k.timestamp not in seen:
                seen.add(k.timestamp)
                unique_klines.append(k)

        return unique_klines[-total_candles:]

    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data dictionary
        """
        params = {
            "category": "linear",
            "symbol": symbol,
        }

        result = self._request("/v5/market/tickers", params)
        tickers = result.get("list", [])

        if not tickers:
            raise ValueError(f"Ticker not found for {symbol}")

        return tickers[0]

    def fetch_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """
        Fetch order book data.

        Args:
            symbol: Trading pair symbol
            limit: Number of levels (1, 25, 50, 100, 200)

        Returns:
            Order book dictionary with bids and asks
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": limit,
        }

        result = self._request("/v5/market/orderbook", params)

        return {
            "bids": [(float(b[0]), float(b[1])) for b in result.get("b", [])],
            "asks": [(float(a[0]), float(a[1])) for a in result.get("a", [])],
            "timestamp": result.get("ts"),
        }

    def get_symbols(self, category: str = "linear") -> List[str]:
        """
        Get list of available trading symbols.

        Args:
            category: Market category (linear, inverse, spot)

        Returns:
            List of symbol names
        """
        params = {
            "category": category,
        }

        result = self._request("/v5/market/instruments-info", params)
        instruments = result.get("list", [])

        return [inst["symbol"] for inst in instruments]

    def to_dataframe(self, klines: List[OHLCV]) -> pd.DataFrame:
        """
        Convert OHLCV list to pandas DataFrame.

        Args:
            klines: List of OHLCV objects

        Returns:
            DataFrame with OHLCV columns
        """
        data = [k.to_dict() for k in klines]
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        return df


class MarketDataAggregator:
    """
    Aggregates market data for multiple symbols.
    """

    def __init__(self, client: Optional[BybitClient] = None):
        """
        Initialize aggregator.

        Args:
            client: BybitClient instance (creates new if None)
        """
        self.client = client or BybitClient()
        self.data: Dict[str, List[OHLCV]] = {}

    def fetch_multi_symbol(
        self,
        symbols: List[str],
        interval: str = "60",
        days: int = 7,
    ) -> Dict[str, List[OHLCV]]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of trading pair symbols
            interval: Kline interval
            days: Number of days of history

        Returns:
            Dictionary mapping symbol to OHLCV list
        """
        for symbol in symbols:
            try:
                logger.info(f"Fetching {symbol}...")
                klines = self.client.fetch_historical_klines(symbol, interval, days)
                self.data[symbol] = klines
                logger.info(f"  Got {len(klines)} candles")
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

        return self.data

    def get_summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics for fetched data.

        Returns:
            Dictionary with summary for each symbol
        """
        summary = {}

        for symbol, klines in self.data.items():
            if not klines:
                continue

            closes = [k.close for k in klines]
            volumes = [k.volume for k in klines]

            summary[symbol] = {
                "candles": len(klines),
                "start": klines[0].datetime.isoformat(),
                "end": klines[-1].datetime.isoformat(),
                "open": klines[0].open,
                "close": klines[-1].close,
                "high": max(k.high for k in klines),
                "low": min(k.low for k in klines),
                "change_pct": (klines[-1].close - klines[0].open) / klines[0].open * 100,
                "avg_volume": sum(volumes) / len(volumes),
            }

        return summary


# Example usage
if __name__ == "__main__":
    print("Bybit Data Client Demo")
    print("=" * 50)

    client = BybitClient()

    # Fetch recent klines
    print("\nFetching BTCUSDT 1-hour klines...")
    klines = client.fetch_klines("BTCUSDT", "60", 10)

    print(f"Got {len(klines)} candles:")
    for k in klines[-5:]:
        print(f"  {k.datetime}: O={k.open:.2f} H={k.high:.2f} L={k.low:.2f} C={k.close:.2f} V={k.volume:.2f}")

    # Fetch ticker
    print("\nFetching BTCUSDT ticker...")
    ticker = client.fetch_ticker("BTCUSDT")
    print(f"  Last Price: {ticker.get('lastPrice')}")
    print(f"  24h Change: {ticker.get('price24hPcnt')}%")
    print(f"  24h Volume: {ticker.get('volume24h')}")

    # Multi-symbol fetch
    print("\nFetching multiple symbols...")
    aggregator = MarketDataAggregator(client)
    data = aggregator.fetch_multi_symbol(["BTCUSDT", "ETHUSDT"], "60", 1)

    print("\nSummary:")
    summary = aggregator.get_summary()
    for symbol, stats in summary.items():
        print(f"\n{symbol}:")
        print(f"  Period: {stats['start']} to {stats['end']}")
        print(f"  Change: {stats['change_pct']:.2f}%")
        print(f"  Range: ${stats['low']:.2f} - ${stats['high']:.2f}")
