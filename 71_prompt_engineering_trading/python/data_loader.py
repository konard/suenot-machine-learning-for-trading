"""
Market Data Loader

Utilities for loading market data from various sources
including Yahoo Finance (stocks) and Bybit (crypto).
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import asyncio


@dataclass
class OHLCV:
    """OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "symbol": self.symbol
        }


@dataclass
class MarketSnapshot:
    """Current market snapshot for analysis."""
    symbol: str
    price: float
    change_pct: float
    volume: float
    high_52w: float
    low_52w: float
    avg_volume: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    timestamp: Optional[datetime] = None


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime
    ) -> List[OHLCV]:
        """Get OHLCV data for symbol."""
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        pass


class YahooFinanceLoader(DataLoader):
    """
    Load market data from Yahoo Finance.

    Supports stocks, ETFs, indices, and some crypto pairs.

    Example:
        >>> loader = YahooFinanceLoader()
        >>> data = await loader.get_ohlcv("AAPL", "1d", start, end)
    """

    def __init__(self):
        """Initialize Yahoo Finance loader."""
        self._yf = None

    def _get_yf(self):
        """Lazy import yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance not installed. Run: pip install yfinance"
                )
        return self._yf

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[OHLCV]:
        """
        Get OHLCV data from Yahoo Finance.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            interval: Time interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
            start: Start datetime
            end: End datetime

        Returns:
            List of OHLCV bars
        """
        yf = self._get_yf()

        if start is None:
            start = datetime.now() - timedelta(days=365)
        if end is None:
            end = datetime.now()

        ticker = yf.Ticker(symbol)

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval
            )
        )

        bars = []
        for idx, row in df.iterrows():
            bars.append(OHLCV(
                timestamp=idx.to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
                symbol=symbol
            ))

        return bars

    async def get_current_price(self, symbol: str) -> float:
        """Get current price from Yahoo Finance."""
        yf = self._get_yf()

        ticker = yf.Ticker(symbol)

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, lambda: ticker.info)

        return info.get("regularMarketPrice", 0.0)

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        """
        Get comprehensive market snapshot.

        Args:
            symbol: Stock ticker

        Returns:
            MarketSnapshot with current metrics
        """
        yf = self._get_yf()

        ticker = yf.Ticker(symbol)

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, lambda: ticker.info)

        return MarketSnapshot(
            symbol=symbol,
            price=info.get("regularMarketPrice", 0.0),
            change_pct=info.get("regularMarketChangePercent", 0.0),
            volume=info.get("regularMarketVolume", 0),
            high_52w=info.get("fiftyTwoWeekHigh", 0.0),
            low_52w=info.get("fiftyTwoWeekLow", 0.0),
            avg_volume=info.get("averageVolume", 0),
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            timestamp=datetime.now()
        )


class BybitLoader(DataLoader):
    """
    Load market data from Bybit cryptocurrency exchange.

    Supports perpetual futures and spot markets.

    Example:
        >>> loader = BybitLoader()
        >>> data = await loader.get_ohlcv("BTCUSDT", "1h", start, end)
    """

    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit loader.

        Args:
            testnet: Use testnet instead of mainnet
        """
        self._ccxt = None
        self.testnet = testnet

    def _get_exchange(self):
        """Lazy import and initialize ccxt."""
        if self._ccxt is None:
            try:
                import ccxt.async_support as ccxt
                self._ccxt = ccxt.bybit({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'linear'  # USDT perpetuals
                    }
                })
                if self.testnet:
                    self._ccxt.set_sandbox_mode(True)
            except ImportError:
                raise ImportError(
                    "ccxt not installed. Run: pip install ccxt"
                )
        return self._ccxt

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[OHLCV]:
        """
        Get OHLCV data from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w)
            start: Start datetime
            end: End datetime
            limit: Maximum bars to fetch

        Returns:
            List of OHLCV bars
        """
        exchange = self._get_exchange()

        # Convert symbol format if needed
        if "/" not in symbol:
            # Convert BTCUSDT to BTC/USDT
            if symbol.endswith("USDT"):
                symbol = symbol[:-4] + "/USDT"
            elif symbol.endswith("USD"):
                symbol = symbol[:-3] + "/USD"

        # Convert interval to ccxt format
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
        }
        timeframe = interval_map.get(interval, interval)

        since = None
        if start:
            since = int(start.timestamp() * 1000)

        ohlcv = await exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )

        bars = []
        for candle in ohlcv:
            ts, o, h, l, c, v = candle
            bar_time = datetime.fromtimestamp(ts / 1000)

            if end and bar_time > end:
                break

            bars.append(OHLCV(
                timestamp=bar_time,
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v),
                symbol=symbol
            ))

        return bars

    async def get_current_price(self, symbol: str) -> float:
        """Get current price from Bybit."""
        exchange = self._get_exchange()

        # Convert symbol format
        if "/" not in symbol:
            if symbol.endswith("USDT"):
                symbol = symbol[:-4] + "/USDT"

        ticker = await exchange.fetch_ticker(symbol)
        return ticker.get("last", 0.0)

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get funding rate for perpetual futures.

        Args:
            symbol: Trading pair

        Returns:
            Funding rate info
        """
        exchange = self._get_exchange()

        if "/" not in symbol:
            if symbol.endswith("USDT"):
                symbol = symbol[:-4] + "/USDT"

        funding = await exchange.fetch_funding_rate(symbol)

        return {
            "symbol": symbol,
            "funding_rate": funding.get("fundingRate", 0),
            "funding_timestamp": funding.get("fundingTimestamp"),
            "next_funding_time": funding.get("nextFundingTimestamp")
        }

    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get orderbook for symbol.

        Args:
            symbol: Trading pair
            limit: Depth limit

        Returns:
            Orderbook with bids and asks
        """
        exchange = self._get_exchange()

        if "/" not in symbol:
            if symbol.endswith("USDT"):
                symbol = symbol[:-4] + "/USDT"

        orderbook = await exchange.fetch_order_book(symbol, limit)

        return {
            "symbol": symbol,
            "bids": orderbook.get("bids", [])[:limit],
            "asks": orderbook.get("asks", [])[:limit],
            "timestamp": datetime.now().isoformat()
        }

    async def close(self):
        """Close exchange connection."""
        if self._ccxt:
            await self._ccxt.close()


class MockDataLoader(DataLoader):
    """
    Mock data loader for testing.

    Generates synthetic price data for backtesting
    without requiring API connections.
    """

    def __init__(self, base_prices: Optional[Dict[str, float]] = None):
        """
        Initialize mock loader.

        Args:
            base_prices: Starting prices for symbols
        """
        self.base_prices = base_prices or {
            "AAPL": 185.0,
            "MSFT": 380.0,
            "GOOGL": 140.0,
            "BTC/USDT": 45000.0,
            "ETH/USDT": 2500.0
        }
        self._volatility = 0.02  # 2% daily volatility

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[OHLCV]:
        """Generate mock OHLCV data."""
        import random

        if start is None:
            start = datetime.now() - timedelta(days=365)
        if end is None:
            end = datetime.now()

        base_price = self.base_prices.get(symbol, 100.0)

        # Determine interval in hours
        interval_hours = {
            "1m": 1/60, "5m": 5/60, "15m": 15/60,
            "1h": 1, "4h": 4, "1d": 24, "1w": 168
        }
        hours = interval_hours.get(interval, 24)

        bars = []
        current_time = start
        current_price = base_price

        while current_time <= end:
            # Random walk with drift
            change = random.gauss(0.0001, self._volatility * (hours/24)**0.5)
            current_price *= (1 + change)

            # Generate OHLCV
            high = current_price * (1 + random.uniform(0, 0.01))
            low = current_price * (1 - random.uniform(0, 0.01))
            open_price = current_price * (1 + random.gauss(0, 0.002))
            volume = random.uniform(100000, 1000000)

            bars.append(OHLCV(
                timestamp=current_time,
                open=open_price,
                high=high,
                low=low,
                close=current_price,
                volume=volume,
                symbol=symbol
            ))

            current_time += timedelta(hours=hours)

        return bars

    async def get_current_price(self, symbol: str) -> float:
        """Get mock current price."""
        import random
        base = self.base_prices.get(symbol, 100.0)
        return base * (1 + random.gauss(0, 0.01))


def prepare_market_data_for_prompt(
    snapshot: MarketSnapshot,
    history: List[OHLCV],
    include_technicals: bool = True
) -> Dict[str, Any]:
    """
    Prepare market data for LLM prompt.

    Args:
        snapshot: Current market snapshot
        history: Historical OHLCV data
        include_technicals: Calculate technical indicators

    Returns:
        Dictionary formatted for prompt templates
    """
    data = {
        "symbol": snapshot.symbol,
        "current_price": snapshot.price,
        "change_pct": snapshot.change_pct,
        "volume": snapshot.volume,
        "avg_volume": snapshot.avg_volume,
        "high_52w": snapshot.high_52w,
        "low_52w": snapshot.low_52w,
        "market_cap": snapshot.market_cap,
        "pe_ratio": snapshot.pe_ratio
    }

    if history and include_technicals:
        closes = [bar.close for bar in history]

        # Simple moving averages
        if len(closes) >= 20:
            data["sma_20"] = sum(closes[-20:]) / 20
        if len(closes) >= 50:
            data["sma_50"] = sum(closes[-50:]) / 50
        if len(closes) >= 200:
            data["sma_200"] = sum(closes[-200:]) / 200

        # Trend
        if len(closes) >= 20:
            recent_trend = (closes[-1] - closes[-20]) / closes[-20] * 100
            data["trend_20d"] = recent_trend

        # Volatility (simple std dev)
        if len(closes) >= 20:
            mean = sum(closes[-20:]) / 20
            variance = sum((x - mean) ** 2 for x in closes[-20:]) / 20
            data["volatility_20d"] = (variance ** 0.5) / mean * 100

        # RSI (simplified)
        if len(closes) >= 15:
            gains = []
            losses = []
            for i in range(-14, 0):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                data["rsi_14"] = 100 - (100 / (1 + rs))
            else:
                data["rsi_14"] = 100

    return data


# Example usage
async def main():
    """Demo of data loaders."""
    print("Data Loader Demo")
    print("=" * 50)

    # Use mock loader for demo
    loader = MockDataLoader()

    # Get OHLCV data
    start = datetime.now() - timedelta(days=30)
    end = datetime.now()

    print("\n1. Getting AAPL OHLCV data...")
    aapl_data = await loader.get_ohlcv("AAPL", "1d", start, end)
    print(f"   Retrieved {len(aapl_data)} bars")
    print(f"   Latest close: ${aapl_data[-1].close:.2f}")

    print("\n2. Getting BTC/USDT data...")
    btc_data = await loader.get_ohlcv("BTC/USDT", "1h", start, end)
    print(f"   Retrieved {len(btc_data)} bars")
    print(f"   Latest close: ${btc_data[-1].close:.2f}")

    print("\n3. Preparing data for LLM prompt...")
    snapshot = MarketSnapshot(
        symbol="AAPL",
        price=aapl_data[-1].close,
        change_pct=((aapl_data[-1].close - aapl_data[-2].close) / aapl_data[-2].close * 100),
        volume=aapl_data[-1].volume,
        high_52w=max(bar.high for bar in aapl_data),
        low_52w=min(bar.low for bar in aapl_data),
        avg_volume=sum(bar.volume for bar in aapl_data) / len(aapl_data)
    )

    prompt_data = prepare_market_data_for_prompt(snapshot, aapl_data)
    print(f"   Prepared data keys: {list(prompt_data.keys())}")

    if "rsi_14" in prompt_data:
        print(f"   RSI(14): {prompt_data['rsi_14']:.2f}")
    if "sma_20" in prompt_data:
        print(f"   SMA(20): ${prompt_data['sma_20']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
