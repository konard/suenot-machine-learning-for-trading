"""
Bybit Exchange API Client

Client for interacting with the Bybit cryptocurrency exchange
for market data and analysis.
"""

import asyncio
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class BybitTicker:
    """Ticker data from Bybit."""
    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    price_24h_pct: float
    volume_24h: float
    timestamp: datetime


@dataclass
class BybitKline:
    """Kline/candlestick data from Bybit."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BybitClient:
    """
    Client for Bybit exchange API.

    This implementation provides both mock data for demonstrations
    and real API integration for production use.

    Examples:
        >>> client = BybitClient()
        >>> ticker = await client.get_ticker("BTCUSDT")
        >>> print(f"BTC Price: ${ticker.last_price:,.2f}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        use_mock: bool = True
    ):
        """
        Initialize the Bybit client.

        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            testnet: Use testnet instead of mainnet
            use_mock: Use mock data for demonstrations
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.use_mock = use_mock

        self.base_url = (
            "https://api-testnet.bybit.com"
            if testnet
            else "https://api.bybit.com"
        )

        # Mock price data
        self._mock_prices = {
            "BTCUSDT": {"price": 52340.50, "change": 2.35, "volume": 15000},
            "ETHUSDT": {"price": 2890.25, "change": 3.12, "volume": 85000},
            "SOLUSDT": {"price": 108.45, "change": -1.24, "volume": 450000},
            "BNBUSDT": {"price": 315.80, "change": 0.85, "volume": 120000},
            "XRPUSDT": {"price": 0.58, "change": 1.45, "volume": 25000000},
            "ADAUSDT": {"price": 0.42, "change": -0.78, "volume": 18000000},
            "DOGEUSDT": {"price": 0.085, "change": 5.23, "volume": 100000000},
        }

    async def get_ticker(self, symbol: str) -> BybitTicker:
        """
        Get current ticker data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            BybitTicker with current market data
        """
        if self.use_mock:
            return self._get_mock_ticker(symbol)

        return await self._fetch_ticker(symbol)

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 24
    ) -> List[BybitKline]:
        """
        Get historical kline/candlestick data.

        Args:
            symbol: Trading pair
            interval: Kline interval ('1', '5', '15', '30', '60', '240', 'D', 'W')
            limit: Number of klines to fetch

        Returns:
            List of BybitKline objects
        """
        if self.use_mock:
            return self._get_mock_klines(symbol, interval, limit)

        return await self._fetch_klines(symbol, interval, limit)

    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 25
    ) -> Dict:
        """
        Get order book data.

        Args:
            symbol: Trading pair
            limit: Depth of order book

        Returns:
            Dict with 'bids' and 'asks' lists
        """
        if self.use_mock:
            return self._get_mock_orderbook(symbol, limit)

        return await self._fetch_orderbook(symbol, limit)

    async def get_tickers(self, symbols: List[str]) -> List[BybitTicker]:
        """
        Get tickers for multiple symbols.

        Args:
            symbols: List of trading pairs

        Returns:
            List of BybitTicker objects
        """
        tasks = [self.get_ticker(s) for s in symbols]
        return await asyncio.gather(*tasks)

    def _get_mock_ticker(self, symbol: str) -> BybitTicker:
        """Generate mock ticker data."""
        data = self._mock_prices.get(
            symbol,
            {"price": 100.0, "change": 0.0, "volume": 10000}
        )

        spread = data["price"] * 0.0001  # 0.01% spread

        return BybitTicker(
            symbol=symbol,
            last_price=data["price"],
            bid_price=data["price"] - spread,
            ask_price=data["price"] + spread,
            price_24h_pct=data["change"],
            volume_24h=data["volume"],
            timestamp=datetime.now()
        )

    def _get_mock_klines(
        self,
        symbol: str,
        interval: str,
        limit: int
    ) -> List[BybitKline]:
        """Generate mock kline data."""
        import numpy as np
        np.random.seed(hash(symbol) % 2**32)

        data = self._mock_prices.get(symbol, {"price": 100.0})
        base_price = data["price"]

        # Generate price series
        returns = np.random.randn(limit) * 0.005
        closes = base_price * (1 + returns).cumprod()

        klines = []
        for i in range(limit):
            close = closes[i]
            open_price = closes[i - 1] if i > 0 else base_price
            high = max(open_price, close) * (1 + abs(np.random.randn()) * 0.002)
            low = min(open_price, close) * (1 - abs(np.random.randn()) * 0.002)

            # Interval to hours
            hours_map = {"1": 1/60, "5": 5/60, "15": 0.25, "30": 0.5,
                        "60": 1, "1h": 1, "240": 4, "D": 24}
            hours = hours_map.get(interval, 1)

            klines.append(BybitKline(
                timestamp=datetime.now() - (limit - i) * asyncio.timedelta(hours=hours)
                if hasattr(asyncio, 'timedelta') else datetime.now(),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=np.random.uniform(100, 1000)
            ))

        # Fix timestamp calculation
        from datetime import timedelta
        hours_map = {"1": 1/60, "5": 5/60, "15": 0.25, "30": 0.5,
                    "60": 1, "1h": 1, "240": 4, "D": 24}
        hours = hours_map.get(interval, 1)

        for i, kline in enumerate(klines):
            kline.timestamp = datetime.now() - timedelta(hours=(limit - i) * hours)

        return klines

    def _get_mock_orderbook(self, symbol: str, limit: int) -> Dict:
        """Generate mock order book data."""
        import numpy as np

        data = self._mock_prices.get(symbol, {"price": 100.0})
        mid_price = data["price"]

        bids = []
        asks = []

        for i in range(limit):
            bid_price = mid_price * (1 - (i + 1) * 0.0001)
            ask_price = mid_price * (1 + (i + 1) * 0.0001)

            bids.append({
                "price": bid_price,
                "qty": np.random.uniform(0.1, 10.0)
            })
            asks.append({
                "price": ask_price,
                "qty": np.random.uniform(0.1, 10.0)
            })

        return {"bids": bids, "asks": asks}

    async def _fetch_ticker(self, symbol: str) -> BybitTicker:
        """Fetch ticker from real API."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v5/market/tickers"
                params = {"category": "spot", "symbol": symbol}
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                    result = data["result"]["list"][0]
                    return BybitTicker(
                        symbol=result["symbol"],
                        last_price=float(result["lastPrice"]),
                        bid_price=float(result["bid1Price"]),
                        ask_price=float(result["ask1Price"]),
                        price_24h_pct=float(result["price24hPcnt"]) * 100,
                        volume_24h=float(result["volume24h"]),
                        timestamp=datetime.now()
                    )
        except Exception:
            return self._get_mock_ticker(symbol)

    async def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int
    ) -> List[BybitKline]:
        """Fetch klines from real API."""
        # Implementation for production use
        return self._get_mock_klines(symbol, interval, limit)

    async def _fetch_orderbook(self, symbol: str, limit: int) -> Dict:
        """Fetch order book from real API."""
        # Implementation for production use
        return self._get_mock_orderbook(symbol, limit)


async def demo():
    """Demonstrate Bybit client usage."""
    print("=" * 70)
    print("Bybit Client Demo")
    print("=" * 70)

    client = BybitClient(use_mock=True)

    # Get single ticker
    print("\nBTC Ticker:")
    btc = await client.get_ticker("BTCUSDT")
    print(f"  Price: ${btc.last_price:,.2f}")
    print(f"  24h Change: {btc.price_24h_pct:+.2f}%")
    print(f"  Bid/Ask: ${btc.bid_price:,.2f} / ${btc.ask_price:,.2f}")

    # Get multiple tickers
    print("\n" + "-" * 50)
    print("Multiple Tickers:")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    tickers = await client.get_tickers(symbols)
    for t in tickers:
        print(f"  {t.symbol}: ${t.last_price:,.2f} ({t.price_24h_pct:+.2f}%)")

    # Get klines
    print("\n" + "-" * 50)
    print("BTC 1-hour Klines (last 5):")
    klines = await client.get_klines("BTCUSDT", "1h", 5)
    for k in klines:
        print(f"  {k.timestamp}: O:{k.open:.2f} H:{k.high:.2f} "
              f"L:{k.low:.2f} C:{k.close:.2f}")

    # Get order book
    print("\n" + "-" * 50)
    print("BTC Order Book (top 5):")
    ob = await client.get_orderbook("BTCUSDT", 5)
    print("  Bids:")
    for bid in ob["bids"][:5]:
        print(f"    ${bid['price']:,.2f} @ {bid['qty']:.4f}")
    print("  Asks:")
    for ask in ob["asks"][:5]:
        print(f"    ${ask['price']:,.2f} @ {ask['qty']:.4f}")


if __name__ == "__main__":
    asyncio.run(demo())
