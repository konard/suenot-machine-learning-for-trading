"""
Bybit Exchange Data Fetcher

Fetches cryptocurrency market data from Bybit exchange API.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import requests


@dataclass
class OHLCV:
    """OHLCV candle data"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    timestamp: int
    bids: List[tuple]  # [(price, quantity), ...]
    asks: List[tuple]  # [(price, quantity), ...]


class BybitDataFetcher:
    """
    Fetches market data from Bybit exchange

    Uses public API endpoints (no authentication required):
    - Kline (candlestick) data
    - Order book depth
    - Recent trades
    - Ticker information
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit data fetcher

        Args:
            testnet: If True, use testnet API
        """
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = self.BASE_URL

    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """Make API request with error handling"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                raise Exception(f"API Error: {data.get('retMsg', 'Unknown error')}")

            return data.get("result", {})
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")

    def get_kline(
        self,
        symbol: str,
        interval: str = "1",
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        category: str = "linear"
    ) -> List[OHLCV]:
        """
        Get kline (candlestick) data

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            limit: Number of candles (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            category: Product type (spot, linear, inverse)

        Returns:
            List of OHLCV candles (oldest first)
        """
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        result = self._make_request("/v5/market/kline", params)

        candles = []
        for item in reversed(result.get("list", [])):  # Reverse to get oldest first
            candles.append(OHLCV(
                timestamp=int(item[0]),
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
                turnover=float(item[6]) if len(item) > 6 else 0.0
            ))

        return candles

    def get_orderbook(
        self,
        symbol: str,
        limit: int = 50,
        category: str = "linear"
    ) -> OrderBookSnapshot:
        """
        Get order book depth

        Args:
            symbol: Trading pair
            limit: Depth limit (1, 25, 50, 100, 200)
            category: Product type

        Returns:
            Order book snapshot
        """
        params = {
            "category": category,
            "symbol": symbol,
            "limit": limit
        }

        result = self._make_request("/v5/market/orderbook", params)

        bids = [(float(b[0]), float(b[1])) for b in result.get("b", [])]
        asks = [(float(a[0]), float(a[1])) for a in result.get("a", [])]

        return OrderBookSnapshot(
            timestamp=int(result.get("ts", time.time() * 1000)),
            bids=bids,
            asks=asks
        )

    def get_recent_trades(
        self,
        symbol: str,
        limit: int = 60,
        category: str = "linear"
    ) -> List[Dict]:
        """
        Get recent trades

        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)
            category: Product type

        Returns:
            List of trade records
        """
        params = {
            "category": category,
            "symbol": symbol,
            "limit": min(limit, 1000)
        }

        result = self._make_request("/v5/market/recent-trade", params)

        trades = []
        for trade in result.get("list", []):
            trades.append({
                "id": trade.get("execId"),
                "price": float(trade.get("price", 0)),
                "quantity": float(trade.get("size", 0)),
                "side": trade.get("side", "").lower(),
                "timestamp": int(trade.get("time", 0))
            })

        return trades

    def get_ticker(
        self,
        symbol: str,
        category: str = "linear"
    ) -> Dict:
        """
        Get ticker information

        Args:
            symbol: Trading pair
            category: Product type

        Returns:
            Ticker data dictionary
        """
        params = {
            "category": category,
            "symbol": symbol
        }

        result = self._make_request("/v5/market/tickers", params)

        ticker_list = result.get("list", [])
        if not ticker_list:
            return {}

        ticker = ticker_list[0]
        return {
            "symbol": ticker.get("symbol"),
            "last_price": float(ticker.get("lastPrice", 0)),
            "bid_price": float(ticker.get("bid1Price", 0)),
            "ask_price": float(ticker.get("ask1Price", 0)),
            "high_24h": float(ticker.get("highPrice24h", 0)),
            "low_24h": float(ticker.get("lowPrice24h", 0)),
            "volume_24h": float(ticker.get("volume24h", 0)),
            "turnover_24h": float(ticker.get("turnover24h", 0)),
            "price_change_24h": float(ticker.get("price24hPcnt", 0)) * 100
        }

    def get_instruments(
        self,
        category: str = "linear",
        status: str = "Trading"
    ) -> List[Dict]:
        """
        Get available trading instruments

        Args:
            category: Product type
            status: Instrument status filter

        Returns:
            List of instrument info
        """
        params = {
            "category": category,
            "status": status
        }

        result = self._make_request("/v5/market/instruments-info", params)

        instruments = []
        for item in result.get("list", []):
            instruments.append({
                "symbol": item.get("symbol"),
                "base_coin": item.get("baseCoin"),
                "quote_coin": item.get("quoteCoin"),
                "status": item.get("status"),
                "tick_size": float(item.get("priceFilter", {}).get("tickSize", 0)),
                "min_qty": float(item.get("lotSizeFilter", {}).get("minOrderQty", 0)),
                "max_qty": float(item.get("lotSizeFilter", {}).get("maxOrderQty", 0))
            })

        return instruments

    def fetch_price_history(
        self,
        symbol: str,
        days: int = 30,
        interval: str = "D",
        category: str = "linear"
    ) -> List[float]:
        """
        Convenience method to fetch price history

        Args:
            symbol: Trading pair
            days: Number of days of history
            interval: Candle interval
            category: Product type

        Returns:
            List of closing prices (oldest first)
        """
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)

        candles = self.get_kline(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            category=category
        )

        return [candle.close for candle in candles]


if __name__ == "__main__":
    # Example usage
    fetcher = BybitDataFetcher()

    print("Fetching BTCUSDT ticker...")
    ticker = fetcher.get_ticker("BTCUSDT")
    print(f"Last Price: ${ticker.get('last_price', 'N/A'):,.2f}")
    print(f"24h Change: {ticker.get('price_change_24h', 0):.2f}%")
    print(f"24h Volume: ${ticker.get('turnover_24h', 0):,.0f}")

    print("\nFetching order book...")
    orderbook = fetcher.get_orderbook("BTCUSDT", limit=5)
    print(f"Best Bid: ${orderbook.bids[0][0]:,.2f}" if orderbook.bids else "No bids")
    print(f"Best Ask: ${orderbook.asks[0][0]:,.2f}" if orderbook.asks else "No asks")

    print("\nFetching recent klines...")
    candles = fetcher.get_kline("BTCUSDT", interval="60", limit=5)
    for candle in candles:
        dt = datetime.fromtimestamp(candle.timestamp / 1000)
        print(f"{dt}: O={candle.open:.2f} H={candle.high:.2f} L={candle.low:.2f} C={candle.close:.2f}")

    print("\nFetching price history (30 days)...")
    prices = fetcher.fetch_price_history("BTCUSDT", days=30)
    print(f"Got {len(prices)} daily prices")
    if prices:
        print(f"First: ${prices[0]:,.2f}, Last: ${prices[-1]:,.2f}")
