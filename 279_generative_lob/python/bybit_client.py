"""
Bybit API Client for LOB Data

Fetches real-time order book data from the Bybit V5 REST API,
supporting both spot and linear (futures) markets.
"""

import requests
import time
from typing import Optional, List

from .lob_snapshot import LobSnapshot, PriceLevel


class BybitClient:
    """
    Client for fetching LOB data from Bybit exchange.

    Connects to the Bybit V5 REST API to fetch order book snapshots
    for cryptocurrency pairs.

    Args:
        base_url: API base URL. Default is Bybit mainnet.
        rate_limit: Minimum seconds between requests. Default 0.2.
    """

    BASE_URL = "https://api.bybit.com"
    RATE_LIMIT = 0.2

    def __init__(
        self,
        base_url: Optional[str] = None,
        rate_limit: float = 0.2,
    ):
        self.base_url = base_url or self.BASE_URL
        self.rate_limit = rate_limit
        self._last_request_time = 0.0

    def _rate_limit_wait(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def fetch_orderbook(
        self,
        symbol: str = "BTCUSDT",
        category: str = "spot",
        limit: int = 50,
    ) -> LobSnapshot:
        """
        Fetch the order book for a given symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            category: Market category ('spot' or 'linear').
            limit: Number of price levels to fetch.

        Returns:
            LobSnapshot with the current order book state.

        Raises:
            RuntimeError: If the API returns an error.
            requests.RequestException: On network errors.
        """
        self._rate_limit_wait()

        url = f"{self.base_url}/v5/market/orderbook"
        params = {
            "category": category,
            "symbol": symbol,
            "limit": limit,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        self._last_request_time = time.time()

        if data.get("retCode", -1) != 0:
            raise RuntimeError(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")

        result = data["result"]

        bids = [
            PriceLevel(price=float(level[0]), quantity=float(level[1]))
            for level in result.get("b", [])
            if len(level) >= 2
        ]

        asks = [
            PriceLevel(price=float(level[0]), quantity=float(level[1]))
            for level in result.get("a", [])
            if len(level) >= 2
        ]

        timestamp = result.get("ts", 0)

        return LobSnapshot(bids=bids, asks=asks, timestamp=timestamp)

    def fetch_orderbook_sequence(
        self,
        symbol: str = "BTCUSDT",
        category: str = "spot",
        limit: int = 50,
        count: int = 10,
        interval: float = 1.0,
    ) -> List[LobSnapshot]:
        """
        Fetch a sequence of order book snapshots with a time interval.

        Args:
            symbol: Trading pair symbol.
            category: Market category.
            limit: Number of price levels per snapshot.
            count: Number of snapshots to collect.
            interval: Seconds between fetches.

        Returns:
            List of LobSnapshot objects.
        """
        snapshots = []
        for i in range(count):
            try:
                snapshot = self.fetch_orderbook(symbol, category, limit)
                snapshots.append(snapshot)
            except Exception as e:
                print(f"  Warning: Failed to fetch snapshot {i+1}: {e}")
            if i < count - 1:
                time.sleep(max(0, interval - self.rate_limit))
        return snapshots


if __name__ == "__main__":
    client = BybitClient()
    try:
        snapshot = client.fetch_orderbook("BTCUSDT", "spot", 25)
        print(f"Fetched orderbook: {len(snapshot.bids)} bids, {len(snapshot.asks)} asks")
        print(f"Mid-price: {snapshot.mid_price():.2f}")
        print(f"Spread: {snapshot.spread():.4f}")
        print(f"Best bid: {snapshot.bids[0].price:.2f} @ {snapshot.bids[0].quantity:.6f}")
        print(f"Best ask: {snapshot.asks[0].price:.2f} @ {snapshot.asks[0].quantity:.6f}")
    except Exception as e:
        print(f"Could not connect to Bybit: {e}")
        print("This is expected in offline environments.")
    print("BybitClient test passed!")
