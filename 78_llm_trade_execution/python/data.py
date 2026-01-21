"""
Market data structures and exchange connectivity.

This module provides:
- Common data types for market data (OHLCV, OrderBook, Trades)
- Bybit exchange client for cryptocurrency data
"""

import asyncio
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import aiohttp


class TimeFrame(Enum):
    """Time frame for OHLCV data."""
    M1 = "1"
    M3 = "3"
    M5 = "5"
    M15 = "15"
    M30 = "30"
    H1 = "60"
    H2 = "120"
    H4 = "240"
    H6 = "360"
    H12 = "720"
    D1 = "D"
    W1 = "W"
    MN = "M"

    def as_seconds(self) -> int:
        """Get the duration in seconds."""
        mapping = {
            "1": 60, "3": 180, "5": 300, "15": 900, "30": 1800,
            "60": 3600, "120": 7200, "240": 14400, "360": 21600,
            "720": 43200, "D": 86400, "W": 604800, "M": 2592000,
        }
        return mapping.get(self.value, 60)


class TradeDirection(Enum):
    """Trade direction."""
    BUY = "Buy"
    SELL = "Sell"


@dataclass
class OhlcvBar:
    """OHLCV bar (candlestick)."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: Optional[float] = None

    def typical_price(self) -> float:
        """Get the typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3.0

    def vwap(self) -> float:
        """Get the VWAP approximation for this bar."""
        return self.typical_price()

    def is_bullish(self) -> bool:
        """Check if the bar is bullish (close > open)."""
        return self.close > self.open

    def range(self) -> float:
        """Get the bar range (high - low)."""
        return self.high - self.low

    def body_size(self) -> float:
        """Get the body size."""
        return abs(self.close - self.open)


@dataclass
class Trade:
    """Individual trade."""
    id: str
    timestamp: datetime
    price: float
    quantity: float
    direction: TradeDirection

    def value(self) -> float:
        """Get the trade value."""
        return self.price * self.quantity


@dataclass
class Ticker:
    """Ticker data."""
    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    bid_qty: float
    ask_qty: float
    high_24h: float
    low_24h: float
    volume_24h: float
    turnover_24h: float
    open_interest: Optional[float] = None
    funding_rate: Optional[float] = None
    next_funding_time: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def mid_price(self) -> float:
        """Get the mid price."""
        return (self.bid_price + self.ask_price) / 2.0

    def spread(self) -> float:
        """Get the spread."""
        return self.ask_price - self.bid_price

    def spread_bps(self) -> float:
        """Get the spread in basis points."""
        mid = self.mid_price()
        if mid > 0:
            return (self.spread() / mid) * 10000.0
        return 0.0


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: float
    quantity: float

    def value(self) -> float:
        """Get the value at this level."""
        return self.price * self.quantity


class OrderBook:
    """Order book with efficient updates."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[float, float] = {}  # price -> quantity
        self.asks: Dict[float, float] = {}  # price -> quantity
        self.last_update = datetime.utcnow()
        self.sequence = 0

    def update_bid(self, price: float, quantity: float) -> None:
        """Update a bid level."""
        if quantity <= 0:
            self.bids.pop(price, None)
        else:
            self.bids[price] = quantity
        self.last_update = datetime.utcnow()

    def update_ask(self, price: float, quantity: float) -> None:
        """Update an ask level."""
        if quantity <= 0:
            self.asks.pop(price, None)
        else:
            self.asks[price] = quantity
        self.last_update = datetime.utcnow()

    def best_bid(self) -> Optional[float]:
        """Get the best bid price."""
        return max(self.bids.keys()) if self.bids else None

    def best_ask(self) -> Optional[float]:
        """Get the best ask price."""
        return min(self.asks.keys()) if self.asks else None

    def mid_price(self) -> Optional[float]:
        """Get the mid price."""
        bid, ask = self.best_bid(), self.best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None

    def spread(self) -> Optional[float]:
        """Get the spread."""
        bid, ask = self.best_bid(), self.best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return None

    def spread_bps(self) -> Optional[float]:
        """Get the spread in basis points."""
        spread = self.spread()
        mid = self.mid_price()
        if spread is not None and mid is not None and mid > 0:
            return (spread / mid) * 10000.0
        return None

    def top_bids(self, n: int) -> List[OrderBookLevel]:
        """Get top N bid levels."""
        sorted_bids = sorted(self.bids.items(), reverse=True)[:n]
        return [OrderBookLevel(p, q) for p, q in sorted_bids]

    def top_asks(self, n: int) -> List[OrderBookLevel]:
        """Get top N ask levels."""
        sorted_asks = sorted(self.asks.items())[:n]
        return [OrderBookLevel(p, q) for p, q in sorted_asks]

    def bid_depth(self, levels: int) -> float:
        """Calculate total bid depth for N levels."""
        return sum(q for _, q in sorted(self.bids.items(), reverse=True)[:levels])

    def ask_depth(self, levels: int) -> float:
        """Calculate total ask depth for N levels."""
        return sum(q for _, q in sorted(self.asks.items())[:levels])

    def imbalance(self, levels: int) -> float:
        """Calculate the imbalance ratio."""
        bid_depth = self.bid_depth(levels)
        ask_depth = self.ask_depth(levels)
        total = bid_depth + ask_depth
        if total > 0:
            return (bid_depth - ask_depth) / total
        return 0.0

    def buy_impact(self, quantity: float) -> Optional[Tuple[float, float]]:
        """Estimate the price impact of buying a given quantity."""
        return self._estimate_impact(self.top_asks(100), quantity)

    def sell_impact(self, quantity: float) -> Optional[Tuple[float, float]]:
        """Estimate the price impact of selling a given quantity."""
        return self._estimate_impact(self.top_bids(100), quantity)

    def _estimate_impact(
        self, levels: List[OrderBookLevel], quantity: float
    ) -> Optional[Tuple[float, float]]:
        """Estimate impact from a list of levels."""
        if not levels or quantity <= 0:
            return None

        remaining = quantity
        total_cost = 0.0
        last_price = levels[0].price

        for level in levels:
            if remaining <= 0:
                break
            fill_qty = min(remaining, level.quantity)
            total_cost += fill_qty * level.price
            remaining -= fill_qty
            last_price = level.price

        if remaining > 0:
            return None  # Not enough liquidity

        avg_price = total_cost / quantity
        impact = abs(last_price - levels[0].price) / levels[0].price
        return avg_price, impact


@dataclass
class BybitConfig:
    """Bybit client configuration."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = False
    timeout: int = 5000
    recv_window: int = 5000

    @property
    def base_url(self) -> str:
        """Get the base URL."""
        if self.testnet:
            return "https://api-testnet.bybit.com"
        return "https://api.bybit.com"

    @classmethod
    def with_credentials(cls, api_key: str, api_secret: str) -> "BybitConfig":
        """Create config with credentials."""
        return cls(api_key=api_key, api_secret=api_secret)

    @classmethod
    def testnet_config(cls) -> "BybitConfig":
        """Create testnet config."""
        return cls(testnet=True)


class BybitClient:
    """Bybit REST API client."""

    def __init__(self, config: Optional[BybitConfig] = None):
        self.config = config or BybitConfig()
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    def _sign(self, params: str) -> str:
        """Generate signature for authenticated requests."""
        if not self.config.api_secret:
            raise ValueError("API secret not configured")
        return hmac.new(
            self.config.api_secret.encode(),
            params.encode(),
            hashlib.sha256
        ).hexdigest()

    async def _get_public(self, endpoint: str, params: Dict) -> Dict:
        """Make a GET request to a public endpoint."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        url = f"{self.config.base_url}{endpoint}"
        async with self._session.get(url, params=params) as response:
            data = await response.json()

            if data.get("retCode") != 0:
                raise Exception(f"API error: {data.get('retMsg')}")

            return data.get("result", {})

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker for a symbol."""
        result = await self._get_public(
            "/v5/market/tickers",
            {"category": "linear", "symbol": symbol}
        )

        data = result.get("list", [{}])[0]

        return Ticker(
            symbol=data.get("symbol", symbol),
            last_price=float(data.get("lastPrice", 0)),
            bid_price=float(data.get("bid1Price", 0)),
            ask_price=float(data.get("ask1Price", 0)),
            bid_qty=float(data.get("bid1Size", 0)),
            ask_qty=float(data.get("ask1Size", 0)),
            high_24h=float(data.get("highPrice24h", 0)),
            low_24h=float(data.get("lowPrice24h", 0)),
            volume_24h=float(data.get("volume24h", 0)),
            turnover_24h=float(data.get("turnover24h", 0)),
            open_interest=float(data.get("openInterest", 0)) or None,
            funding_rate=float(data.get("fundingRate", 0)) or None,
        )

    async def get_orderbook(
        self, symbol: str, limit: int = 50
    ) -> OrderBook:
        """Get order book for a symbol."""
        result = await self._get_public(
            "/v5/market/orderbook",
            {"category": "linear", "symbol": symbol, "limit": limit}
        )

        book = OrderBook(symbol)

        for bid in result.get("b", []):
            book.update_bid(float(bid[0]), float(bid[1]))

        for ask in result.get("a", []):
            book.update_ask(float(ask[0]), float(ask[1]))

        book.sequence = result.get("u", 0)
        return book

    async def get_klines(
        self,
        symbol: str,
        interval: TimeFrame,
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> List[OhlcvBar]:
        """Get OHLCV bars (klines)."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval.value,
            "limit": limit,
        }

        if start:
            params["start"] = start
        if end:
            params["end"] = end

        result = await self._get_public("/v5/market/kline", params)

        bars = []
        for row in result.get("list", []):
            if len(row) >= 6:
                bars.append(OhlcvBar(
                    timestamp=datetime.fromtimestamp(int(row[0]) / 1000),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    turnover=float(row[6]) if len(row) > 6 else None,
                ))

        return bars

    async def get_trades(
        self, symbol: str, limit: int = 500
    ) -> List[Trade]:
        """Get recent trades."""
        result = await self._get_public(
            "/v5/market/recent-trade",
            {"category": "linear", "symbol": symbol, "limit": limit}
        )

        trades = []
        for data in result.get("list", []):
            trades.append(Trade(
                id=data.get("execId", ""),
                timestamp=datetime.fromtimestamp(int(data.get("time", 0)) / 1000),
                price=float(data.get("price", 0)),
                quantity=float(data.get("size", 0)),
                direction=TradeDirection.BUY if data.get("side") == "Buy" else TradeDirection.SELL,
            ))

        return trades

    def is_authenticated(self) -> bool:
        """Check if the client has authentication configured."""
        return bool(self.config.api_key and self.config.api_secret)
