"""
Order Book Implementation for Market Simulation

A realistic limit order book with price-time priority matching.
Supports market orders, limit orders, and partial fills.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import heapq
from datetime import datetime


class OrderType(Enum):
    """Type of order"""
    MARKET = "market"
    LIMIT = "limit"


class Side(Enum):
    """Side of the order (buy or sell)"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """
    Represents a trading order

    Attributes:
        order_type: Type of order (market or limit)
        side: Side of the order (buy or sell)
        quantity: Number of shares to trade
        price: Limit price (None for market orders)
        agent_id: Identifier of the agent placing the order
        timestamp: Time when order was placed
        order_id: Unique identifier for the order
    """
    order_type: OrderType
    side: Side
    quantity: int
    price: Optional[float] = None
    agent_id: str = ""
    timestamp: int = 0
    order_id: int = 0


@dataclass
class OrderResult:
    """
    Result of submitting an order

    Attributes:
        filled_quantity: Number of shares filled
        average_price: Average price of fills
        remaining_quantity: Number of shares not filled
        status: Status of the order ("filled", "partial", "pending", "rejected")
        fills: List of individual fills (price, quantity)
    """
    filled_quantity: int
    average_price: float
    remaining_quantity: int
    status: str
    fills: List[Tuple[float, int]] = field(default_factory=list)


@dataclass
class BookLevel:
    """Represents a price level in the order book"""
    price: float
    orders: List[Tuple[int, int, str]]  # (order_id, quantity, agent_id)


class OrderBook:
    """
    Limit Order Book with Price-Time Priority

    Implements a realistic order book that matches orders based on
    price-time priority. Supports limit orders, market orders,
    partial fills, and order cancellation.

    Examples:
        >>> book = OrderBook(tick_size=0.01)
        >>> order = Order(OrderType.LIMIT, Side.BUY, 100, price=50.0, agent_id="agent1")
        >>> result = book.add_limit_order(order)
        >>> print(result.status)
        'pending'
    """

    def __init__(self, tick_size: float = 0.01):
        """
        Initialize order book

        Args:
            tick_size: Minimum price increment
        """
        self.tick_size = tick_size
        self.bids: List[Tuple[float, int, int, str]] = []  # (-price, timestamp, qty, agent_id)
        self.asks: List[Tuple[float, int, int, str]] = []  # (price, timestamp, qty, agent_id)
        self.order_id_counter = 0
        self.timestamp_counter = 0
        self.trade_history: List[dict] = []

    def _next_order_id(self) -> int:
        """Get next unique order ID"""
        self.order_id_counter += 1
        return self.order_id_counter

    def _next_timestamp(self) -> int:
        """Get next timestamp for ordering"""
        self.timestamp_counter += 1
        return self.timestamp_counter

    def add_limit_order(self, order: Order) -> OrderResult:
        """
        Add a limit order to the book

        Args:
            order: Limit order to add

        Returns:
            OrderResult with fill information
        """
        if order.order_type != OrderType.LIMIT or order.price is None:
            return OrderResult(0, 0.0, order.quantity, "rejected", [])

        order.order_id = self._next_order_id()
        order.timestamp = self._next_timestamp()

        if order.side == Side.BUY:
            return self._process_buy_limit(order)
        else:
            return self._process_sell_limit(order)

    def _process_buy_limit(self, order: Order) -> OrderResult:
        """Process a buy limit order"""
        fills = []
        filled_qty = 0
        remaining = order.quantity

        # Try to match against asks
        while remaining > 0 and self.asks:
            best_ask = self.asks[0]
            if order.price >= best_ask[0]:  # Can trade
                trade_qty = min(remaining, best_ask[2])
                trade_price = best_ask[0]

                fills.append((trade_price, trade_qty))
                filled_qty += trade_qty
                remaining -= trade_qty

                # Record trade
                self.trade_history.append({
                    "price": trade_price,
                    "quantity": trade_qty,
                    "buyer": order.agent_id,
                    "seller": best_ask[3],
                    "timestamp": order.timestamp
                })

                # Update or remove the ask
                if trade_qty >= best_ask[2]:
                    heapq.heappop(self.asks)
                else:
                    # Update remaining quantity
                    self.asks[0] = (best_ask[0], best_ask[1], best_ask[2] - trade_qty, best_ask[3])
            else:
                break

        # Add remaining to bid book
        if remaining > 0:
            heapq.heappush(self.bids, (-order.price, order.timestamp, remaining, order.agent_id))

        # Calculate average price
        avg_price = sum(p * q for p, q in fills) / filled_qty if filled_qty > 0 else 0.0

        if remaining == 0:
            status = "filled"
        elif filled_qty > 0:
            status = "partial"
        else:
            status = "pending"

        return OrderResult(filled_qty, avg_price, remaining, status, fills)

    def _process_sell_limit(self, order: Order) -> OrderResult:
        """Process a sell limit order"""
        fills = []
        filled_qty = 0
        remaining = order.quantity

        # Try to match against bids
        while remaining > 0 and self.bids:
            best_bid = self.bids[0]
            bid_price = -best_bid[0]  # Convert back from negative
            if order.price <= bid_price:  # Can trade
                trade_qty = min(remaining, best_bid[2])
                trade_price = bid_price

                fills.append((trade_price, trade_qty))
                filled_qty += trade_qty
                remaining -= trade_qty

                # Record trade
                self.trade_history.append({
                    "price": trade_price,
                    "quantity": trade_qty,
                    "buyer": best_bid[3],
                    "seller": order.agent_id,
                    "timestamp": order.timestamp
                })

                # Update or remove the bid
                if trade_qty >= best_bid[2]:
                    heapq.heappop(self.bids)
                else:
                    # Update remaining quantity
                    self.bids[0] = (best_bid[0], best_bid[1], best_bid[2] - trade_qty, best_bid[3])
            else:
                break

        # Add remaining to ask book
        if remaining > 0:
            heapq.heappush(self.asks, (order.price, order.timestamp, remaining, order.agent_id))

        # Calculate average price
        avg_price = sum(p * q for p, q in fills) / filled_qty if filled_qty > 0 else 0.0

        if remaining == 0:
            status = "filled"
        elif filled_qty > 0:
            status = "partial"
        else:
            status = "pending"

        return OrderResult(filled_qty, avg_price, remaining, status, fills)

    def execute_market_order(self, order: Order) -> OrderResult:
        """
        Execute a market order immediately

        Args:
            order: Market order to execute

        Returns:
            OrderResult with fill information
        """
        order.order_id = self._next_order_id()
        order.timestamp = self._next_timestamp()

        if order.side == Side.BUY:
            return self._execute_market_buy(order)
        else:
            return self._execute_market_sell(order)

    def _execute_market_buy(self, order: Order) -> OrderResult:
        """Execute a market buy order"""
        fills = []
        filled_qty = 0
        remaining = order.quantity

        while remaining > 0 and self.asks:
            best_ask = self.asks[0]
            trade_qty = min(remaining, best_ask[2])
            trade_price = best_ask[0]

            fills.append((trade_price, trade_qty))
            filled_qty += trade_qty
            remaining -= trade_qty

            self.trade_history.append({
                "price": trade_price,
                "quantity": trade_qty,
                "buyer": order.agent_id,
                "seller": best_ask[3],
                "timestamp": order.timestamp
            })

            if trade_qty >= best_ask[2]:
                heapq.heappop(self.asks)
            else:
                self.asks[0] = (best_ask[0], best_ask[1], best_ask[2] - trade_qty, best_ask[3])

        avg_price = sum(p * q for p, q in fills) / filled_qty if filled_qty > 0 else 0.0
        status = "filled" if remaining == 0 else ("partial" if filled_qty > 0 else "rejected")

        return OrderResult(filled_qty, avg_price, remaining, status, fills)

    def _execute_market_sell(self, order: Order) -> OrderResult:
        """Execute a market sell order"""
        fills = []
        filled_qty = 0
        remaining = order.quantity

        while remaining > 0 and self.bids:
            best_bid = self.bids[0]
            bid_price = -best_bid[0]
            trade_qty = min(remaining, best_bid[2])

            fills.append((bid_price, trade_qty))
            filled_qty += trade_qty
            remaining -= trade_qty

            self.trade_history.append({
                "price": bid_price,
                "quantity": trade_qty,
                "buyer": best_bid[3],
                "seller": order.agent_id,
                "timestamp": order.timestamp
            })

            if trade_qty >= best_bid[2]:
                heapq.heappop(self.bids)
            else:
                self.bids[0] = (best_bid[0], best_bid[1], best_bid[2] - trade_qty, best_bid[3])

        avg_price = sum(p * q for p, q in fills) / filled_qty if filled_qty > 0 else 0.0
        status = "filled" if remaining == 0 else ("partial" if filled_qty > 0 else "rejected")

        return OrderResult(filled_qty, avg_price, remaining, status, fills)

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        if self.bids:
            return -self.bids[0][0]
        return None

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        if self.asks:
            return self.asks[0][0]
        return None

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        bid, ask = self.get_best_bid(), self.get_best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return None

    def get_mid_price(self) -> Optional[float]:
        """Get mid-market price"""
        bid, ask = self.get_best_bid(), self.get_best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None

    def get_bid_depth(self, levels: int = 5) -> List[Tuple[float, int]]:
        """Get bid side depth (price, total quantity)"""
        depth = {}
        for neg_price, _, qty, _ in self.bids:
            price = -neg_price
            depth[price] = depth.get(price, 0) + qty

        sorted_depth = sorted(depth.items(), reverse=True)[:levels]
        return sorted_depth

    def get_ask_depth(self, levels: int = 5) -> List[Tuple[float, int]]:
        """Get ask side depth (price, total quantity)"""
        depth = {}
        for price, _, qty, _ in self.asks:
            depth[price] = depth.get(price, 0) + qty

        sorted_depth = sorted(depth.items())[:levels]
        return sorted_depth

    def get_last_trade_price(self) -> Optional[float]:
        """Get the price of the last trade"""
        if self.trade_history:
            return self.trade_history[-1]["price"]
        return None

    def clear(self):
        """Clear all orders from the book"""
        self.bids.clear()
        self.asks.clear()


if __name__ == "__main__":
    # Example usage
    book = OrderBook(tick_size=0.01)

    # Add some limit orders
    buy_order = Order(OrderType.LIMIT, Side.BUY, 100, price=99.50, agent_id="buyer1")
    result = book.add_limit_order(buy_order)
    print(f"Buy order: {result.status}")

    sell_order = Order(OrderType.LIMIT, Side.SELL, 50, price=100.50, agent_id="seller1")
    result = book.add_limit_order(sell_order)
    print(f"Sell order: {result.status}")

    print(f"Best bid: {book.get_best_bid()}")
    print(f"Best ask: {book.get_best_ask()}")
    print(f"Spread: {book.get_spread()}")

    # Execute a market order
    market_buy = Order(OrderType.MARKET, Side.BUY, 30, agent_id="market_buyer")
    result = book.execute_market_order(market_buy)
    print(f"Market buy: filled {result.filled_quantity} @ {result.average_price:.2f}")
