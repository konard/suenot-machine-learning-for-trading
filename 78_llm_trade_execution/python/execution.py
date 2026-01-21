"""
Execution engine and order management.

This module provides:
- Parent and child order structures
- Execution engine for managing order flow
- Execution configuration and results
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from .data import OrderBook


class Side(Enum):
    """Order side."""
    BUY = "Buy"
    SELL = "Sell"

    def opposite(self) -> "Side":
        """Get the opposite side."""
        return Side.SELL if self == Side.BUY else Side.BUY

    def sign(self) -> float:
        """Get the sign for calculations."""
        return 1.0 if self == Side.BUY else -1.0


class ParentOrderStatus(Enum):
    """Parent order status."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (
            ParentOrderStatus.COMPLETED,
            ParentOrderStatus.CANCELLED,
            ParentOrderStatus.FAILED,
        )


class ChildOrderStatus(Enum):
    """Child order status."""
    PENDING = "pending"
    SENT = "sent"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ParentOrder:
    """Parent order - the high-level order to be executed."""
    symbol: str
    side: Side
    total_quantity: float
    time_horizon: int  # seconds
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filled_quantity: float = 0.0
    status: ParentOrderStatus = ParentOrderStatus.PENDING
    arrival_price: Optional[float] = None
    average_price: Optional[float] = None
    max_participation: float = 0.10
    urgency: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    limit_price: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return max(0.0, self.total_quantity - self.filled_quantity)

    def fill_rate(self) -> float:
        """Get fill rate (fraction filled)."""
        if self.total_quantity > 0:
            return self.filled_quantity / self.total_quantity
        return 0.0

    def remaining_time(self) -> int:
        """Get remaining time in seconds."""
        if self.started_at:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds()
            return max(0, int(self.time_horizon - elapsed))
        return self.time_horizon

    def elapsed_time(self) -> int:
        """Get elapsed time in seconds."""
        if self.started_at:
            return max(0, int((datetime.utcnow() - self.started_at).total_seconds()))
        return 0

    def is_time_expired(self) -> bool:
        """Check if the order should complete based on time."""
        return self.remaining_time() == 0

    def record_fill(self, quantity: float, price: float) -> None:
        """Record a fill."""
        new_filled = self.filled_quantity + quantity
        old_value = (self.average_price or 0.0) * self.filled_quantity
        new_value = old_value + price * quantity

        self.filled_quantity = new_filled
        self.average_price = new_value / new_filled

        if self.remaining_quantity() <= 0:
            self.status = ParentOrderStatus.COMPLETED
            self.completed_at = datetime.utcnow()

    def start(self, arrival_price: float) -> None:
        """Start execution."""
        self.status = ParentOrderStatus.ACTIVE
        self.started_at = datetime.utcnow()
        self.arrival_price = arrival_price

    def cancel(self) -> None:
        """Cancel the order."""
        if not self.status.is_terminal():
            self.status = ParentOrderStatus.CANCELLED
            self.completed_at = datetime.utcnow()


@dataclass
class ChildOrder:
    """Child order - individual slice sent to exchange."""
    parent_id: str
    symbol: str
    side: Side
    quantity: float
    limit_price: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filled_quantity: float = 0.0
    fill_price: Optional[float] = None
    status: ChildOrderStatus = ChildOrderStatus.PENDING
    exchange_order_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    slice_index: Optional[int] = None

    def remaining_quantity(self) -> float:
        """Get remaining quantity."""
        return max(0.0, self.quantity - self.filled_quantity)

    def is_market(self) -> bool:
        """Check if this is a market order."""
        return self.limit_price is None

    def record_fill(self, quantity: float, price: float) -> None:
        """Record a fill."""
        new_filled = self.filled_quantity + quantity
        old_value = (self.fill_price or 0.0) * self.filled_quantity
        new_value = old_value + price * quantity

        self.filled_quantity = new_filled
        self.fill_price = new_value / new_filled

        if self.remaining_quantity() <= 0:
            self.status = ChildOrderStatus.FILLED
        else:
            self.status = ChildOrderStatus.PARTIALLY_FILLED


@dataclass
class ExecutionConfig:
    """Execution engine configuration."""
    min_slice_size: float = 0.001
    max_slice_size: float = 1.0
    min_slice_interval_ms: int = 1000
    max_slice_interval_ms: int = 60000
    use_llm: bool = False
    llm_interval_ms: int = 5000
    max_impact_bps: float = 50.0
    max_participation_rate: float = 0.25
    adaptive_sizing: bool = True
    verbose: bool = False

    @classmethod
    def aggressive(cls) -> "ExecutionConfig":
        """Create config for aggressive execution."""
        return cls(
            max_participation_rate=0.35,
            max_impact_bps=100.0,
            min_slice_interval_ms=500,
        )

    @classmethod
    def passive(cls) -> "ExecutionConfig":
        """Create config for passive execution."""
        return cls(
            max_participation_rate=0.10,
            max_impact_bps=20.0,
            min_slice_interval_ms=5000,
        )

    @classmethod
    def with_llm(cls) -> "ExecutionConfig":
        """Create config for LLM-assisted execution."""
        return cls(use_llm=True, adaptive_sizing=True)


@dataclass
class ExecutionResult:
    """Execution result metrics."""
    order_id: str
    symbol: str
    side: Side
    total_quantity: float
    filled_quantity: float
    child_order_count: int
    average_price: float
    arrival_price: float
    market_vwap: float
    implementation_shortfall: float  # in bps
    vwap_slippage: float  # in bps
    participation_rate: float
    start_time: datetime
    end_time: datetime
    duration_secs: int
    status: ParentOrderStatus
    llm_decisions: List[Any] = field(default_factory=list)

    def execution_cost(self) -> float:
        """Calculate execution cost in quote currency."""
        return self.filled_quantity * self.average_price

    def theoretical_cost(self) -> float:
        """Calculate theoretical cost at arrival price."""
        return self.filled_quantity * self.arrival_price

    def absolute_slippage(self) -> float:
        """Calculate absolute slippage."""
        sign = self.side.sign()
        return (self.average_price - self.arrival_price) * self.filled_quantity * sign

    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return (
            self.status == ParentOrderStatus.COMPLETED
            and self.filled_quantity >= self.total_quantity * 0.99
        )


class ExecutionEngine:
    """Execution engine for managing order lifecycle."""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.parent_orders: Dict[str, ParentOrder] = {}
        self.child_orders: Dict[str, List[ChildOrder]] = {}
        self.llm_adapter = None
        self.impact_estimator = None

    def with_llm_adapter(self, adapter: Any) -> "ExecutionEngine":
        """Set the LLM adapter."""
        self.llm_adapter = adapter
        return self

    def with_impact_estimator(self, estimator: Any) -> "ExecutionEngine":
        """Set the impact estimator."""
        self.impact_estimator = estimator
        return self

    async def execute(
        self,
        order: ParentOrder,
        strategy: Any,
    ) -> ExecutionResult:
        """Execute a parent order with a given strategy."""
        import random

        # Validate order
        if order.total_quantity <= 0:
            raise ValueError("Quantity must be positive")
        if order.time_horizon <= 0:
            raise ValueError("Time horizon must be positive")

        # Get arrival price (simulated)
        arrival_price = self._get_simulated_price(order.symbol)
        order.start(arrival_price)

        # Store order
        self.parent_orders[order.id] = order
        self.child_orders[order.id] = []

        # Execute loop
        slice_count = 0
        max_slices = 1000
        llm_decisions = []

        while slice_count < max_slices:
            parent = self.parent_orders[order.id]

            if parent.remaining_quantity() <= 0:
                break

            if parent.is_time_expired():
                break

            # Get simulated orderbook
            orderbook = self._get_simulated_orderbook(parent.symbol)

            # Get next slice from strategy
            slice_spec = strategy.next_slice(parent, orderbook)

            if slice_spec.quantity > 0:
                # Create child order
                child = ChildOrder(
                    parent_id=order.id,
                    symbol=parent.symbol,
                    side=parent.side,
                    quantity=slice_spec.quantity,
                    limit_price=slice_spec.limit_price,
                    slice_index=slice_count,
                )

                # Simulate fill
                fill_price = self._simulate_fill(child, orderbook)

                # Record fill
                parent.record_fill(child.quantity, fill_price)
                child.record_fill(child.quantity, fill_price)

                self.child_orders[order.id].append(child)
                slice_count += 1

            # Wait before next slice
            await asyncio.sleep(strategy.slice_interval_ms() / 1000.0)

        # Build result
        parent = self.parent_orders[order.id]
        end_time = datetime.utcnow()
        duration = int((end_time - parent.started_at).total_seconds()) if parent.started_at else 0

        avg_price = parent.average_price or arrival_price
        is_bps = ((avg_price - arrival_price) / arrival_price * 10000.0) * parent.side.sign()

        market_vwap = arrival_price * (1.0 + (random.random() - 0.5) * 0.001)
        vwap_slippage = ((avg_price - market_vwap) / market_vwap * 10000.0) * parent.side.sign()

        return ExecutionResult(
            order_id=order.id,
            symbol=parent.symbol,
            side=parent.side,
            total_quantity=parent.total_quantity,
            filled_quantity=parent.filled_quantity,
            child_order_count=slice_count,
            average_price=avg_price,
            arrival_price=arrival_price,
            market_vwap=market_vwap,
            implementation_shortfall=is_bps,
            vwap_slippage=vwap_slippage,
            participation_rate=0.0,
            start_time=parent.started_at or parent.created_at,
            end_time=end_time,
            duration_secs=duration,
            status=parent.status,
            llm_decisions=llm_decisions,
        )

    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated price for a symbol."""
        import random

        if "BTC" in symbol:
            return 50000.0 + (random.random() - 0.5) * 100.0
        elif "ETH" in symbol:
            return 3000.0 + (random.random() - 0.5) * 10.0
        return 100.0 + (random.random() - 0.5) * 1.0

    def _get_simulated_orderbook(self, symbol: str) -> OrderBook:
        """Get simulated orderbook."""
        import random

        mid_price = self._get_simulated_price(symbol)
        book = OrderBook(symbol)

        for i in range(1, 21):
            bid_price = mid_price * (1.0 - 0.0001 * i)
            ask_price = mid_price * (1.0 + 0.0001 * i)
            qty = 1.0 + random.random() * 5.0

            book.update_bid(bid_price, qty)
            book.update_ask(ask_price, qty)

        return book

    def _simulate_fill(self, order: ChildOrder, orderbook: OrderBook) -> float:
        """Simulate a fill."""
        if order.side == Side.BUY:
            result = orderbook.buy_impact(order.quantity)
        else:
            result = orderbook.sell_impact(order.quantity)

        if result:
            avg_price, _ = result
            return avg_price

        return orderbook.mid_price() or 0.0
