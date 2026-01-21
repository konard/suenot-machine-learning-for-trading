"""
Execution strategy implementations.

This module provides various execution strategies:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Implementation Shortfall (IS)
- Adaptive strategy
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from .data import OrderBook
from .execution import ParentOrder, Side
from .impact import AlmgrenChrissModel, AlmgrenChrissParams


@dataclass
class ExecutionSlice:
    """Execution slice specification."""
    quantity: float
    limit_price: Optional[float] = None
    urgency: float = 0.5

    @classmethod
    def market(cls, quantity: float) -> "ExecutionSlice":
        """Create a market order slice."""
        return cls(quantity=quantity, limit_price=None, urgency=1.0)

    @classmethod
    def limit(cls, quantity: float, price: float) -> "ExecutionSlice":
        """Create a limit order slice."""
        return cls(quantity=quantity, limit_price=price, urgency=0.5)


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    slice_interval_secs: int = 60
    min_slice_size: float = 0.001
    max_slice_size: float = 10.0
    use_limit_orders: bool = True
    limit_offset_bps: float = 2.0
    max_spread_bps: float = 50.0


class ExecutionStrategy(ABC):
    """Base class for execution strategies."""

    @abstractmethod
    def next_slice(
        self, order: ParentOrder, orderbook: OrderBook
    ) -> ExecutionSlice:
        """Get the next slice to execute."""
        pass

    @abstractmethod
    def slice_interval_ms(self) -> int:
        """Get the recommended interval between slices (milliseconds)."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        pass

    def is_complete(self, order: ParentOrder) -> bool:
        """Check if strategy is complete."""
        return order.remaining_quantity() <= 0

    def reset(self) -> None:
        """Reset strategy state."""
        pass


def calculate_limit_price(
    orderbook: OrderBook,
    side: Side,
    aggressiveness: float,  # -1 (passive) to +1 (aggressive)
) -> Optional[float]:
    """Calculate limit price based on side and aggressiveness."""
    mid = orderbook.mid_price()
    spread = orderbook.spread()

    if mid is None or spread is None:
        return None

    offset = spread * 0.5 * aggressiveness

    if side == Side.BUY:
        return mid + offset
    else:
        return mid - offset


class TwapStrategy(ExecutionStrategy):
    """TWAP (Time-Weighted Average Price) execution strategy."""

    def __init__(
        self,
        slice_interval_secs: int = 60,
        config: Optional[StrategyConfig] = None,
    ):
        self.config = config or StrategyConfig(slice_interval_secs=slice_interval_secs)

    def next_slice(
        self, order: ParentOrder, orderbook: OrderBook
    ) -> ExecutionSlice:
        remaining = order.remaining_quantity()

        if remaining <= 0:
            return ExecutionSlice.market(0.0)

        # Check spread
        spread_bps = orderbook.spread_bps()
        if spread_bps is not None and spread_bps > self.config.max_spread_bps:
            return ExecutionSlice.market(0.0)

        # Calculate number of remaining slices
        remaining_time = order.remaining_time()
        num_slices = max(1, remaining_time // self.config.slice_interval_secs)

        # Uniform distribution
        slice_qty = remaining / num_slices
        slice_qty = min(slice_qty, self.config.max_slice_size)
        slice_qty = max(slice_qty, self.config.min_slice_size)
        slice_qty = min(slice_qty, remaining)

        if self.config.use_limit_orders:
            aggressiveness = order.urgency * 0.5
            price = calculate_limit_price(orderbook, order.side, aggressiveness)
            if price is not None:
                return ExecutionSlice.limit(slice_qty, price)

        return ExecutionSlice.market(slice_qty)

    def slice_interval_ms(self) -> int:
        return self.config.slice_interval_secs * 1000

    def name(self) -> str:
        return "TWAP"


@dataclass
class VolumeProfile:
    """Intraday volume profile."""
    weights: List[float]
    period_minutes: int = 60

    @classmethod
    def uniform(cls, num_periods: int) -> "VolumeProfile":
        """Create a uniform volume profile."""
        weight = 1.0 / num_periods
        return cls(weights=[weight] * num_periods)

    @classmethod
    def equity_u_shape(cls) -> "VolumeProfile":
        """Create a typical equity market U-shaped profile."""
        weights = [
            0.12, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05,
            0.06, 0.07, 0.08, 0.09, 0.10, 0.12,
        ]
        return cls(weights=weights, period_minutes=30)

    @classmethod
    def crypto_24h(cls) -> "VolumeProfile":
        """Create a crypto 24h profile."""
        weights = [
            0.035, 0.035, 0.040, 0.045, 0.050, 0.050,
            0.045, 0.045, 0.050, 0.055, 0.055, 0.050,
            0.045, 0.045, 0.050, 0.055, 0.060, 0.055,
            0.050, 0.045, 0.040, 0.035, 0.030, 0.030,
        ]
        return cls(weights=weights, period_minutes=60)


class VwapStrategy(ExecutionStrategy):
    """VWAP (Volume-Weighted Average Price) execution strategy."""

    def __init__(
        self,
        num_periods: int = 10,
        profile: Optional[VolumeProfile] = None,
        config: Optional[StrategyConfig] = None,
    ):
        self.profile = profile or VolumeProfile.uniform(num_periods)
        self.config = config or StrategyConfig()
        self.current_period = 0

    @classmethod
    def equity(cls) -> "VwapStrategy":
        """Create for equity markets."""
        return cls(profile=VolumeProfile.equity_u_shape())

    @classmethod
    def crypto(cls) -> "VwapStrategy":
        """Create for crypto markets."""
        return cls(profile=VolumeProfile.crypto_24h())

    def next_slice(
        self, order: ParentOrder, orderbook: OrderBook
    ) -> ExecutionSlice:
        remaining = order.remaining_quantity()

        if remaining <= 0:
            return ExecutionSlice.market(0.0)

        # Check spread
        spread_bps = orderbook.spread_bps()
        if spread_bps is not None and spread_bps > self.config.max_spread_bps:
            return ExecutionSlice.market(0.0)

        # Get weight for current period
        period = self.current_period % len(self.profile.weights)
        weight = self.profile.weights[period]

        # Calculate target quantity
        period_target = order.total_quantity * weight

        # Adjust based on progress
        time_progress = order.elapsed_time() / order.time_horizon if order.time_horizon > 0 else 0
        fill_progress = order.fill_rate()

        adjustment = 1.0
        if fill_progress < time_progress:
            adjustment = 1.2  # Behind schedule
        elif fill_progress > time_progress + 0.1:
            adjustment = 0.8  # Ahead of schedule

        slice_qty = period_target * adjustment
        slice_qty = min(slice_qty, self.config.max_slice_size)
        slice_qty = max(slice_qty, self.config.min_slice_size)
        slice_qty = min(slice_qty, remaining)

        if self.config.use_limit_orders:
            aggressiveness = 0.7 if fill_progress < time_progress else 0.3
            aggressiveness *= order.urgency
            price = calculate_limit_price(orderbook, order.side, aggressiveness)
            if price is not None:
                return ExecutionSlice.limit(slice_qty, price)

        return ExecutionSlice.market(slice_qty)

    def slice_interval_ms(self) -> int:
        return self.config.slice_interval_secs * 1000

    def name(self) -> str:
        return "VWAP"


class ImplementationShortfallStrategy(ExecutionStrategy):
    """Implementation Shortfall (IS) execution strategy."""

    def __init__(
        self,
        params: Optional[AlmgrenChrissParams] = None,
        config: Optional[StrategyConfig] = None,
    ):
        self.model = AlmgrenChrissModel(params)
        self.config = config or StrategyConfig()
        self.trajectory: Optional[List[float]] = None
        self.current_step = 0

    def next_slice(
        self, order: ParentOrder, orderbook: OrderBook
    ) -> ExecutionSlice:
        remaining = order.remaining_quantity()

        if remaining <= 0:
            return ExecutionSlice.market(0.0)

        # Check spread
        spread_bps = orderbook.spread_bps()
        if spread_bps is not None and spread_bps > self.config.max_spread_bps:
            return ExecutionSlice.market(0.0)

        # Calculate trajectory if not set
        if self.trajectory is None:
            num_steps = max(1, order.remaining_time() // self.config.slice_interval_secs)
            self.trajectory = self.model.optimal_trajectory(remaining, num_steps)

        # Get slice from trajectory
        if self.current_step < len(self.trajectory):
            base_qty = self.trajectory[self.current_step]
            self.current_step += 1
        else:
            base_qty = remaining

        # Adjust for market conditions
        depth = orderbook.ask_depth(5) if order.side == Side.BUY else orderbook.bid_depth(5)
        if depth > 0:
            base_qty = min(base_qty, depth * 0.1)

        slice_qty = min(base_qty, self.config.max_slice_size)
        slice_qty = max(slice_qty, self.config.min_slice_size)
        slice_qty = min(slice_qty, remaining)

        # Calculate aggressiveness
        time_progress = order.elapsed_time() / order.time_horizon if order.time_horizon > 0 else 0
        fill_progress = order.fill_rate()
        schedule_diff = fill_progress - time_progress

        aggressiveness = (order.urgency - 0.5) * 2.0 - schedule_diff * 2.0
        aggressiveness = max(-1.0, min(1.0, aggressiveness))

        if self.config.use_limit_orders and aggressiveness < 0.8:
            price = calculate_limit_price(orderbook, order.side, aggressiveness)
            if price is not None:
                return ExecutionSlice.limit(slice_qty, price)

        return ExecutionSlice.market(slice_qty)

    def slice_interval_ms(self) -> int:
        return self.config.slice_interval_secs * 1000

    def name(self) -> str:
        return "ImplementationShortfall"

    def reset(self) -> None:
        self.trajectory = None
        self.current_step = 0


class AdaptiveStrategy(ExecutionStrategy):
    """Adaptive execution strategy based on market conditions."""

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        spread_threshold_bps: float = 10.0,
        imbalance_threshold: float = 0.3,
    ):
        self.config = config or StrategyConfig()
        self.spread_threshold_bps = spread_threshold_bps
        self.imbalance_threshold = imbalance_threshold

    def next_slice(
        self, order: ParentOrder, orderbook: OrderBook
    ) -> ExecutionSlice:
        remaining = order.remaining_quantity()

        if remaining <= 0:
            return ExecutionSlice.market(0.0)

        spread_bps = orderbook.spread_bps() or 100.0
        imbalance = orderbook.imbalance(10)

        # Determine favorable conditions
        favorable = (
            (order.side == Side.BUY and imbalance < -self.imbalance_threshold)
            or (order.side == Side.SELL and imbalance > self.imbalance_threshold)
        )

        # Time pressure
        time_remaining_ratio = order.remaining_time() / order.time_horizon if order.time_horizon > 0 else 1.0
        urgent = time_remaining_ratio < 0.2 or order.urgency > 0.7

        # Decision logic
        if urgent:
            mode = "aggressive"
        elif spread_bps > self.spread_threshold_bps:
            mode = "wait"
        elif favorable:
            mode = "aggressive"
        elif spread_bps < self.spread_threshold_bps / 2.0:
            mode = "opportunistic"
        else:
            mode = "passive"

        if mode == "wait":
            return ExecutionSlice.market(0.0)

        # Calculate quantity
        num_slices = max(1, order.remaining_time() // self.config.slice_interval_secs)
        base_qty = remaining / num_slices

        mode_factor = {
            "aggressive": 1.5,
            "opportunistic": 1.2,
            "passive": 0.8,
        }.get(mode, 1.0)

        slice_qty = base_qty * mode_factor
        slice_qty = min(slice_qty, self.config.max_slice_size)
        slice_qty = max(slice_qty, self.config.min_slice_size)
        slice_qty = min(slice_qty, remaining)

        # Depth limit
        depth = orderbook.ask_depth(5) if order.side == Side.BUY else orderbook.bid_depth(5)
        if depth > 0:
            slice_qty = min(slice_qty, depth * 0.1)

        aggressiveness = {
            "aggressive": 0.8,
            "opportunistic": 0.4,
            "passive": -0.3,
        }.get(mode, 0.0)

        if self.config.use_limit_orders and mode != "aggressive":
            price = calculate_limit_price(orderbook, order.side, aggressiveness)
            if price is not None:
                return ExecutionSlice.limit(slice_qty, price)

        return ExecutionSlice.market(slice_qty)

    def slice_interval_ms(self) -> int:
        return self.config.slice_interval_secs * 1000

    def name(self) -> str:
        return "Adaptive"
