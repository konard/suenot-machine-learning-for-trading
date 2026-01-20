"""
Market impact models and estimation.

This module provides:
- Almgren-Chriss optimal execution model
- Temporary and permanent impact estimation
- Impact calibration utilities
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from .data import OrderBook


@dataclass
class ImpactComponent:
    """Market impact component."""
    permanent: float
    temporary: float

    @property
    def total(self) -> float:
        """Get total impact."""
        return self.permanent + self.temporary

    def as_bps(self) -> float:
        """Get impact in basis points."""
        return self.total * 10000.0


@dataclass
class AlmgrenChrissParams:
    """Almgren-Chriss model parameters."""
    sigma: float = 0.02        # Daily volatility
    daily_volume: float = 1e6  # Daily trading volume
    gamma: float = 0.1         # Permanent impact coefficient
    eta: float = 0.2           # Temporary impact coefficient
    lambda_: float = 1e-6      # Risk aversion parameter
    spread: float = 0.0005     # Bid-ask spread (as fraction)

    @classmethod
    def liquid(cls) -> "AlmgrenChrissParams":
        """Create parameters for a liquid asset."""
        return cls(
            sigma=0.015,
            daily_volume=1e7,
            gamma=0.05,
            eta=0.1,
            lambda_=1e-7,
            spread=0.0002,
        )

    @classmethod
    def illiquid(cls) -> "AlmgrenChrissParams":
        """Create parameters for an illiquid asset."""
        return cls(
            sigma=0.04,
            daily_volume=1e5,
            gamma=0.3,
            eta=0.5,
            lambda_=1e-5,
            spread=0.002,
        )

    @classmethod
    def crypto(cls) -> "AlmgrenChrissParams":
        """Create parameters for crypto markets."""
        return cls(
            sigma=0.04,
            daily_volume=1e8,
            gamma=0.08,
            eta=0.15,
            lambda_=1e-6,
            spread=0.0003,
        )


class AlmgrenChrissModel:
    """Almgren-Chriss optimal execution model."""

    def __init__(self, params: Optional[AlmgrenChrissParams] = None):
        self.params = params or AlmgrenChrissParams()

    def _kappa(self, time_horizon: float) -> float:
        """Calculate the urgency parameter."""
        lambda_ = self.params.lambda_
        sigma = self.params.sigma
        eta = self.params.eta

        if eta <= 0 or time_horizon <= 0:
            return 0.0

        return math.sqrt((lambda_ * sigma ** 2) / eta) / math.sqrt(time_horizon)

    def calculate_impact(
        self, quantity: float, time_horizon: float
    ) -> ImpactComponent:
        """Calculate the impact of executing a given quantity."""
        fraction = quantity / self.params.daily_volume
        rate = fraction / time_horizon if time_horizon > 0 else fraction

        # Permanent impact
        permanent = self.params.gamma * fraction

        # Temporary impact
        temporary = self.params.eta * rate + self.params.spread / 2.0

        return ImpactComponent(permanent=permanent, temporary=temporary)

    def optimal_trajectory(
        self, total_quantity: float, num_periods: int
    ) -> List[float]:
        """Calculate the optimal execution trajectory."""
        if num_periods <= 0:
            return []

        time_horizon = float(num_periods)
        kappa = self._kappa(time_horizon)

        trajectory = []
        remaining = total_quantity

        for t in range(num_periods):
            time = float(t)
            time_remaining = time_horizon - time

            # Optimal trading rate from Almgren-Chriss
            sinh_kt = math.sinh(kappa * time_remaining) if kappa > 0 else time_remaining
            sinh_kT = math.sinh(kappa * time_horizon) if kappa > 0 else time_horizon

            if sinh_kT > 0:
                trade_fraction = sinh_kt / sinh_kT
            else:
                trade_fraction = 1.0 / num_periods

            if t == num_periods - 1:
                trade_quantity = remaining
            else:
                target = total_quantity * (1.0 - trade_fraction)
                trade_quantity = (total_quantity - remaining) - target + total_quantity / num_periods
                trade_quantity = max(0.0, min(remaining, trade_quantity))

            trajectory.append(trade_quantity)
            remaining -= trade_quantity

        return trajectory

    def expected_cost(self, quantity: float, time_horizon: float) -> float:
        """Calculate expected execution cost."""
        impact = self.calculate_impact(quantity, time_horizon)
        return quantity * (impact.permanent + impact.temporary)

    def execution_risk(self, quantity: float, time_horizon: float) -> float:
        """Calculate execution risk (variance)."""
        sigma = self.params.sigma
        fraction = quantity / self.params.daily_volume
        return sigma ** 2 * fraction ** 2 * time_horizon


class MarketImpactEstimator:
    """Market impact estimator with calibration."""

    def __init__(self, params: Optional[AlmgrenChrissParams] = None):
        self.model = AlmgrenChrissModel(params)
        self.use_orderbook_depth = True

    @classmethod
    def crypto(cls) -> "MarketImpactEstimator":
        """Create estimator for crypto markets."""
        return cls(AlmgrenChrissParams.crypto())

    @classmethod
    def liquid_equity(cls) -> "MarketImpactEstimator":
        """Create estimator for liquid equity markets."""
        return cls(AlmgrenChrissParams.liquid())

    @classmethod
    def illiquid(cls) -> "MarketImpactEstimator":
        """Create estimator for illiquid markets."""
        return cls(AlmgrenChrissParams.illiquid())

    def estimate(
        self,
        quantity: float,
        time_horizon: float,
        orderbook: Optional[OrderBook] = None,
    ) -> "ImpactEstimate":
        """Estimate impact for a given quantity and time horizon."""
        impact = self.model.calculate_impact(quantity, time_horizon)

        # Adjust based on order book depth if available
        if self.use_orderbook_depth and orderbook:
            impact = self._adjust_for_orderbook(impact, quantity, orderbook)

        expected_cost = self.model.expected_cost(quantity, time_horizon)
        execution_risk = self.model.execution_risk(quantity, time_horizon)

        return ImpactEstimate(
            impact=impact,
            confidence=0.7,
            expected_cost=expected_cost,
            execution_risk=execution_risk,
            model_name="AlmgrenChriss",
        )

    def _adjust_for_orderbook(
        self,
        impact: ImpactComponent,
        quantity: float,
        orderbook: OrderBook,
    ) -> ImpactComponent:
        """Adjust impact estimate based on order book depth."""
        bid_depth = orderbook.bid_depth(20)
        ask_depth = orderbook.ask_depth(20)
        total_depth = bid_depth + ask_depth

        if total_depth <= 0:
            return impact

        depth_ratio = quantity / total_depth

        if depth_ratio > 0.5:
            depth_multiplier = 2.0
        elif depth_ratio > 0.2:
            depth_multiplier = 1.5
        elif depth_ratio > 0.1:
            depth_multiplier = 1.2
        else:
            depth_multiplier = 1.0

        return ImpactComponent(
            permanent=impact.permanent,
            temporary=impact.temporary * depth_multiplier,
        )

    def optimal_trajectory(
        self, total_quantity: float, num_periods: int
    ) -> List[float]:
        """Get the optimal execution trajectory."""
        return self.model.optimal_trajectory(total_quantity, num_periods)

    def estimate_book_impact(
        self,
        quantity: float,
        orderbook: OrderBook,
        is_buy: bool,
    ) -> Optional[ImpactComponent]:
        """Estimate impact of executing through the order book."""
        if is_buy:
            result = orderbook.buy_impact(quantity)
        else:
            result = orderbook.sell_impact(quantity)

        if result is None:
            return None

        avg_price, price_impact = result
        mid_price = orderbook.mid_price()

        if mid_price is None or mid_price <= 0:
            return None

        impact_fraction = abs(avg_price - mid_price) / mid_price

        return ImpactComponent(
            permanent=impact_fraction * 0.3,
            temporary=impact_fraction * 0.7,
        )


@dataclass
class ImpactEstimate:
    """Impact estimation result."""
    impact: ImpactComponent
    confidence: float
    expected_cost: float
    execution_risk: float
    model_name: str


def optimal_execution_time(
    quantity: float,
    daily_volume: float,
    volatility: float,
    risk_aversion: float,
    eta: float,
) -> float:
    """Calculate the optimal execution time given risk aversion."""
    fraction = quantity / daily_volume
    if risk_aversion <= 0 or volatility <= 0:
        return 1.0
    return math.sqrt(eta / (risk_aversion * volatility ** 2)) * math.sqrt(fraction)


def required_participation_rate(
    quantity: float,
    daily_volume: float,
    time_horizon: float,
) -> float:
    """Calculate participation rate needed to complete in given time."""
    volume_per_hour = daily_volume / 6.5  # 6.5 trading hours per day
    if volume_per_hour <= 0 or time_horizon <= 0:
        return 0.0
    return (quantity / time_horizon) / volume_per_hour
