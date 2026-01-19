"""
Chain-of-Thought Position Sizing

This module provides position sizing with explicit reasoning chains,
making risk management decisions transparent and auditable.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation with reasoning."""
    symbol: str
    recommended_size: float
    size_in_units: int
    risk_amount: float
    reasoning_chain: List[str]
    risk_metrics: Dict[str, float]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'recommended_size': self.recommended_size,
            'size_in_units': self.size_in_units,
            'risk_amount': self.risk_amount,
            'reasoning_chain': self.reasoning_chain,
            'risk_metrics': self.risk_metrics,
            'timestamp': self.timestamp.isoformat()
        }


class CoTPositionSizer:
    """
    Chain-of-Thought position sizing for trading.

    Uses explicit reasoning steps to determine optimal position sizes
    based on risk tolerance, account size, and market conditions.

    Attributes:
        account_size: Total account value
        max_risk_per_trade: Maximum risk per trade as fraction
        max_position_size: Maximum single position as fraction
        max_correlated_exposure: Maximum exposure to correlated assets

    Example:
        >>> sizer = CoTPositionSizer(account_size=100000)
        >>> result = sizer.calculate_position_size(
        ...     symbol="BTCUSDT",
        ...     entry_price=43250,
        ...     stop_loss=42000,
        ...     signal_confidence=0.75
        ... )
        >>> print(result.size_in_units, result.risk_amount)
    """

    def __init__(
        self,
        account_size: float,
        max_risk_per_trade: float = 0.02,
        max_position_size: float = 0.10,
        max_correlated_exposure: float = 0.25
    ):
        """
        Initialize position sizer.

        Args:
            account_size: Total account value
            max_risk_per_trade: Max risk per trade (default 2%)
            max_position_size: Max position size (default 10%)
            max_correlated_exposure: Max correlated exposure (default 25%)
        """
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.max_correlated_exposure = max_correlated_exposure
        self.reasoning_chain: List[str] = []

    def _add_reasoning(self, step: str):
        """Add a step to the reasoning chain."""
        self.reasoning_chain.append(step)

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        signal_confidence: float,
        current_positions: Optional[Dict[str, float]] = None,
        correlations: Optional[Dict[str, float]] = None,
        volatility: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calculate position size with explicit reasoning chain.

        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss: Stop loss price
            signal_confidence: Confidence in the signal (0-1)
            current_positions: Dict of symbol -> position value
            correlations: Dict of symbol -> correlation with new trade
            volatility: Current volatility (e.g., ATR as % of price)

        Returns:
            PositionSizeResult with recommended size and reasoning
        """
        self.reasoning_chain = []
        current_positions = current_positions or {}
        correlations = correlations or {}

        # Step 1: Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        risk_percent = risk_per_share / entry_price if entry_price > 0 else 0
        self._add_reasoning(
            f"STEP 1 - RISK PER SHARE: "
            f"Entry ${entry_price:,.2f}, Stop ${stop_loss:,.2f}. "
            f"Risk per share: ${risk_per_share:,.2f} ({risk_percent:.2%} of entry price)."
        )

        # Step 2: Calculate base position size from risk tolerance
        risk_amount = self.account_size * self.max_risk_per_trade
        base_shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        base_position_value = base_shares * entry_price
        base_position_pct = base_position_value / self.account_size if self.account_size > 0 else 0
        self._add_reasoning(
            f"STEP 2 - BASE POSITION SIZE: "
            f"Max risk per trade: {self.max_risk_per_trade:.1%} = ${risk_amount:,.2f}. "
            f"Base position: {base_shares:,} shares (${base_position_value:,.2f}, "
            f"{base_position_pct:.1%} of portfolio)."
        )

        # Step 3: Apply confidence adjustment
        confidence_multiplier = self._calculate_confidence_multiplier(signal_confidence)
        confidence_adjusted_shares = int(base_shares * confidence_multiplier)
        self._add_reasoning(
            f"STEP 3 - CONFIDENCE ADJUSTMENT: "
            f"Signal confidence: {signal_confidence:.1%}. "
            f"Confidence multiplier: {confidence_multiplier:.2f}. "
            f"Adjusted shares: {confidence_adjusted_shares:,}."
        )

        # Step 4: Apply volatility adjustment
        if volatility is not None:
            vol_multiplier = self._calculate_volatility_multiplier(volatility)
            volatility_adjusted_shares = int(confidence_adjusted_shares * vol_multiplier)
            self._add_reasoning(
                f"STEP 4 - VOLATILITY ADJUSTMENT: "
                f"Current volatility: {volatility:.2%}. "
                f"Volatility multiplier: {vol_multiplier:.2f}. "
                f"Vol-adjusted shares: {volatility_adjusted_shares:,}."
            )
        else:
            volatility_adjusted_shares = confidence_adjusted_shares
            self._add_reasoning(
                f"STEP 4 - VOLATILITY ADJUSTMENT: "
                f"No volatility data provided. Skipping adjustment."
            )

        # Step 5: Check correlation constraints
        corr_adjusted_shares, corr_reasoning = self._apply_correlation_constraint(
            volatility_adjusted_shares,
            entry_price,
            current_positions,
            correlations
        )
        self._add_reasoning(f"STEP 5 - CORRELATION CHECK: {corr_reasoning}")

        # Step 6: Apply maximum position size constraint
        max_shares = int((self.account_size * self.max_position_size) / entry_price)
        final_shares = min(corr_adjusted_shares, max_shares)
        if corr_adjusted_shares > max_shares:
            self._add_reasoning(
                f"STEP 6 - MAX SIZE CONSTRAINT: "
                f"Position capped at {self.max_position_size:.0%} of portfolio. "
                f"Reduced from {corr_adjusted_shares:,} to {final_shares:,} shares."
            )
        else:
            self._add_reasoning(
                f"STEP 6 - MAX SIZE CONSTRAINT: "
                f"Position ({final_shares:,} shares) within max limit "
                f"({max_shares:,} shares). No adjustment needed."
            )

        # Step 7: Final recommendation
        final_position_value = final_shares * entry_price
        final_position_pct = final_position_value / self.account_size if self.account_size > 0 else 0
        final_risk = final_shares * risk_per_share
        final_risk_pct = final_risk / self.account_size if self.account_size > 0 else 0

        self._add_reasoning(
            f"STEP 7 - FINAL RECOMMENDATION: "
            f"Buy {final_shares:,} shares at ${entry_price:,.2f} = "
            f"${final_position_value:,.2f} ({final_position_pct:.2%} of portfolio). "
            f"Risk: ${final_risk:,.2f} ({final_risk_pct:.2%} of portfolio)."
        )

        risk_metrics = {
            'risk_per_share': risk_per_share,
            'risk_percent_of_entry': risk_percent,
            'total_risk_amount': final_risk,
            'total_risk_pct': final_risk_pct,
            'position_pct': final_position_pct,
            'confidence_multiplier': confidence_multiplier,
            'base_shares': base_shares,
            'final_shares': final_shares
        }

        return PositionSizeResult(
            symbol=symbol,
            recommended_size=final_position_pct,
            size_in_units=final_shares,
            risk_amount=final_risk,
            reasoning_chain=self.reasoning_chain.copy(),
            risk_metrics=risk_metrics
        )

    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """
        Calculate position multiplier based on signal confidence.

        Args:
            confidence: Signal confidence (0-1)

        Returns:
            Multiplier for position size
        """
        if confidence >= 0.9:
            return 1.0
        elif confidence >= 0.8:
            return 0.8
        elif confidence >= 0.7:
            return 0.6
        elif confidence >= 0.6:
            return 0.4
        else:
            return 0.2

    def _calculate_volatility_multiplier(self, volatility: float) -> float:
        """
        Adjust position size based on volatility.

        Higher volatility = smaller position size.

        Args:
            volatility: Volatility as decimal (e.g., 0.02 for 2%)

        Returns:
            Multiplier for position size
        """
        # Assume "normal" volatility is 2%
        normal_vol = 0.02

        if volatility <= normal_vol * 0.5:
            return 1.2  # Low vol, can size up slightly
        elif volatility <= normal_vol:
            return 1.0
        elif volatility <= normal_vol * 1.5:
            return 0.75
        elif volatility <= normal_vol * 2:
            return 0.5
        else:
            return 0.25  # Very high vol, reduce significantly

    def _apply_correlation_constraint(
        self,
        shares: int,
        entry_price: float,
        current_positions: Dict[str, float],
        correlations: Dict[str, float]
    ) -> Tuple[int, str]:
        """
        Apply correlation-based position constraints.

        Args:
            shares: Proposed number of shares
            entry_price: Entry price
            current_positions: Existing positions
            correlations: Correlations with existing positions

        Returns:
            Tuple of (adjusted shares, reasoning string)
        """
        if not current_positions or not correlations:
            return shares, "No existing positions or correlation data. No adjustment needed."

        # Calculate correlated exposure
        correlated_exposure = 0
        position_value = shares * entry_price

        for symbol, pos_value in current_positions.items():
            corr = correlations.get(symbol, 0)
            if corr > 0.5:  # Only count significantly correlated positions
                correlated_exposure += pos_value * corr

        # Add new position's contribution
        new_total_correlated = correlated_exposure + position_value
        max_correlated = self.account_size * self.max_correlated_exposure

        if new_total_correlated > max_correlated:
            # Need to reduce position
            available_for_new = max_correlated - correlated_exposure
            adjusted_shares = int(max(0, available_for_new / entry_price))
            reasoning = (
                f"Correlated exposure would be ${new_total_correlated:,.2f} "
                f"(limit: ${max_correlated:,.2f}). "
                f"Reduced from {shares:,} to {adjusted_shares:,} shares."
            )
            return adjusted_shares, reasoning
        else:
            reasoning = (
                f"Correlated exposure: ${new_total_correlated:,.2f} "
                f"(within {self.max_correlated_exposure:.0%} limit). Position size OK."
            )
            return shares, reasoning

    def update_account_size(self, new_size: float):
        """
        Update the account size.

        Args:
            new_size: New account value
        """
        self.account_size = new_size


class KellyPositionSizer(CoTPositionSizer):
    """
    Position sizer using Kelly Criterion with CoT reasoning.

    The Kelly Criterion calculates the optimal position size to
    maximize long-term growth rate.
    """

    def __init__(
        self,
        account_size: float,
        max_risk_per_trade: float = 0.02,
        max_position_size: float = 0.10,
        kelly_fraction: float = 0.5  # Half-Kelly for safety
    ):
        """
        Initialize Kelly position sizer.

        Args:
            account_size: Total account value
            max_risk_per_trade: Maximum risk per trade
            max_position_size: Maximum position size
            kelly_fraction: Fraction of Kelly to use (default 0.5)
        """
        super().__init__(account_size, max_risk_per_trade, max_position_size)
        self.kelly_fraction = kelly_fraction

    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly-optimal position size.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)

        Returns:
            Optimal position size as fraction of account
        """
        if avg_loss == 0:
            return 0

        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate

        kelly = (b * p - q) / b

        # Apply Kelly fraction for safety
        return max(0, kelly * self.kelly_fraction)


if __name__ == "__main__":
    print("Chain-of-Thought Position Sizing Demo")
    print("=" * 50)

    # Initialize position sizer
    sizer = CoTPositionSizer(
        account_size=100000,
        max_risk_per_trade=0.02,
        max_position_size=0.10,
        max_correlated_exposure=0.25
    )

    # Calculate position for a trade
    result = sizer.calculate_position_size(
        symbol="BTCUSDT",
        entry_price=43250,
        stop_loss=42000,
        signal_confidence=0.75,
        current_positions={
            "ETHUSDT": 15000,
            "SOLUSDT": 5000
        },
        correlations={
            "ETHUSDT": 0.85,  # High correlation
            "SOLUSDT": 0.70   # Moderate correlation
        },
        volatility=0.025  # 2.5% ATR
    )

    print(f"\nPosition Sizing for: {result.symbol}")
    print("\n" + "=" * 50)
    print("REASONING CHAIN:")
    print("=" * 50)

    for step in result.reasoning_chain:
        print(f"\n{step}")

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    print(f"Recommended Position: {result.size_in_units:,} units "
          f"({result.recommended_size:.2%} of portfolio)")
    print(f"Risk Amount: ${result.risk_amount:,.2f}")
    print(f"\nRisk Metrics:")
    for key, value in result.risk_metrics.items():
        if isinstance(value, float):
            if value < 1:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value:,}")
