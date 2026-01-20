"""
Trading signal generation from regime classification.

This module converts regime classifications into actionable trading signals
with position sizing and risk management recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .classifier import MarketRegime, RegimeResult


class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class TradingSignal:
    """Trading signal with position sizing."""
    signal_type: SignalType
    position_size: float  # -1 to 1 (negative = short)
    stop_loss: Optional[float]  # As decimal (e.g., 0.05 = 5%)
    take_profit: Optional[float]
    confidence: float
    regime: MarketRegime
    reasoning: str


class RegimeSignalGenerator:
    """
    Generate trading signals based on market regime classification.

    Maps regimes to appropriate position sizes and risk parameters.
    """

    def __init__(
        self,
        regime_positions: Optional[Dict[MarketRegime, Dict]] = None,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize signal generator.

        Args:
            regime_positions: Custom mapping of regimes to position parameters
            confidence_threshold: Minimum confidence to generate signals
        """
        self.confidence_threshold = confidence_threshold

        # Default regime-to-position mapping
        self.regime_positions = regime_positions or {
            MarketRegime.BULL: {
                'signal': SignalType.BUY,
                'position': 1.0,
                'stop_loss': 0.05,
                'take_profit': 0.15
            },
            MarketRegime.BEAR: {
                'signal': SignalType.SELL,
                'position': -0.5,
                'stop_loss': 0.03,
                'take_profit': 0.10
            },
            MarketRegime.SIDEWAYS: {
                'signal': SignalType.HOLD,
                'position': 0.3,
                'stop_loss': 0.03,
                'take_profit': 0.05
            },
            MarketRegime.HIGH_VOLATILITY: {
                'signal': SignalType.HOLD,
                'position': 0.2,
                'stop_loss': 0.02,
                'take_profit': 0.04
            },
            MarketRegime.CRISIS: {
                'signal': SignalType.STRONG_SELL,
                'position': 0.0,
                'stop_loss': None,
                'take_profit': None
            }
        }

    def generate_signal(self, regime_result: RegimeResult) -> TradingSignal:
        """
        Generate trading signal from regime classification.

        Args:
            regime_result: Result from regime classifier

        Returns:
            TradingSignal with recommendations
        """
        regime = regime_result.regime
        confidence = regime_result.confidence

        # Get base parameters for regime
        params = self.regime_positions.get(
            regime,
            self.regime_positions[MarketRegime.SIDEWAYS]
        )

        # Adjust position size by confidence
        if confidence < self.confidence_threshold:
            adjusted_position = params['position'] * 0.5
            signal_type = SignalType.HOLD
        else:
            adjusted_position = params['position'] * confidence
            signal_type = params['signal']

        return TradingSignal(
            signal_type=signal_type,
            position_size=adjusted_position,
            stop_loss=params['stop_loss'],
            take_profit=params['take_profit'],
            confidence=confidence,
            regime=regime,
            reasoning=f"Regime: {regime.value}, Confidence: {confidence:.1%}, "
                     f"Position: {adjusted_position:.1%}"
        )

    def generate_signals_series(
        self,
        regime_results: List[RegimeResult]
    ) -> List[TradingSignal]:
        """
        Generate signals for a series of regime classifications.

        Args:
            regime_results: List of regime results over time

        Returns:
            List of trading signals
        """
        return [self.generate_signal(r) for r in regime_results]


class RegimeTransitionDetector:
    """
    Detect regime transitions for signal generation.

    Adds hysteresis and confirmation to prevent excessive trading.
    """

    def __init__(
        self,
        confirmation_periods: int = 3,
        hysteresis_threshold: float = 0.1
    ):
        """
        Initialize transition detector.

        Args:
            confirmation_periods: Periods to confirm regime change
            hysteresis_threshold: Probability threshold for transition
        """
        self.confirmation_periods = confirmation_periods
        self.hysteresis_threshold = hysteresis_threshold

        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_count = 0
        self.history: List[RegimeResult] = []

    def update(self, result: RegimeResult) -> Tuple[bool, Optional[MarketRegime]]:
        """
        Update with new regime result and detect transitions.

        Args:
            result: New regime classification result

        Returns:
            Tuple of (transition_detected, new_regime)
        """
        self.history.append(result)

        # Check if regime is different from current
        if result.regime != self.current_regime:
            self.regime_count += 1

            # Check for confirmation
            if self.regime_count >= self.confirmation_periods:
                # Also check probability threshold
                if result.probability >= self.hysteresis_threshold:
                    old_regime = self.current_regime
                    self.current_regime = result.regime
                    self.regime_count = 0
                    return True, result.regime
        else:
            self.regime_count = 0

        return False, None

    def get_transition_history(self) -> List[Dict]:
        """Get history of regime transitions."""
        transitions = []
        current = None

        for i, result in enumerate(self.history):
            if result.regime != current:
                transitions.append({
                    'index': i,
                    'from_regime': current,
                    'to_regime': result.regime,
                    'probability': result.probability
                })
                current = result.regime

        return transitions


class PositionSizer:
    """
    Calculate position sizes based on regime and risk parameters.
    """

    def __init__(
        self,
        max_position: float = 1.0,
        risk_per_trade: float = 0.02,
        volatility_scaling: bool = True
    ):
        """
        Initialize position sizer.

        Args:
            max_position: Maximum position as fraction of capital
            risk_per_trade: Target risk per trade (e.g., 0.02 = 2%)
            volatility_scaling: Whether to scale by volatility
        """
        self.max_position = max_position
        self.risk_per_trade = risk_per_trade
        self.volatility_scaling = volatility_scaling

    def calculate_position(
        self,
        signal: TradingSignal,
        capital: float,
        current_price: float,
        volatility: float
    ) -> Dict:
        """
        Calculate position size in units and dollars.

        Args:
            signal: Trading signal
            capital: Available capital
            current_price: Current asset price
            volatility: Current volatility (annualized)

        Returns:
            Dictionary with position details
        """
        base_position = signal.position_size * capital

        # Volatility scaling
        if self.volatility_scaling and volatility > 0:
            # Target constant volatility contribution
            target_vol = 0.15  # 15% portfolio vol contribution
            vol_scalar = target_vol / volatility
            vol_scalar = np.clip(vol_scalar, 0.5, 2.0)
            base_position *= vol_scalar

        # Apply maximum position constraint
        max_dollars = self.max_position * capital
        position_dollars = np.clip(base_position, -max_dollars, max_dollars)

        # Calculate units
        position_units = position_dollars / current_price if current_price > 0 else 0

        # Stop loss levels
        if signal.stop_loss is not None:
            stop_price = current_price * (1 - signal.stop_loss if position_dollars > 0
                                          else 1 + signal.stop_loss)
        else:
            stop_price = None

        return {
            'position_dollars': position_dollars,
            'position_units': position_units,
            'position_pct': position_dollars / capital if capital > 0 else 0,
            'entry_price': current_price,
            'stop_loss_price': stop_price,
            'stop_loss_pct': signal.stop_loss,
            'take_profit_pct': signal.take_profit,
            'regime': signal.regime.value,
            'confidence': signal.confidence
        }
