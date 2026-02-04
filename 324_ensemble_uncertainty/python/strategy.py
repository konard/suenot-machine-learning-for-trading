"""
Trading Strategy module with uncertainty-aware decisions.

This module implements trading strategies that use ensemble uncertainty
to make more robust trading decisions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Represents a trading signal with uncertainty."""
    symbol: str
    signal_type: SignalType
    prediction: float
    uncertainty: float
    confidence: float
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None

    def __repr__(self):
        return (f"TradingSignal({self.symbol}, {self.signal_type.value}, "
                f"pred={self.prediction:.4f}, unc={self.uncertainty:.4f}, "
                f"conf={self.confidence:.2%}, size={self.position_size:.2%})")


class UncertaintyStrategy:
    """
    Trading strategy that incorporates uncertainty in decision making.

    Features:
    - Position sizing based on confidence
    - Dynamic stop-loss based on uncertainty
    - Signal filtering by uncertainty threshold
    - Kelly criterion with uncertainty adjustment
    """

    def __init__(
        self,
        min_prediction_threshold: float = 0.001,
        max_uncertainty_threshold: float = 0.05,
        min_confidence_threshold: float = 0.6,
        base_position_size: float = 1.0,
        use_kelly: bool = True,
        kelly_fraction: float = 0.5
    ):
        """
        Initialize the uncertainty-aware strategy.

        Args:
            min_prediction_threshold: Minimum prediction magnitude to trade
            max_uncertainty_threshold: Maximum uncertainty to allow trading
            min_confidence_threshold: Minimum confidence to trade
            base_position_size: Base position size (1.0 = 100%)
            use_kelly: Whether to use Kelly criterion
            kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly)
        """
        self.min_prediction_threshold = min_prediction_threshold
        self.max_uncertainty_threshold = max_uncertainty_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.base_position_size = base_position_size
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction

    def calculate_confidence(
        self,
        prediction: float,
        uncertainty: float
    ) -> float:
        """
        Calculate confidence score from prediction and uncertainty.

        Args:
            prediction: Expected return prediction
            uncertainty: Prediction uncertainty (std)

        Returns:
            Confidence score between 0 and 1
        """
        if uncertainty >= self.max_uncertainty_threshold:
            return 0.0

        # Base confidence from uncertainty level
        uncertainty_conf = 1.0 - (uncertainty / self.max_uncertainty_threshold)

        # Boost confidence for stronger predictions
        pred_strength = min(abs(prediction) / 0.02, 1.0)  # Cap at 2% prediction

        # Combined confidence
        confidence = uncertainty_conf * (0.5 + 0.5 * pred_strength)

        return np.clip(confidence, 0.0, 1.0)

    def calculate_position_size(
        self,
        prediction: float,
        uncertainty: float,
        confidence: float,
        historical_accuracy: Optional[float] = None,
        historical_avg_win: Optional[float] = None,
        historical_avg_loss: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on uncertainty and confidence.

        Args:
            prediction: Expected return prediction
            uncertainty: Prediction uncertainty
            confidence: Calculated confidence
            historical_accuracy: Optional historical win rate
            historical_avg_win: Optional average win size
            historical_avg_loss: Optional average loss size

        Returns:
            Position size (0 to base_position_size)
        """
        # Basic position sizing by confidence
        position_size = self.base_position_size * confidence

        # Apply Kelly criterion if enabled and historical data available
        if self.use_kelly and historical_accuracy is not None:
            kelly_size = self._kelly_criterion(
                historical_accuracy,
                historical_avg_win or abs(prediction),
                historical_avg_loss or uncertainty,
                uncertainty
            )
            # Use minimum of confidence-based and Kelly-based size
            position_size = min(position_size, kelly_size)

        # Scale by prediction strength
        pred_factor = min(abs(prediction) / self.min_prediction_threshold, 2.0) / 2.0
        position_size *= pred_factor

        return np.clip(position_size, 0.0, self.base_position_size)

    def _kelly_criterion(
        self,
        win_prob: float,
        win_size: float,
        loss_size: float,
        uncertainty: float
    ) -> float:
        """
        Calculate Kelly criterion position size with uncertainty adjustment.

        Args:
            win_prob: Probability of winning
            win_size: Expected size of win
            loss_size: Expected size of loss
            uncertainty: Model uncertainty

        Returns:
            Kelly position size
        """
        # Standard Kelly formula: (p*b - q) / b
        # where p = win prob, q = 1-p, b = win/loss ratio
        if loss_size <= 0:
            return 0.0

        b = win_size / (loss_size + 1e-8)
        q = 1 - win_prob
        kelly = (win_prob * b - q) / b

        # Uncertainty adjustment (reduce position when uncertain)
        uncertainty_factor = 1.0 / (1.0 + uncertainty * 10)

        # Apply fractional Kelly
        fractional_kelly = kelly * self.kelly_fraction * uncertainty_factor

        return max(0.0, fractional_kelly)

    def calculate_stop_loss(
        self,
        prediction: float,
        uncertainty: float,
        current_price: float,
        signal_type: SignalType,
        multiplier: float = 2.0
    ) -> float:
        """
        Calculate dynamic stop-loss based on uncertainty.

        Args:
            prediction: Expected return prediction
            uncertainty: Prediction uncertainty
            current_price: Current asset price
            signal_type: LONG or SHORT
            multiplier: Uncertainty multiplier for stop distance

        Returns:
            Stop-loss price
        """
        # Stop distance based on uncertainty (wider stops when uncertain)
        stop_distance = current_price * uncertainty * multiplier

        if signal_type == SignalType.LONG:
            return current_price - stop_distance
        elif signal_type == SignalType.SHORT:
            return current_price + stop_distance
        else:
            return current_price

    def calculate_take_profit(
        self,
        prediction: float,
        uncertainty: float,
        current_price: float,
        signal_type: SignalType,
        risk_reward_ratio: float = 1.5
    ) -> float:
        """
        Calculate take-profit based on prediction and uncertainty.

        Args:
            prediction: Expected return prediction
            uncertainty: Prediction uncertainty
            current_price: Current asset price
            signal_type: LONG or SHORT
            risk_reward_ratio: Desired risk/reward ratio

        Returns:
            Take-profit price
        """
        # Take profit at predicted return, adjusted by risk-reward ratio
        tp_distance = current_price * abs(prediction) * risk_reward_ratio

        if signal_type == SignalType.LONG:
            return current_price + tp_distance
        elif signal_type == SignalType.SHORT:
            return current_price - tp_distance
        else:
            return current_price

    def generate_signal(
        self,
        symbol: str,
        prediction: float,
        uncertainty: float,
        current_price: Optional[float] = None,
        timestamp: Optional[pd.Timestamp] = None
    ) -> TradingSignal:
        """
        Generate a trading signal with uncertainty information.

        Args:
            symbol: Trading symbol
            prediction: Expected return prediction
            uncertainty: Prediction uncertainty
            current_price: Optional current price for stop/TP calculation
            timestamp: Optional timestamp

        Returns:
            TradingSignal object
        """
        # Calculate confidence
        confidence = self.calculate_confidence(prediction, uncertainty)

        # Determine signal type
        if confidence < self.min_confidence_threshold:
            signal_type = SignalType.HOLD
        elif prediction > self.min_prediction_threshold:
            signal_type = SignalType.LONG
        elif prediction < -self.min_prediction_threshold:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.HOLD

        # Calculate position size
        if signal_type == SignalType.HOLD:
            position_size = 0.0
        else:
            position_size = self.calculate_position_size(
                prediction, uncertainty, confidence
            )

        # Calculate stop-loss and take-profit if price provided
        stop_loss = None
        take_profit = None
        if current_price is not None and signal_type != SignalType.HOLD:
            stop_loss = self.calculate_stop_loss(
                prediction, uncertainty, current_price, signal_type
            )
            take_profit = self.calculate_take_profit(
                prediction, uncertainty, current_price, signal_type
            )

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            prediction=prediction,
            uncertainty=uncertainty,
            confidence=confidence,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=timestamp
        )

    def generate_signals_batch(
        self,
        symbols: List[str],
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        current_prices: Optional[np.ndarray] = None,
        timestamp: Optional[pd.Timestamp] = None
    ) -> List[TradingSignal]:
        """
        Generate trading signals for multiple symbols.

        Args:
            symbols: List of trading symbols
            predictions: Array of predictions
            uncertainties: Array of uncertainties
            current_prices: Optional array of current prices
            timestamp: Optional timestamp

        Returns:
            List of TradingSignal objects
        """
        signals = []
        for i, symbol in enumerate(symbols):
            current_price = current_prices[i] if current_prices is not None else None
            signal = self.generate_signal(
                symbol=symbol,
                prediction=predictions[i],
                uncertainty=uncertainties[i],
                current_price=current_price,
                timestamp=timestamp
            )
            signals.append(signal)

        return signals

    def filter_signals(
        self,
        signals: List[TradingSignal],
        min_confidence: Optional[float] = None,
        max_positions: Optional[int] = None
    ) -> List[TradingSignal]:
        """
        Filter signals by confidence and limit number of positions.

        Args:
            signals: List of trading signals
            min_confidence: Minimum confidence threshold
            max_positions: Maximum number of positions to take

        Returns:
            Filtered list of signals
        """
        if min_confidence is None:
            min_confidence = self.min_confidence_threshold

        # Filter by confidence
        filtered = [s for s in signals if s.confidence >= min_confidence]

        # Filter out HOLDs
        filtered = [s for s in filtered if s.signal_type != SignalType.HOLD]

        # Sort by confidence and take top N
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        if max_positions is not None:
            filtered = filtered[:max_positions]

        return filtered


class PortfolioManager:
    """
    Manages portfolio allocation with uncertainty-aware positioning.
    """

    def __init__(
        self,
        max_position_per_asset: float = 0.2,
        max_total_exposure: float = 1.0,
        max_correlation_overlap: float = 0.7
    ):
        """
        Initialize portfolio manager.

        Args:
            max_position_per_asset: Maximum position size per asset
            max_total_exposure: Maximum total portfolio exposure
            max_correlation_overlap: Maximum correlation to allow between positions
        """
        self.max_position_per_asset = max_position_per_asset
        self.max_total_exposure = max_total_exposure
        self.max_correlation_overlap = max_correlation_overlap

    def allocate_positions(
        self,
        signals: List[TradingSignal],
        correlations: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Allocate portfolio positions based on signals and correlations.

        Args:
            signals: List of trading signals
            correlations: Optional correlation matrix

        Returns:
            Dictionary mapping symbol to position size
        """
        if not signals:
            return {}

        # Sort by confidence
        signals = sorted(signals, key=lambda x: x.confidence, reverse=True)

        allocations = {}
        total_exposure = 0.0
        selected_indices = []

        for i, signal in enumerate(signals):
            # Check total exposure limit
            proposed_size = min(signal.position_size, self.max_position_per_asset)
            if total_exposure + proposed_size > self.max_total_exposure:
                proposed_size = self.max_total_exposure - total_exposure

            # Check correlation with existing positions
            if correlations is not None and selected_indices:
                max_corr = max(
                    abs(correlations[i, j]) for j in selected_indices
                )
                if max_corr > self.max_correlation_overlap:
                    continue  # Skip correlated asset

            if proposed_size > 0.01:  # Minimum position threshold
                allocations[signal.symbol] = proposed_size
                total_exposure += proposed_size
                selected_indices.append(i)

            if total_exposure >= self.max_total_exposure:
                break

        return allocations


def main():
    """Example usage of the uncertainty-aware strategy."""
    np.random.seed(42)

    print("=" * 60)
    print("Uncertainty-Aware Trading Strategy")
    print("=" * 60)

    # Initialize strategy
    strategy = UncertaintyStrategy(
        min_prediction_threshold=0.001,
        max_uncertainty_threshold=0.05,
        min_confidence_threshold=0.6,
        base_position_size=1.0,
        use_kelly=True
    )

    # Simulate predictions and uncertainties
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
    predictions = np.array([0.025, -0.015, 0.008, 0.002, -0.030])
    uncertainties = np.array([0.010, 0.020, 0.035, 0.040, 0.015])
    prices = np.array([50000, 3000, 100, 300, 0.50])

    print("\n--- Input Data ---")
    for sym, pred, unc, price in zip(symbols, predictions, uncertainties, prices):
        print(f"{sym}: pred={pred:.2%}, unc={unc:.2%}, price=${price}")

    # Generate signals
    print("\n--- Generated Signals ---")
    signals = strategy.generate_signals_batch(
        symbols, predictions, uncertainties, prices
    )

    for signal in signals:
        print(signal)
        if signal.stop_loss:
            print(f"  Stop-Loss: ${signal.stop_loss:.2f}, Take-Profit: ${signal.take_profit:.2f}")

    # Filter signals
    print("\n--- Filtered Signals (top 3 by confidence) ---")
    filtered = strategy.filter_signals(signals, max_positions=3)
    for signal in filtered:
        print(signal)

    # Portfolio allocation
    print("\n--- Portfolio Allocation ---")
    portfolio = PortfolioManager(
        max_position_per_asset=0.25,
        max_total_exposure=0.8
    )

    allocations = portfolio.allocate_positions(filtered)
    print("Allocations:")
    for symbol, size in allocations.items():
        print(f"  {symbol}: {size:.2%}")
    print(f"Total exposure: {sum(allocations.values()):.2%}")

    # Demonstrate confidence calculation
    print("\n--- Confidence vs Uncertainty ---")
    for unc in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        conf = strategy.calculate_confidence(0.02, unc)
        print(f"  Uncertainty={unc:.2%} -> Confidence={conf:.2%}")


if __name__ == '__main__':
    main()
