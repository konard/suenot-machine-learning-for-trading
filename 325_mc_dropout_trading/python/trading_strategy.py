"""
Trading Strategy with MC Dropout Uncertainty
=============================================

This module implements trading strategies that use MC Dropout
uncertainty estimation for position sizing and signal filtering.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from mc_dropout_model import MCDropoutModel, MCDropoutRegressor, create_mc_dropout_model


class SignalType(Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """
    Trading signal with uncertainty information.
    """
    signal_type: SignalType
    predicted_return: float
    uncertainty: float
    confidence: float  # z-score
    position_size: float
    reason: str = ""
    timestamp: Optional[pd.Timestamp] = None

    def __repr__(self):
        return (f"Signal({self.signal_type.value}, "
                f"return={self.predicted_return:.4f}, "
                f"uncertainty={self.uncertainty:.4f}, "
                f"confidence={self.confidence:.2f}, "
                f"size={self.position_size:.4f})")


@dataclass
class StrategyConfig:
    """Configuration for the trading strategy."""
    # Confidence thresholds
    confidence_threshold: float = 1.5      # Minimum z-score to trade
    uncertainty_threshold: float = 0.03    # Maximum uncertainty to trade

    # Position sizing
    max_position_size: float = 0.1         # Maximum position size (fraction)
    min_position_size: float = 0.01        # Minimum position size
    use_kelly: bool = True                 # Use Kelly criterion for sizing

    # Risk management
    max_daily_trades: int = 10             # Maximum trades per day
    max_portfolio_uncertainty: float = 0.05  # Portfolio uncertainty limit

    # MC Dropout settings
    n_mc_samples: int = 50                 # Number of MC samples
    min_return_threshold: float = 0.001    # Minimum expected return to trade


class MCDropoutTradingStrategy:
    """
    Trading strategy using MC Dropout for uncertainty-aware decisions.
    """

    def __init__(self,
                 model: MCDropoutModel,
                 config: Optional[StrategyConfig] = None):
        """
        Initialize strategy.

        Args:
            model: Trained MC Dropout model
            config: Strategy configuration
        """
        self.model = model
        self.config = config or StrategyConfig()
        self.daily_trade_count = 0
        self.last_trade_date = None

    def _reset_daily_count(self, current_date: pd.Timestamp):
        """Reset daily trade counter if new day."""
        if self.last_trade_date is None or current_date.date() != self.last_trade_date.date():
            self.daily_trade_count = 0
            self.last_trade_date = current_date

    def _calculate_z_score(self, mean: float, std: float) -> float:
        """Calculate z-score (confidence measure)."""
        if std <= 0:
            return 0.0
        return abs(mean) / std

    def _kelly_position_size(self,
                             predicted_return: float,
                             uncertainty: float) -> float:
        """
        Calculate position size using Kelly criterion.

        Kelly fraction: f = (p*b - q) / b = excess_return / variance

        Args:
            predicted_return: Expected return
            uncertainty: Standard deviation of return

        Returns:
            Position size fraction
        """
        if uncertainty <= 0:
            return 0.0

        # Simple Kelly: f = mu / sigma^2
        kelly = predicted_return / (uncertainty ** 2)

        # Fractional Kelly (more conservative)
        kelly *= 0.5  # Half Kelly

        # Apply uncertainty penalty
        confidence_factor = 1.0 / (1.0 + uncertainty)
        kelly *= confidence_factor

        return max(0.0, min(self.config.max_position_size, kelly))

    def _simple_position_size(self,
                              confidence: float,
                              uncertainty: float) -> float:
        """
        Calculate position size based on confidence.

        Higher confidence = larger position
        Higher uncertainty = smaller position

        Args:
            confidence: Z-score
            uncertainty: Standard deviation

        Returns:
            Position size fraction
        """
        # Base size from confidence
        base_size = min(confidence / 5.0, 1.0)  # Scale to [0, 1] for z-score up to 5

        # Reduce based on uncertainty
        uncertainty_penalty = 1.0 - min(uncertainty / self.config.uncertainty_threshold, 1.0)

        size = base_size * uncertainty_penalty * self.config.max_position_size

        return max(self.config.min_position_size, min(self.config.max_position_size, size))

    def generate_signal(self,
                        features: np.ndarray,
                        timestamp: Optional[pd.Timestamp] = None) -> TradingSignal:
        """
        Generate trading signal with uncertainty-based filtering.

        Args:
            features: Input features as numpy array
            timestamp: Current timestamp

        Returns:
            TradingSignal with direction, size, and confidence
        """
        # Reset daily counter if needed
        if timestamp:
            self._reset_daily_count(timestamp)

        # Check daily trade limit
        if self.daily_trade_count >= self.config.max_daily_trades:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                predicted_return=0.0,
                uncertainty=0.0,
                confidence=0.0,
                position_size=0.0,
                reason="Daily trade limit reached",
                timestamp=timestamp
            )

        # Convert to tensor
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        if len(features.shape) == 1:
            features = features.unsqueeze(0)

        # Get predictions with uncertainty
        result = self.model.predict_with_uncertainty(features, self.config.n_mc_samples)

        mean_pred = result['mean'].item()
        std_pred = result['std'].item()

        # Calculate confidence (z-score)
        z_score = self._calculate_z_score(mean_pred, std_pred)

        # Filter by uncertainty
        if std_pred > self.config.uncertainty_threshold:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                predicted_return=mean_pred,
                uncertainty=std_pred,
                confidence=z_score,
                position_size=0.0,
                reason=f"Uncertainty too high: {std_pred:.4f} > {self.config.uncertainty_threshold}",
                timestamp=timestamp
            )

        # Filter by confidence
        if z_score < self.config.confidence_threshold:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                predicted_return=mean_pred,
                uncertainty=std_pred,
                confidence=z_score,
                position_size=0.0,
                reason=f"Confidence too low: {z_score:.2f} < {self.config.confidence_threshold}",
                timestamp=timestamp
            )

        # Filter by minimum return
        if abs(mean_pred) < self.config.min_return_threshold:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                predicted_return=mean_pred,
                uncertainty=std_pred,
                confidence=z_score,
                position_size=0.0,
                reason=f"Expected return too small: {mean_pred:.4f}",
                timestamp=timestamp
            )

        # Determine direction
        signal_type = SignalType.LONG if mean_pred > 0 else SignalType.SHORT

        # Calculate position size
        if self.config.use_kelly:
            position_size = self._kelly_position_size(abs(mean_pred), std_pred)
        else:
            position_size = self._simple_position_size(z_score, std_pred)

        # Update trade counter
        self.daily_trade_count += 1

        return TradingSignal(
            signal_type=signal_type,
            predicted_return=mean_pred,
            uncertainty=std_pred,
            confidence=z_score,
            position_size=position_size,
            reason="Signal generated",
            timestamp=timestamp
        )

    def generate_batch_signals(self,
                               features_batch: np.ndarray,
                               timestamps: Optional[List[pd.Timestamp]] = None) -> List[TradingSignal]:
        """
        Generate signals for a batch of features.

        Args:
            features_batch: Batch of features [n_samples, n_features]
            timestamps: List of timestamps

        Returns:
            List of TradingSignal objects
        """
        signals = []
        n_samples = len(features_batch)

        if timestamps is None:
            timestamps = [None] * n_samples

        for i in range(n_samples):
            signal = self.generate_signal(features_batch[i], timestamps[i])
            signals.append(signal)

        return signals


class UncertaintyBasedRiskManager:
    """
    Risk manager that uses uncertainty for position management.
    """

    def __init__(self,
                 max_position_size: float = 0.1,
                 max_portfolio_uncertainty: float = 0.05,
                 max_correlated_positions: int = 3):
        """
        Args:
            max_position_size: Maximum single position size
            max_portfolio_uncertainty: Maximum portfolio-level uncertainty
            max_correlated_positions: Maximum positions in correlated assets
        """
        self.max_position_size = max_position_size
        self.max_portfolio_uncertainty = max_portfolio_uncertainty
        self.max_correlated_positions = max_correlated_positions
        self.current_positions: Dict[str, Dict] = {}

    def can_add_position(self,
                         symbol: str,
                         signal: TradingSignal) -> Tuple[bool, str]:
        """
        Check if a new position can be added.

        Args:
            symbol: Trading symbol
            signal: Proposed trading signal

        Returns:
            Tuple of (can_add, reason)
        """
        # Check if already have position in this symbol
        if symbol in self.current_positions:
            return False, f"Already have position in {symbol}"

        # Check portfolio uncertainty
        current_uncertainty = self._calculate_portfolio_uncertainty()
        new_uncertainty = self._estimate_new_portfolio_uncertainty(signal)

        if new_uncertainty > self.max_portfolio_uncertainty:
            return False, f"Portfolio uncertainty would exceed limit: {new_uncertainty:.4f}"

        return True, "Position allowed"

    def _calculate_portfolio_uncertainty(self) -> float:
        """Calculate current portfolio uncertainty."""
        if not self.current_positions:
            return 0.0

        uncertainties = [pos['uncertainty'] * pos['size']
                        for pos in self.current_positions.values()]

        # Simplified: assume uncorrelated
        return np.sqrt(sum(u**2 for u in uncertainties))

    def _estimate_new_portfolio_uncertainty(self, new_signal: TradingSignal) -> float:
        """Estimate portfolio uncertainty after adding new position."""
        current = self._calculate_portfolio_uncertainty()
        new_contribution = new_signal.uncertainty * new_signal.position_size

        # Simplified: assume uncorrelated
        return np.sqrt(current**2 + new_contribution**2)

    def add_position(self,
                     symbol: str,
                     signal: TradingSignal):
        """Add a position to tracking."""
        self.current_positions[symbol] = {
            'signal_type': signal.signal_type,
            'size': signal.position_size,
            'uncertainty': signal.uncertainty,
            'entry_return': signal.predicted_return,
        }

    def remove_position(self, symbol: str):
        """Remove a position from tracking."""
        if symbol in self.current_positions:
            del self.current_positions[symbol]

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary."""
        return {
            'n_positions': len(self.current_positions),
            'total_size': sum(pos['size'] for pos in self.current_positions.values()),
            'portfolio_uncertainty': self._calculate_portfolio_uncertainty(),
            'positions': dict(self.current_positions),
        }


def uncertainty_adjusted_kelly(predicted_return: float,
                               uncertainty: float,
                               risk_free_rate: float = 0.0,
                               max_leverage: float = 1.0) -> float:
    """
    Calculate position size using uncertainty-adjusted Kelly criterion.

    Args:
        predicted_return: Expected return
        uncertainty: Standard deviation of return
        risk_free_rate: Risk-free rate
        max_leverage: Maximum allowed leverage

    Returns:
        Position size fraction
    """
    if uncertainty <= 0:
        return 0.0

    excess_return = predicted_return - risk_free_rate

    # Kelly fraction
    kelly = excess_return / (uncertainty ** 2)

    # Apply half-Kelly for safety
    kelly *= 0.5

    # Apply uncertainty penalty
    confidence_factor = 1.0 / (1.0 + 2 * uncertainty)
    kelly *= confidence_factor

    # Clip to allowed range
    return max(0.0, min(max_leverage, kelly))


if __name__ == '__main__':
    print("=== MC Dropout Trading Strategy Example ===\n")

    # Create a simple trained model for demonstration
    from mc_dropout_model import create_mc_dropout_model

    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20).astype(np.float32)
    y = (X[:, 0] * 0.01 + np.random.randn(1000) * 0.005).astype(np.float32)

    # Train model
    model = create_mc_dropout_model(
        input_dim=20,
        hidden_dims=[64, 32],
        dropout_rate=0.1,
        n_mc_samples=50,
        task='regression'
    )

    print("Training model...")
    model.fit(X[:800], y[:800], X[800:], y[800:], epochs=30, patience=5)

    # Create strategy
    config = StrategyConfig(
        confidence_threshold=1.5,
        uncertainty_threshold=0.03,
        max_position_size=0.1,
        use_kelly=True,
        n_mc_samples=50
    )

    strategy = MCDropoutTradingStrategy(model, config)

    # Generate signals for test data
    print("\nGenerating trading signals...")
    signals = []
    for i in range(10):
        signal = strategy.generate_signal(X[800+i])
        signals.append(signal)
        print(f"  {signal}")

    # Count signal types
    longs = sum(1 for s in signals if s.signal_type == SignalType.LONG)
    shorts = sum(1 for s in signals if s.signal_type == SignalType.SHORT)
    holds = sum(1 for s in signals if s.signal_type == SignalType.HOLD)

    print(f"\nSignal Summary:")
    print(f"  LONG:  {longs}")
    print(f"  SHORT: {shorts}")
    print(f"  HOLD:  {holds}")

    # Test risk manager
    print("\n=== Risk Manager Example ===")
    risk_manager = UncertaintyBasedRiskManager(
        max_position_size=0.1,
        max_portfolio_uncertainty=0.05
    )

    for i, signal in enumerate(signals):
        if signal.signal_type != SignalType.HOLD:
            symbol = f"ASSET_{i}"
            can_add, reason = risk_manager.can_add_position(symbol, signal)
            if can_add:
                risk_manager.add_position(symbol, signal)
                print(f"Added position: {symbol}")

    summary = risk_manager.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    print(f"  Positions: {summary['n_positions']}")
    print(f"  Total Size: {summary['total_size']:.4f}")
    print(f"  Portfolio Uncertainty: {summary['portfolio_uncertainty']:.4f}")
