"""
Trading Strategy using Variational Inference
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from config import StrategyConfig
from vae_model import MarketVAE


class SignalType(Enum):
    """Trading signal types"""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Trading signal with uncertainty information"""
    signal_type: SignalType
    confidence: float
    expected_return: float
    return_std: float
    prob_positive: float
    regime: int
    regime_probs: np.ndarray
    timestamp: Optional[str] = None

    def __repr__(self):
        return (
            f"TradingSignal({self.signal_type.value}, "
            f"conf={self.confidence:.2f}, "
            f"E[r]={self.expected_return:.4f}, "
            f"std={self.return_std:.4f}, "
            f"P(r>0)={self.prob_positive:.2f})"
        )


class VariationalTradingStrategy:
    """
    Trading strategy using Variational Autoencoder predictions

    Features:
    - Uncertainty-aware position sizing
    - Regime detection
    - Confidence-based signal filtering
    """

    def __init__(
        self,
        model: MarketVAE,
        config: Optional[StrategyConfig] = None,
        device: str = "auto"
    ):
        self.model = model
        self.config = config or StrategyConfig()

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.set_to_eval_mode()

    def generate_signal(
        self,
        features: np.ndarray,
        timestamp: Optional[str] = None
    ) -> TradingSignal:
        """
        Generate trading signal from market features

        Args:
            features: Market features array of shape (seq_len, n_features)
            timestamp: Optional timestamp for the signal

        Returns:
            TradingSignal with predictions and uncertainty
        """
        # Prepare input
        if features.ndim == 2:
            features = features[np.newaxis, ...]  # Add batch dimension

        x = torch.FloatTensor(features).to(self.device)

        # Sample predictions for uncertainty estimation
        return_samples, regime_samples, _ = self.model.sample_predictions(
            x, n_samples=self.config.n_samples
        )

        # Aggregate predictions
        return_samples = return_samples.squeeze()  # (n_samples,)
        expected_return = float(np.mean(return_samples))
        return_std = float(np.std(return_samples))

        # Probability of positive return
        prob_positive = float(np.mean(return_samples > 0))

        # Regime prediction (average probabilities)
        avg_regime_probs = np.mean(regime_samples, axis=0).squeeze()
        regime = int(np.argmax(avg_regime_probs))

        # Determine signal based on rules
        signal_type, confidence = self._determine_signal(
            expected_return, return_std, prob_positive
        )

        return TradingSignal(
            signal_type=signal_type,
            confidence=confidence,
            expected_return=expected_return,
            return_std=return_std,
            prob_positive=prob_positive,
            regime=regime,
            regime_probs=avg_regime_probs,
            timestamp=timestamp
        )

    def _determine_signal(
        self,
        expected_return: float,
        return_std: float,
        prob_positive: float
    ) -> Tuple[SignalType, float]:
        """
        Determine trading signal based on predictions

        Args:
            expected_return: Expected return prediction
            return_std: Standard deviation of return predictions
            prob_positive: Probability of positive return

        Returns:
            Tuple of (SignalType, confidence)
        """
        # Check if uncertainty is too high
        if return_std > self.config.uncertainty_threshold:
            return SignalType.HOLD, 0.5

        # Long signal
        if prob_positive > self.config.confidence_threshold:
            confidence = prob_positive
            return SignalType.LONG, confidence

        # Short signal
        if prob_positive < (1 - self.config.confidence_threshold):
            confidence = 1 - prob_positive
            return SignalType.SHORT, confidence

        # Neutral / Hold
        return SignalType.HOLD, 0.5

    def calculate_position_size(
        self,
        signal: TradingSignal,
        portfolio_value: float
    ) -> float:
        """
        Calculate position size based on signal confidence and uncertainty

        Uses Kelly-inspired sizing adjusted for uncertainty:
        - Higher confidence -> larger position
        - Higher uncertainty -> smaller position

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value

        Returns:
            Position size as fraction of portfolio
        """
        if signal.signal_type == SignalType.HOLD:
            return 0.0

        # Base size from confidence
        base_size = signal.confidence * self.config.max_position_size

        # Adjust for uncertainty (reduce size when uncertain)
        uncertainty_factor = max(
            0.1,
            1.0 - signal.return_std / self.config.uncertainty_threshold
        )
        adjusted_size = base_size * uncertainty_factor

        # Clamp to limits
        position_size = np.clip(
            adjusted_size,
            self.config.min_position_size,
            self.config.max_position_size
        )

        return float(position_size)

    def get_stop_loss_take_profit(
        self,
        signal: TradingSignal,
        entry_price: float
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels

        Adjusts levels based on predicted volatility

        Args:
            signal: Trading signal
            entry_price: Entry price for the trade

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Adjust stops based on predicted volatility
        vol_multiplier = 1 + signal.return_std * 10  # Scale by uncertainty

        if signal.signal_type == SignalType.LONG:
            stop_loss = entry_price * (1 - self.config.stop_loss * vol_multiplier)
            take_profit = entry_price * (1 + self.config.take_profit * vol_multiplier)
        else:  # SHORT
            stop_loss = entry_price * (1 + self.config.stop_loss * vol_multiplier)
            take_profit = entry_price * (1 - self.config.take_profit * vol_multiplier)

        return stop_loss, take_profit

    def batch_generate_signals(
        self,
        features_batch: np.ndarray,
        timestamps: Optional[List[str]] = None
    ) -> List[TradingSignal]:
        """
        Generate signals for multiple timesteps

        Args:
            features_batch: Array of shape (n_timesteps, seq_len, n_features)
            timestamps: Optional list of timestamps

        Returns:
            List of TradingSignals
        """
        signals = []
        n_timesteps = features_batch.shape[0]

        for i in range(n_timesteps):
            ts = timestamps[i] if timestamps else None
            signal = self.generate_signal(features_batch[i], ts)
            signals.append(signal)

        return signals


class RegimeAwareStrategy(VariationalTradingStrategy):
    """
    Extended strategy that adjusts behavior based on detected market regime
    """

    def __init__(
        self,
        model: MarketVAE,
        config: Optional[StrategyConfig] = None,
        device: str = "auto"
    ):
        super().__init__(model, config, device)

        # Regime-specific parameters
        # Regime 0: Bull, Regime 1: Bear, Regime 2: Sideways
        self.regime_confidence_multipliers = {
            0: 1.2,  # More confident in bull market
            1: 1.1,  # Slightly more confident in bear market
            2: 0.7   # Less confident in sideways market
        }

        self.regime_position_multipliers = {
            0: 1.2,  # Larger positions in bull market
            1: 0.8,  # Smaller positions in bear market
            2: 0.5   # Smallest positions in sideways
        }

    def calculate_position_size(
        self,
        signal: TradingSignal,
        portfolio_value: float
    ) -> float:
        """
        Calculate position size with regime adjustment
        """
        base_size = super().calculate_position_size(signal, portfolio_value)

        # Adjust for regime
        regime_mult = self.regime_position_multipliers.get(signal.regime, 1.0)
        adjusted_size = base_size * regime_mult

        return float(np.clip(
            adjusted_size,
            self.config.min_position_size,
            self.config.max_position_size
        ))


class PortfolioOptimizer:
    """
    Portfolio optimization using variational predictions

    Implements uncertainty-aware mean-variance optimization
    """

    def __init__(
        self,
        strategy: VariationalTradingStrategy,
        risk_aversion: float = 1.0
    ):
        self.strategy = strategy
        self.risk_aversion = risk_aversion

    def optimize_weights(
        self,
        signals: Dict[str, TradingSignal],
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights based on signals

        Args:
            signals: Dictionary mapping asset names to signals
            current_weights: Current portfolio weights (optional)

        Returns:
            Dictionary of optimized weights
        """
        assets = list(signals.keys())
        n_assets = len(assets)

        if n_assets == 0:
            return {}

        # Extract expected returns and uncertainties
        expected_returns = np.array([s.expected_return for s in signals.values()])
        uncertainties = np.array([s.return_std for s in signals.values()])
        confidences = np.array([s.confidence for s in signals.values()])

        # Build uncertainty-adjusted covariance matrix (diagonal approximation)
        cov_matrix = np.diag(uncertainties ** 2)

        # Mean-variance optimization with uncertainty penalty
        # max w'mu - lambda/2 * w'Sigma*w
        # With analytical solution for no constraints:
        # w* = (1/lambda) * Sigma^-1 * mu

        # Add regularization for numerical stability
        cov_inv = np.linalg.inv(cov_matrix + 1e-6 * np.eye(n_assets))

        # Optimal weights (unconstrained)
        raw_weights = (1 / self.risk_aversion) * cov_inv @ expected_returns

        # Adjust for signal direction
        for i, signal in enumerate(signals.values()):
            if signal.signal_type == SignalType.HOLD:
                raw_weights[i] = 0
            elif signal.signal_type == SignalType.SHORT:
                raw_weights[i] = -abs(raw_weights[i])

        # Normalize weights to sum to 1 (long-only constraint simplification)
        # For a more realistic approach, we'd use proper constrained optimization
        positive_weights = np.maximum(raw_weights, 0)
        weight_sum = positive_weights.sum()

        if weight_sum > 0:
            normalized_weights = positive_weights / weight_sum
        else:
            normalized_weights = np.ones(n_assets) / n_assets

        return dict(zip(assets, normalized_weights))


if __name__ == "__main__":
    # Test strategy with synthetic data
    from vae_model import create_model, ModelConfig
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Create model
    model_config = ModelConfig(
        input_dim=13,
        latent_dim=8,
        hidden_dims=[64, 32]
    )
    model = create_model(model_config, sequence_length=60, device="cpu")

    # Create strategy
    strategy_config = StrategyConfig(
        confidence_threshold=0.6,
        uncertainty_threshold=0.05,
        n_samples=50
    )
    strategy = VariationalTradingStrategy(model, strategy_config, device="cpu")

    # Generate synthetic features
    features = np.random.randn(60, 13).astype(np.float32)

    # Generate signal
    signal = strategy.generate_signal(features)
    print(f"Generated signal: {signal}")

    # Calculate position size
    position_size = strategy.calculate_position_size(signal, portfolio_value=10000)
    print(f"Position size: {position_size:.2%}")

    # Test batch generation
    features_batch = np.random.randn(10, 60, 13).astype(np.float32)
    signals = strategy.batch_generate_signals(features_batch)
    print(f"\nGenerated {len(signals)} signals")
    for i, s in enumerate(signals[:3]):
        print(f"  Signal {i}: {s.signal_type.value}, conf={s.confidence:.2f}")

    # Test portfolio optimizer
    print("\nTesting portfolio optimizer...")
    multi_signals = {
        "BTC/USDT": signals[0],
        "ETH/USDT": signals[1],
        "SOL/USDT": signals[2]
    }
    optimizer = PortfolioOptimizer(strategy, risk_aversion=1.0)
    weights = optimizer.optimize_weights(multi_signals)
    print(f"Optimized weights: {weights}")
