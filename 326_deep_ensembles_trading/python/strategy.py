"""
Trading Strategy with Deep Ensembles

This module implements an uncertainty-aware trading strategy using
Deep Ensembles for position sizing and risk management.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from deep_ensemble import DeepEnsemble, EnsemblePrediction


class SignalType(Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Trading signal with uncertainty information."""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0 to 1
    predicted_return: float
    total_uncertainty: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float


@dataclass
class Position:
    """Trading position."""
    symbol: str
    size: float  # Positive for long, negative for short
    entry_price: float
    entry_time: int


class UncertaintyAwareStrategy:
    """
    Trading strategy that uses Deep Ensemble uncertainty
    for position sizing and signal filtering.
    """

    def __init__(
        self,
        ensemble: DeepEnsemble,
        confidence_threshold: float = 0.6,
        epistemic_threshold: float = 0.5,
        max_position_size: float = 0.1,
        signal_strength_threshold: float = 1.5
    ):
        """
        Initialize the strategy.

        Args:
            ensemble: Trained Deep Ensemble model
            confidence_threshold: Minimum confidence for trading
            epistemic_threshold: Maximum epistemic/total ratio for trading
            max_position_size: Maximum position size (fraction of capital)
            signal_strength_threshold: Minimum signal strength (return/std)
        """
        self.ensemble = ensemble
        self.confidence_threshold = confidence_threshold
        self.epistemic_threshold = epistemic_threshold
        self.max_position_size = max_position_size
        self.signal_strength_threshold = signal_strength_threshold

    def generate_signals(
        self,
        features: np.ndarray,
        symbols: List[str]
    ) -> List[Signal]:
        """
        Generate trading signals from features.

        Args:
            features: Feature matrix [n_samples, n_features]
            symbols: List of symbol names

        Returns:
            List of trading signals
        """
        # Get ensemble predictions
        predictions = self.ensemble.predict(features)

        signals = []

        for i, symbol in enumerate(symbols):
            mean_pred = predictions.mean[i]
            total_std = predictions.total_std[i]
            epistemic_std = predictions.epistemic_std[i]
            aleatoric_std = predictions.aleatoric_std[i]

            # Calculate signal strength (like Sharpe ratio)
            signal_strength = mean_pred / total_std if total_std > 0 else 0

            # Calculate epistemic ratio
            epistemic_ratio = epistemic_std / total_std if total_std > 0 else 1

            # Determine signal type
            if epistemic_ratio > self.epistemic_threshold:
                # High model uncertainty - don't trade
                signal_type = SignalType.HOLD
                confidence = 0
            elif signal_strength > self.signal_strength_threshold:
                signal_type = SignalType.LONG
                confidence = min(0.95, 1 - epistemic_ratio)
            elif signal_strength < -self.signal_strength_threshold:
                signal_type = SignalType.SHORT
                confidence = min(0.95, 1 - epistemic_ratio)
            else:
                signal_type = SignalType.HOLD
                confidence = 0

            signals.append(Signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                predicted_return=mean_pred,
                total_uncertainty=total_std,
                epistemic_uncertainty=epistemic_std,
                aleatoric_uncertainty=aleatoric_std
            ))

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        capital: float
    ) -> float:
        """
        Calculate position size based on uncertainty.

        Uses a Kelly-inspired approach with uncertainty adjustment.

        Args:
            signal: Trading signal
            capital: Available capital

        Returns:
            Position size (can be negative for shorts)
        """
        if signal.signal_type == SignalType.HOLD:
            return 0

        # Base Kelly fraction
        kelly_fraction = signal.confidence / (1 + signal.total_uncertainty)

        # Reduce position if epistemic uncertainty is high
        epistemic_penalty = 1 - min(1, signal.epistemic_uncertainty / signal.total_uncertainty)

        # Final position size
        position_fraction = kelly_fraction * epistemic_penalty * self.max_position_size

        # Position in capital units
        position = capital * position_fraction

        if signal.signal_type == SignalType.SHORT:
            position = -position

        return position


class MarketRegimeDetector:
    """
    Detects market regime changes using uncertainty dynamics.
    """

    def __init__(
        self,
        window_size: int = 20,
        epistemic_trend_threshold: float = 0.1,
        aleatoric_trend_threshold: float = 0.1
    ):
        """
        Initialize the regime detector.

        Args:
            window_size: Window size for trend calculation
            epistemic_trend_threshold: Threshold for epistemic trend
            aleatoric_trend_threshold: Threshold for aleatoric trend
        """
        self.window_size = window_size
        self.epistemic_trend_threshold = epistemic_trend_threshold
        self.aleatoric_trend_threshold = aleatoric_trend_threshold

        self.epistemic_history = []
        self.aleatoric_history = []

    def update(self, predictions: EnsemblePrediction):
        """
        Update regime detector with new predictions.

        Args:
            predictions: Ensemble predictions
        """
        self.epistemic_history.append(np.mean(predictions.epistemic_std))
        self.aleatoric_history.append(np.mean(predictions.aleatoric_std))

        # Keep only window_size history
        if len(self.epistemic_history) > self.window_size * 2:
            self.epistemic_history = self.epistemic_history[-self.window_size * 2:]
            self.aleatoric_history = self.aleatoric_history[-self.window_size * 2:]

    def detect_regime(self) -> str:
        """
        Detect current market regime.

        Returns:
            Regime string: 'STABLE', 'NORMAL', 'REGIME_CHANGE', or 'VOLATILITY_SPIKE'
        """
        if len(self.epistemic_history) < self.window_size:
            return "UNKNOWN"

        recent_epistemic = np.array(self.epistemic_history[-self.window_size:])
        recent_aleatoric = np.array(self.aleatoric_history[-self.window_size:])

        # Calculate trends using linear regression
        x = np.arange(self.window_size)
        epistemic_trend = np.polyfit(x, recent_epistemic, 1)[0]
        aleatoric_trend = np.polyfit(x, recent_aleatoric, 1)[0]

        if epistemic_trend > self.epistemic_trend_threshold:
            return "REGIME_CHANGE"  # Model becoming uncertain
        elif aleatoric_trend > self.aleatoric_trend_threshold:
            return "VOLATILITY_SPIKE"  # Market becoming chaotic
        elif np.mean(recent_epistemic) < 0.1 and np.mean(recent_aleatoric) < 0.1:
            return "STABLE"
        else:
            return "NORMAL"


class PortfolioManager:
    """
    Manages a portfolio of positions with risk controls.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        max_drawdown: float = 0.1,
        max_positions: int = 5
    ):
        """
        Initialize portfolio manager.

        Args:
            initial_capital: Starting capital
            max_drawdown: Maximum allowed drawdown
            max_positions: Maximum number of positions
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_drawdown = max_drawdown
        self.max_positions = max_positions

        self.positions: Dict[str, Position] = {}
        self.equity_curve = [initial_capital]
        self.peak_equity = initial_capital
        self.current_drawdown = 0

    def can_open_position(self) -> bool:
        """Check if a new position can be opened."""
        return (
            len(self.positions) < self.max_positions and
            self.current_drawdown < self.max_drawdown
        )

    def open_position(
        self,
        symbol: str,
        size: float,
        price: float,
        timestamp: int
    ):
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            size: Position size (negative for short)
            price: Entry price
            timestamp: Entry timestamp
        """
        if symbol in self.positions:
            print(f"Warning: Position already exists for {symbol}")
            return

        if not self.can_open_position():
            print("Warning: Cannot open new position")
            return

        self.positions[symbol] = Position(
            symbol=symbol,
            size=size,
            entry_price=price,
            entry_time=timestamp
        )

    def close_position(
        self,
        symbol: str,
        price: float
    ) -> float:
        """
        Close a position.

        Args:
            symbol: Trading symbol
            price: Exit price

        Returns:
            PnL from the trade
        """
        if symbol not in self.positions:
            print(f"Warning: No position for {symbol}")
            return 0

        position = self.positions[symbol]
        pnl = position.size * (price - position.entry_price)

        self.capital += pnl
        del self.positions[symbol]

        return pnl

    def update_equity(self, current_prices: Dict[str, float]):
        """
        Update equity curve with current prices.

        Args:
            current_prices: Dictionary of current prices by symbol
        """
        # Calculate unrealized PnL
        unrealized_pnl = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                unrealized_pnl += position.size * (current_prices[symbol] - position.entry_price)

        current_equity = self.capital + unrealized_pnl
        self.equity_curve.append(current_equity)

        # Update peak and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity


def compute_trading_metrics(
    equity_curve: List[float],
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Compute trading performance metrics.

    Args:
        equity_curve: List of equity values
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of metrics
    """
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    # Sharpe Ratio (annualized, assuming daily returns)
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
    sortino = np.sqrt(252) * np.mean(excess_returns) / downside_std

    # Maximum Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_drawdown = np.max(drawdown)

    # Win Rate
    winning_days = np.sum(returns > 0)
    total_days = len(returns)
    win_rate = winning_days / total_days if total_days > 0 else 0

    # Profit Factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Total Return
    total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] > 0 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': total_days,
        'final_equity': equity[-1]
    }


if __name__ == "__main__":
    from deep_ensemble import DeepEnsemble
    from data_fetcher import generate_synthetic_data

    print("Uncertainty-Aware Trading Strategy - Example")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    features, targets = generate_synthetic_data(n_samples=2000, n_features=8)

    # Split data
    train_size = int(0.6 * len(features))
    val_size = int(0.2 * len(features))

    X_train = features[:train_size]
    y_train = targets[:train_size]
    X_val = features[train_size:train_size+val_size]
    y_val = targets[train_size:train_size+val_size]
    X_test = features[train_size+val_size:]
    y_test = targets[train_size+val_size:]

    # Train ensemble
    print("\n2. Training Deep Ensemble...")
    ensemble = DeepEnsemble(
        input_dim=8,
        hidden_dims=[64, 32],
        num_models=5
    )
    ensemble.train(X_train, y_train, X_val, y_val, epochs=30, verbose=False)

    # Create strategy
    print("\n3. Creating trading strategy...")
    strategy = UncertaintyAwareStrategy(
        ensemble=ensemble,
        confidence_threshold=0.6,
        epistemic_threshold=0.5,
        max_position_size=0.1
    )

    # Create regime detector
    regime_detector = MarketRegimeDetector()

    # Create portfolio manager
    portfolio = PortfolioManager(initial_capital=100000)

    # Simulate trading
    print("\n4. Simulating trading...")
    symbols = ["BTC/USDT"]

    for i in range(len(X_test)):
        # Get predictions
        predictions = ensemble.predict(X_test[i:i+1])
        regime_detector.update(predictions)

        # Generate signals
        signals = strategy.generate_signals(X_test[i:i+1], symbols)

        # Detect regime
        regime = regime_detector.detect_regime()

        # Simulate price (using target as proxy)
        simulated_price = 100 * (1 + y_test[i])

        # Process signals
        for signal in signals:
            if regime == "REGIME_CHANGE":
                # Close all positions during regime change
                if signal.symbol in portfolio.positions:
                    pnl = portfolio.close_position(signal.symbol, simulated_price)
                continue

            if signal.signal_type == SignalType.LONG and portfolio.can_open_position():
                position_size = strategy.calculate_position_size(signal, portfolio.capital)
                if position_size > 0:
                    portfolio.open_position(signal.symbol, position_size, simulated_price, i)

            elif signal.signal_type == SignalType.SHORT and signal.symbol in portfolio.positions:
                portfolio.close_position(signal.symbol, simulated_price)

        # Update equity
        portfolio.update_equity({symbols[0]: simulated_price})

    # Compute metrics
    print("\n5. Trading Results:")
    metrics = compute_trading_metrics(portfolio.equity_curve)

    for metric, value in metrics.items():
        if 'return' in metric or 'drawdown' in metric or 'win_rate' in metric:
            print(f"   {metric}: {value*100:.2f}%")
        else:
            print(f"   {metric}: {value:.2f}")
