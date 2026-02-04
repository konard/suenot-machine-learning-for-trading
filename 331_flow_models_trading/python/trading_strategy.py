#!/usr/bin/env python3
"""
Trading Strategy using Flow Models

Implements trading signals based on flow model likelihood and regime detection.
Provides backtesting framework and signal generation for live trading.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from flow_model import NormalizingFlow, AnomalyDetector, RegimeDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"


@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    signal_type: SignalType
    confidence: float
    reason: str
    timestamp: Optional[pd.Timestamp] = None
    log_likelihood: Optional[float] = None
    regime: Optional[str] = None


@dataclass
class Trade:
    """Executed trade record"""
    timestamp: pd.Timestamp
    signal: SignalType
    price: float
    size: float
    pnl: float = 0.0


class FlowTradingStrategy:
    """
    Trading strategy using normalizing flow for signal generation.

    Uses:
    - Likelihood for anomaly detection (reduce exposure when unusual)
    - Latent space for regime detection (adapt to market state)
    - Conditional generation for scenario analysis
    """

    def __init__(
        self,
        flow_model: NormalizingFlow,
        anomaly_threshold_percentile: float = 5.0,
        n_regimes: int = 4,
        confidence_threshold: float = 0.6
    ):
        self.model = flow_model
        self.confidence_threshold = confidence_threshold

        # Initialize detectors
        self.anomaly_detector = AnomalyDetector(flow_model, anomaly_threshold_percentile)
        self.regime_detector = RegimeDetector(flow_model, n_regimes)

        # Regime labels based on market characteristics
        self.regime_names = [
            "High Vol Bull",
            "High Vol Bear",
            "Low Vol Bull",
            "Low Vol Bear"
        ]

        self.is_fitted = False

    def fit(self, training_data: torch.Tensor, returns: Optional[np.ndarray] = None):
        """
        Fit strategy components on training data

        Args:
            training_data: Feature data for flow model
            returns: Optional return data for regime labeling
        """
        logger.info("Fitting strategy components...")

        # Fit anomaly detector threshold
        self.anomaly_detector.fit(training_data)

        # Fit regime detector
        self.regime_detector.fit(training_data)

        # If returns provided, analyze and label regimes
        if returns is not None:
            self._label_regimes(training_data, returns)

        self.is_fitted = True
        logger.info("Strategy fitting complete")

    def _label_regimes(self, data: torch.Tensor, returns: np.ndarray):
        """Analyze and label regimes based on return characteristics"""
        regimes, _ = self.regime_detector.detect(data)

        regime_stats = {}
        for i in range(self.regime_detector.n_regimes):
            mask = regimes == i
            if mask.sum() > 0:
                regime_returns = returns[mask]
                regime_stats[i] = {
                    'mean': np.mean(regime_returns),
                    'std': np.std(regime_returns),
                    'sharpe': np.mean(regime_returns) / (np.std(regime_returns) + 1e-8)
                }

        # Sort regimes by characteristics and assign labels
        sorted_regimes = sorted(
            regime_stats.items(),
            key=lambda x: (x[1]['std'], x[1]['mean']),
            reverse=True
        )

        labels = []
        for i, (regime_idx, stats) in enumerate(sorted_regimes):
            if stats['std'] > np.median([s['std'] for s in regime_stats.values()]):
                vol = "High Vol"
            else:
                vol = "Low Vol"

            if stats['mean'] > 0:
                trend = "Bull"
            else:
                trend = "Bear"

            labels.append(f"{vol} {trend}")

        self.regime_detector.regime_labels = labels
        logger.info(f"Regime labels: {labels}")

    @torch.no_grad()
    def generate_signal(
        self,
        market_data: torch.Tensor,
        timestamp: Optional[pd.Timestamp] = None
    ) -> TradingSignal:
        """
        Generate trading signal from current market data

        Args:
            market_data: Current market features [1, input_dim]
            timestamp: Current timestamp

        Returns:
            TradingSignal with recommendation
        """
        if not self.is_fitted:
            raise RuntimeError("Strategy must be fitted before generating signals")

        self.model.eval()

        # Ensure correct shape
        if market_data.dim() == 1:
            market_data = market_data.unsqueeze(0)

        # 1. Compute likelihood for anomaly detection
        log_likelihood = self.model.log_prob(market_data).item()
        is_anomaly, _ = self.anomaly_detector.detect(market_data)

        # 2. Detect current regime
        regimes, distances = self.regime_detector.detect(market_data)
        current_regime = regimes[0]
        regime_label = self.regime_detector.regime_labels[current_regime]

        # Compute regime confidence (inverse of distance to cluster)
        min_dist = distances[0, current_regime]
        regime_confidence = 1.0 / (1.0 + min_dist)

        # 3. Generate signal based on conditions

        # Anomaly detected - reduce exposure
        if is_anomaly[0]:
            return TradingSignal(
                signal_type=SignalType.REDUCE_EXPOSURE,
                confidence=0.9,
                reason=f"Anomalous market state detected (log_p={log_likelihood:.2f})",
                timestamp=timestamp,
                log_likelihood=log_likelihood,
                regime=regime_label
            )

        # Generate signal based on regime
        if "Bull" in regime_label and regime_confidence > self.confidence_threshold:
            signal_type = SignalType.LONG
            confidence = regime_confidence * 0.8
            reason = f"Bullish regime detected with confidence {regime_confidence:.2f}"

        elif "Bear" in regime_label and regime_confidence > self.confidence_threshold:
            signal_type = SignalType.SHORT
            confidence = regime_confidence * 0.8
            reason = f"Bearish regime detected with confidence {regime_confidence:.2f}"

        else:
            signal_type = SignalType.NEUTRAL
            confidence = 0.5
            reason = f"Uncertain regime ({regime_label}, conf={regime_confidence:.2f})"

        return TradingSignal(
            signal_type=signal_type,
            confidence=confidence,
            reason=reason,
            timestamp=timestamp,
            log_likelihood=log_likelihood,
            regime=regime_label
        )

    def set_model_mode(self, training: bool = False):
        """Set model to training or inference mode"""
        if training:
            self.model.train()
        else:
            self.model.eval()


class FlowBacktester:
    """
    Backtesting framework for flow-based trading strategy.
    Simulates trading and computes performance metrics.
    """

    def __init__(
        self,
        strategy: FlowTradingStrategy,
        initial_capital: float = 100000.0,
        position_size: float = 0.1,
        transaction_cost_bps: float = 10.0
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost_bps / 10000

        # State
        self.capital = initial_capital
        self.position = 0.0
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

    def reset(self):
        """Reset backtester state"""
        self.capital = self.initial_capital
        self.position = 0.0
        self.trades = []
        self.equity_curve = []

    def run(
        self,
        features: torch.Tensor,
        prices: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> Dict:
        """
        Run backtest on historical data

        Args:
            features: Feature data [N, input_dim]
            prices: Price array [N]
            timestamps: Timestamp index

        Returns:
            Dictionary with backtest results
        """
        self.reset()
        n_samples = len(features)

        logger.info(f"Running backtest on {n_samples} samples")

        for i in range(n_samples):
            current_features = features[i:i+1]
            current_price = prices[i]
            current_time = timestamps[i]

            # Generate signal
            signal = self.strategy.generate_signal(current_features, current_time)

            # Execute trading logic
            self._execute_signal(signal, current_price, current_time)

            # Update equity
            equity = self.capital + self.position * current_price
            self.equity_curve.append(equity)

        # Close any remaining position
        if self.position != 0:
            self._close_position(prices[-1], timestamps[-1])

        # Compute metrics
        return self._compute_metrics(prices, timestamps)

    def _execute_signal(
        self,
        signal: TradingSignal,
        price: float,
        timestamp: pd.Timestamp
    ):
        """Execute trading signal"""

        if signal.signal_type == SignalType.REDUCE_EXPOSURE:
            # Close position if anomaly detected
            if self.position != 0:
                self._close_position(price, timestamp)

        elif signal.signal_type == SignalType.LONG:
            # Go long if not already long
            if self.position <= 0:
                if self.position < 0:
                    self._close_position(price, timestamp)
                self._open_position(1, price, timestamp, signal.confidence)

        elif signal.signal_type == SignalType.SHORT:
            # Go short if not already short
            if self.position >= 0:
                if self.position > 0:
                    self._close_position(price, timestamp)
                self._open_position(-1, price, timestamp, signal.confidence)

    def _open_position(
        self,
        direction: int,
        price: float,
        timestamp: pd.Timestamp,
        confidence: float
    ):
        """Open a new position"""
        # Size based on confidence
        size = self.capital * self.position_size * confidence / price
        size *= direction

        # Apply transaction cost
        cost = abs(size * price * self.transaction_cost)
        self.capital -= cost

        self.position = size
        self.trades.append(Trade(
            timestamp=timestamp,
            signal=SignalType.LONG if direction > 0 else SignalType.SHORT,
            price=price,
            size=abs(size)
        ))

    def _close_position(self, price: float, timestamp: pd.Timestamp):
        """Close current position"""
        if self.position == 0:
            return

        # Calculate P&L
        trade_value = self.position * price
        cost = abs(trade_value * self.transaction_cost)

        self.capital += trade_value - cost

        # Update last trade P&L
        if self.trades:
            entry_price = self.trades[-1].price
            pnl = (price - entry_price) * self.position - cost
            self.trades[-1].pnl = pnl

        self.position = 0.0

    def _compute_metrics(
        self,
        prices: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> Dict:
        """Compute backtest performance metrics"""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Basic metrics
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital

        # Annualized metrics (assuming daily data)
        n_days = (timestamps[-1] - timestamps[0]).days or 1
        annual_factor = 365 / n_days

        annual_return = (1 + total_return) ** annual_factor - 1
        annual_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0

        # Risk metrics
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Downside metrics
        negative_returns = returns[returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino = annual_return / downside_vol if downside_vol > 0 else 0

        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()

        # Trade statistics
        if self.trades:
            wins = sum(1 for t in self.trades if t.pnl > 0)
            win_rate = wins / len(self.trades)
            avg_pnl = np.mean([t.pnl for t in self.trades])
        else:
            win_rate = 0
            avg_pnl = 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'avg_trade_pnl': avg_pnl,
            'final_equity': equity[-1],
            'equity_curve': equity.tolist()
        }


def create_synthetic_market_data(n_samples: int, dim: int) -> np.ndarray:
    """Create synthetic market-like data for testing"""
    data1 = np.random.multivariate_normal(
        mean=np.zeros(dim) + 0.001,
        cov=np.eye(dim) * 0.02,
        size=n_samples // 2
    )
    data2 = np.random.multivariate_normal(
        mean=np.zeros(dim) - 0.001,
        cov=np.eye(dim) * 0.03,
        size=n_samples // 2
    )
    data = np.vstack([data1, data2]).astype(np.float32)
    np.random.shuffle(data)
    return data


def main():
    """Example: Run flow trading strategy backtest"""
    print("=" * 60)
    print("Flow Trading Strategy Backtest")
    print("=" * 60)

    # Import flow model
    from flow_model import NormalizingFlow, FlowModelTrainer

    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate synthetic data
    n_samples = 2000
    dim = 8

    print("\n1. Generating synthetic market data...")
    features = create_synthetic_market_data(n_samples, dim)
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Generate synthetic prices (random walk with drift)
    returns = np.random.randn(n_samples) * 0.02 + 0.0001  # 2% daily vol, slight upward drift
    prices = 100 * np.cumprod(1 + returns)
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')

    # Split data
    train_size = int(n_samples * 0.7)
    train_features = features_tensor[:train_size]
    train_returns = returns[:train_size]

    test_features = features_tensor[train_size:]
    test_prices = prices[train_size:]
    test_timestamps = timestamps[train_size:]

    print(f"   Train samples: {train_size}")
    print(f"   Test samples: {n_samples - train_size}")

    # Train flow model
    print("\n2. Training flow model...")
    model = NormalizingFlow(input_dim=dim, num_layers=4, hidden_dim=64)
    trainer = FlowModelTrainer(model, learning_rate=1e-3)
    trainer.train(train_features, epochs=30, batch_size=64, patience=5)

    # Create strategy
    print("\n3. Creating trading strategy...")
    strategy = FlowTradingStrategy(
        flow_model=model,
        anomaly_threshold_percentile=5.0,
        n_regimes=4,
        confidence_threshold=0.5
    )

    # Fit strategy
    print("\n4. Fitting strategy on training data...")
    strategy.fit(train_features, train_returns)

    # Run backtest
    print("\n5. Running backtest...")
    backtester = FlowBacktester(
        strategy=strategy,
        initial_capital=100000,
        position_size=0.1,
        transaction_cost_bps=10
    )

    results = backtester.run(test_features, test_prices, test_timestamps)

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Return:      {results['total_return']*100:>8.2f}%")
    print(f"Annual Return:     {results['annual_return']*100:>8.2f}%")
    print(f"Annual Volatility: {results['annual_volatility']*100:>8.2f}%")
    print(f"Sharpe Ratio:      {results['sharpe_ratio']:>8.2f}")
    print(f"Sortino Ratio:     {results['sortino_ratio']:>8.2f}")
    print(f"Max Drawdown:      {results['max_drawdown']*100:>8.2f}%")
    print(f"Win Rate:          {results['win_rate']*100:>8.2f}%")
    print(f"Number of Trades:  {results['num_trades']:>8d}")
    print(f"Avg Trade PnL:     ${results['avg_trade_pnl']:>7.2f}")
    print(f"Final Equity:      ${results['final_equity']:>10,.2f}")
    print("=" * 60)

    # Example signal generation
    print("\n6. Example signal generation...")
    sample_data = test_features[0:1]
    signal = strategy.generate_signal(sample_data, test_timestamps[0])
    print(f"   Signal: {signal.signal_type.value}")
    print(f"   Confidence: {signal.confidence:.2f}")
    print(f"   Regime: {signal.regime}")
    print(f"   Reason: {signal.reason}")

    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
