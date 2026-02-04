"""
Backtesting Framework for Deep Ensemble Trading Strategy

This module provides a backtesting framework for testing
Deep Ensemble trading strategies on historical data.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from deep_ensemble import DeepEnsemble, EnsemblePrediction
from strategy import (
    UncertaintyAwareStrategy,
    Signal,
    SignalType,
    MarketRegimeDetector,
    compute_trading_metrics
)


@dataclass
class Trade:
    """Record of a single trade."""
    symbol: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    holding_period: int
    signal_confidence: float
    epistemic_at_entry: float


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    initial_capital: float
    final_capital: float
    equity_curve: List[float]
    trades: List[Trade]
    metrics: Dict[str, float]
    regime_history: List[str]
    uncertainty_history: List[Dict[str, float]]


class BacktestEngine:
    """
    Backtesting engine for Deep Ensemble trading strategies.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        trading_fee: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize the backtest engine.

        Args:
            initial_capital: Starting capital
            trading_fee: Trading fee as fraction (e.g., 0.001 = 0.1%)
            slippage: Slippage as fraction
        """
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.slippage = slippage

    def run(
        self,
        strategy: UncertaintyAwareStrategy,
        features: np.ndarray,
        prices: np.ndarray,
        symbols: List[str],
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            strategy: Trading strategy to test
            features: Feature matrix [n_timesteps, n_features]
            prices: Price array [n_timesteps]
            symbols: List of symbols (length 1 for single asset)
            verbose: Whether to print progress

        Returns:
            BacktestResult with all metrics and trade records
        """
        n_steps = len(features)
        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        regime_history = []
        uncertainty_history = []

        # Track position
        position = None
        position_entry_time = None
        position_entry_price = None
        position_signal = None

        # Regime detector
        regime_detector = MarketRegimeDetector()

        for t in range(n_steps):
            current_price = prices[t]

            # Get prediction and signal
            prediction = strategy.ensemble.predict(features[t:t+1])
            signals = strategy.generate_signals(features[t:t+1], symbols)
            signal = signals[0] if signals else None

            # Update regime
            regime_detector.update(prediction)
            regime = regime_detector.detect_regime()
            regime_history.append(regime)

            # Track uncertainty
            uncertainty_history.append({
                'epistemic': float(np.mean(prediction.epistemic_std)),
                'aleatoric': float(np.mean(prediction.aleatoric_std)),
                'total': float(np.mean(prediction.total_std))
            })

            # Trading logic
            if position is not None:
                # Check exit conditions
                should_exit = False

                if regime == "REGIME_CHANGE":
                    should_exit = True
                elif signal and signal.signal_type == SignalType.SHORT and position > 0:
                    should_exit = True
                elif signal and signal.signal_type == SignalType.LONG and position < 0:
                    should_exit = True

                if should_exit:
                    # Close position
                    exit_price = current_price * (1 - np.sign(position) * self.slippage)
                    fee = abs(position) * exit_price * self.trading_fee

                    pnl = position * (exit_price - position_entry_price) - fee

                    capital += pnl

                    trades.append(Trade(
                        symbol=symbols[0],
                        entry_time=position_entry_time,
                        exit_time=t,
                        entry_price=position_entry_price,
                        exit_price=exit_price,
                        size=position,
                        pnl=pnl,
                        return_pct=pnl / (abs(position) * position_entry_price),
                        holding_period=t - position_entry_time,
                        signal_confidence=position_signal.confidence if position_signal else 0,
                        epistemic_at_entry=position_signal.epistemic_uncertainty if position_signal else 0
                    ))

                    position = None
                    position_entry_time = None
                    position_entry_price = None
                    position_signal = None

            # Check entry conditions
            if position is None and signal and regime not in ["REGIME_CHANGE", "UNKNOWN"]:
                if signal.signal_type == SignalType.LONG:
                    # Calculate position size
                    position_value = strategy.calculate_position_size(signal, capital)
                    if position_value > 0:
                        entry_price = current_price * (1 + self.slippage)
                        fee = position_value * self.trading_fee

                        position = position_value / entry_price
                        position_entry_time = t
                        position_entry_price = entry_price
                        position_signal = signal
                        capital -= fee

                elif signal.signal_type == SignalType.SHORT:
                    # Short position
                    position_value = strategy.calculate_position_size(signal, capital)
                    if position_value < 0:
                        entry_price = current_price * (1 - self.slippage)
                        fee = abs(position_value) * self.trading_fee

                        position = position_value / entry_price
                        position_entry_time = t
                        position_entry_price = entry_price
                        position_signal = signal
                        capital -= fee

            # Update equity curve
            if position is not None:
                unrealized_pnl = position * (current_price - position_entry_price)
                equity_curve.append(capital + unrealized_pnl)
            else:
                equity_curve.append(capital)

            if verbose and (t + 1) % 100 == 0:
                print(f"  Step {t+1}/{n_steps}, Equity: ${equity_curve[-1]:.2f}")

        # Close any remaining position
        if position is not None:
            exit_price = prices[-1]
            pnl = position * (exit_price - position_entry_price)
            capital += pnl
            equity_curve[-1] = capital

        # Compute metrics
        metrics = compute_trading_metrics(equity_curve)

        # Add trade-specific metrics
        if trades:
            metrics['num_trades'] = len(trades)
            metrics['avg_trade_pnl'] = np.mean([t.pnl for t in trades])
            metrics['avg_holding_period'] = np.mean([t.holding_period for t in trades])
            metrics['avg_confidence'] = np.mean([t.signal_confidence for t in trades])
            metrics['avg_epistemic_at_entry'] = np.mean([t.epistemic_at_entry for t in trades])

            # Win rate
            winning_trades = [t for t in trades if t.pnl > 0]
            metrics['trade_win_rate'] = len(winning_trades) / len(trades)
        else:
            metrics['num_trades'] = 0

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=capital,
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            regime_history=regime_history,
            uncertainty_history=uncertainty_history
        )


def generate_synthetic_prices(
    n_steps: int,
    initial_price: float = 100,
    drift: float = 0.0001,
    volatility: float = 0.02,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic price series using geometric Brownian motion.

    Args:
        n_steps: Number of time steps
        initial_price: Starting price
        drift: Daily drift (mean return)
        volatility: Daily volatility
        seed: Random seed

    Returns:
        Price array
    """
    np.random.seed(seed)

    returns = np.random.normal(drift, volatility, n_steps)
    log_prices = np.log(initial_price) + np.cumsum(returns)
    prices = np.exp(log_prices)

    return prices


def generate_synthetic_features(
    prices: np.ndarray,
    n_features: int = 8
) -> np.ndarray:
    """
    Generate synthetic features from price series.

    Args:
        prices: Price array
        n_features: Number of features

    Returns:
        Feature matrix
    """
    n_steps = len(prices)
    features = np.zeros((n_steps, n_features))

    # Returns at different lookbacks
    for i, lookback in enumerate([1, 5, 10, 20]):
        if i < n_features:
            returns = np.zeros(n_steps)
            returns[lookback:] = (prices[lookback:] - prices[:-lookback]) / prices[:-lookback]
            features[:, i] = returns

    # Volatility
    if n_features > 4:
        returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
        for i, window in enumerate([10, 20]):
            if 4 + i < n_features:
                vol = np.zeros(n_steps)
                for t in range(window, n_steps):
                    vol[t] = np.std(returns[t-window:t])
                features[:, 4 + i] = vol

    # Moving average deviation
    if n_features > 6:
        ma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
        features[:, 6] = (prices - ma_20) / ma_20

    # Random feature (noise)
    if n_features > 7:
        features[:, 7] = np.random.randn(n_steps) * 0.01

    # Fill NaN with zeros
    features = np.nan_to_num(features)

    return features


def print_backtest_report(result: BacktestResult):
    """
    Print a formatted backtest report.

    Args:
        result: BacktestResult object
    """
    print("\n" + "=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)

    print(f"\nCapital:")
    print(f"  Initial: ${result.initial_capital:,.2f}")
    print(f"  Final:   ${result.final_capital:,.2f}")
    print(f"  Return:  {(result.final_capital/result.initial_capital - 1)*100:.2f}%")

    print(f"\nPerformance Metrics:")
    for key, value in result.metrics.items():
        if 'return' in key or 'drawdown' in key or 'win_rate' in key:
            print(f"  {key}: {value*100:.2f}%")
        elif 'ratio' in key or 'factor' in key:
            print(f"  {key}: {value:.2f}")
        elif 'equity' in key or 'capital' in key:
            print(f"  {key}: ${value:,.2f}")
        else:
            print(f"  {key}: {value:.2f}")

    print(f"\nTrading Activity:")
    print(f"  Total trades: {len(result.trades)}")
    if result.trades:
        winning = [t for t in result.trades if t.pnl > 0]
        losing = [t for t in result.trades if t.pnl <= 0]
        print(f"  Winning trades: {len(winning)}")
        print(f"  Losing trades: {len(losing)}")
        print(f"  Average PnL: ${np.mean([t.pnl for t in result.trades]):.2f}")
        print(f"  Best trade: ${max(t.pnl for t in result.trades):.2f}")
        print(f"  Worst trade: ${min(t.pnl for t in result.trades):.2f}")

    print(f"\nRegime Distribution:")
    regimes, counts = np.unique(result.regime_history, return_counts=True)
    for regime, count in zip(regimes, counts):
        print(f"  {regime}: {count} ({count/len(result.regime_history)*100:.1f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Deep Ensemble Trading Backtest")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic market data...")
    n_steps = 1000
    prices = generate_synthetic_prices(n_steps, initial_price=100)
    features = generate_synthetic_features(prices, n_features=8)

    print(f"   Generated {n_steps} time steps")
    print(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")

    # Split data
    train_end = int(0.6 * n_steps)
    val_end = int(0.8 * n_steps)

    X_train = features[:train_end]
    y_train = np.diff(np.log(prices[:train_end+1]))
    X_val = features[train_end:val_end]
    y_val = np.diff(np.log(prices[train_end:val_end+1]))
    X_test = features[val_end:]
    prices_test = prices[val_end:]

    # Train ensemble
    print("\n2. Training Deep Ensemble...")
    ensemble = DeepEnsemble(
        input_dim=8,
        hidden_dims=[64, 32],
        num_models=5
    )
    ensemble.train(X_train, y_train, X_val, y_val, epochs=50, verbose=False)
    print("   Training complete")

    # Create strategy
    print("\n3. Creating trading strategy...")
    strategy = UncertaintyAwareStrategy(
        ensemble=ensemble,
        confidence_threshold=0.6,
        epistemic_threshold=0.5,
        max_position_size=0.15,
        signal_strength_threshold=1.0
    )

    # Run backtest
    print("\n4. Running backtest...")
    engine = BacktestEngine(
        initial_capital=100000,
        trading_fee=0.001,
        slippage=0.0005
    )

    result = engine.run(
        strategy=strategy,
        features=X_test,
        prices=prices_test,
        symbols=["BTC/USDT"],
        verbose=True
    )

    # Print report
    print_backtest_report(result)

    # Compare with buy-and-hold
    print("\nComparison with Buy-and-Hold:")
    bh_return = (prices_test[-1] - prices_test[0]) / prices_test[0]
    strategy_return = (result.final_capital - result.initial_capital) / result.initial_capital
    print(f"  Strategy Return: {strategy_return*100:.2f}%")
    print(f"  Buy-and-Hold Return: {bh_return*100:.2f}%")
    print(f"  Outperformance: {(strategy_return - bh_return)*100:.2f}%")
