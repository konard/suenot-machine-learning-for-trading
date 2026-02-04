"""
Backtesting Engine for Conformal Prediction Trading

Provides comprehensive backtesting with performance metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

from conformal_predictor import SplitConformalPredictor, compute_coverage_metrics
from trading_strategy import ConformalTradingStrategy, PortfolioManager, TradingSignal


@dataclass
class BacktestResult:
    """Container for backtest results."""
    equity_curve: np.ndarray
    returns: np.ndarray
    signals: List[TradingSignal]
    trades: List[dict]
    coverage_metrics: dict
    performance_metrics: dict
    timestamps: pd.DatetimeIndex


class ConformalBacktester:
    """
    Backtesting engine for conformal prediction trading strategies.
    """

    def __init__(
        self,
        conformal_predictor: SplitConformalPredictor,
        strategy_config: Optional[dict] = None,
        portfolio_config: Optional[dict] = None
    ):
        """
        Initialize backtester.

        Args:
            conformal_predictor: Calibrated conformal predictor
            strategy_config: Strategy configuration
            portfolio_config: Portfolio configuration
        """
        self.conformal_predictor = conformal_predictor

        # Default strategy config
        self.strategy_config = strategy_config or {
            'max_position': 1.0,
            'min_confidence': 0.4,
            'volatility_threshold': 0.03,
            'position_scaling': 'inverse_width'
        }

        # Default portfolio config
        self.portfolio_config = portfolio_config or {
            'initial_capital': 10000.0,
            'max_leverage': 1.0,
            'transaction_cost': 0.001,
            'slippage': 0.0005
        }

    def run_backtest(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        prices: np.ndarray,
        timestamps: pd.DatetimeIndex,
        y_train: Optional[np.ndarray] = None
    ) -> BacktestResult:
        """
        Run full backtest.

        Args:
            X_test: Test features
            y_test: Test targets (actual returns)
            prices: Price series
            timestamps: Timestamps for test data
            y_train: Training targets (for volatility estimation)

        Returns:
            BacktestResult with all metrics
        """
        # Create strategy
        strategy = ConformalTradingStrategy(
            self.conformal_predictor,
            **self.strategy_config
        )

        # Set historical volatility
        if y_train is not None:
            strategy.set_historical_volatility(y_train)
        else:
            strategy.set_historical_volatility(y_test[:100])

        # Generate signals
        signals = strategy.generate_signals_batch(X_test, timestamps)

        # Create portfolio
        portfolio = PortfolioManager(**self.portfolio_config)

        # Execute backtest
        equity_curve = [portfolio.capital]

        for signal, price in zip(signals, prices):
            portfolio.execute_signal(signal, price)
            equity = portfolio.update_equity(price)
            equity_curve.append(equity)

        equity_curve = np.array(equity_curve[1:])  # Remove initial

        # Calculate returns
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Get predictions for coverage analysis
        point_preds, lower_bounds, upper_bounds = self.conformal_predictor.predict(X_test)

        # Compute coverage metrics
        coverage_metrics = compute_coverage_metrics(
            y_test, lower_bounds, upper_bounds,
            target_alpha=self.conformal_predictor.alpha
        )

        # Get performance metrics
        performance_metrics = portfolio.get_performance_metrics()

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            signals=signals,
            trades=portfolio.trades,
            coverage_metrics=coverage_metrics,
            performance_metrics=performance_metrics,
            timestamps=timestamps
        )

    def run_walk_forward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prices: np.ndarray,
        timestamps: pd.DatetimeIndex,
        train_size: int = 500,
        cal_size: int = 200,
        step_size: int = 50
    ) -> List[BacktestResult]:
        """
        Run walk-forward backtesting.

        Args:
            X: All features
            y: All targets
            prices: All prices
            timestamps: All timestamps
            train_size: Training window size
            cal_size: Calibration window size
            step_size: Step size for walking forward

        Returns:
            List of BacktestResult for each window
        """
        results = []
        n = len(X)

        start_idx = 0
        while start_idx + train_size + cal_size + step_size < n:
            train_end = start_idx + train_size
            cal_end = train_end + cal_size
            test_end = min(cal_end + step_size, n)

            # Get data splits
            X_train = X[start_idx:train_end]
            y_train = y[start_idx:train_end]
            X_cal = X[train_end:cal_end]
            y_cal = y[train_end:cal_end]
            X_test = X[cal_end:test_end]
            y_test = y[cal_end:test_end]

            # Retrain and recalibrate
            self.conformal_predictor.fit(X_train, y_train)
            self.conformal_predictor.calibrate(X_cal, y_cal)

            # Run backtest for this window
            result = self.run_backtest(
                X_test, y_test,
                prices[cal_end:test_end],
                timestamps[cal_end:test_end],
                y_train
            )

            results.append(result)

            start_idx += step_size

        return results


def plot_backtest_results(result: BacktestResult, save_path: Optional[str] = None):
    """
    Create visualization of backtest results.

    Args:
        result: BacktestResult to visualize
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # 1. Equity curve
    ax = axes[0, 0]
    ax.plot(result.timestamps[:len(result.equity_curve)], result.equity_curve)
    ax.set_title('Equity Curve')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value')
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    peak = np.maximum.accumulate(result.equity_curve)
    drawdown = (peak - result.equity_curve) / peak
    ax.fill_between(
        result.timestamps[:len(drawdown)],
        drawdown,
        alpha=0.5,
        color='red'
    )
    ax.set_title('Drawdown')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown %')
    ax.grid(True, alpha=0.3)

    # 3. Signal distribution
    ax = axes[1, 0]
    signal_types = [s.direction for s in result.signals]
    unique, counts = np.unique(signal_types, return_counts=True)
    ax.bar(unique, counts, color=['green', 'gray', 'red'])
    ax.set_title('Signal Distribution')
    ax.set_xlabel('Signal Type')
    ax.set_ylabel('Count')

    # 4. Position sizes over time
    ax = axes[1, 1]
    positions = [s.position_size for s in result.signals]
    ax.plot(result.timestamps[:len(positions)], positions, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Position Size Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Position Size')
    ax.grid(True, alpha=0.3)

    # 5. Confidence distribution
    ax = axes[2, 0]
    confidences = [s.confidence for s in result.signals]
    ax.hist(confidences, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(
        x=result.performance_metrics.get('win_rate', 0.5),
        color='red',
        linestyle='--',
        label='Win Rate'
    )
    ax.set_title('Confidence Distribution')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.legend()

    # 6. Performance metrics text
    ax = axes[2, 1]
    ax.axis('off')

    metrics_text = "Performance Metrics\n" + "=" * 30 + "\n"
    for key, value in result.performance_metrics.items():
        if isinstance(value, float):
            metrics_text += f"{key}: {value:.4f}\n"
        else:
            metrics_text += f"{key}: {value}\n"

    metrics_text += "\nCoverage Metrics\n" + "=" * 30 + "\n"
    for key, value in result.coverage_metrics.items():
        if isinstance(value, float):
            metrics_text += f"{key}: {value:.4f}\n"
        else:
            metrics_text += f"{key}: {value}\n"

    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def print_backtest_summary(result: BacktestResult):
    """Print summary of backtest results."""
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    print("\nPerformance Metrics:")
    print("-" * 40)
    perf = result.performance_metrics
    print(f"  Total Return:     {perf['total_return']*100:>10.2f}%")
    print(f"  Sharpe Ratio:     {perf['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:    {perf['sortino_ratio']:>10.2f}")
    print(f"  Max Drawdown:     {perf['max_drawdown']*100:>10.2f}%")
    print(f"  Win Rate:         {perf['win_rate']*100:>10.2f}%")
    print(f"  Number of Trades: {perf['num_trades']:>10d}")
    print(f"  Final Capital:    ${perf['final_capital']:>10,.2f}")

    print("\nCoverage Metrics:")
    print("-" * 40)
    cov = result.coverage_metrics
    print(f"  Actual Coverage:  {cov['coverage']*100:>10.2f}%")
    print(f"  Target Coverage:  {cov['target_coverage']*100:>10.2f}%")
    print(f"  Coverage Gap:     {cov['coverage_gap']*100:>10.2f}%")
    print(f"  Mean Int. Width:  {cov['mean_width']:>10.6f}")
    print(f"  Median Int. Width:{cov['median_width']:>10.6f}")

    print("\nSignal Statistics:")
    print("-" * 40)
    signal_types = [s.direction for s in result.signals]
    for signal_type in ['LONG', 'SHORT', 'HOLD']:
        count = signal_types.count(signal_type)
        pct = count / len(signal_types) * 100
        print(f"  {signal_type:>10s}: {count:>6d} ({pct:>6.2f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage with synthetic data
    from sklearn.ensemble import GradientBoostingRegressor
    from conformal_predictor import SplitConformalPredictor

    np.random.seed(42)

    # Generate synthetic market data
    n = 2000
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    returns = np.diff(prices) / prices[:-1]

    # Create features
    lookback = 10
    X = np.column_stack([
        np.array([returns[i:i+lookback].mean() for i in range(len(returns)-lookback)]),
        np.array([returns[i:i+lookback].std() for i in range(len(returns)-lookback)]),
        np.array([returns[i+lookback-1] for i in range(len(returns)-lookback)]),
    ])
    y = returns[lookback:]

    # Align prices and timestamps
    prices = prices[lookback+1:]
    timestamps = pd.date_range('2024-01-01', periods=len(X), freq='1h')

    # Split data
    train_end = int(0.6 * len(X))
    cal_end = int(0.8 * len(X))

    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[train_end:cal_end], y[train_end:cal_end]
    X_test, y_test = X[cal_end:], y[cal_end:]

    # Create and calibrate conformal predictor
    base_model = GradientBoostingRegressor(n_estimators=100, max_depth=4)
    cp = SplitConformalPredictor(base_model, alpha=0.1)
    cp.fit_calibrate(X_train, y_train, X_cal, y_cal)

    # Run backtest
    backtester = ConformalBacktester(
        cp,
        strategy_config={
            'max_position': 1.0,
            'min_confidence': 0.3,
            'volatility_threshold': 0.05,
            'position_scaling': 'inverse_width'
        },
        portfolio_config={
            'initial_capital': 10000.0,
            'transaction_cost': 0.001,
            'slippage': 0.0005
        }
    )

    result = backtester.run_backtest(
        X_test, y_test,
        prices[cal_end:],
        timestamps[cal_end:],
        y_train
    )

    # Print and plot results
    print_backtest_summary(result)

    # Uncomment to show plots
    # plot_backtest_results(result)
