"""
Trading Strategy and Backtesting

This module provides trading strategy implementation and backtesting
utilities for evaluating time series models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    LONG = 1
    NEUTRAL = 0
    SHORT = -1


@dataclass
class BacktestResult:
    """
    Container for backtest results.

    Attributes:
        returns: Daily/period returns
        cumulative_returns: Cumulative returns
        equity_curve: Portfolio value over time
        sharpe_ratio: Annualized Sharpe ratio
        sortino_ratio: Sortino ratio
        max_drawdown: Maximum drawdown
        total_return: Total period return
        win_rate: Percentage of winning trades
        n_trades: Number of trades
        metrics: Additional metrics dictionary
    """
    returns: np.ndarray
    cumulative_returns: np.ndarray
    equity_curve: np.ndarray
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    n_trades: int
    metrics: Dict


class TradingStrategy:
    """
    Base trading strategy class.

    Converts model predictions to trading signals and manages positions.

    Args:
        threshold: Signal threshold for position entry
        max_position: Maximum position size (1.0 = 100%)
        transaction_cost: Transaction cost per trade (fraction)
        stop_loss: Stop loss threshold (fraction)
        take_profit: Take profit threshold (fraction)

    Example:
        >>> strategy = TradingStrategy(threshold=0.001)
        >>> signals = strategy.generate_signals(predictions, threshold=0.001)
    """

    def __init__(
        self,
        threshold: float = 0.001,
        max_position: float = 1.0,
        transaction_cost: float = 0.001,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.threshold = threshold
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signals(
        self,
        predictions: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate trading signals from predictions.

        Args:
            predictions: Model predictions (returns or directions)
            threshold: Override default threshold

        Returns:
            Array of signals (-1, 0, 1)
        """
        if threshold is None:
            threshold = self.threshold

        signals = np.zeros_like(predictions)
        signals[predictions > threshold] = 1
        signals[predictions < -threshold] = -1

        return signals

    def generate_position_sizes(
        self,
        signals: np.ndarray,
        confidences: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate position sizes from signals.

        Args:
            signals: Trading signals (-1, 0, 1)
            confidences: Optional confidence scores

        Returns:
            Position sizes
        """
        if confidences is not None:
            # Scale position by confidence
            positions = signals * confidences * self.max_position
        else:
            positions = signals * self.max_position

        return np.clip(positions, -self.max_position, self.max_position)


def calculate_metrics(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate trading performance metrics.

    Args:
        returns: Array of period returns
        periods_per_year: Number of periods per year (252 for daily, 8760 for hourly)

    Returns:
        Dictionary of performance metrics
    """
    # Filter out NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0
        }

    # Total return
    cumulative = np.cumprod(1 + returns)
    total_return = cumulative[-1] - 1

    # Annualized return
    n_periods = len(returns)
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # Volatility
    volatility = np.std(returns) * np.sqrt(periods_per_year)

    # Sharpe ratio (assuming 0 risk-free rate)
    if volatility > 0:
        sharpe_ratio = annual_return / volatility
    else:
        sharpe_ratio = 0.0

    # Sortino ratio (downside volatility)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_vol = np.std(downside_returns) * np.sqrt(periods_per_year)
        sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0.0
    else:
        sortino_ratio = sharpe_ratio * 1.5  # Approximate if no downside

    # Maximum drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    max_drawdown = np.min(drawdowns)

    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Win rate
    winning_trades = np.sum(returns > 0)
    total_trades = np.sum(returns != 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate
    }


def run_backtest(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    strategy: Optional[TradingStrategy] = None,
    initial_capital: float = 100000,
    periods_per_year: int = 8760  # Hourly data
) -> BacktestResult:
    """
    Run backtest on model predictions.

    Args:
        predictions: Model return predictions
        actual_returns: Actual market returns
        strategy: Trading strategy (uses default if None)
        initial_capital: Starting capital
        periods_per_year: Periods per year for annualization

    Returns:
        BacktestResult with performance metrics

    Example:
        >>> predictions = model(test_data)['predictions'].numpy()
        >>> result = run_backtest(predictions, actual_returns)
        >>> print(f"Sharpe: {result.sharpe_ratio:.2f}")
    """
    if strategy is None:
        strategy = TradingStrategy()

    # Ensure arrays are 1D
    if predictions.ndim > 1:
        predictions = predictions[:, 0]
    if actual_returns.ndim > 1:
        actual_returns = actual_returns[:, 0]

    # Align lengths
    min_len = min(len(predictions), len(actual_returns))
    predictions = predictions[:min_len]
    actual_returns = actual_returns[:min_len]

    # Generate signals
    signals = strategy.generate_signals(predictions)
    positions = strategy.generate_position_sizes(signals)

    # Calculate strategy returns
    strategy_returns = positions * actual_returns

    # Apply transaction costs
    position_changes = np.abs(np.diff(positions, prepend=0))
    transaction_costs = position_changes * strategy.transaction_cost
    strategy_returns = strategy_returns - transaction_costs

    # Calculate equity curve
    equity_curve = initial_capital * np.cumprod(1 + strategy_returns)

    # Calculate metrics
    metrics = calculate_metrics(strategy_returns, periods_per_year)

    # Cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1

    # Count trades
    n_trades = np.sum(position_changes > 0)

    return BacktestResult(
        returns=strategy_returns,
        cumulative_returns=cumulative_returns,
        equity_curve=equity_curve,
        sharpe_ratio=metrics['sharpe_ratio'],
        sortino_ratio=metrics['sortino_ratio'],
        max_drawdown=metrics['max_drawdown'],
        total_return=metrics['total_return'],
        win_rate=metrics['win_rate'],
        n_trades=n_trades,
        metrics=metrics
    )


def compare_strategies(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    thresholds: List[float] = [0.0005, 0.001, 0.002],
    periods_per_year: int = 8760
) -> pd.DataFrame:
    """
    Compare different strategy thresholds.

    Args:
        predictions: Model predictions
        actual_returns: Actual returns
        thresholds: List of thresholds to test
        periods_per_year: Periods per year

    Returns:
        DataFrame comparing strategy performance
    """
    results = []

    for threshold in thresholds:
        strategy = TradingStrategy(threshold=threshold)
        result = run_backtest(
            predictions, actual_returns, strategy,
            periods_per_year=periods_per_year
        )

        results.append({
            'threshold': threshold,
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'n_trades': result.n_trades
        })

    return pd.DataFrame(results)


def calculate_buy_and_hold(
    actual_returns: np.ndarray,
    initial_capital: float = 100000,
    periods_per_year: int = 8760
) -> BacktestResult:
    """
    Calculate buy and hold benchmark.

    Args:
        actual_returns: Market returns
        initial_capital: Starting capital
        periods_per_year: Periods per year

    Returns:
        BacktestResult for buy and hold strategy
    """
    if actual_returns.ndim > 1:
        actual_returns = actual_returns[:, 0]

    equity_curve = initial_capital * np.cumprod(1 + actual_returns)
    cumulative_returns = np.cumprod(1 + actual_returns) - 1
    metrics = calculate_metrics(actual_returns, periods_per_year)

    return BacktestResult(
        returns=actual_returns,
        cumulative_returns=cumulative_returns,
        equity_curve=equity_curve,
        sharpe_ratio=metrics['sharpe_ratio'],
        sortino_ratio=metrics['sortino_ratio'],
        max_drawdown=metrics['max_drawdown'],
        total_return=metrics['total_return'],
        win_rate=metrics['win_rate'],
        n_trades=1,
        metrics=metrics
    )


if __name__ == "__main__":
    # Test strategy and backtesting
    print("Testing Trading Strategy and Backtesting...")
    print("=" * 60)

    # Generate synthetic predictions and returns
    np.random.seed(42)
    n_samples = 1000

    # Simulate model predictions with some signal
    true_signal = np.random.normal(0, 0.01, n_samples)
    predictions = true_signal + np.random.normal(0, 0.005, n_samples)
    actual_returns = true_signal + np.random.normal(0, 0.02, n_samples)

    # Test strategy
    print("\n1. Testing TradingStrategy...")
    strategy = TradingStrategy(threshold=0.001, transaction_cost=0.001)
    signals = strategy.generate_signals(predictions)
    positions = strategy.generate_position_sizes(signals)
    print(f"   Long signals: {np.sum(signals > 0)}")
    print(f"   Short signals: {np.sum(signals < 0)}")
    print(f"   Neutral: {np.sum(signals == 0)}")

    # Test backtest
    print("\n2. Running backtest...")
    result = run_backtest(predictions, actual_returns, strategy)
    print(f"   Total Return: {result.total_return:.2%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown:.2%}")
    print(f"   Win Rate: {result.win_rate:.2%}")
    print(f"   Trades: {result.n_trades}")

    # Compare with buy and hold
    print("\n3. Buy and Hold benchmark...")
    bh_result = calculate_buy_and_hold(actual_returns)
    print(f"   Total Return: {bh_result.total_return:.2%}")
    print(f"   Sharpe Ratio: {bh_result.sharpe_ratio:.2f}")

    # Compare thresholds
    print("\n4. Comparing thresholds...")
    comparison = compare_strategies(predictions, actual_returns)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 60)
    print("Strategy tests completed!")
