"""
Trading strategy and backtesting utilities for Multi-Scale Attention.

This module provides tools for backtesting trading strategies based on
multi-scale attention predictions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class BacktestResult:
    """Container for backtest results."""

    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    equity_curve: pd.Series
    trades: pd.DataFrame


def backtest_multi_scale_strategy(
    predictions: Dict[str, np.ndarray],
    prices: pd.DataFrame,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    confidence_threshold: float = 0.55,
    position_sizing: str = "confidence",
    max_position: float = 1.0,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
) -> BacktestResult:
    """
    Backtest a multi-scale attention trading strategy.

    Args:
        predictions: Model predictions dictionary with keys:
            - 'direction': Direction probabilities [n_samples, 1]
            - 'short_term': Short-term predictions [n_samples, 1]
            - 'uncertainty': Uncertainty estimates [n_samples, 1]
        prices: DataFrame with 'close' column
        initial_capital: Starting capital
        transaction_cost: Cost per transaction (fraction)
        confidence_threshold: Minimum direction confidence to trade
        position_sizing: 'fixed', 'confidence', or 'kelly'
        max_position: Maximum position size (fraction of capital)
        stop_loss: Stop loss threshold (fraction)
        take_profit: Take profit threshold (fraction)

    Returns:
        BacktestResult with performance metrics
    """
    n_samples = len(predictions["direction"])
    prices = prices.iloc[:n_samples].copy()

    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    equity = [capital]
    returns = []
    trades = []

    for i in range(n_samples - 1):
        current_price = prices["close"].iloc[i]
        next_price = prices["close"].iloc[i + 1]

        # Get signals
        direction_prob = predictions["direction"][i, 0]
        uncertainty = (
            predictions.get("uncertainty", np.ones_like(predictions["direction"]))[i, 0]
        )

        # Determine confidence
        confidence = abs(direction_prob - 0.5) * 2

        # Check stop loss / take profit
        if position != 0 and entry_price > 0:
            unrealized_return = (current_price - entry_price) / entry_price * np.sign(
                position
            )

            if stop_loss is not None and unrealized_return < -stop_loss:
                # Stop loss triggered
                position_change = -position
                cost = abs(position_change) * transaction_cost * capital
                capital -= cost

                trades.append(
                    {
                        "index": i,
                        "type": "stop_loss",
                        "price": current_price,
                        "position": position,
                        "cost": cost,
                    }
                )
                position = 0
                entry_price = 0

            elif take_profit is not None and unrealized_return > take_profit:
                # Take profit triggered
                position_change = -position
                cost = abs(position_change) * transaction_cost * capital
                capital -= cost

                trades.append(
                    {
                        "index": i,
                        "type": "take_profit",
                        "price": current_price,
                        "position": position,
                        "cost": cost,
                    }
                )
                position = 0
                entry_price = 0

        # Determine target position
        if confidence > confidence_threshold:
            if position_sizing == "fixed":
                target_position = max_position if direction_prob > 0.5 else -max_position
            elif position_sizing == "confidence":
                target_position = confidence * max_position
                if direction_prob < 0.5:
                    target_position = -target_position
            elif position_sizing == "kelly":
                # Simplified Kelly criterion
                win_prob = direction_prob if direction_prob > 0.5 else (1 - direction_prob)
                kelly_fraction = (2 * win_prob - 1) / 1  # Assuming 1:1 payoff
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                target_position = kelly_fraction * max_position
                if direction_prob < 0.5:
                    target_position = -target_position
            else:
                target_position = 0.0

            # Adjust for uncertainty (reduce position if uncertain)
            if uncertainty > 0:
                target_position = target_position / (1 + uncertainty)
        else:
            target_position = 0.0

        # Calculate position change
        position_change = target_position - position

        # Transaction costs
        if abs(position_change) > 0.01:
            cost = abs(position_change) * transaction_cost * capital
            capital -= cost

            if position_change != 0:
                entry_price = current_price

            trades.append(
                {
                    "index": i,
                    "type": "long" if target_position > 0 else "short",
                    "price": current_price,
                    "position": target_position,
                    "cost": cost,
                }
            )

        position = target_position

        # Calculate return
        price_return = (next_price - current_price) / current_price
        pnl = position * price_return * capital
        capital += pnl

        if capital > 0:
            returns.append(pnl / equity[-1])
        else:
            returns.append(0)

        equity.append(capital)

    # Calculate metrics
    equity_series = pd.Series(equity)
    returns_series = pd.Series(returns)

    total_return = (capital - initial_capital) / initial_capital

    # Annualized Sharpe ratio (assuming hourly data)
    periods_per_year = 24 * 365
    if len(returns_series) > 1 and returns_series.std() > 0:
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(periods_per_year)
    else:
        sharpe = 0

    # Sortino ratio
    downside_returns = returns_series[returns_series < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino = (
            returns_series.mean() / downside_returns.std() * np.sqrt(periods_per_year)
        )
    else:
        sortino = 0

    # Max drawdown
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    winning_returns = returns_series[returns_series > 0]
    win_rate = len(winning_returns) / len(returns_series) if len(returns_series) > 0 else 0

    # Profit factor
    gross_profit = winning_returns.sum() if len(winning_returns) > 0 else 0
    losing_returns = returns_series[returns_series < 0]
    gross_loss = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Volatility (annualized)
    volatility = returns_series.std() * np.sqrt(periods_per_year)

    # Calmar ratio
    calmar_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else 0

    # Average trade return
    avg_trade_return = returns_series.mean() if len(returns_series) > 0 else 0

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_trades=len(trades),
        avg_trade_return=avg_trade_return,
        volatility=volatility,
        calmar_ratio=calmar_ratio,
        equity_curve=equity_series,
        trades=pd.DataFrame(trades) if trades else pd.DataFrame(),
    )


def compare_scale_strategies(
    predictions: Dict[str, np.ndarray],
    prices: pd.DataFrame,
    **backtest_kwargs,
) -> pd.DataFrame:
    """
    Compare strategies using different horizon predictions.

    Args:
        predictions: Model predictions dictionary
        prices: DataFrame with price data
        **backtest_kwargs: Additional arguments for backtest_multi_scale_strategy

    Returns:
        DataFrame comparing different strategies
    """
    results = {}

    horizons = ["short_term", "medium_term", "long_term"]

    for horizon in horizons:
        if horizon not in predictions:
            continue

        # Create predictions using only this horizon for direction
        single_horizon_preds = {
            "direction": (predictions[horizon] > 0).astype(float),
            "short_term": predictions[horizon],
        }

        if "uncertainty" in predictions:
            single_horizon_preds["uncertainty"] = predictions["uncertainty"]

        result = backtest_multi_scale_strategy(
            single_horizon_preds, prices, **backtest_kwargs
        )

        results[horizon] = {
            "Total Return": f"{result.total_return * 100:.2f}%",
            "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{result.sortino_ratio:.2f}",
            "Max Drawdown": f"{result.max_drawdown * 100:.2f}%",
            "Win Rate": f"{result.win_rate * 100:.1f}%",
            "Profit Factor": f"{result.profit_factor:.2f}",
            "Trades": result.n_trades,
            "Calmar Ratio": f"{result.calmar_ratio:.2f}",
        }

    # Add combined strategy
    result = backtest_multi_scale_strategy(predictions, prices, **backtest_kwargs)
    results["multi_scale"] = {
        "Total Return": f"{result.total_return * 100:.2f}%",
        "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
        "Sortino Ratio": f"{result.sortino_ratio:.2f}",
        "Max Drawdown": f"{result.max_drawdown * 100:.2f}%",
        "Win Rate": f"{result.win_rate * 100:.1f}%",
        "Profit Factor": f"{result.profit_factor:.2f}",
        "Trades": result.n_trades,
        "Calmar Ratio": f"{result.calmar_ratio:.2f}",
    }

    return pd.DataFrame(results).T


def calculate_information_ratio(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 24 * 365,
) -> float:
    """
    Calculate the Information Ratio.

    Args:
        strategy_returns: Strategy returns array
        benchmark_returns: Benchmark returns array
        periods_per_year: Number of periods per year

    Returns:
        Information Ratio
    """
    excess_returns = strategy_returns - benchmark_returns
    tracking_error = np.std(excess_returns)

    if tracking_error == 0:
        return 0

    return np.mean(excess_returns) / tracking_error * np.sqrt(periods_per_year)


def calculate_var(
    returns: np.ndarray, confidence: float = 0.95, method: str = "historical"
) -> float:
    """
    Calculate Value at Risk.

    Args:
        returns: Returns array
        confidence: Confidence level
        method: 'historical' or 'parametric'

    Returns:
        VaR value (negative number representing loss)
    """
    if method == "historical":
        return np.percentile(returns, (1 - confidence) * 100)
    elif method == "parametric":
        from scipy import stats

        mean = np.mean(returns)
        std = np.std(returns)
        return mean + std * stats.norm.ppf(1 - confidence)
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_expected_shortfall(
    returns: np.ndarray, confidence: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).

    Args:
        returns: Returns array
        confidence: Confidence level

    Returns:
        Expected Shortfall value
    """
    var = calculate_var(returns, confidence)
    return np.mean(returns[returns <= var])


def generate_performance_report(result: BacktestResult) -> str:
    """
    Generate a text performance report.

    Args:
        result: BacktestResult from backtesting

    Returns:
        Formatted performance report string
    """
    report = """
================================================================================
                        MULTI-SCALE ATTENTION STRATEGY REPORT
================================================================================

PERFORMANCE METRICS
-------------------
Total Return:        {total_return:>10.2%}
Sharpe Ratio:        {sharpe:>10.2f}
Sortino Ratio:       {sortino:>10.2f}
Calmar Ratio:        {calmar:>10.2f}

RISK METRICS
------------
Maximum Drawdown:    {max_dd:>10.2%}
Volatility (Ann.):   {volatility:>10.2%}

TRADING METRICS
---------------
Total Trades:        {n_trades:>10d}
Win Rate:            {win_rate:>10.2%}
Profit Factor:       {profit_factor:>10.2f}
Avg Trade Return:    {avg_trade:>10.4%}

================================================================================
    """.format(
        total_return=result.total_return,
        sharpe=result.sharpe_ratio,
        sortino=result.sortino_ratio,
        calmar=result.calmar_ratio,
        max_dd=result.max_drawdown,
        volatility=result.volatility,
        n_trades=result.n_trades,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
        avg_trade=result.avg_trade_return,
    )

    return report


def plot_equity_curve(
    result: BacktestResult,
    title: str = "Multi-Scale Attention Strategy",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the equity curve.

    Args:
        result: BacktestResult from backtesting
        title: Plot title
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Equity curve
    axes[0].plot(result.equity_curve, color="steelblue", linewidth=1)
    axes[0].fill_between(
        range(len(result.equity_curve)),
        result.equity_curve.iloc[0],
        result.equity_curve,
        alpha=0.3,
    )
    axes[0].set_title(title)
    axes[0].set_ylabel("Portfolio Value")
    axes[0].grid(True, alpha=0.3)

    # Drawdown
    running_max = result.equity_curve.cummax()
    drawdown = (result.equity_curve - running_max) / running_max
    axes[1].fill_between(
        range(len(drawdown)), 0, drawdown * 100, color="red", alpha=0.5
    )
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close()
