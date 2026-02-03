"""
Backtesting Module

This module provides functionality for backtesting rule-based
trading strategies on historical data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from rule_extractor import RuleSet


@dataclass
class Trade:
    """Record of a single trade."""
    idx: int
    trade_type: str  # "BUY" or "SELL"
    price: float
    size: float
    explanation: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Results from backtesting a strategy."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    avg_trade_return: float
    trades: List[Trade]
    equity_curve: np.ndarray
    returns: np.ndarray


class RuleBasedStrategy:
    """Trading strategy using extracted rules."""

    def __init__(self, ruleset: RuleSet, signal_threshold: float = 0.5):
        self.ruleset = ruleset
        self.signal_threshold = signal_threshold

    def generate_signal(self, features: np.ndarray) -> int:
        """
        Generate trading signal.

        Returns:
            1 for BUY, -1 for SELL, 0 for HOLD
        """
        return self.ruleset.predict(features)

    def explain_signal(self, features: np.ndarray) -> List[str]:
        """Get explanation for signal."""
        return self.ruleset.explain(features)


def backtest(
    strategy: RuleBasedStrategy,
    features: np.ndarray,
    prices: np.ndarray,
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.001
) -> BacktestResult:
    """
    Backtest a rule-based trading strategy.

    Args:
        strategy: Trading strategy to test
        features: Feature matrix (n_samples, n_features)
        prices: Price series
        initial_capital: Starting capital
        transaction_cost: Transaction cost as fraction

    Returns:
        BacktestResult with performance metrics
    """
    assert len(features) == len(prices), "Features and prices must have same length"

    capital = initial_capital
    position = 0.0  # Positive for long, negative for short
    entry_price = 0.0

    returns = []
    trades = []
    trade_returns = []
    equity_curve = [initial_capital]

    for i in range(len(features) - 1):
        signal = strategy.generate_signal(features[i])
        price = prices[i]
        next_price = prices[i + 1]

        # Execute trades based on signal
        current_pos_sign = 1 if position > 0 else (-1 if position < 0 else 0)

        if signal == 1 and current_pos_sign <= 0:  # BUY signal
            # Close short position if any
            if position < 0:
                pnl = position * (entry_price - price) * (1 - transaction_cost)
                capital += pnl
                trade_returns.append(pnl / abs(position * entry_price))

            # Open long position
            shares = capital * (1 - transaction_cost) / price
            position = shares
            entry_price = price
            capital = 0.0

            trades.append(Trade(
                idx=i,
                trade_type="BUY",
                price=price,
                size=shares,
                explanation=strategy.explain_signal(features[i])
            ))

        elif signal == -1 and current_pos_sign >= 0:  # SELL signal
            # Close long position if any
            if position > 0:
                capital = position * price * (1 - transaction_cost)
                pnl = position * (price - entry_price) - position * price * transaction_cost
                trade_returns.append(pnl / (position * entry_price))
                position = 0.0

            # Open short position
            shares = capital * (1 - transaction_cost) / price
            position = -shares
            entry_price = price
            capital = 0.0

            trades.append(Trade(
                idx=i,
                trade_type="SELL",
                price=price,
                size=shares,
                explanation=strategy.explain_signal(features[i])
            ))

        # Calculate period return
        if position > 0:
            period_return = (next_price - price) / price
        elif position < 0:
            period_return = (price - next_price) / price
        else:
            period_return = 0.0

        returns.append(period_return)

        # Update equity curve
        if position > 0:
            current_equity = position * next_price
        elif position < 0:
            current_equity = capital + abs(position) * (entry_price - next_price)
        else:
            current_equity = capital

        equity_curve.append(max(current_equity, 0.0))

    # Close any remaining position at the end
    if position != 0:
        final_price = prices[-1]
        if position > 0:
            capital = position * final_price * (1 - transaction_cost)
        else:
            capital += position * (entry_price - final_price) * (1 - transaction_cost)

    returns = np.array(returns)
    equity_curve = np.array(equity_curve)

    # Calculate metrics
    total_return = (equity_curve[-1] / initial_capital) - 1 if len(equity_curve) > 0 else 0
    sharpe_ratio = calculate_sharpe_ratio(returns)
    max_drawdown = calculate_max_drawdown(equity_curve)

    win_rate = np.mean([r > 0 for r in trade_returns]) if trade_returns else 0.0
    avg_trade_return = np.mean(trade_returns) if trade_returns else 0.0

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        n_trades=len(trades),
        win_rate=win_rate,
        avg_trade_return=avg_trade_return,
        trades=trades,
        equity_curve=equity_curve,
        returns=returns
    )


def calculate_sharpe_ratio(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    return (mean_return / std_return) * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculate maximum drawdown from equity curve."""
    if len(equity_curve) == 0:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity

        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return max_dd


def buy_and_hold_benchmark(
    prices: np.ndarray,
    initial_capital: float = 100000.0
) -> BacktestResult:
    """
    Calculate buy-and-hold benchmark performance.

    Args:
        prices: Price series
        initial_capital: Starting capital

    Returns:
        BacktestResult for benchmark
    """
    if len(prices) == 0:
        return BacktestResult(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            n_trades=0,
            win_rate=0.0,
            avg_trade_return=0.0,
            trades=[],
            equity_curve=np.array([initial_capital]),
            returns=np.array([])
        )

    shares = initial_capital / prices[0]
    equity_curve = shares * prices

    returns = np.diff(prices) / prices[:-1]

    total_return = (equity_curve[-1] / initial_capital) - 1
    sharpe_ratio = calculate_sharpe_ratio(returns)
    max_drawdown = calculate_max_drawdown(equity_curve)

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        n_trades=1,
        win_rate=1.0 if total_return > 0 else 0.0,
        avg_trade_return=total_return,
        trades=[Trade(
            idx=0,
            trade_type="BUY",
            price=prices[0],
            size=shares,
            explanation=["Buy and hold benchmark"]
        )],
        equity_curve=equity_curve,
        returns=returns
    )


def print_backtest_results(
    result: BacktestResult,
    benchmark: Optional[BacktestResult] = None,
    title: str = "Backtest Results"
) -> None:
    """Print formatted backtest results."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

    print(f"\n{'--- Strategy Performance ---':^50}")
    print(f"{'Total Return:':<25} {result.total_return*100:>10.2f}%")
    print(f"{'Sharpe Ratio:':<25} {result.sharpe_ratio:>10.2f}")
    print(f"{'Max Drawdown:':<25} {result.max_drawdown*100:>10.2f}%")
    print(f"{'Number of Trades:':<25} {result.n_trades:>10}")
    print(f"{'Win Rate:':<25} {result.win_rate*100:>10.1f}%")
    print(f"{'Avg Trade Return:':<25} {result.avg_trade_return*100:>10.2f}%")
    print(f"{'Final Equity:':<25} ${result.equity_curve[-1]:>9.2f}")

    if benchmark is not None:
        print(f"\n{'--- Benchmark (Buy & Hold) ---':^50}")
        print(f"{'Total Return:':<25} {benchmark.total_return*100:>10.2f}%")
        print(f"{'Sharpe Ratio:':<25} {benchmark.sharpe_ratio:>10.2f}")
        print(f"{'Max Drawdown:':<25} {benchmark.max_drawdown*100:>10.2f}%")

        excess = result.total_return - benchmark.total_return
        sign = "+" if excess >= 0 else ""
        print(f"\n{'--- Comparison ---':^50}")
        print(f"{'Excess Return:':<25} {sign}{excess*100:>9.2f}%")
        print(f"{'Sharpe Difference:':<25} {result.sharpe_ratio - benchmark.sharpe_ratio:>+10.2f}")


def print_trades(trades: List[Trade], max_trades: int = 10) -> None:
    """Print trade history."""
    if not trades:
        print("\nNo trades executed.")
        return

    print(f"\n{'--- Trade History ---':^50}")
    print(f"{'Index':>6} {'Type':>8} {'Price':>12} {'Size':>12}")
    print("-" * 50)

    for trade in trades[:max_trades]:
        print(f"{trade.idx:>6} {trade.trade_type:>8} {trade.price:>12.2f} {trade.size:>12.4f}")

    if len(trades) > max_trades:
        print(f"... ({len(trades) - max_trades} more trades)")


if __name__ == "__main__":
    # Example usage
    from rule_extractor import Rule, Condition, RuleSet
    import numpy as np

    print("=== Backtest Example ===")

    # Generate sample data
    np.random.seed(42)
    n = 500

    # Simulated price data with trends
    prices = np.zeros(n)
    prices[0] = 100
    trend = 0

    for i in range(1, n):
        if i % 50 == 0:
            trend = np.random.uniform(-0.002, 0.002)
        prices[i] = prices[i-1] * (1 + trend + np.random.randn() * 0.02)

    prices = np.clip(prices, 50, 200)

    # Generate features
    rsi = 50 + np.cumsum(np.random.randn(n) * 2)
    rsi = np.clip(rsi, 10, 90)

    macd = np.cumsum(np.random.randn(n) * 0.1)
    macd = np.clip(macd, -2, 2)

    sma_ratio = 1 + np.cumsum(np.random.randn(n) * 0.002)
    sma_ratio = np.clip(sma_ratio, 0.95, 1.05)

    volatility = 0.02 + np.abs(np.random.randn(n) * 0.01)

    volume_change = np.random.randn(n) * 0.2

    features = np.column_stack([rsi, macd, sma_ratio, volatility, volume_change])
    feature_names = ['RSI', 'MACD', 'SMA_Ratio', 'Volatility', 'Volume_Change']

    # Create trading rules
    rules = [
        Rule(
            conditions=[
                Condition(0, '<=', 35.0, 'RSI'),
                Condition(1, '>', 0.0, 'MACD')
            ],
            prediction=1,
            confidence=0.75
        ),
        Rule(
            conditions=[
                Condition(0, '>', 65.0, 'RSI'),
                Condition(1, '<=', 0.0, 'MACD')
            ],
            prediction=-1,
            confidence=0.75
        ),
        Rule(
            conditions=[
                Condition(2, '>', 1.02, 'SMA_Ratio')
            ],
            prediction=1,
            confidence=0.65
        ),
        Rule(
            conditions=[
                Condition(2, '<=', 0.98, 'SMA_Ratio'),
                Condition(3, '>', 0.02, 'Volatility')
            ],
            prediction=-1,
            confidence=0.70
        ),
    ]

    ruleset = RuleSet(rules, feature_names)
    strategy = RuleBasedStrategy(ruleset)

    print("\nTrading Rules:")
    for i, rule in enumerate(rules):
        print(f"  {i+1}. {rule.to_string(feature_names)}")

    # Run backtest
    print("\nRunning backtest...")
    result = backtest(strategy, features, prices, initial_capital=100000)
    benchmark = buy_and_hold_benchmark(prices, initial_capital=100000)

    # Print results
    print_backtest_results(result, benchmark, "Rule-Based Strategy Backtest")
    print_trades(result.trades)

    # Show trade explanations
    print("\n--- Trade Explanations (first 3) ---")
    for trade in result.trades[:3]:
        print(f"\n[{trade.idx}] {trade.trade_type} at ${trade.price:.2f}")
        for exp in trade.explanation:
            print(f"  - {exp}")
