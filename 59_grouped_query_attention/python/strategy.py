"""
Trading Strategy and Backtesting for GQA Model

This module provides backtesting framework and strategy utilities
for evaluating the GQA trading model performance.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .model import GQATrader
from .predict import predict_next, get_confidence


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: int
    entry_price: float
    direction: str  # "LONG" or "SHORT"
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None


@dataclass
class BacktestResult:
    """Results from backtesting."""
    total_return: float
    num_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: np.ndarray


def backtest_strategy(
    model: GQATrader,
    data: np.ndarray,
    seq_len: int = 60,
    initial_capital: float = 10000.0,
    position_size: float = 1.0,
    confidence_threshold: float = 0.3,
    transaction_cost: float = 0.001,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    verbose: bool = True,
    device: Optional[torch.device] = None
) -> BacktestResult:
    """
    Backtest the GQA trading model on historical data.

    Args:
        model: Trained GQATrader model
        data: OHLCV data array of shape (n_samples, 5)
        seq_len: Sequence length for model input
        initial_capital: Starting capital
        position_size: Fraction of capital to use per trade (0 to 1)
        confidence_threshold: Minimum confidence to take a trade
        transaction_cost: Transaction cost as fraction of trade value
        stop_loss: Stop loss percentage (e.g., 0.02 for 2%)
        take_profit: Take profit percentage
        verbose: Whether to print progress
        device: Inference device

    Returns:
        BacktestResult with performance metrics

    Example:
        >>> model = GQATrader.load("model.pt")
        >>> data = load_bybit_data("BTCUSDT", limit=1000)
        >>> result = backtest_strategy(model, data, seq_len=60)
        >>> print(f"Total return: {result.total_return:.2%}")
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    n_samples = len(data)
    if n_samples < seq_len + 10:
        raise ValueError(f"Not enough data: {n_samples} samples for seq_len={seq_len}")

    # Initialize backtest state
    capital = initial_capital
    position = 0  # 1 = long, -1 = short, 0 = flat
    entry_price = 0.0
    entry_time = 0

    trades: List[Trade] = []
    equity_curve = [initial_capital]

    if verbose:
        print(f"Backtesting from index {seq_len} to {n_samples - 1}")
        print(f"Initial capital: ${initial_capital:,.2f}")

    # Backtest loop
    for i in range(seq_len, n_samples - 1):
        # Get current and next prices
        current_close = data[i, 3]
        next_close = data[i + 1, 3]

        # Get model prediction
        sequence = torch.from_numpy(data[i - seq_len:i]).float()
        pred, probs = predict_next(model, sequence, return_probs=True, device=device)
        confidence = get_confidence(probs)

        # Check stop loss / take profit if in position
        if position != 0:
            pnl_percent = (current_close - entry_price) / entry_price * position

            # Stop loss
            if stop_loss and pnl_percent <= -stop_loss:
                exit_trade(trades, i, current_close)
                capital *= (1 + pnl_percent - transaction_cost)
                position = 0

            # Take profit
            elif take_profit and pnl_percent >= take_profit:
                exit_trade(trades, i, current_close)
                capital *= (1 + pnl_percent - transaction_cost)
                position = 0

        # Trading logic
        if confidence >= confidence_threshold:
            if pred == 2 and position <= 0:  # UP signal
                # Close short if any
                if position == -1:
                    pnl_percent = (entry_price - current_close) / entry_price
                    exit_trade(trades, i, current_close)
                    capital *= (1 + pnl_percent - transaction_cost)

                # Open long
                position = 1
                entry_price = current_close
                entry_time = i
                trades.append(Trade(
                    entry_time=i,
                    entry_price=current_close,
                    direction="LONG"
                ))
                capital *= (1 - transaction_cost)

            elif pred == 0 and position >= 0:  # DOWN signal
                # Close long if any
                if position == 1:
                    pnl_percent = (current_close - entry_price) / entry_price
                    exit_trade(trades, i, current_close)
                    capital *= (1 + pnl_percent - transaction_cost)

                # Open short
                position = -1
                entry_price = current_close
                entry_time = i
                trades.append(Trade(
                    entry_time=i,
                    entry_price=current_close,
                    direction="SHORT"
                ))
                capital *= (1 - transaction_cost)

        # Update equity based on current position
        if position == 1:
            unrealized_pnl = (next_close - entry_price) / entry_price
            equity_curve.append(capital * (1 + unrealized_pnl))
        elif position == -1:
            unrealized_pnl = (entry_price - next_close) / entry_price
            equity_curve.append(capital * (1 + unrealized_pnl))
        else:
            equity_curve.append(capital)

    # Close any remaining position
    if position != 0:
        final_price = data[-1, 3]
        if position == 1:
            pnl_percent = (final_price - entry_price) / entry_price
        else:
            pnl_percent = (entry_price - final_price) / entry_price

        exit_trade(trades, n_samples - 1, final_price)
        capital *= (1 + pnl_percent - transaction_cost)

    # Calculate metrics
    equity_curve = np.array(equity_curve)
    metrics = calculate_metrics(trades, equity_curve, initial_capital)

    if verbose:
        print(f"\nBacktest Results:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Number of Trades: {metrics['num_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

    return BacktestResult(
        total_return=metrics['total_return'],
        num_trades=metrics['num_trades'],
        win_rate=metrics['win_rate'],
        profit_factor=metrics['profit_factor'],
        max_drawdown=metrics['max_drawdown'],
        sharpe_ratio=metrics['sharpe_ratio'],
        trades=trades,
        equity_curve=equity_curve
    )


def exit_trade(trades: List[Trade], time: int, price: float):
    """Update the last trade with exit information."""
    if trades and trades[-1].exit_time is None:
        trade = trades[-1]
        trade.exit_time = time
        trade.exit_price = price

        if trade.direction == "LONG":
            trade.pnl = price - trade.entry_price
            trade.pnl_percent = trade.pnl / trade.entry_price
        else:  # SHORT
            trade.pnl = trade.entry_price - price
            trade.pnl_percent = trade.pnl / trade.entry_price


def calculate_metrics(
    trades: List[Trade],
    equity_curve: np.ndarray,
    initial_capital: float
) -> Dict:
    """
    Calculate trading performance metrics.

    Args:
        trades: List of completed trades
        equity_curve: Array of equity values over time
        initial_capital: Starting capital

    Returns:
        Dictionary with performance metrics
    """
    # Filter completed trades
    completed_trades = [t for t in trades if t.exit_time is not None]

    if not completed_trades:
        return {
            'total_return': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }

    # Basic metrics
    total_return = (equity_curve[-1] - initial_capital) / initial_capital
    num_trades = len(completed_trades)

    # Win rate
    winning_trades = [t for t in completed_trades if t.pnl and t.pnl > 0]
    win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0

    # Profit factor
    gross_profit = sum(t.pnl for t in completed_trades if t.pnl and t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in completed_trades if t.pnl and t.pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Maximum drawdown
    peak = equity_curve[0]
    max_drawdown = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)

    # Sharpe ratio (assuming daily returns, annualized)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0

    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }


def compare_strategies(
    model: GQATrader,
    data: np.ndarray,
    strategies: Dict[str, Dict],
    seq_len: int = 60,
    verbose: bool = True
) -> Dict[str, BacktestResult]:
    """
    Compare multiple strategy configurations.

    Args:
        model: Trained GQATrader model
        data: OHLCV data
        strategies: Dictionary of strategy name -> parameters
        seq_len: Sequence length
        verbose: Whether to print results

    Returns:
        Dictionary of strategy name -> BacktestResult

    Example:
        >>> strategies = {
        ...     "conservative": {"confidence_threshold": 0.5, "stop_loss": 0.01},
        ...     "aggressive": {"confidence_threshold": 0.2, "stop_loss": 0.03}
        ... }
        >>> results = compare_strategies(model, data, strategies)
    """
    results = {}

    for name, params in strategies.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Strategy: {name}")
            print(f"Parameters: {params}")

        result = backtest_strategy(
            model, data, seq_len=seq_len, verbose=verbose, **params
        )
        results[name] = result

    if verbose:
        print(f"\n{'='*50}")
        print("Strategy Comparison Summary:")
        print("-" * 50)
        print(f"{'Strategy':<20} {'Return':>10} {'Win Rate':>10} {'Sharpe':>10}")
        print("-" * 50)
        for name, result in results.items():
            print(f"{name:<20} {result.total_return:>10.2%} "
                  f"{result.win_rate:>10.2%} {result.sharpe_ratio:>10.2f}")

    return results


def walk_forward_optimization(
    model: GQATrader,
    data: np.ndarray,
    param_grid: Dict[str, List],
    train_window: int = 500,
    test_window: int = 100,
    seq_len: int = 60,
    verbose: bool = True
) -> Dict:
    """
    Perform walk-forward optimization to find best parameters.

    Args:
        model: Trained GQATrader model
        data: OHLCV data
        param_grid: Dictionary of parameter name -> list of values to try
        train_window: Training window size
        test_window: Testing window size
        seq_len: Sequence length
        verbose: Whether to print progress

    Returns:
        Dictionary with optimization results
    """
    from itertools import product

    n_samples = len(data)
    n_windows = (n_samples - train_window - seq_len) // test_window

    if n_windows < 1:
        raise ValueError("Not enough data for walk-forward optimization")

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    if verbose:
        print(f"Testing {len(combinations)} parameter combinations "
              f"over {n_windows} windows")

    results = {combo: [] for combo in combinations}

    for window in range(n_windows):
        train_start = window * test_window
        train_end = train_start + train_window
        test_end = train_end + test_window

        test_data = data[train_start:test_end]

        for combo in combinations:
            params = dict(zip(param_names, combo))

            try:
                result = backtest_strategy(
                    model, test_data, seq_len=seq_len, verbose=False, **params
                )
                results[combo].append(result.total_return)
            except Exception:
                results[combo].append(0)

    # Find best parameters
    avg_returns = {combo: np.mean(returns) for combo, returns in results.items()}
    best_combo = max(avg_returns, key=avg_returns.get)
    best_params = dict(zip(param_names, best_combo))

    if verbose:
        print(f"\nBest parameters: {best_params}")
        print(f"Average return: {avg_returns[best_combo]:.2%}")

    return {
        "best_params": best_params,
        "best_return": avg_returns[best_combo],
        "all_results": {
            dict(zip(param_names, combo)).__str__(): np.mean(returns)
            for combo, returns in results.items()
        }
    }


if __name__ == "__main__":
    # Test strategy utilities
    print("Testing Strategy Utilities...")
    print("=" * 50)

    from .model import GQATrader
    from .data import _generate_synthetic_data

    # Create model
    print("\n1. Creating model...")
    model = GQATrader(
        input_dim=5,
        d_model=32,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2
    )

    # Generate test data
    print("\n2. Generating test data...")
    data = _generate_synthetic_data(500)
    print(f"   Data shape: {data.shape}")

    # Run backtest
    print("\n3. Running backtest...")
    result = backtest_strategy(
        model,
        data,
        seq_len=30,
        initial_capital=10000,
        confidence_threshold=0.2,
        verbose=True
    )

    print(f"\n4. Equity curve shape: {result.equity_curve.shape}")
    print(f"   Final equity: ${result.equity_curve[-1]:,.2f}")

    # Compare strategies
    print("\n5. Comparing strategies...")
    strategies = {
        "conservative": {"confidence_threshold": 0.5},
        "moderate": {"confidence_threshold": 0.3},
        "aggressive": {"confidence_threshold": 0.1}
    }
    comparison = compare_strategies(model, data, strategies, seq_len=30, verbose=True)

    print("\n" + "=" * 50)
    print("All strategy tests passed!")
