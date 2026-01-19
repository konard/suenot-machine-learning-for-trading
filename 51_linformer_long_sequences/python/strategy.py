"""
Trading strategy and backtesting utilities for Linformer.

This module provides:
- Backtesting framework for Linformer predictions
- Performance metrics calculation (Sharpe, Sortino, Max Drawdown)
- Signal generation from model predictions
- Portfolio allocation strategies
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def generate_signals(
    predictions: np.ndarray,
    threshold: float = 0.001,
    signal_type: str = 'binary'
) -> np.ndarray:
    """
    Generate trading signals from model predictions.

    Args:
        predictions: Model predictions array
        threshold: Threshold for generating signals
        signal_type: Type of signal ('binary', 'ternary', 'continuous')

    Returns:
        Signal array with values:
            - binary: 1 (buy) or 0 (hold)
            - ternary: 1 (buy), 0 (hold), -1 (sell)
            - continuous: raw predictions clipped to [-1, 1]
    """
    predictions = np.array(predictions).flatten()

    if signal_type == 'binary':
        # Long only
        signals = (predictions > threshold).astype(float)

    elif signal_type == 'ternary':
        # Long, short, or flat
        signals = np.zeros_like(predictions)
        signals[predictions > threshold] = 1
        signals[predictions < -threshold] = -1

    elif signal_type == 'continuous':
        # Continuous position sizing
        signals = np.clip(predictions / threshold, -1, 1)

    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    return signals


def backtest_linformer_strategy(
    model: torch.nn.Module,
    test_data: Dict,
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.001,
    position_sizing: str = 'equal',
    signal_threshold: float = 0.001,
    max_position: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> pd.DataFrame:
    """
    Backtest a Linformer-based trading strategy.

    Args:
        model: Trained Linformer model
        test_data: Test data dictionary with 'X' and 'y' keys
        initial_capital: Starting capital
        transaction_cost: Transaction cost per trade (as fraction)
        position_sizing: 'equal' or 'proportional'
        signal_threshold: Threshold for generating trading signals
        max_position: Maximum position size (1.0 = 100%)
        device: Computation device

    Returns:
        DataFrame with backtesting results including:
            - capital: Portfolio value at each step
            - position: Current position
            - prediction: Model prediction
            - actual_return: Actual market return
            - portfolio_return: Portfolio return
            - trade_cost: Transaction costs
    """
    model.eval()
    model = model.to(device)

    X = torch.FloatTensor(test_data['X']).to(device)
    actual_returns = test_data['y'].flatten()

    capital = initial_capital
    position = 0.0
    results = []

    logger.info(f"Running backtest with {len(X)} samples...")

    with torch.no_grad():
        for i in range(len(X)):
            # Get model prediction
            prediction = model(X[i:i+1]).cpu().numpy().flatten()[0]

            # Generate signal
            if prediction > signal_threshold:
                target_position = min(1.0, max_position)
            elif prediction < -signal_threshold:
                target_position = max(-1.0, -max_position)
            else:
                target_position = 0.0

            # Position sizing
            if position_sizing == 'proportional':
                # Scale position by prediction confidence
                target_position *= min(abs(prediction) / signal_threshold, 1.0)

            # Calculate trade cost
            position_change = abs(target_position - position)
            trade_cost = position_change * transaction_cost * capital

            # Update position
            position = target_position

            # Calculate return
            actual_return = actual_returns[i]
            portfolio_return = position * actual_return

            # Update capital
            capital = capital * (1 + portfolio_return) - trade_cost

            results.append({
                'step': i,
                'capital': capital,
                'position': position,
                'prediction': prediction,
                'actual_return': actual_return,
                'portfolio_return': portfolio_return,
                'trade_cost': trade_cost,
                'cumulative_return': (capital - initial_capital) / initial_capital
            })

    df = pd.DataFrame(results)

    # Add timestamps if available
    if 'timestamps' in test_data:
        df['timestamp'] = test_data['timestamps'][:len(df)]

    # Calculate and log summary metrics
    metrics = calculate_performance_metrics(df, initial_capital)

    logger.info("\n" + "="*50)
    logger.info("BACKTEST RESULTS")
    logger.info("="*50)
    logger.info(f"Total Return: {metrics['total_return']:.2f}%")
    logger.info(f"Annualized Return: {metrics['annualized_return']:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    logger.info(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
    logger.info(f"Profit Factor: {metrics['profit_factor']:.3f}")
    logger.info(f"Final Capital: ${capital:,.2f}")
    logger.info("="*50)

    return df


def calculate_performance_metrics(
    results: pd.DataFrame,
    initial_capital: float = 100000.0,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252 * 24  # hourly data
) -> Dict:
    """
    Calculate comprehensive performance metrics.

    Args:
        results: Backtest results DataFrame
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (for annualization)

    Returns:
        Dictionary of performance metrics
    """
    returns = results['portfolio_return'].values

    # Total and annualized return
    total_return = (results['capital'].iloc[-1] / initial_capital - 1) * 100
    n_periods = len(results)
    annualized_return = ((1 + total_return/100) ** (periods_per_year / n_periods) - 1) * 100

    # Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)

    # Sortino Ratio
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)

    # Max Drawdown
    max_drawdown = calculate_max_drawdown(results['capital'])

    # Win Rate
    winning_trades = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # Profit Factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    # Calmar Ratio
    calmar_ratio = (annualized_return / max_drawdown) if max_drawdown > 0 else float('inf')

    # Average trade
    avg_trade = returns.mean() * 100

    # Standard deviation of returns
    volatility = returns.std() * np.sqrt(periods_per_year) * 100

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'calmar_ratio': calmar_ratio,
        'avg_trade': avg_trade,
        'volatility': volatility,
        'total_trades': total_trades
    }


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252 * 24
) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev of Returns

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Convert annual risk-free rate to per-period rate
    rf_per_period = risk_free_rate / periods_per_year

    excess_returns = returns - rf_per_period
    sharpe = excess_returns.mean() / excess_returns.std()

    # Annualize
    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252 * 24
) -> float:
    """
    Calculate annualized Sortino Ratio.

    Sortino Ratio = (Mean Return - Risk Free Rate) / Downside Std Dev

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) == 0:
        return 0.0

    # Convert annual risk-free rate to per-period rate
    rf_per_period = risk_free_rate / periods_per_year

    excess_returns = returns - rf_per_period

    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0

    downside_std = np.sqrt((downside_returns ** 2).mean())
    sortino = excess_returns.mean() / downside_std

    # Annualize
    return sortino * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown percentage.

    Max Drawdown = (Peak - Trough) / Peak

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Maximum drawdown as percentage
    """
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return abs(drawdown.min()) * 100


def calculate_information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252 * 24
) -> float:
    """
    Calculate Information Ratio.

    Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year

    Returns:
        Annualized Information Ratio
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark must have same length")

    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std()

    if tracking_error == 0:
        return 0.0

    ir = excess_returns.mean() / tracking_error
    return ir * np.sqrt(periods_per_year)


def compare_strategies(
    results_dict: Dict[str, pd.DataFrame],
    initial_capital: float = 100000.0
) -> pd.DataFrame:
    """
    Compare multiple backtesting strategies.

    Args:
        results_dict: Dictionary mapping strategy names to results DataFrames
        initial_capital: Starting capital

    Returns:
        Comparison DataFrame with metrics for each strategy
    """
    comparison = []

    for name, results in results_dict.items():
        metrics = calculate_performance_metrics(results, initial_capital)
        metrics['strategy'] = name
        comparison.append(metrics)

    return pd.DataFrame(comparison).set_index('strategy')


def plot_backtest_results(
    results: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    title: str = "Linformer Strategy Backtest"
) -> None:
    """
    Plot backtest results.

    Creates a multi-panel figure with:
    1. Equity curve
    2. Drawdown
    3. Position over time
    4. Return distribution

    Args:
        results: Backtest results DataFrame
        benchmark: Optional benchmark equity curve
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    # Equity curve
    ax1 = axes[0, 0]
    ax1.plot(results['capital'], label='Strategy', linewidth=1.5)
    if benchmark is not None:
        ax1.plot(benchmark, label='Benchmark', linewidth=1.5, alpha=0.7)
    ax1.set_title('Equity Curve')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Capital')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[0, 1]
    rolling_max = results['capital'].expanding().max()
    drawdown = (results['capital'] - rolling_max) / rolling_max * 100
    ax2.fill_between(results.index, drawdown, 0, alpha=0.5, color='red')
    ax2.set_title('Drawdown')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # Position
    ax3 = axes[1, 0]
    ax3.fill_between(results.index, results['position'], 0, alpha=0.7)
    ax3.set_title('Position Over Time')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Position')
    ax3.grid(True, alpha=0.3)

    # Return distribution
    ax4 = axes[1, 1]
    returns = results['portfolio_return'] * 100
    ax4.hist(returns, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.3f}%')
    ax4.set_title('Return Distribution')
    ax4.set_xlabel('Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150)
    plt.show()


def walk_forward_backtest(
    model_class,
    model_config: Dict,
    data: Dict,
    train_window: int = 1000,
    test_window: int = 100,
    step_size: int = 100,
    initial_capital: float = 100000.0,
    **backtest_kwargs
) -> pd.DataFrame:
    """
    Perform walk-forward backtesting.

    This method trains on a rolling window and tests on the next period,
    simulating real-world trading conditions more accurately.

    Args:
        model_class: Linformer model class
        model_config: Model configuration dictionary
        data: Data dictionary with 'X' and 'y'
        train_window: Number of samples for training
        test_window: Number of samples for testing
        step_size: Step size for rolling window
        initial_capital: Starting capital
        **backtest_kwargs: Additional arguments for backtest function

    Returns:
        Combined backtest results DataFrame
    """
    X = data['X']
    y = data['y']
    n_samples = len(X)

    all_results = []
    capital = initial_capital

    for start in range(0, n_samples - train_window - test_window, step_size):
        train_end = start + train_window
        test_end = train_end + test_window

        logger.info(f"Walk-forward: training on {start}:{train_end}, testing on {train_end}:{test_end}")

        # Create train/test data
        train_data = {'X': X[start:train_end], 'y': y[start:train_end]}
        test_data = {'X': X[train_end:test_end], 'y': y[train_end:test_end]}

        # Train model
        model = model_class(**model_config)
        # Note: You would need to implement training here
        # For simplicity, we assume model is pre-trained

        # Backtest
        results = backtest_linformer_strategy(
            model, test_data,
            initial_capital=capital,
            **backtest_kwargs
        )

        # Update capital for next window
        capital = results['capital'].iloc[-1]

        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)
