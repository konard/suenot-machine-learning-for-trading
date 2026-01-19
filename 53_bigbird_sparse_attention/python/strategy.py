"""
Trading Strategy and Backtesting for BigBird Model

Provides:
- generate_signals: Generate trading signals from predictions
- backtest_strategy: Run backtest with position management
- calculate_metrics: Calculate performance metrics
- visualize_results: Plot backtest results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import torch
from datetime import datetime
import warnings

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000.0
    position_size: float = 0.1  # Fraction of capital per trade
    max_position: float = 1.0   # Maximum position (1.0 = 100% long, -1.0 = 100% short)
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005   # 0.05% slippage
    signal_threshold: float = 0.001  # Minimum prediction for signal


@dataclass
class BacktestResults:
    """Results from backtesting"""
    results_df: pd.DataFrame
    metrics: Dict[str, float]
    trades: List[Dict]


def generate_signals(
    predictions: np.ndarray,
    threshold_long: float = 0.001,
    threshold_short: float = -0.001,
    use_quantiles: bool = False,
    quantile_long: float = 0.7,
    quantile_short: float = 0.3
) -> np.ndarray:
    """
    Generate trading signals from model predictions.

    Args:
        predictions: Model predictions [n_samples] or [n_samples, n_quantiles]
        threshold_long: Threshold for long signal
        threshold_short: Threshold for short signal
        use_quantiles: Whether predictions are quantiles
        quantile_long: Quantile threshold for long (if use_quantiles)
        quantile_short: Quantile threshold for short (if use_quantiles)

    Returns:
        signals: Array of signals (-1 = short, 0 = flat, 1 = long)

    Example:
        signals = generate_signals(predictions, threshold_long=0.001)
    """
    n_samples = len(predictions)
    signals = np.zeros(n_samples, dtype=np.int32)

    if use_quantiles and predictions.ndim == 2:
        # Use median (middle quantile) for signal
        median_pred = predictions[:, predictions.shape[1] // 2]
        # Confidence from quantile range
        confidence = predictions[:, -1] - predictions[:, 0]
        high_confidence = confidence < np.percentile(confidence, 70)

        signals[median_pred > threshold_long] = 1
        signals[median_pred < threshold_short] = -1
        signals[~high_confidence] = 0
    else:
        # Simple threshold-based signals
        if predictions.ndim == 2:
            predictions = predictions[:, 0]

        signals[predictions > threshold_long] = 1
        signals[predictions < threshold_short] = -1

    return signals


def backtest_strategy(
    model,
    test_data: pd.DataFrame,
    seq_len: int = 256,
    features: Optional[List[str]] = None,
    config: Optional[BacktestConfig] = None,
    device: Optional[torch.device] = None
) -> BacktestResults:
    """
    Backtest BigBird prediction strategy.

    Args:
        model: Trained BigBird model
        test_data: DataFrame with features and 'log_return' column
        seq_len: Input sequence length
        features: Feature columns (None = auto-detect)
        config: Backtest configuration
        device: Device for model inference

    Returns:
        BacktestResults with results DataFrame, metrics, and trade list

    Example:
        results = backtest_strategy(model, test_data, seq_len=256)
        print(f"Sharpe Ratio: {results.metrics['sharpe_ratio']:.2f}")
    """
    if config is None:
        config = BacktestConfig()

    if device is None:
        device = next(model.parameters()).device

    if features is None:
        features = [c for c in test_data.columns
                   if c not in ['open', 'high', 'low', 'close', 'volume']]

    model.eval()

    # Initialize tracking variables
    capital = config.initial_capital
    position = 0.0  # Current position (-1 to 1)

    results = {
        'timestamp': [],
        'capital': [],
        'position': [],
        'prediction': [],
        'signal': [],
        'actual_return': [],
        'trade_return': [],
        'equity': []
    }

    trades = []

    # Get data as numpy array
    data = test_data[features].values
    returns = test_data['log_return'].values if 'log_return' in test_data.columns else np.zeros(len(test_data))

    # Run backtest
    for i in range(seq_len, len(data) - 1):
        # Prepare input
        x = torch.FloatTensor(data[i-seq_len:i]).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(x)
            pred = output['predictions'].cpu().numpy().flatten()[0]

        # Generate signal
        if pred > config.signal_threshold:
            signal = 1
        elif pred < -config.signal_threshold:
            signal = -1
        else:
            signal = 0

        # Get actual return for this period
        actual_return = returns[i]

        # Calculate position change
        target_position = signal * config.position_size
        target_position = np.clip(target_position, -config.max_position, config.max_position)
        position_change = target_position - position

        # Apply transaction costs
        if abs(position_change) > 0.001:
            cost_rate = (config.transaction_cost + config.slippage)
            cost = abs(position_change) * cost_rate * capital
            capital -= cost

            # Record trade
            trades.append({
                'timestamp': test_data.index[i] if hasattr(test_data.index, '__getitem__') else i,
                'type': 'long' if position_change > 0 else 'short',
                'size': abs(position_change),
                'cost': cost
            })

        # Update position
        position = target_position

        # Calculate trade return
        trade_return = position * actual_return

        # Update capital
        capital *= (1 + trade_return)

        # Record results
        results['timestamp'].append(
            test_data.index[i] if hasattr(test_data.index, '__getitem__') else i
        )
        results['capital'].append(capital)
        results['position'].append(position)
        results['prediction'].append(pred)
        results['signal'].append(signal)
        results['actual_return'].append(actual_return)
        results['trade_return'].append(trade_return)
        results['equity'].append(capital)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    if hasattr(test_data.index, 'dtype'):
        results_df.set_index('timestamp', inplace=True)

    # Calculate metrics
    metrics = calculate_metrics(results_df, config.initial_capital)

    return BacktestResults(
        results_df=results_df,
        metrics=metrics,
        trades=trades
    )


def calculate_metrics(
    results_df: pd.DataFrame,
    initial_capital: float = 100000.0,
    risk_free_rate: float = 0.02  # Annual risk-free rate
) -> Dict[str, float]:
    """
    Calculate performance metrics from backtest results.

    Args:
        results_df: DataFrame with 'capital', 'trade_return' columns
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        Dictionary of performance metrics
    """
    # Total return
    final_capital = results_df['capital'].iloc[-1]
    total_return = (final_capital - initial_capital) / initial_capital

    # Daily returns
    daily_returns = results_df['trade_return']

    # Annualized metrics (assuming hourly data, 8760 hours/year)
    periods_per_year = 8760  # Adjust based on your data frequency
    ann_factor = np.sqrt(periods_per_year)

    # Sharpe Ratio
    excess_returns = daily_returns - risk_free_rate / periods_per_year
    sharpe_ratio = ann_factor * excess_returns.mean() / (daily_returns.std() + 1e-10)

    # Sortino Ratio (using downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-10
    sortino_ratio = ann_factor * excess_returns.mean() / (downside_std + 1e-10)

    # Maximum Drawdown
    cummax = results_df['capital'].cummax()
    drawdown = (results_df['capital'] - cummax) / cummax
    max_drawdown = drawdown.min()

    # Calmar Ratio
    ann_return = total_return * (periods_per_year / len(results_df))
    calmar_ratio = ann_return / (abs(max_drawdown) + 1e-10)

    # Win Rate
    winning_trades = (daily_returns > 0).sum()
    total_trades = (daily_returns != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Profit Factor
    gross_profit = daily_returns[daily_returns > 0].sum()
    gross_loss = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gross_profit / (gross_loss + 1e-10)

    # Average Trade
    avg_trade = daily_returns[daily_returns != 0].mean() if total_trades > 0 else 0

    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_trade': avg_trade,
        'num_trades': total_trades,
        'final_capital': final_capital
    }


def compare_with_benchmark(
    results: BacktestResults,
    benchmark_returns: np.ndarray,
    initial_capital: float = 100000.0
) -> Dict[str, float]:
    """
    Compare strategy performance with benchmark (e.g., buy-and-hold).

    Args:
        results: Backtest results
        benchmark_returns: Benchmark returns series
        initial_capital: Starting capital

    Returns:
        Dictionary with comparison metrics
    """
    # Calculate benchmark equity
    benchmark_equity = initial_capital * np.cumprod(1 + benchmark_returns)
    benchmark_final = benchmark_equity[-1]
    benchmark_return = (benchmark_final - initial_capital) / initial_capital

    # Strategy metrics
    strategy_return = results.metrics['total_return']
    strategy_sharpe = results.metrics['sharpe_ratio']

    # Benchmark metrics
    periods_per_year = 8760
    ann_factor = np.sqrt(periods_per_year)
    benchmark_sharpe = ann_factor * benchmark_returns.mean() / (benchmark_returns.std() + 1e-10)

    # Alpha and Beta (simple calculation)
    strategy_returns = results.results_df['trade_return'].values
    if len(strategy_returns) == len(benchmark_returns):
        cov_matrix = np.cov(strategy_returns, benchmark_returns)
        beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-10)
        alpha = strategy_returns.mean() - beta * benchmark_returns.mean()
        alpha_ann = alpha * periods_per_year
    else:
        beta = 0
        alpha_ann = 0

    return {
        'strategy_return': strategy_return,
        'benchmark_return': benchmark_return,
        'excess_return': strategy_return - benchmark_return,
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'beta': beta,
        'alpha_annualized': alpha_ann,
        'information_ratio': (strategy_return - benchmark_return) / (
            np.std(strategy_returns - benchmark_returns) * ann_factor + 1e-10
        )
    }


def visualize_results(
    results: BacktestResults,
    benchmark_equity: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualize backtest results.

    Args:
        results: Backtest results
        benchmark_equity: Optional benchmark equity curve
        save_path: Path to save figure (None = display)
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not installed. Cannot visualize results.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 1. Equity Curve
    ax1 = axes[0]
    ax1.plot(results.results_df['capital'], label='Strategy', linewidth=1.5)
    if benchmark_equity is not None:
        ax1.plot(benchmark_equity, label='Benchmark', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Capital')
    ax1.set_title('Equity Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = axes[1]
    cummax = results.results_df['capital'].cummax()
    drawdown = (results.results_df['capital'] - cummax) / cummax
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red')
    ax2.set_ylabel('Drawdown')
    ax2.set_title(f"Drawdown (Max: {results.metrics['max_drawdown']*100:.2f}%)")
    ax2.grid(True, alpha=0.3)

    # 3. Position and Signals
    ax3 = axes[2]
    ax3.plot(results.results_df['position'], label='Position', linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Position')
    ax3.set_title('Position Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def print_metrics(metrics: Dict[str, float], title: str = "Performance Metrics"):
    """Print formatted performance metrics."""
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)
    print(f"  Total Return:       {metrics['total_return']*100:>10.2f}%")
    print(f"  Annualized Return:  {metrics['annualized_return']*100:>10.2f}%")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:      {metrics['sortino_ratio']:>10.2f}")
    print(f"  Calmar Ratio:       {metrics['calmar_ratio']:>10.2f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown']*100:>10.2f}%")
    print(f"  Win Rate:           {metrics['win_rate']*100:>10.2f}%")
    print(f"  Profit Factor:      {metrics['profit_factor']:>10.2f}")
    print(f"  Avg Trade:          {metrics['avg_trade']*100:>10.4f}%")
    print(f"  Number of Trades:   {metrics['num_trades']:>10.0f}")
    print(f"  Final Capital:      ${metrics['final_capital']:>10,.2f}")
    print("=" * 50)


if __name__ == "__main__":
    # Test strategy utilities with synthetic data
    print("Testing strategy utilities...")

    # Create synthetic results for testing
    np.random.seed(42)
    n_periods = 1000

    # Simulated equity curve with positive drift
    returns = np.random.randn(n_periods) * 0.001 + 0.0001
    capital = 100000 * np.cumprod(1 + returns)

    results_df = pd.DataFrame({
        'capital': capital,
        'position': np.random.choice([-1, 0, 1], n_periods),
        'prediction': np.random.randn(n_periods) * 0.01,
        'signal': np.random.choice([-1, 0, 1], n_periods),
        'actual_return': returns,
        'trade_return': returns * np.random.choice([-1, 0, 1], n_periods),
        'equity': capital
    })

    # Calculate metrics
    metrics = calculate_metrics(results_df)
    print_metrics(metrics, "Test Strategy Metrics")

    # Test signal generation
    predictions = np.random.randn(100) * 0.01
    signals = generate_signals(predictions, threshold_long=0.005, threshold_short=-0.005)
    print(f"\nSignal distribution: long={np.sum(signals==1)}, flat={np.sum(signals==0)}, short={np.sum(signals==-1)}")

    print("\nAll tests completed!")
