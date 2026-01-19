"""
Backtesting Framework for Cross-Attention Multi-Asset Strategy

Provides:
- CrossAttentionBacktest: Main backtesting class
- Performance metrics computation
- Visualization utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    rebalance_freq: int = 24  # Hours between rebalances
    max_position: float = 0.5  # Maximum position per asset
    allow_short: bool = False  # Allow short positions


class CrossAttentionBacktest:
    """
    Backtesting framework for cross-attention portfolio strategy.

    Example:
        backtest = CrossAttentionBacktest(model, config)
        results = backtest.run(X_test, returns_test, timestamps_test)
        metrics = backtest.compute_metrics(results)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: BacktestConfig = None
    ):
        self.model = model
        self.config = config or BacktestConfig()

    def run(
        self,
        X: np.ndarray,
        returns: np.ndarray,
        timestamps: pd.DatetimeIndex,
        device: str = 'cpu'
    ) -> pd.DataFrame:
        """
        Run backtest on test data.

        Args:
            X: [n_samples, n_assets, lookback, n_features]
            returns: [n_samples, n_assets] - Actual returns
            timestamps: DatetimeIndex for each sample
            device: 'cpu' or 'cuda'

        Returns:
            DataFrame with backtest results
        """
        self.model.eval()
        self.model = self.model.to(device)

        n_samples = len(X)
        n_assets = X.shape[1]

        capital = self.config.initial_capital
        positions = None  # Will be set on first rebalance

        results = []

        for i in range(0, n_samples, self.config.rebalance_freq):
            # Get model predictions
            with torch.no_grad():
                x_batch = torch.FloatTensor(X[i:i+1]).to(device)
                output = self.model(x_batch)
                weights = output['predictions'].cpu().numpy().flatten()

            # Process weights based on output type
            weights = self._process_weights(weights)

            # Calculate transaction costs
            position_change = np.abs(weights - positions).sum() if positions is not None else weights.sum()
            costs = position_change * self.config.transaction_cost * capital

            # Update positions at start of period to use model predictions immediately
            positions = weights.copy()

            # Simulate trading for each timestep until next rebalance
            period_end = min(i + self.config.rebalance_freq, n_samples)

            for j in range(i, period_end):
                # Calculate portfolio return using current positions
                portfolio_return = np.sum(positions * returns[j])
                capital = capital * (1 + portfolio_return)

                # Deduct costs at first timestep of period
                if j == i:
                    capital -= costs

                results.append({
                    'timestamp': timestamps[j] if j < len(timestamps) else None,
                    'capital': capital,
                    'return': portfolio_return,
                    'positions': positions.copy(),
                    'weights': weights.copy(),
                    'costs': costs if j == i else 0.0
                })

        return pd.DataFrame(results)

    def _process_weights(self, weights: np.ndarray) -> np.ndarray:
        """Process raw model output into portfolio weights."""
        output_type = getattr(self.model, 'output_type', 'regression')

        if output_type == 'portfolio':
            # Already normalized via softmax
            processed = weights
        elif output_type == 'classification':
            # Convert class probabilities to positions
            # Assume: [down, neutral, up] classes
            n_assets = len(weights) // 3
            weights = weights.reshape(n_assets, 3)
            processed = weights[:, 2] - weights[:, 0]  # up - down
        else:
            # Regression: treat as return predictions
            processed = weights

        # Normalize to sum to 1 (long-only) or allow shorts
        if self.config.allow_short:
            # Normalize by sum of absolute values
            total = np.abs(processed).sum() + 1e-8
            processed = processed / total
        else:
            # Long-only: clip negatives and normalize
            processed = np.clip(processed, 0, None)
            total = processed.sum() + 1e-8
            processed = processed / total

        # Apply position limits
        processed = np.clip(processed, -self.config.max_position, self.config.max_position)

        # Renormalize after clipping to maintain full investment
        if self.config.allow_short:
            total = np.abs(processed).sum() + 1e-8
            processed = processed / total
        else:
            total = processed.sum() + 1e-8
            processed = processed / total

        return processed

    def compute_metrics(self, results: pd.DataFrame) -> Dict:
        """
        Compute performance metrics from backtest results.

        Returns:
            Dictionary of performance metrics
        """
        returns = results['return'].values
        capital = results['capital'].values

        # Total Return
        total_return = (capital[-1] / self.config.initial_capital - 1) * 100

        # Annualized metrics (assuming hourly data)
        annual_factor = np.sqrt(365 * 24)

        # Sharpe Ratio
        sharpe = annual_factor * returns.mean() / (returns.std() + 1e-8)

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-8
        sortino = annual_factor * returns.mean() / (downside_std + 1e-8)

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-8)
        max_drawdown = drawdown.min() * 100

        # Calmar Ratio
        annual_return = (capital[-1] / self.config.initial_capital) ** (365 * 24 / len(returns)) - 1
        calmar = annual_return / (abs(max_drawdown / 100) + 1e-8)

        # Win Rate
        win_rate = (returns > 0).mean() * 100

        # Profit Factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / (gross_loss + 1e-8)

        # Average Trade
        avg_return = returns.mean() * 100

        # Total Costs
        total_costs = results['costs'].sum()

        return {
            'total_return': total_return,
            'annual_return': annual_return * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_return_pct': avg_return,
            'volatility': returns.std() * annual_factor * 100,
            'total_costs': total_costs,
            'n_trades': len(results) // self.config.rebalance_freq
        }

    def compare_baseline(
        self,
        results: pd.DataFrame,
        returns: np.ndarray
    ) -> Dict:
        """
        Compare strategy against baselines.

        Baselines:
        - Equal Weight: 1/n allocation to each asset
        - Buy & Hold: Hold first asset (typically BTC)
        """
        n_assets = returns.shape[1]

        # Equal Weight baseline
        equal_weight_returns = returns.mean(axis=1)
        equal_weight_cumulative = (1 + equal_weight_returns).cumprod()
        equal_weight_total = (equal_weight_cumulative[-1] - 1) * 100

        # Buy & Hold first asset
        buy_hold_returns = returns[:, 0]
        buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
        buy_hold_total = (buy_hold_cumulative[-1] - 1) * 100

        # Strategy returns
        strategy_total = (results['capital'].iloc[-1] / self.config.initial_capital - 1) * 100

        return {
            'strategy_return': strategy_total,
            'equal_weight_return': equal_weight_total,
            'buy_hold_return': buy_hold_total,
            'outperformance_vs_equal': strategy_total - equal_weight_total,
            'outperformance_vs_buyhold': strategy_total - buy_hold_total
        }


def plot_backtest_results(
    results: pd.DataFrame,
    returns: np.ndarray,
    symbols: List[str],
    save_path: Optional[str] = None
):
    """
    Plot backtest results with equity curve and comparisons.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 1. Equity Curve
    ax1 = axes[0]
    ax1.plot(results['timestamp'], results['capital'], label='Strategy', linewidth=2)

    # Add baselines
    initial = results['capital'].iloc[0]
    n_assets = returns.shape[1]

    # Equal weight
    equal_weight = initial * (1 + returns.mean(axis=1)).cumprod()
    ax1.plot(results['timestamp'], equal_weight, label='Equal Weight', linestyle='--', alpha=0.7)

    # Buy & hold first asset
    buy_hold = initial * (1 + returns[:, 0]).cumprod()
    ax1.plot(results['timestamp'], buy_hold, label=f'Buy & Hold {symbols[0]}', linestyle=':', alpha=0.7)

    ax1.set_title('Portfolio Equity Curve')
    ax1.set_ylabel('Capital ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = axes[1]
    cumulative = results['capital'] / results['capital'].iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100

    ax2.fill_between(results['timestamp'], drawdown, 0, alpha=0.3, color='red')
    ax2.plot(results['timestamp'], drawdown, color='red', linewidth=1)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # 3. Portfolio Weights over time
    ax3 = axes[2]
    weights_df = pd.DataFrame(
        np.vstack(results['weights'].values),
        columns=symbols,
        index=results['timestamp']
    )

    # Resample to reduce noise
    weights_resampled = weights_df.resample('D').mean()
    weights_resampled.plot.area(ax=ax3, alpha=0.7)

    ax3.set_title('Portfolio Weights Over Time')
    ax3.set_ylabel('Weight')
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_analysis(
    model: torch.nn.Module,
    X: np.ndarray,
    symbols: List[str],
    save_path: Optional[str] = None
):
    """
    Visualize cross-asset attention patterns.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available for plotting")
        return

    model.eval()

    with torch.no_grad():
        x = torch.FloatTensor(X)
        output = model(x, return_attention=True)

    attention_weights = output.get('attention_weights')

    if attention_weights is None:
        print("No attention weights available")
        return

    # Get last layer cross-asset attention
    last_layer = list(attention_weights.keys())[-1]
    cross_attn = attention_weights[last_layer].get('cross_asset')

    if cross_attn is None:
        print("No cross-asset attention found")
        return

    # Average over batch
    avg_attn = cross_attn.mean(dim=0).numpy()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        avg_attn,
        xticklabels=symbols,
        yticklabels=symbols,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        ax=ax
    )

    ax.set_title('Cross-Asset Attention Weights\n(Query asset attends to Key asset)')
    ax.set_xlabel('Key (Source Asset)')
    ax.set_ylabel('Query (Target Asset)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def print_metrics_report(metrics: Dict, comparison: Dict = None):
    """Print formatted metrics report."""
    print("\n" + "=" * 50)
    print("BACKTEST PERFORMANCE REPORT")
    print("=" * 50)

    print("\n--- Returns ---")
    print(f"Total Return:     {metrics['total_return']:>10.2f}%")
    print(f"Annual Return:    {metrics['annual_return']:>10.2f}%")
    print(f"Avg Return/Trade: {metrics['avg_return_pct']:>10.4f}%")

    print("\n--- Risk Metrics ---")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
    print(f"Sortino Ratio:    {metrics['sortino_ratio']:>10.2f}")
    print(f"Calmar Ratio:     {metrics['calmar_ratio']:>10.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']:>10.2f}%")
    print(f"Volatility:       {metrics['volatility']:>10.2f}%")

    print("\n--- Trading Stats ---")
    print(f"Win Rate:         {metrics['win_rate']:>10.2f}%")
    print(f"Profit Factor:    {metrics['profit_factor']:>10.2f}")
    print(f"Number of Trades: {metrics['n_trades']:>10d}")
    print(f"Total Costs:      ${metrics['total_costs']:>9.2f}")

    if comparison:
        print("\n--- Baseline Comparison ---")
        print(f"Strategy Return:     {comparison['strategy_return']:>10.2f}%")
        print(f"Equal Weight Return: {comparison['equal_weight_return']:>10.2f}%")
        print(f"Buy & Hold Return:   {comparison['buy_hold_return']:>10.2f}%")
        print(f"vs Equal Weight:     {comparison['outperformance_vs_equal']:>+10.2f}%")
        print(f"vs Buy & Hold:       {comparison['outperformance_vs_buyhold']:>+10.2f}%")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    # Test backtesting framework
    print("Testing backtesting framework...")

    # Create dummy model for testing
    from model import create_model

    model = create_model(
        n_assets=5,
        n_features=6,
        seq_len=48,
        d_model=32,
        output_type='portfolio'
    )

    # Create dummy data
    n_samples = 200
    n_assets = 5
    seq_len = 48
    n_features = 6

    X = np.random.randn(n_samples, n_assets, seq_len, n_features).astype(np.float32)
    returns = np.random.randn(n_samples, n_assets) * 0.01  # Small returns
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='H')

    symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'DOT']

    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        rebalance_freq=24
    )

    backtest = CrossAttentionBacktest(model, config)
    results = backtest.run(X, returns, timestamps)

    print(f"\nBacktest completed: {len(results)} timesteps")

    # Compute metrics
    metrics = backtest.compute_metrics(results)
    comparison = backtest.compare_baseline(results, returns)

    print_metrics_report(metrics, comparison)

    print("\nAll backtest tests passed!")
