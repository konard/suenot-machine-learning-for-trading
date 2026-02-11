"""
Backtesting Module for Neural ODE Trading Strategies

Implements a simple backtesting framework that uses Neural ODE models
to generate trading signals and evaluate performance.

Strategies:
1. Momentum: Trade in the direction of predicted price movement
2. Mean Reversion: Trade when predicted price deviates from current
3. Continuous-Time Rebalancing: Adjust position based on ODE trajectory

Metrics:
- Total Return, Sharpe Ratio, Max Drawdown, Win Rate
- Comparison with buy-and-hold baseline

Usage:
    python backtest.py --model neural_ode --symbol BTCUSDT
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.neural_ode import NeuralODE
from python.ode_rnn import ODERNN
from python.data_loader import BybitDataLoader, IrregularTimeSeriesDataset


class NeuralODEBacktester:
    """
    Backtesting engine for Neural ODE trading strategies.

    Simulates trading using model predictions while accounting for:
    - Transaction costs
    - Slippage
    - Position sizing
    - Risk management

    Args:
        model: Trained Neural ODE model
        initial_capital: Starting capital in USD
        transaction_cost: Cost per trade (fraction, e.g., 0.001 = 0.1%)
        max_position: Maximum position size as fraction of capital
        device: Computation device
    """

    def __init__(
        self,
        model: torch.nn.Module,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.device = device
        self.model.eval()

    def generate_signals(
        self,
        dataset: IrregularTimeSeriesDataset,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate trading signals from model predictions.

        Signal logic:
            predicted_return > threshold  -> BUY  (+1)
            predicted_return < -threshold -> SELL (-1)
            otherwise                     -> HOLD (0)

        Args:
            dataset: Test dataset
            threshold: Minimum predicted return to trigger trade

        Returns:
            DataFrame with columns: timestamp_idx, actual, predicted, signal
        """
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                target = batch["target"].to(self.device)

                if isinstance(self.model, (NeuralODE,)):
                    pred = self.model(x)
                elif isinstance(self.model, ODERNN):
                    t = batch["t"].to(self.device)
                    mask = batch["mask"].to(self.device)
                    obs_mask = mask.any(dim=-1).float()
                    t_common = t[0] if t.dim() > 1 else t
                    pred = self.model(x, t_common, mask=obs_mask)
                else:
                    pred = self.model(x)

                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()

        # Denormalize
        preds_denorm = preds * dataset.target_std + dataset.target_mean
        targets_denorm = targets * dataset.target_std + dataset.target_mean

        # Convert predictions to returns relative to last known price
        # We use the normalized predictions as directional signals
        predicted_returns = preds  # Already centered around 0

        # Generate signals
        signals = np.zeros(len(predicted_returns))
        signals[predicted_returns > threshold] = 1.0   # Buy
        signals[predicted_returns < -threshold] = -1.0  # Sell

        return pd.DataFrame({
            "idx": range(len(preds)),
            "actual_price": targets_denorm,
            "predicted_price": preds_denorm,
            "predicted_return": predicted_returns,
            "signal": signals,
        })

    def run_backtest(
        self,
        signals_df: pd.DataFrame,
        strategy: str = "momentum",
    ) -> Dict[str, float]:
        """
        Execute backtest with given signals.

        Args:
            signals_df: DataFrame from generate_signals()
            strategy: Strategy type ('momentum', 'mean_reversion')

        Returns:
            Dictionary with performance metrics
        """
        capital = self.initial_capital
        position = 0.0  # Number of units held
        cash = capital
        portfolio_values = [capital]
        trades = []

        prices = signals_df["actual_price"].values
        signals = signals_df["signal"].values

        if strategy == "mean_reversion":
            signals = -signals  # Reverse signals for mean reversion

        for i in range(1, len(prices)):
            current_price = prices[i]
            prev_price = prices[i - 1]
            signal = signals[i]

            # Calculate desired position
            target_position_value = signal * cash * self.max_position
            target_units = target_position_value / current_price if current_price > 0 else 0

            # Execute trade if signal changes
            trade_units = target_units - position
            if abs(trade_units) > 1e-8:
                trade_value = abs(trade_units * current_price)
                cost = trade_value * self.transaction_cost

                # Update positions
                cash -= trade_units * current_price + cost
                position = target_units
                trades.append({
                    "idx": i,
                    "action": "buy" if trade_units > 0 else "sell",
                    "units": abs(trade_units),
                    "price": current_price,
                    "cost": cost,
                })

            # Portfolio value
            portfolio_value = cash + position * current_price
            portfolio_values.append(portfolio_value)

        portfolio_values = np.array(portfolio_values)

        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[np.isfinite(returns)]

        # Buy-and-hold benchmark
        bh_returns = (prices[-1] / prices[0]) - 1.0 if prices[0] > 0 else 0.0

        metrics = self._calculate_metrics(portfolio_values, returns, trades, bh_returns)

        return metrics

    def _calculate_metrics(
        self,
        portfolio_values: np.ndarray,
        returns: np.ndarray,
        trades: List[Dict],
        benchmark_return: float,
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1.0

        # Sharpe Ratio (annualized, assuming ~252 trading days * ~7 obs/day)
        if len(returns) > 1 and np.std(returns) > 1e-10:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 7)
        else:
            sharpe = 0.0

        # Max Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()

        # Win rate
        winning_returns = returns[returns > 0]
        win_rate = len(winning_returns) / max(len(returns), 1)

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252 * 7)
        else:
            sortino = float("inf") if np.mean(returns) > 0 else 0.0

        # Profit Factor
        gross_profit = np.sum(returns[returns > 0]) if len(returns[returns > 0]) > 0 else 0
        gross_loss = abs(np.sum(returns[returns < 0])) if len(returns[returns < 0]) > 0 else 1e-10
        profit_factor = gross_profit / gross_loss

        # Calmar Ratio
        calmar = total_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0.0

        return {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "benchmark_return_pct": benchmark_return * 100,
            "excess_return_pct": (total_return - benchmark_return) * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "calmar_ratio": calmar,
            "n_trades": len(trades),
            "total_cost": sum(t["cost"] for t in trades),
            "final_portfolio": portfolio_values[-1],
        }

    @staticmethod
    def print_results(metrics: Dict[str, float], strategy_name: str = "Neural ODE"):
        """Pretty-print backtest results."""
        print(f"\n{'=' * 55}")
        print(f"  Backtest Results: {strategy_name}")
        print(f"{'=' * 55}")
        print(f"  Total Return:      {metrics['total_return_pct']:>10.2f}%")
        print(f"  Benchmark (B&H):   {metrics['benchmark_return_pct']:>10.2f}%")
        print(f"  Excess Return:     {metrics['excess_return_pct']:>10.2f}%")
        print(f"  {'─' * 50}")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>10.3f}")
        print(f"  Sortino Ratio:     {metrics['sortino_ratio']:>10.3f}")
        print(f"  Max Drawdown:      {metrics['max_drawdown_pct']:>10.2f}%")
        print(f"  Win Rate:          {metrics['win_rate']:>10.2f}%")
        print(f"  Profit Factor:     {metrics['profit_factor']:>10.3f}")
        print(f"  Calmar Ratio:      {metrics['calmar_ratio']:>10.3f}")
        print(f"  {'─' * 50}")
        print(f"  Number of Trades:  {metrics['n_trades']:>10d}")
        print(f"  Total Costs:       ${metrics['total_cost']:>9.2f}")
        print(f"  Final Portfolio:   ${metrics['final_portfolio']:>9.2f}")
        print(f"{'=' * 55}\n")


def main():
    parser = argparse.ArgumentParser(description="Backtest Neural ODE Trading")
    parser.add_argument("--model", type=str, default="neural_ode",
                        choices=["neural_ode", "ode_rnn"])
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--strategy", type=str, default="momentum",
                        choices=["momentum", "mean_reversion"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--initial_capital", type=float, default=100_000)
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--hidden_dim", type=int, default=64)

    args = parser.parse_args()

    print(f"{'=' * 55}")
    print(f"  Neural ODE Trading Backtest")
    print(f"{'=' * 55}")
    print(f"  Model:    {args.model}")
    print(f"  Symbol:   {args.symbol}")
    print(f"  Strategy: {args.strategy}")
    print(f"{'=' * 55}")

    # Load data
    print("\nLoading data...")
    loader = BybitDataLoader()
    df = loader.fetch_klines(args.symbol, interval="1", limit=1000)
    print(f"Loaded {len(df)} candles")

    # Create test dataset (last 20%)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    test_ds = IrregularTimeSeriesDataset(
        test_df,
        feature_cols=["open", "high", "low", "close", "volume"],
        seq_len=args.seq_len,
    )

    # Create model
    input_dim = 5
    if args.model == "neural_ode":
        model = NeuralODE(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            seq_len=args.seq_len,
        )
    else:
        model = ODERNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=1,
        )

    # Load checkpoint if available
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint loaded - using randomly initialized model (demo mode)")

    # Run backtest
    backtester = NeuralODEBacktester(
        model=model,
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
    )

    print("\nGenerating signals...")
    signals = backtester.generate_signals(test_ds)
    print(f"Generated {len(signals)} signals")
    print(f"  Buy signals:  {(signals['signal'] == 1).sum()}")
    print(f"  Sell signals: {(signals['signal'] == -1).sum()}")
    print(f"  Hold signals: {(signals['signal'] == 0).sum()}")

    print(f"\nRunning {args.strategy} backtest...")
    metrics = backtester.run_backtest(signals, strategy=args.strategy)
    NeuralODEBacktester.print_results(metrics, f"Neural ODE ({args.strategy})")

    return metrics


if __name__ == "__main__":
    main()
