"""
Example: Backtest the Bayesian Neural Network trading strategy.

Usage:
    python -m examples.backtest --model bnn_model.pt --symbol BTC/USDT
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from bnn import BayesianNN, BayesianNNConfig
from data import BybitFetcher
from features import FeatureEngineer
from strategy import TradingStrategy


def plot_results(result, save_path: str = 'backtest_results.png'):
    """Plot backtest results."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Equity curve
    ax1 = axes[0, 0]
    result.equity_curve.plot(ax=ax1)
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Portfolio Value')
    ax1.grid(True)

    # Returns distribution
    ax2 = axes[0, 1]
    result.returns.hist(bins=50, ax=ax2)
    ax2.axvline(0, color='r', linestyle='--')
    ax2.set_title('Returns Distribution')
    ax2.set_xlabel('Daily Return')

    # Drawdown
    ax3 = axes[1, 0]
    cumulative = (1 + result.returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    drawdown.plot(ax=ax3, color='red')
    ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax3.set_title('Drawdown')
    ax3.set_ylabel('Drawdown %')
    ax3.grid(True)

    # Trade PnL by confidence
    ax4 = axes[1, 1]
    if result.trades:
        confidences = [t.entry_confidence for t in result.trades]
        pnls = [t.pnl_pct for t in result.trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax4.scatter(confidences, pnls, c=colors, alpha=0.6)
        ax4.axhline(0, color='black', linestyle='--')
        ax4.set_xlabel('Entry Confidence')
        ax4.set_ylabel('Trade PnL %')
        ax4.set_title('PnL vs Confidence')
        ax4.grid(True)

    # Uncertainty distribution
    ax5 = axes[2, 0]
    uncertainties = [s.uncertainty for s in result.signals]
    ax5.hist(uncertainties, bins=50)
    ax5.set_title('Uncertainty Distribution')
    ax5.set_xlabel('Epistemic Uncertainty')

    # Win rate by confidence quartile
    ax6 = axes[2, 1]
    if result.trades:
        trades_df = pd.DataFrame([
            {'confidence': t.entry_confidence, 'win': t.pnl > 0}
            for t in result.trades
        ])
        trades_df['conf_quartile'] = pd.qcut(
            trades_df['confidence'], 4,
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
        )
        win_rates = trades_df.groupby('conf_quartile')['win'].mean()
        win_rates.plot(kind='bar', ax=ax6)
        ax6.set_title('Win Rate by Confidence Quartile')
        ax6.set_ylabel('Win Rate')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
        ax6.axhline(0.5, color='r', linestyle='--', label='50%')
        ax6.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Backtest BNN Trading Strategy')
    parser.add_argument('--model', type=str, default='bnn_model.pt', help='Model path')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe')
    parser.add_argument('--days', type=int, default=30, help='Days for backtest')
    parser.add_argument('--prob-threshold', type=float, default=0.55, help='Probability threshold')
    parser.add_argument('--conf-threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--output', type=str, default='backtest_results.png', help='Output plot')

    args = parser.parse_args()

    logger.info(f"Backtesting on {args.symbol}")

    # Load model
    logger.info(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=args.device)
    config = checkpoint['config']
    model = BayesianNN(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)

    # Fetch data
    logger.info("Fetching data...")
    fetcher = BybitFetcher()
    ohlcv_data = fetcher.fetch_ohlcv(
        args.symbol,
        timeframe=args.timeframe,
        days=args.days
    )

    # Create features
    logger.info("Creating features...")
    engineer = FeatureEngineer(
        lookback=20,
        prediction_horizon=5,
        return_threshold=0.002
    )
    features = engineer.create_features(ohlcv_data.df)

    # Initialize strategy
    strategy = TradingStrategy(
        model=model,
        prob_threshold=args.prob_threshold,
        confidence_threshold=args.conf_threshold,
        max_position_size=0.25,
        use_kelly=True,
        stop_loss=0.02,
        take_profit=0.04,
        device=args.device
    )

    # Run backtest
    logger.info("Running backtest...")
    prices = ohlcv_data.df.loc[features.timestamps, 'close'].values

    result = strategy.backtest(
        features=features.X,
        prices=prices,
        timestamps=features.timestamps,
        initial_capital=10000.0,
        transaction_cost=0.001
    )

    # Print results
    print(result.summary())

    # Plot results
    plot_results(result, save_path=args.output)


if __name__ == '__main__':
    main()
