#!/usr/bin/env python3
"""
Example 04: Backtesting Demo

This example demonstrates how to backtest LLM-generated alpha factors
using realistic trading simulations.

Run with: python 04_backtest_demo.py
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
from data_loader import generate_synthetic_data
from alpha_generator import AlphaExpressionParser, PREDEFINED_FACTORS
from backtest import Backtester, backtest_factor, BacktestResult


def main():
    print("=" * 60)
    print("LLM Alpha Mining - Backtesting Demo")
    print("=" * 60)

    # Load data
    print("\n1. LOADING DATA")
    print("-" * 40)

    data = generate_synthetic_data(["BTCUSDT"], days=500, seed=42)
    btc_data = data["BTCUSDT"].ohlcv
    print(f"Loaded {len(btc_data)} records for BTCUSDT")
    print(f"Date range: {btc_data.index[0].date()} to {btc_data.index[-1].date()}")

    # Initialize backtester
    print("\n2. INITIALIZING BACKTESTER")
    print("-" * 40)

    backtester = Backtester(
        initial_capital=100000,
        position_size=0.5,      # 50% of capital per trade
        commission=0.001,       # 0.1% commission
        slippage=0.0005,        # 0.05% slippage
        periods_per_year=252    # Daily data
    )

    print(f"Initial capital: ${backtester.initial_capital:,.0f}")
    print(f"Position size: {backtester.position_size:.0%}")
    print(f"Commission: {backtester.commission:.2%}")
    print(f"Slippage: {backtester.slippage:.2%}")

    # Initialize parser
    parser = AlphaExpressionParser()

    # Backtest predefined factors
    print("\n3. BACKTESTING PREDEFINED FACTORS")
    print("-" * 40)

    results = {}
    for factor in PREDEFINED_FACTORS[:5]:
        try:
            # Calculate factor values
            factor_values = parser.evaluate(factor.expression, btc_data)

            # Normalize to z-score
            factor_zscore = (factor_values - factor_values.mean()) / factor_values.std()

            # Run backtest
            result = backtester.run(
                signals=factor_zscore,
                prices=btc_data["close"],
                long_threshold=0.5,
                short_threshold=-0.5,
                max_holding_periods=10,
                stop_loss=0.05,
                take_profit=0.10
            )

            results[factor.name] = result

            print(f"\n{factor.name}:")
            print(f"  Total Return: {result.total_return:+.2%}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {result.max_drawdown:.2%}")
            print(f"  Win Rate: {result.win_rate:.2%}")
            print(f"  Profit Factor: {result.profit_factor:.2f}")
            print(f"  Total Trades: {result.total_trades}")

        except Exception as e:
            print(f"\n{factor.name}: Error - {e}")

    # Compare all factors
    print("\n4. FACTOR COMPARISON")
    print("-" * 40)

    if results:
        comparison = []
        for name, r in results.items():
            comparison.append({
                "Factor": name,
                "Return": f"{r.total_return:+.1%}",
                "Sharpe": f"{r.sharpe_ratio:.2f}",
                "MaxDD": f"{r.max_drawdown:.1%}",
                "WinRate": f"{r.win_rate:.0%}",
                "Trades": r.total_trades
            })

        df = pd.DataFrame(comparison)
        print("\n" + df.to_string(index=False))

    # Detailed analysis of best factor
    print("\n5. DETAILED ANALYSIS")
    print("-" * 40)

    if results:
        # Find best by Sharpe
        best_name = max(results, key=lambda x: results[x].sharpe_ratio)
        best = results[best_name]

        print(f"\nBest factor: {best_name}")
        print(best.summary())

        # Trade analysis
        print("\nTrade Statistics:")
        winning_trades = [t for t in best.trades if t.pnl > 0]
        losing_trades = [t for t in best.trades if t.pnl <= 0]

        if winning_trades:
            avg_win = np.mean([t.pnl for t in winning_trades])
            max_win = max([t.pnl for t in winning_trades])
            print(f"  Avg winning trade: ${avg_win:.2f}")
            print(f"  Max winning trade: ${max_win:.2f}")

        if losing_trades:
            avg_loss = np.mean([t.pnl for t in losing_trades])
            max_loss = min([t.pnl for t in losing_trades])
            print(f"  Avg losing trade: ${avg_loss:.2f}")
            print(f"  Max losing trade: ${max_loss:.2f}")

        # Recent trades
        print("\nRecent Trades:")
        for trade in best.trades[-5:]:
            direction = "LONG" if trade.side == "long" else "SHORT"
            exit_reason = trade.metadata.get('exit_reason', 'unknown')
            print(f"  {trade.timestamp.date()}: {direction} @ ${trade.entry_price:.2f} -> "
                  f"${trade.exit_price:.2f}, PnL: ${trade.pnl:+.2f} ({exit_reason})")

    # Walk-forward analysis
    print("\n6. WALK-FORWARD ANALYSIS")
    print("-" * 40)

    if results:
        factor_values = parser.evaluate(
            PREDEFINED_FACTORS[0].expression,
            btc_data
        )
        factor_zscore = (factor_values - factor_values.mean()) / factor_values.std()

        wf_results = backtester.walk_forward(
            signals=factor_zscore,
            prices=btc_data["close"],
            train_periods=100,
            test_periods=50,
            long_threshold=0.5,
            short_threshold=-0.5
        )

        print(f"\nWalk-forward windows: {len(wf_results)}")
        print("\nWindow Results:")
        for i, r in enumerate(wf_results):
            status = "+" if r.total_return > 0 else "-"
            print(f"  {status} Window {i + 1}: Return={r.total_return:+.1%}, "
                  f"Sharpe={r.sharpe_ratio:.2f}, Trades={r.total_trades}")

        # Aggregate statistics
        wf_returns = [r.total_return for r in wf_results]
        wf_sharpes = [r.sharpe_ratio for r in wf_results]

        print(f"\nAggregate Walk-Forward Statistics:")
        print(f"  Mean Return: {np.mean(wf_returns):+.2%}")
        print(f"  Std Return: {np.std(wf_returns):.2%}")
        print(f"  Mean Sharpe: {np.mean(wf_sharpes):.2f}")
        print(f"  Win Rate (windows): {np.mean([r > 0 for r in wf_returns]):.0%}")

    # Monte Carlo simulation
    print("\n7. MONTE CARLO SIMULATION")
    print("-" * 40)

    if results and best.total_trades >= 10:
        mc_result = backtester.monte_carlo(
            signals=factor_zscore,
            prices=btc_data["close"],
            n_simulations=1000,
            long_threshold=0.5,
            short_threshold=-0.5
        )

        print(f"\nMonte Carlo Analysis (1000 simulations):")
        print(f"  Original Return: {mc_result['original_return']:+.2%}")
        print(f"  Mean Simulated: {mc_result['mean_return']:+.2%}")
        print(f"  Std Simulated: {mc_result['std_return']:.2%}")
        print(f"\nReturn Distribution:")
        print(f"  5th percentile: {mc_result['percentile_5']:+.2%}")
        print(f"  25th percentile: {mc_result['percentile_25']:+.2%}")
        print(f"  50th percentile: {mc_result['percentile_50']:+.2%}")
        print(f"  75th percentile: {mc_result['percentile_75']:+.2%}")
        print(f"  95th percentile: {mc_result['percentile_95']:+.2%}")
        print(f"\nProbability of Profit: {mc_result['probability_profit']:.1%}")

    # Quick backtest function
    print("\n8. QUICK BACKTEST FUNCTION")
    print("-" * 40)

    expression = "(close - ts_mean(close, 10)) / ts_std(close, 10)"
    print(f"\nExpression: {expression}")

    result = backtest_factor(
        expression,
        btc_data,
        long_threshold=1.0,
        short_threshold=-1.0,
        initial_capital=100000,
        commission=0.001
    )

    print(f"\nResults:")
    print(f"  Total Return: {result.total_return:+.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Total Trades: {result.total_trades}")

    # Different strategies
    print("\n9. STRATEGY VARIATIONS")
    print("-" * 40)

    expression = "ts_delta(close, 5) / ts_delay(close, 5)"
    factor_values = parser.evaluate(expression, btc_data)
    factor_zscore = (factor_values - factor_values.mean()) / factor_values.std()

    strategies = [
        ("Long Only", {"long_threshold": 0.5, "short_threshold": None}),
        ("Long-Short", {"long_threshold": 0.5, "short_threshold": -0.5}),
        ("High Conviction", {"long_threshold": 1.5, "short_threshold": -1.5}),
        ("With Stop Loss", {"long_threshold": 0.5, "short_threshold": -0.5, "stop_loss": 0.03}),
    ]

    print(f"\nFactor: {expression}")
    print("\nStrategy Comparison:")

    for name, params in strategies:
        try:
            result = backtester.run(
                signals=factor_zscore,
                prices=btc_data["close"],
                max_holding_periods=10,
                **params
            )
            print(f"  {name:20s}: Return={result.total_return:+6.1%}, "
                  f"Sharpe={result.sharpe_ratio:5.2f}, Trades={result.total_trades:3d}")
        except Exception as e:
            print(f"  {name:20s}: Error - {e}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
