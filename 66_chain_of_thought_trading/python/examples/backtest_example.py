#!/usr/bin/env python3
"""
Backtesting Example for Chain-of-Thought Trading

This example demonstrates how to backtest a CoT trading strategy
and analyze the results with full audit trail.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, "..")

from signal_generator import MultiStepSignalGenerator
from backtest import CoTBacktester, BacktestConfig
from data_loader import MockDataLoader, add_technical_indicators


def main():
    """Run backtest example."""
    print("=" * 60)
    print("Chain-of-Thought Trading Backtest")
    print("=" * 60)

    # Configuration
    symbol = "AAPL"
    initial_capital = 100000.0

    print(f"\nConfiguration:")
    print(f"  Symbol:           {symbol}")
    print(f"  Initial Capital:  ${initial_capital:,.2f}")

    # Load historical data
    print("\nLoading historical data...")
    loader = MockDataLoader(seed=42)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    df = loader.load(symbol, start_date, end_date, "1d")
    df = add_technical_indicators(df)
    print(f"  Loaded {len(df)} data points")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Prepare DataFrame with datetime index for backtester
    prices_df = df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']]

    # Initialize components
    print("\nInitializing trading components...")
    signal_gen = MultiStepSignalGenerator()

    # Configure backtest
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_pct=0.001,  # 0.1% commission
        slippage_pct=0.0005,   # 0.05% slippage
        max_position_pct=0.2,
    )

    # Run backtest
    print("\nRunning backtest...")
    backtester = CoTBacktester(config=config)

    result = backtester.run(
        prices=prices_df,
        signal_generator=signal_gen,
    )

    # Display results
    print("\n" + "=" * 60)
    print("Backtest Results")
    print("=" * 60)

    metrics = result.metrics
    print(f"\nPerformance Metrics:")
    print(f"  Total Return:      {metrics.get('total_return', 0):.2%}")
    print(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown:      {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Win Rate:          {metrics.get('win_rate', 0):.1%}")
    print(f"  Profit Factor:     {metrics.get('profit_factor', 0):.2f}")

    winning_trades = len([t for t in result.trades if t.pnl > 0])
    losing_trades = len([t for t in result.trades if t.pnl <= 0])
    print(f"\nTrading Statistics:")
    print(f"  Total Trades:      {metrics.get('num_trades', len(result.trades))}")
    print(f"  Winning Trades:    {winning_trades}")
    print(f"  Losing Trades:     {losing_trades}")

    final_capital = metrics.get('final_equity', result.equity_curve.iloc[-1] if len(result.equity_curve) > 0 else initial_capital)
    print(f"\nCapital:")
    print(f"  Initial:           ${initial_capital:,.2f}")
    print(f"  Final:             ${final_capital:,.2f}")
    print(f"  Profit/Loss:       ${final_capital - initial_capital:,.2f}")

    # Display trade log
    if result.trades:
        print(f"\nTrade Log (showing first 10 trades):")
        print("-" * 80)
        print(f"{'#':<4} {'Date':<12} {'Type':<6} {'Entry':<10} {'Exit':<10} {'P/L':<12} {'Return':<8}")
        print("-" * 80)

        for i, trade in enumerate(result.trades[:10], 1):
            trade_type = trade.direction.name
            entry_date = trade.entry_time.strftime("%Y-%m-%d")
            pnl = trade.pnl
            ret = (trade.exit_price - trade.entry_price) / trade.entry_price
            if trade.direction.value == -1:  # SHORT
                ret = -ret

            print(f"{i:<4} {entry_date:<12} {trade_type:<6} ${trade.entry_price:<9.2f} "
                  f"${trade.exit_price:<9.2f} ${pnl:<11.2f} {ret:>6.2%}")

        if len(result.trades) > 10:
            print(f"... and {len(result.trades) - 10} more trades")

    # Display sample reasoning chain
    if result.trades:
        print(f"\nSample Trade Reasoning Chain:")
        print("-" * 60)
        sample_trade = result.trades[0]
        print(f"Trade: {sample_trade.direction.name} at ${sample_trade.entry_price:.2f}")
        if sample_trade.reasoning_chain:
            print(f"\nReasoning:")
            for i, reason in enumerate(sample_trade.reasoning_chain[:5], 1):
                print(f"  {i}. {reason}")
        else:
            print("  (No reasoning chain available)")

    # Equity curve summary
    print(f"\nEquity Curve (monthly snapshots):")
    print("-" * 40)

    # Show monthly equity values (equity_curve is a pandas Series with datetime index)
    if len(result.equity_curve) > 0:
        equity_len = len(result.equity_curve)
        step = max(1, equity_len // 12)

        for i in range(0, equity_len, step):
            date = result.equity_curve.index[i]
            equity = result.equity_curve.iloc[i]
            print(f"  {date.strftime('%Y-%m')}: ${equity:,.2f}")

    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)
    print("\nNote: This backtest uses mock data and mock AI analysis.")
    print("Results are for demonstration purposes only.")


if __name__ == "__main__":
    main()
