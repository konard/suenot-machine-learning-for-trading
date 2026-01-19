#!/usr/bin/env python3
"""
FinGPT Backtesting Demo

This example demonstrates how to backtest a FinGPT-based trading strategy
using historical price data and simulated news sentiment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fingpt_sentiment import FinGPTSentimentAnalyzer
from backtest import FinGPTBacktester, BacktestConfig, generate_mock_data


def main():
    """Demo of backtesting FinGPT trading strategies."""
    # Initialize components
    analyzer = FinGPTSentimentAnalyzer(use_mock=True)

    print("=" * 70)
    print("FinGPT Strategy Backtesting Demo")
    print("=" * 70)

    # Generate mock historical data
    print("\nGenerating mock historical data...")
    signals, prices = generate_mock_data(
        symbols=["BTCUSDT", "ETHUSDT", "AAPL", "NVDA"],
        start_date="2024-01-01",
        end_date="2024-06-30",
        n_signals=50
    )

    print(f"  Price data points: {len(prices)}")
    print(f"  Signal events: {len(signals)}")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000.0,
        max_position_pct=0.1,
        transaction_cost_bps=10,
        slippage_bps=5,
        signal_threshold=0.3,
        signal_decay_hours=24.0
    )

    print(f"\nBacktest Configuration:")
    print(f"  Initial Capital: ${config.initial_capital:,.2f}")
    print(f"  Position Size: {config.max_position_pct:.0%}")
    print(f"  Signal Threshold: {config.signal_threshold}")
    print(f"  Transaction Cost: {config.transaction_cost_bps} bps")
    print(f"  Slippage: {config.slippage_bps} bps")
    print(f"  Signal Decay: {config.signal_decay_hours} hours")

    # Run backtest
    print("\n" + "=" * 70)
    print("Running Backtest...")
    print("=" * 70)

    backtester = FinGPTBacktester(config)
    result = backtester.run_backtest(signals, prices)

    # Display results
    print(f"\n{'BACKTEST RESULTS':=^70}")

    print(f"\nPerformance Summary:")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Annualized Return: {result.annualized_return:.2%}")
    print(f"  Final Portfolio Value: ${result.portfolio_values.iloc[-1]:,.2f}")
    print(f"  Profit/Loss: ${result.portfolio_values.iloc[-1] - config.initial_capital:+,.2f}")

    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Volatility (Ann.): {result.volatility:.2%}")

    print(f"\nTrading Statistics:")
    print(f"  Total Trades: {result.num_trades}")
    winning = len([t for t in result.trades if t.get('pnl', 0) > 0])
    losing = len([t for t in result.trades if t.get('pnl', 0) <= 0])
    print(f"  Winning Trades: {winning}")
    print(f"  Losing Trades: {losing}")
    print(f"  Win Rate: {result.win_rate:.1%}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")

    # Display individual trades
    if result.trades:
        print(f"\n{'TRADE LOG':=^70}")
        print("\n{:<12} {:<10} {:<8} {:<10} {:<12} {:<10}".format(
            "Date", "Symbol", "Type", "Shares", "Price", "Value"
        ))
        print("-" * 70)

        for trade in result.trades[:10]:  # Show first 10 trades
            print(f"{trade['date'].strftime('%Y-%m-%d'):<12} "
                  f"{trade['symbol']:<10} "
                  f"{trade['type']:<8} "
                  f"{trade['shares']:<10.4f} "
                  f"${trade['price']:<10,.2f} "
                  f"${trade['value']:,.2f}")

        if len(result.trades) > 10:
            print(f"... and {len(result.trades) - 10} more trades")

    # Compare strategies
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    # Test different configurations
    strategies = [
        ("Conservative", BacktestConfig(
            initial_capital=100000.0,
            max_position_pct=0.05,
            signal_threshold=0.5,
            transaction_cost_bps=10
        )),
        ("Moderate", BacktestConfig(
            initial_capital=100000.0,
            max_position_pct=0.10,
            signal_threshold=0.3,
            transaction_cost_bps=10
        )),
        ("Aggressive", BacktestConfig(
            initial_capital=100000.0,
            max_position_pct=0.20,
            signal_threshold=0.2,
            transaction_cost_bps=10
        )),
    ]

    print("\n{:<15} {:<12} {:<12} {:<12} {:<12}".format(
        "Strategy", "Return", "Sharpe", "Max DD", "Win Rate"
    ))
    print("-" * 70)

    for name, cfg in strategies:
        bt = FinGPTBacktester(cfg)
        res = bt.run_backtest(signals, prices)

        print(f"{name:<15} "
              f"{res.total_return:+.1%}        "
              f"{res.sharpe_ratio:.2f}         "
              f"{res.max_drawdown:.1%}        "
              f"{res.win_rate:.0%}")

    # Monthly breakdown
    print("\n" + "=" * 70)
    print("MONTHLY PERFORMANCE")
    print("=" * 70)

    if len(result.portfolio_values) > 0:
        # Resample to monthly
        monthly_values = result.portfolio_values.resample('ME').last()

        print("\n{:<12} {:<15} {:<12}".format(
            "Month", "End Value", "Return"
        ))
        print("-" * 45)

        prev_value = config.initial_capital
        for month_end, value in monthly_values.items():
            ret = (value - prev_value) / prev_value
            print(f"{month_end.strftime('%Y-%m'):<12} "
                  f"${value:<13,.2f} "
                  f"{ret:+.2%}")
            prev_value = value

    print("\n" + "=" * 70)
    print("DISCLAIMER")
    print("=" * 70)
    print("""
This backtest uses simulated data and mock sentiment analysis.
Past performance does not guarantee future results.
This is for educational purposes only - not financial advice.
Always do your own research before trading.
""")


if __name__ == "__main__":
    main()
