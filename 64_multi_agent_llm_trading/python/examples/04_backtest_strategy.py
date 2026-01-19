#!/usr/bin/env python3
"""
Example 4: Backtesting Multi-Agent Trading Strategy

This example demonstrates how to backtest a multi-agent trading
strategy against historical data and compare it to benchmarks.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from agents import (
    TechnicalAgent,
    FundamentalsAgent,
    SentimentAgent,
    BullAgent,
    BearAgent,
    RiskManagerAgent,
    TraderAgent,
)
from backtest import MultiAgentBacktester, BenchmarkComparison


def create_realistic_market_data(days: int = 504) -> pd.DataFrame:
    """
    Create realistic market data with various market regimes.

    Simulates: bull market, correction, recovery, and sideways periods.
    """
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=days, freq="B")

    # Create different market regimes
    regime_length = days // 4

    # Regime 1: Bull market
    bull = np.linspace(0, 0.3, regime_length) + np.random.randn(regime_length).cumsum() * 0.01

    # Regime 2: Sharp correction
    correction = np.linspace(0.3, 0.1, regime_length) + np.random.randn(regime_length).cumsum() * 0.015

    # Regime 3: Recovery
    recovery = np.linspace(0.1, 0.35, regime_length) + np.random.randn(regime_length).cumsum() * 0.012

    # Regime 4: Sideways/consolidation
    sideways = np.linspace(0.35, 0.38, days - 3 * regime_length) + np.random.randn(days - 3 * regime_length).cumsum() * 0.008

    # Combine regimes
    full_trend = np.concatenate([bull, correction, recovery, sideways])
    close = 100 * np.exp(full_trend)

    return pd.DataFrame({
        "open": close * (1 + np.random.randn(days) * 0.005),
        "high": close * (1 + abs(np.random.randn(days) * 0.012)),
        "low": close * (1 - abs(np.random.randn(days) * 0.012)),
        "close": close,
        "volume": np.random.randint(5e6, 5e7, days).astype(float)
    }, index=dates)


def main():
    print("=" * 60)
    print("Multi-Agent LLM Trading - Backtesting Example")
    print("=" * 60)

    # Create market data
    print("\n1. Creating market data with multiple regimes...")
    data = create_realistic_market_data(days=504)  # ~2 years

    print(f"   Period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"   Start price: ${data['close'].iloc[0]:.2f}")
    print(f"   End price: ${data['close'].iloc[-1]:.2f}")
    print(f"   Total return: {(data['close'].iloc[-1] / data['close'].iloc[0] - 1):.1%}")
    print(f"   Max drawdown: {((data['close'] / data['close'].cummax()).min() - 1):.1%}")

    # Create agent team
    print("\n2. Creating agent team...")

    agents = [
        TechnicalAgent("Tech-Analyst"),
        FundamentalsAgent("Fund-Analyst"),
        SentimentAgent("Sent-Analyst"),
        BullAgent("Optimist"),
        BearAgent("Skeptic"),
        RiskManagerAgent("Risk-Mgr", max_position_pct=0.05),
    ]

    trader = TraderAgent("Head-Trader")

    print(f"   {len(agents)} analysis agents created")

    # Run backtest with different configurations
    print("\n3. Running backtests...")
    print("-" * 60)

    # Configuration 1: Standard multi-agent
    print("\n   [A] Standard Multi-Agent Strategy (rebalance weekly)...")
    backtester_standard = MultiAgentBacktester(
        agents=agents,
        trader_agent=trader,
        initial_capital=100000,
        position_size_pct=0.2,
        transaction_cost_pct=0.001,
        use_debate=False
    )

    result_standard = backtester_standard.run(
        "DEMO", data,
        lookback=50,
        step=5  # Rebalance every 5 days
    )

    print(f"      Total Return: {result_standard.total_return:.2f}%")
    print(f"      Sharpe Ratio: {result_standard.sharpe_ratio:.2f}")
    print(f"      Max Drawdown: {result_standard.max_drawdown:.2f}%")
    print(f"      Trades: {result_standard.num_trades}")

    # Configuration 2: With debate
    print("\n   [B] Multi-Agent with Bull/Bear Debate (rebalance weekly)...")
    backtester_debate = MultiAgentBacktester(
        agents=agents,
        trader_agent=trader,
        initial_capital=100000,
        position_size_pct=0.2,
        transaction_cost_pct=0.001,
        use_debate=True,
        debate_rounds=2
    )

    result_debate = backtester_debate.run(
        "DEMO", data,
        lookback=50,
        step=5
    )

    print(f"      Total Return: {result_debate.total_return:.2f}%")
    print(f"      Sharpe Ratio: {result_debate.sharpe_ratio:.2f}")
    print(f"      Max Drawdown: {result_debate.max_drawdown:.2f}%")
    print(f"      Trades: {result_debate.num_trades}")

    # Configuration 3: More aggressive
    print("\n   [C] Aggressive Strategy (larger positions, daily rebalance)...")
    backtester_aggressive = MultiAgentBacktester(
        agents=agents,
        trader_agent=trader,
        initial_capital=100000,
        position_size_pct=0.4,
        transaction_cost_pct=0.001,
        use_debate=False
    )

    result_aggressive = backtester_aggressive.run(
        "DEMO", data,
        lookback=50,
        step=1  # Daily rebalance
    )

    print(f"      Total Return: {result_aggressive.total_return:.2f}%")
    print(f"      Sharpe Ratio: {result_aggressive.sharpe_ratio:.2f}")
    print(f"      Max Drawdown: {result_aggressive.max_drawdown:.2f}%")
    print(f"      Trades: {result_aggressive.num_trades}")

    # Benchmark comparison
    print("\n4. Comparing to Buy & Hold benchmark...")
    print("-" * 60)

    benchmark = BenchmarkComparison.buy_and_hold(data, initial_capital=100000)

    print(f"\n   Buy & Hold:")
    print(f"      Total Return: {benchmark.total_return:.2f}%")
    print(f"      Sharpe Ratio: {benchmark.sharpe_ratio:.2f}")
    print(f"      Max Drawdown: {benchmark.max_drawdown:.2f}%")

    # Comparison
    comparison = BenchmarkComparison.compare(result_standard, benchmark)

    print(f"\n   Standard Strategy vs Benchmark:")
    print(f"      Excess Return: {comparison['excess_return']:+.2f}%")
    print(f"      Sharpe Difference: {comparison['sharpe_difference']:+.2f}")
    print(f"      Drawdown Improvement: {comparison['drawdown_improvement']:+.2f}%")

    # Summary table
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    results = [
        ("Buy & Hold", benchmark),
        ("Standard Multi-Agent", result_standard),
        ("With Debate", result_debate),
        ("Aggressive", result_aggressive),
    ]

    print(f"\n{'Strategy':<25} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Trades':>8}")
    print("-" * 65)

    for name, result in results:
        print(f"{name:<25} {result.total_return:>9.2f}% {result.sharpe_ratio:>10.2f} "
              f"{result.max_drawdown:>9.2f}% {result.num_trades:>8}")

    # Trade analysis
    print("\n" + "=" * 60)
    print("TRADE ANALYSIS (Standard Strategy)")
    print("=" * 60)

    if result_standard.trades:
        winning_trades = [t for t in result_standard.trades if t.pnl > 0]
        losing_trades = [t for t in result_standard.trades if t.pnl < 0]

        print(f"\n   Total Trades: {len(result_standard.trades)}")
        print(f"   Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(result_standard.trades)*100:.1f}%)")
        print(f"   Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(result_standard.trades)*100:.1f}%)")

        if winning_trades:
            avg_win = np.mean([t.pnl for t in winning_trades])
            print(f"   Average Win: ${avg_win:.2f}")

        if losing_trades:
            avg_loss = np.mean([t.pnl for t in losing_trades])
            print(f"   Average Loss: ${avg_loss:.2f}")

        print("\n   Recent Trades:")
        for trade in result_standard.trades[-5:]:
            pnl_str = f"+${trade.pnl:.2f}" if trade.pnl > 0 else f"-${abs(trade.pnl):.2f}"
            print(f"      {trade.timestamp.date()}: {trade.action} @ ${trade.price:.2f} [{pnl_str}]")

    # Conclusions
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print("""
   Key Takeaways from Backtest:

   1. MULTI-AGENT VS BUY & HOLD
      - Multi-agent strategies can potentially reduce drawdowns
      - More consistent returns across different market regimes
      - But may underperform in strong bull markets (cash drag)

   2. DEBATE MECHANISM
      - Bull vs Bear debate adds a layer of validation
      - May reduce false signals by challenging assumptions
      - Slight performance impact from additional analysis

   3. POSITION SIZING
      - Aggressive sizing increases both returns AND risk
      - Conservative sizing (5-20%) recommended for real trading
      - Risk management is crucial in volatile markets

   4. REBALANCING FREQUENCY
      - Daily rebalancing: More trades, higher costs, faster reaction
      - Weekly rebalancing: Fewer trades, lower costs, smoother equity curve
      - Monthly rebalancing: Risk of missing important signals

   IMPORTANT: This is a demonstration with simulated data.
   Real trading involves additional risks and complexities.
   Always paper trade before using real capital.
    """)


if __name__ == "__main__":
    main()
