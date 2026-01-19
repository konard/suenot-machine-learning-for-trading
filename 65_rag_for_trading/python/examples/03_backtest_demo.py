"""
Example 03: Backtesting RAG Trading Signals

This example demonstrates how to backtest a trading strategy
based on RAG-generated signals using historical price data.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from retriever import Document, SimpleRetriever
from signals import RAGTradingSignalGenerator
from backtest import RAGBacktester, run_backtest_comparison
from data_loader import generate_mock_data, calculate_features


def create_sample_documents():
    """Create sample documents for the backtest."""
    return [
        # Bullish documents
        Document.create(
            text="Strong bullish momentum with increasing volume. Technical "
                 "indicators suggest continued uptrend. Analysts upgrade to buy. "
                 "Revenue growth exceeded expectations.",
            ticker="DEMO",
            source="TechAnalysis",
            date=datetime.now() - timedelta(days=1),
            doc_type="news"
        ),
        Document.create(
            text="Company reports record quarterly earnings, beating consensus "
                 "by 20%. Management raises full-year guidance. Strong demand "
                 "across all segments. Profit margins expanding.",
            ticker="DEMO",
            source="EarningsReport",
            date=datetime.now() - timedelta(days=5),
            doc_type="news"
        ),
        Document.create(
            text="New product launch exceeds expectations. Pre-orders surge "
                 "past initial estimates. Analysts see significant upside "
                 "potential. Market share gains accelerating.",
            ticker="DEMO",
            source="ProductNews",
            date=datetime.now() - timedelta(days=10),
            doc_type="news"
        ),
    ]


def main():
    print("=" * 60)
    print("RAG Backtest Demo")
    print("=" * 60)

    # Generate synthetic price data
    print("\nGenerating synthetic price data...")

    price_data = generate_mock_data(
        symbol="DEMO",
        initial_price=100.0,
        days=252,  # One year of trading days
        volatility=0.02,
        seed=42
    )

    df = price_data.ohlcv
    print(f"Price data: {len(df)} trading days")
    print(f"Period: {price_data.start_date.date()} to {price_data.end_date.date()}")
    print(f"Start price: ${df['close'].iloc[0]:.2f}")
    print(f"End price: ${df['close'].iloc[-1]:.2f}")

    buy_hold_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1
    print(f"Buy & Hold Return: {buy_hold_return:.2%}")

    # Calculate technical features
    print("\nCalculating technical features...")
    features = calculate_features(df)
    print(f"Features: {list(features.columns)}")

    # Initialize RAG components
    print("\nInitializing RAG components...")
    documents = create_sample_documents()
    retriever = SimpleRetriever()
    retriever.add_documents(documents)

    generator = RAGTradingSignalGenerator(retriever)

    # Run backtest
    print("\n" + "=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)

    backtester = RAGBacktester(
        signal_generator=generator,
        initial_capital=100000.0,
        position_size=0.95,
        transaction_cost=0.001,  # 0.1% per trade
        allow_short=False
    )

    # Weekly signal frequency
    result = backtester.backtest(
        ticker="DEMO",
        price_data=df,
        signal_frequency="weekly"
    )

    # Print results
    print(result.summary())

    # Additional metrics
    print("\nAdditional Metrics:")
    print("-" * 40)
    print(f"Initial Capital:    ${backtester.initial_capital:,.2f}")
    print(f"Final Equity:       ${result.metadata.get('final_equity', 0):,.2f}")
    print(f"Winning Trades:     {result.metadata.get('winning_trades', 0)}")
    print(f"Losing Trades:      {result.metadata.get('losing_trades', 0)}")

    # Trade details
    if result.trades:
        print("\n" + "=" * 60)
        print("TRADE HISTORY (Last 5 trades)")
        print("=" * 60)
        print(f"\n{'Entry Date':<12} {'Exit Date':<12} {'Shares':>8} {'Entry $':>10} "
              f"{'Exit $':>10} {'P&L':>12} {'Return':>10}")
        print("-" * 82)

        for trade in result.trades[-5:]:
            if trade.is_closed:
                print(f"{trade.entry_date.strftime('%Y-%m-%d'):<12} "
                      f"{trade.exit_date.strftime('%Y-%m-%d'):<12} "
                      f"{trade.shares:>8.0f} "
                      f"${trade.entry_price:>9.2f} "
                      f"${trade.exit_price:>9.2f} "
                      f"${trade.pnl:>11.2f} "
                      f"{trade.return_pct:>9.2%}")

    # Compare different frequencies
    print("\n" + "=" * 60)
    print("FREQUENCY COMPARISON")
    print("=" * 60)

    comparison = run_backtest_comparison(
        signal_generator=generator,
        price_data=df,
        ticker="DEMO",
        frequencies=["daily", "weekly", "monthly"]
    )

    print(f"\n{'Frequency':<12} {'Return':>10} {'Sharpe':>10} {'Max DD':>10} "
          f"{'Win Rate':>10} {'Trades':>8}")
    print("-" * 70)

    for freq, res in comparison.items():
        print(f"{freq:<12} {res.total_return:>10.2%} {res.sharpe_ratio:>10.2f} "
              f"{res.max_drawdown:>10.2%} {res.win_rate:>10.2%} {res.total_trades:>8}")

    # Equity curve analysis
    if result.equity_curve:
        print("\n" + "=" * 60)
        print("EQUITY CURVE ANALYSIS")
        print("=" * 60)

        equity = pd.Series(result.equity_curve)
        print(f"\nEquity Curve Statistics:")
        print(f"  Min:    ${equity.min():,.2f}")
        print(f"  Max:    ${equity.max():,.2f}")
        print(f"  Mean:   ${equity.mean():,.2f}")
        print(f"  Final:  ${equity.iloc[-1]:,.2f}")

        # Rolling statistics
        if len(equity) > 20:
            rolling_max = equity.rolling(20).max()
            rolling_min = equity.rolling(20).min()
            print(f"\n20-period Rolling:")
            print(f"  Peak:   ${rolling_max.max():,.2f}")
            print(f"  Trough: ${rolling_min.min():,.2f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
