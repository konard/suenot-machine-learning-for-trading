#!/usr/bin/env python3
"""
Example 01: Stock Market Regime Classification

This example demonstrates how to classify market regimes
for stock market data using S&P 500 (SPY) ETF.

Usage:
    python 01_stock_regime.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python.data_loader import YahooFinanceLoader
from python.classifier import (
    StatisticalRegimeClassifier,
    HybridRegimeClassifier,
    MarketRegime
)
from python.signals import RegimeSignalGenerator
from python.backtest import RegimeBacktester
from python.evaluate import generate_report


def main():
    """Run stock market regime classification example."""
    print("=" * 60)
    print("Stock Market Regime Classification Example")
    print("=" * 60)
    print()

    # Step 1: Load data
    print("Step 1: Loading SPY data from Yahoo Finance...")
    loader = YahooFinanceLoader()

    try:
        spy_data = loader.get_daily("SPY", period="2y")
        print(f"  Loaded {len(spy_data)} days of data")
        print(f"  Date range: {spy_data.index[0].date()} to {spy_data.index[-1].date()}")
        print(f"  Price range: ${spy_data['close'].min():.2f} - ${spy_data['close'].max():.2f}")
    except Exception as e:
        print(f"  Error loading data: {e}")
        print("  Using mock data for demonstration...")
        spy_data = loader._generate_mock_data("SPY", 500, "1d")
    print()

    # Step 2: Initialize classifier
    print("Step 2: Initializing regime classifier...")
    classifier = StatisticalRegimeClassifier(
        lookback_window=20,
        volatility_threshold_high=0.25,
        trend_threshold=0.0005
    )

    # Fit the classifier
    classifier.fit(spy_data)
    print("  Classifier fitted successfully")
    print()

    # Step 3: Classify recent market regime
    print("Step 3: Classifying current market regime...")
    result = classifier.classify(spy_data)

    print(f"  Current Regime: {result.regime.value}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Probability: {result.probability:.3f}")
    print(f"  Explanation: {result.explanation}")
    print()

    # Step 4: Generate historical regime classifications
    print("Step 4: Generating historical regime classifications...")

    # Classify each day using rolling window
    regime_history = []
    window = 20

    for i in range(window, len(spy_data)):
        window_data = spy_data.iloc[i-window:i+1]
        regime_result = classifier.classify(window_data)
        regime_history.append(regime_result)

    # Count regime occurrences
    regime_counts = {}
    for r in MarketRegime:
        count = sum(1 for rh in regime_history if rh.regime == r)
        regime_counts[r] = count

    print("  Regime Distribution:")
    total = len(regime_history)
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        print(f"    {regime.value:15s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Step 5: Generate trading signals
    print("Step 5: Generating trading signals...")
    signal_gen = RegimeSignalGenerator(confidence_threshold=0.6)

    # Get signal for current regime
    signal = signal_gen.generate_signal(result)

    print(f"  Signal Type: {signal.signal_type.value}")
    print(f"  Position Size: {signal.position_size:.1%}")
    print(f"  Stop Loss: {signal.stop_loss:.1%}" if signal.stop_loss else "  Stop Loss: None")
    print(f"  Take Profit: {signal.take_profit:.1%}" if signal.take_profit else "  Take Profit: None")
    print(f"  Reasoning: {signal.reasoning}")
    print()

    # Step 6: Run backtest
    print("Step 6: Running backtest...")
    backtester = RegimeBacktester(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )

    # Get prices for backtest
    prices = spy_data['close'].iloc[window:]

    # Run backtest
    backtest_result = backtester.run(prices, regime_history, signal_gen)

    print(f"  Total Return: {backtest_result.total_return:.2%}")
    print(f"  Annualized Return: {backtest_result.annualized_return:.2%}")
    print(f"  Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {backtest_result.max_drawdown:.2%}")
    print(f"  Win Rate: {backtest_result.win_rate:.1%}")
    print(f"  Number of Trades: {backtest_result.num_trades}")
    print()

    # Step 7: Generate evaluation report
    print("Step 7: Generating evaluation report...")
    report = generate_report(regime_history, prices)
    print()
    print(report)

    print()
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
