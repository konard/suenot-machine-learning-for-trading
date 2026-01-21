#!/usr/bin/env python3
"""
Example 02: Cryptocurrency Regime Classification (Bybit)

This example demonstrates how to classify market regimes
for cryptocurrency data using Bybit exchange data.

Usage:
    python 02_crypto_regime.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python.data_loader import BybitDataLoader
from python.classifier import (
    StatisticalRegimeClassifier,
    CryptoRegimeClassifier,
    MarketRegime
)
from python.signals import RegimeSignalGenerator, PositionSizer
from python.backtest import RegimeBacktester
from python.evaluate import generate_report


def main():
    """Run cryptocurrency regime classification example."""
    print("=" * 60)
    print("Cryptocurrency Regime Classification Example (Bybit)")
    print("=" * 60)
    print()

    # Step 1: Load Bybit data
    print("Step 1: Loading BTC/USDT data from Bybit...")
    loader = BybitDataLoader()

    try:
        btc_data = loader.get_klines("BTCUSDT", interval="1h", limit=1000)
        print(f"  Loaded {len(btc_data)} hourly bars")
        print(f"  Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        print(f"  Price range: ${btc_data['close'].min():.2f} - ${btc_data['close'].max():.2f}")
    except Exception as e:
        print(f"  Error loading data: {e}")
        print("  Using mock data for demonstration...")
        btc_data = loader._generate_mock_crypto_data("BTCUSDT", 1000, "1h")
    print()

    # Also load ETH data for comparison
    print("  Loading ETH/USDT data...")
    try:
        eth_data = loader.get_klines("ETHUSDT", interval="1h", limit=1000)
        print(f"  Loaded {len(eth_data)} hourly bars for ETH")
    except Exception as e:
        print(f"  Using mock data for ETH...")
        eth_data = loader._generate_mock_crypto_data("ETHUSDT", 1000, "1h")
    print()

    # Step 2: Initialize crypto-specific classifier
    print("Step 2: Initializing crypto regime classifier...")
    print("  (Note: Crypto uses higher volatility thresholds)")

    # Use crypto-specific classifier with adjusted thresholds
    classifier = CryptoRegimeClassifier(
        lookback=24,  # 24 hours
        volatility_threshold=0.5,  # Higher threshold for crypto
        trend_threshold=0.02  # 2% threshold
    )

    # Fit the classifier on Bitcoin data
    classifier.fit(btc_data)
    print("  Classifier fitted successfully")
    print()

    # Step 3: Classify current regime
    print("Step 3: Classifying current BTC regime...")
    btc_result = classifier.classify(btc_data)

    print(f"  BTC Current Regime: {btc_result.regime.value}")
    print(f"  Confidence: {btc_result.confidence:.1%}")
    print(f"  Explanation: {btc_result.explanation}")
    print()

    # Also classify ETH
    print("  Classifying current ETH regime...")
    eth_classifier = CryptoRegimeClassifier(lookback=24)
    eth_classifier.fit(eth_data)
    eth_result = eth_classifier.classify(eth_data)

    print(f"  ETH Current Regime: {eth_result.regime.value}")
    print(f"  Confidence: {eth_result.confidence:.1%}")
    print()

    # Step 4: Generate historical classifications
    print("Step 4: Generating historical regime classifications...")

    regime_history = []
    window = 24  # 24 hours

    for i in range(window, len(btc_data)):
        window_data = btc_data.iloc[i-window:i+1]
        regime_result = classifier.classify(window_data)
        regime_history.append(regime_result)

    # Count regime occurrences
    regime_counts = {}
    for r in MarketRegime:
        count = sum(1 for rh in regime_history if rh.regime == r)
        regime_counts[r] = count

    print("  BTC Regime Distribution (hourly):")
    total = len(regime_history)
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        print(f"    {regime.value:15s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Step 5: Calculate position sizing
    print("Step 5: Calculating position sizes...")

    signal_gen = RegimeSignalGenerator(confidence_threshold=0.5)
    signal = signal_gen.generate_signal(btc_result)

    position_sizer = PositionSizer(
        max_position=0.5,  # Max 50% of capital in crypto
        risk_per_trade=0.02,
        volatility_scaling=True
    )

    # Calculate volatility
    returns = btc_data['close'].pct_change().dropna()
    current_vol = returns.tail(24).std() * (24 ** 0.5) * (365 ** 0.5)  # Annualized

    position = position_sizer.calculate_position(
        signal=signal,
        capital=100000,
        current_price=btc_data['close'].iloc[-1],
        volatility=current_vol
    )

    print(f"  Signal: {signal.signal_type.value}")
    print(f"  Current BTC Price: ${position['entry_price']:.2f}")
    print(f"  Annualized Volatility: {current_vol:.1%}")
    print(f"  Position Size: ${position['position_dollars']:.2f} ({position['position_pct']:.1%} of capital)")
    print(f"  BTC Units: {position['position_units']:.6f}")
    if position['stop_loss_price']:
        print(f"  Stop Loss Price: ${position['stop_loss_price']:.2f}")
    print()

    # Step 6: Run backtest
    print("Step 6: Running backtest on BTC...")
    backtester = RegimeBacktester(
        initial_capital=100000,
        commission=0.001,
        slippage=0.001  # Higher slippage for crypto
    )

    prices = btc_data['close'].iloc[window:]
    backtest_result = backtester.run(prices, regime_history, signal_gen)

    print(f"  Total Return: {backtest_result.total_return:.2%}")
    print(f"  Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {backtest_result.max_drawdown:.2%}")
    print(f"  Win Rate: {backtest_result.win_rate:.1%}")
    print(f"  Number of Trades: {backtest_result.num_trades}")
    print()

    # Step 7: Compare BTC vs ETH regimes
    print("Step 7: Comparing BTC vs ETH regimes...")

    # Get ETH regime history
    eth_history = []
    for i in range(window, min(len(eth_data), len(btc_data))):
        window_data = eth_data.iloc[i-window:i+1]
        eth_result = eth_classifier.classify(window_data)
        eth_history.append(eth_result)

    # Calculate agreement
    min_len = min(len(regime_history), len(eth_history))
    agreement = sum(
        1 for i in range(min_len)
        if regime_history[i].regime == eth_history[i].regime
    ) / min_len if min_len > 0 else 0

    print(f"  BTC-ETH Regime Agreement: {agreement:.1%}")
    print()

    # Cross-correlation analysis
    btc_regimes = [r.regime.value for r in regime_history[:min_len]]
    eth_regimes = [r.regime.value for r in eth_history[:min_len]]

    # Count disagreements
    disagreements = []
    for i in range(min_len):
        if btc_regimes[i] != eth_regimes[i]:
            disagreements.append({
                'index': i,
                'btc': btc_regimes[i],
                'eth': eth_regimes[i]
            })

    print(f"  Number of disagreements: {len(disagreements)}")
    if len(disagreements) > 0 and len(disagreements) <= 5:
        print("  Recent disagreements:")
        for d in disagreements[-5:]:
            print(f"    Period {d['index']}: BTC={d['btc']}, ETH={d['eth']}")
    print()

    # Step 8: Generate report
    print("Step 8: Generating evaluation report...")
    report = generate_report(regime_history, prices)
    print()
    print(report)

    print()
    print("=" * 60)
    print("Crypto regime classification example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
