#!/usr/bin/env python3
"""
Example 03: Multi-Asset Regime Classification

This example demonstrates how to classify market regimes
across multiple assets (stocks and crypto) and detect
cross-market regime relationships.

Usage:
    python 03_multi_asset_regime.py
"""

import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from python.data_loader import YahooFinanceLoader, BybitDataLoader
from python.classifier import (
    StatisticalRegimeClassifier,
    CryptoRegimeClassifier,
    HybridRegimeClassifier,
    MarketRegime,
    RegimeResult
)
from python.signals import RegimeSignalGenerator, RegimeTransitionDetector
from python.evaluate import RegimeEvaluator


def main():
    """Run multi-asset regime classification example."""
    print("=" * 60)
    print("Multi-Asset Regime Classification Example")
    print("=" * 60)
    print()

    # Step 1: Load multiple asset data
    print("Step 1: Loading multi-asset data...")

    stock_loader = YahooFinanceLoader()
    crypto_loader = BybitDataLoader()

    assets = {}

    # Load stock data
    stock_symbols = ["SPY", "QQQ", "IWM"]  # S&P 500, Nasdaq, Russell 2000
    for symbol in stock_symbols:
        try:
            data = stock_loader.get_daily(symbol, period="1y")
            assets[symbol] = {"data": data, "type": "stock"}
            print(f"  Loaded {symbol}: {len(data)} days")
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            data = stock_loader._generate_mock_data(symbol, 252, "1d")
            assets[symbol] = {"data": data, "type": "stock"}
            print(f"  Using mock data for {symbol}")

    # Load crypto data (daily aggregation from hourly)
    crypto_symbols = ["BTCUSDT", "ETHUSDT"]
    for symbol in crypto_symbols:
        try:
            # Get hourly data and resample to daily
            hourly = crypto_loader.get_klines(symbol, interval="1h", limit=1000)
            # Resample to daily
            daily = hourly.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            assets[symbol] = {"data": daily, "type": "crypto"}
            print(f"  Loaded {symbol}: {len(daily)} days")
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            data = crypto_loader._generate_mock_crypto_data(symbol, 252, "1d")
            assets[symbol] = {"data": data, "type": "crypto"}
            print(f"  Using mock data for {symbol}")

    print()

    # Step 2: Initialize classifiers for each asset type
    print("Step 2: Initializing classifiers...")

    stock_classifier = StatisticalRegimeClassifier(
        lookback_window=20,
        volatility_threshold_high=0.25,
        trend_threshold=0.0005
    )

    crypto_classifier = CryptoRegimeClassifier(
        lookback=20,
        volatility_threshold=0.50,
        trend_threshold=0.02
    )

    print("  Stock classifier: 20-day window, 2% vol threshold")
    print("  Crypto classifier: 20-day window, 5% vol threshold")
    print()

    # Step 3: Classify each asset
    print("Step 3: Classifying current regimes for all assets...")

    current_regimes = {}
    for symbol, info in assets.items():
        data = info["data"]
        asset_type = info["type"]

        if asset_type == "stock":
            classifier = stock_classifier
        else:
            classifier = crypto_classifier

        classifier.fit(data)
        result = classifier.classify(data)
        current_regimes[symbol] = result

        print(f"  {symbol:10s}: {result.regime.value:15s} (confidence: {result.confidence:.1%})")
    print()

    # Step 4: Analyze cross-asset regime relationships
    print("Step 4: Analyzing cross-asset regime relationships...")

    # Group by regime
    regime_groups = {}
    for symbol, result in current_regimes.items():
        regime = result.regime
        if regime not in regime_groups:
            regime_groups[regime] = []
        regime_groups[regime].append(symbol)

    print("  Assets grouped by current regime:")
    for regime, symbols in regime_groups.items():
        print(f"    {regime.value}: {', '.join(symbols)}")
    print()

    # Calculate regime agreement
    regimes = [r.regime for r in current_regimes.values()]
    most_common = Counter(regimes).most_common(1)[0]
    agreement = most_common[1] / len(regimes)
    print(f"  Market regime agreement: {agreement:.1%}")
    print(f"  Dominant regime: {most_common[0].value}")
    print()

    # Step 5: Generate historical regime matrices
    print("Step 5: Generating historical regime analysis...")

    window = 20
    history_length = 100  # Analyze last 100 days

    regime_histories = {}
    for symbol, info in assets.items():
        data = info["data"]
        asset_type = info["type"]

        if asset_type == "stock":
            classifier = StatisticalRegimeClassifier(lookback_window=window)
        else:
            classifier = CryptoRegimeClassifier(lookback=window)

        classifier.fit(data)

        history = []
        start_idx = max(window, len(data) - history_length)
        for i in range(start_idx, len(data)):
            window_data = data.iloc[i-window:i+1]
            result = classifier.classify(window_data)
            history.append(result)

        regime_histories[symbol] = history

    # Calculate pairwise agreement
    print("  Pairwise regime agreement matrix:")
    symbols = list(assets.keys())
    print(f"    {'':10s}", end="")
    for s in symbols:
        print(f"{s:10s}", end="")
    print()

    for s1 in symbols:
        print(f"    {s1:10s}", end="")
        for s2 in symbols:
            if s1 == s2:
                print(f"{'--':10s}", end="")
            else:
                min_len = min(len(regime_histories[s1]), len(regime_histories[s2]))
                agree = sum(
                    1 for i in range(min_len)
                    if regime_histories[s1][i].regime == regime_histories[s2][i].regime
                ) / min_len if min_len > 0 else 0
                print(f"{agree:.0%}".ljust(10), end="")
        print()
    print()

    # Step 6: Detect regime transitions
    print("Step 6: Detecting regime transitions...")

    for symbol, history in regime_histories.items():
        detector = RegimeTransitionDetector(
            confirmation_periods=3,
            hysteresis_threshold=0.1
        )

        transitions = []
        for result in history:
            changed, new_regime = detector.update(result)
            if changed:
                transitions.append(new_regime)

        print(f"  {symbol}: {len(transitions)} regime transitions detected")
    print()

    # Step 7: Generate portfolio signals based on consensus
    print("Step 7: Generating portfolio signals based on regime consensus...")

    signal_gen = RegimeSignalGenerator(confidence_threshold=0.6)

    # Create consensus regime
    latest_regimes = [h[-1].regime for h in regime_histories.values() if h]
    consensus_regime = Counter(latest_regimes).most_common(1)[0][0]
    consensus_confidence = sum(
        h[-1].confidence for h in regime_histories.values() if h
    ) / len(regime_histories)

    consensus_result = RegimeResult(
        regime=consensus_regime,
        probability=consensus_confidence,
        confidence=consensus_confidence,
        explanation=f"Consensus across {len(assets)} assets",
        supporting_factors=[f"{symbol}: {h[-1].regime.value}" for symbol, h in regime_histories.items() if h]
    )

    signal = signal_gen.generate_signal(consensus_result)

    print(f"  Consensus Regime: {consensus_regime.value}")
    print(f"  Average Confidence: {consensus_confidence:.1%}")
    print(f"  Portfolio Signal: {signal.signal_type.value}")
    print(f"  Suggested Position: {signal.position_size:.1%}")
    print()

    # Step 8: Risk analysis by regime
    print("Step 8: Analyzing risk by market regime...")

    # Calculate returns for each asset
    for symbol, info in assets.items():
        data = info["data"]
        returns = data['close'].pct_change().dropna()

        # Get regime at each point
        history = regime_histories[symbol]

        # Align returns with regimes
        aligned_returns = returns.iloc[-len(history):]

        print(f"  {symbol}:")
        for regime in MarketRegime:
            regime_returns = [
                aligned_returns.iloc[i]
                for i in range(len(history))
                if history[i].regime == regime and i < len(aligned_returns)
            ]

            if regime_returns:
                avg_ret = np.mean(regime_returns) * 252  # Annualized
                vol = np.std(regime_returns) * np.sqrt(252)  # Annualized
                print(f"    {regime.value:15s}: Return={avg_ret:+.1%}, Vol={vol:.1%}")
        print()

    # Step 9: Summary
    print("=" * 60)
    print("MULTI-ASSET REGIME SUMMARY")
    print("=" * 60)
    print()

    print("Current Market State:")
    print(f"  Dominant Regime: {consensus_regime.value}")
    print(f"  Agreement Level: {agreement:.1%}")
    print()

    print("Recommended Actions:")
    if consensus_regime == MarketRegime.BULL:
        print("  - Markets aligned in BULL regime")
        print("  - Consider increasing equity exposure")
        print("  - Trend-following strategies may work well")
    elif consensus_regime == MarketRegime.BEAR:
        print("  - Markets aligned in BEAR regime")
        print("  - Consider reducing risk exposure")
        print("  - Defensive positioning recommended")
    elif consensus_regime == MarketRegime.HIGH_VOLATILITY:
        print("  - High volatility across markets")
        print("  - Reduce position sizes")
        print("  - Wait for clearer direction")
    elif consensus_regime == MarketRegime.SIDEWAYS:
        print("  - Markets in sideways consolidation")
        print("  - Range-trading strategies may work")
        print("  - Avoid strong directional bets")
    elif consensus_regime == MarketRegime.CRISIS:
        print("  - CRISIS regime detected!")
        print("  - Maximum defensive positioning")
        print("  - Consider safe haven assets")
    print()

    if agreement < 0.5:
        print("  NOTE: Low agreement between assets")
        print("  Markets may be transitioning between regimes")
        print("  Consider asset-specific strategies")
    print()

    print("=" * 60)
    print("Multi-asset regime analysis completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
