#!/usr/bin/env python3
"""
Example 01: Data Loading Demo

This example demonstrates how to load market data from various sources
including Bybit for cryptocurrency data.

Run with: python 01_data_loading_demo.py
"""

import sys
sys.path.insert(0, '..')

from data_loader import (
    DataLoader,
    BybitLoader,
    YahooFinanceLoader,
    generate_synthetic_data,
    calculate_features,
    combine_prices
)


def main():
    print("=" * 60)
    print("LLM Alpha Mining - Data Loading Demo")
    print("=" * 60)

    # 1. Generate synthetic data (no API calls needed)
    print("\n1. SYNTHETIC DATA")
    print("-" * 40)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    synthetic_data = generate_synthetic_data(symbols, days=180, seed=42)

    for symbol, market_data in synthetic_data.items():
        df = market_data.ohlcv
        ret = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        print(f"\n{symbol} ({market_data.source}):")
        print(f"  Period: {market_data.start_date.date()} to {market_data.end_date.date()}")
        print(f"  Records: {len(df)}")
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"  Total return: {ret:+.2f}%")

    # 2. Calculate technical features
    print("\n2. TECHNICAL FEATURES")
    print("-" * 40)

    btc_data = synthetic_data["BTCUSDT"].ohlcv
    features = calculate_features(btc_data)

    print(f"\nCalculated {len(features.columns)} features for BTCUSDT:")
    print(f"Features: {', '.join(features.columns[:10])}...")

    # Show recent feature values
    print("\nRecent feature values (last 5 rows):")
    display_cols = ['close', 'return', 'momentum_5', 'rsi_14', 'volume_ratio']
    print(features[display_cols].tail().round(4).to_string())

    # 3. Combine multiple assets
    print("\n3. COMBINED PRICE MATRIX")
    print("-" * 40)

    combined = combine_prices(synthetic_data)
    print(f"\nCombined matrix shape: {combined.shape}")

    # Calculate correlations
    returns = combined.pct_change().dropna()
    correlations = returns.corr()

    print("\nReturn correlations:")
    print(correlations.round(3).to_string())

    # 4. Using the unified DataLoader
    print("\n4. UNIFIED DATA LOADER")
    print("-" * 40)

    loader = DataLoader()

    # Auto-detect source based on symbol
    print("\nAuto-detection examples:")
    print("  'BTCUSDT' -> would use: bybit (crypto)")
    print("  'AAPL' -> would use: yahoo (stock)")
    print("  'ETHPERP' -> would use: bybit (crypto)")

    # 5. Live data example (commented out to avoid API calls)
    print("\n5. LIVE DATA (Example Code)")
    print("-" * 40)

    print("""
# To fetch live Bybit data:
from data_loader import BybitLoader

loader = BybitLoader()
btc = loader.load("BTCUSDT", interval="60", days=30)
print(f"Loaded {len(btc.ohlcv)} hourly candles")

# To fetch live Yahoo Finance data:
from data_loader import YahooFinanceLoader

loader = YahooFinanceLoader()
aapl = loader.load("AAPL", period="1y")
print(f"Loaded {len(aapl.ohlcv)} daily candles")
""")

    # 6. Feature statistics
    print("\n6. FEATURE STATISTICS")
    print("-" * 40)

    stats = features.describe()
    print("\nFeature statistics (selected):")
    stat_cols = ['close', 'volume', 'rsi_14', 'macd', 'volatility_20']
    print(stats[stat_cols].round(4).to_string())

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
