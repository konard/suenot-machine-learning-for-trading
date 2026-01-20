#!/usr/bin/env python3
"""
Example 1: Basic Anomaly Detection

This example demonstrates basic anomaly detection on stock and crypto data
using the StatisticalAnomalyDetector.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import YahooFinanceLoader, BybitDataLoader
from detector import StatisticalAnomalyDetector, AnomalyType


def detect_stock_anomalies():
    """Detect anomalies in stock market data."""
    print("=" * 60)
    print("STOCK MARKET ANOMALY DETECTION")
    print("=" * 60)

    # Load SPY data
    loader = YahooFinanceLoader()
    data = loader.get_ohlcv("SPY", interval="1d", limit=500)

    if data.empty:
        print("Failed to load stock data")
        return

    # Add features
    data = loader.compute_features(data)
    print(f"Loaded {len(data)} days of SPY data")

    # Create and train detector
    detector = StatisticalAnomalyDetector(
        z_threshold=2.5,
        contamination=0.05,
        methods=["zscore", "isolation_forest"],
    )

    # Train on first 400 days
    train_data = data.iloc[:400]
    test_data = data.iloc[400:]

    detector.fit(train_data)
    print(f"Trained on {len(train_data)} days")

    # Detect anomalies in test data
    print(f"\nAnalyzing last {len(test_data)} days...")
    anomalies_found = 0

    for idx, row in test_data.iterrows():
        result = detector.detect_single(row)

        if result.is_anomaly:
            anomalies_found += 1
            date = row.get("timestamp", idx)
            print(f"\nANOMALY on {date}:")
            print(f"  Type: {result.anomaly_type.value}")
            print(f"  Score: {result.score:.3f}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Explanation: {result.explanation}")

            # Show market data
            print(f"  Close: ${row.get('close', 0):.2f}")
            if "returns" in row:
                print(f"  Return: {row['returns']*100:.2f}%")
            if "volume_ratio" in row:
                print(f"  Volume ratio: {row['volume_ratio']:.2f}x")

    print(f"\nTotal anomalies found: {anomalies_found} / {len(test_data)}")


def detect_crypto_anomalies():
    """Detect anomalies in cryptocurrency data from Bybit."""
    print("\n" + "=" * 60)
    print("CRYPTOCURRENCY ANOMALY DETECTION (Bybit)")
    print("=" * 60)

    # Load BTC data from Bybit
    loader = BybitDataLoader()
    data = loader.get_ohlcv("BTCUSDT", interval="1h", limit=500)

    if data.empty:
        print("Failed to load crypto data")
        return

    # Add features
    data = loader.compute_features(data)
    print(f"Loaded {len(data)} hours of BTCUSDT data")

    # Create detector with crypto-specific settings
    detector = StatisticalAnomalyDetector(
        z_threshold=3.0,  # Higher threshold for volatile crypto
        contamination=0.03,
        methods=["zscore", "isolation_forest", "lof"],
    )

    # Train on first 400 hours
    train_data = data.iloc[:400]
    test_data = data.iloc[400:]

    detector.fit(train_data)
    print(f"Trained on {len(train_data)} hours")

    # Detect anomalies
    print(f"\nAnalyzing last {len(test_data)} hours...")
    anomalies_found = 0

    for idx, row in test_data.iterrows():
        result = detector.detect_single(row)

        if result.is_anomaly:
            anomalies_found += 1
            timestamp = row.get("timestamp", idx)
            print(f"\nANOMALY at {timestamp}:")
            print(f"  Type: {result.anomaly_type.value}")
            print(f"  Score: {result.score:.3f}")
            print(f"  Explanation: {result.explanation}")
            print(f"  BTC Price: ${row.get('close', 0):,.2f}")

    print(f"\nTotal anomalies found: {anomalies_found} / {len(test_data)}")


def compare_multiple_symbols():
    """Compare anomaly detection across multiple symbols."""
    print("\n" + "=" * 60)
    print("MULTI-SYMBOL COMPARISON")
    print("=" * 60)

    # Bybit symbols to analyze
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    loader = BybitDataLoader()

    results = {}

    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        data = loader.get_ohlcv(symbol, interval="4h", limit=200)

        if data.empty:
            print(f"  Failed to load {symbol}")
            continue

        data = loader.compute_features(data)

        # Train and detect
        detector = StatisticalAnomalyDetector(z_threshold=2.5)
        detector.fit(data.iloc[:150])

        # Count anomalies in last 50 candles
        anomaly_count = 0
        for _, row in data.iloc[150:].iterrows():
            result = detector.detect_single(row)
            if result.is_anomaly:
                anomaly_count += 1

        results[symbol] = {
            "total_candles": 50,
            "anomalies": anomaly_count,
            "anomaly_rate": anomaly_count / 50 * 100,
        }

        print(f"  Anomalies: {anomaly_count}/50 ({results[symbol]['anomaly_rate']:.1f}%)")

    # Summary
    print("\n" + "-" * 40)
    print("Summary:")
    for symbol, stats in results.items():
        print(f"  {symbol}: {stats['anomaly_rate']:.1f}% anomaly rate")


if __name__ == "__main__":
    print("LLM Anomaly Detection - Basic Example")
    print("=" * 60)

    # Run examples
    detect_stock_anomalies()
    detect_crypto_anomalies()
    compare_multiple_symbols()

    print("\n" + "=" * 60)
    print("Example completed!")
