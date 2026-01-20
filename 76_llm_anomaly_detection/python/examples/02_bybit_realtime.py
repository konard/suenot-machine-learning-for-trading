#!/usr/bin/env python3
"""
Example 2: Real-time Anomaly Detection for Bybit

This example demonstrates how to monitor Bybit markets for anomalies
in real-time, useful for detecting market manipulation or unusual activity.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import BybitDataLoader
from detector import StatisticalAnomalyDetector, AnomalyResult, AnomalyType
from signals import AnomalySignalGenerator, SignalType


class BybitAnomalyMonitor:
    """
    Real-time anomaly monitor for Bybit markets.

    Continuously monitors specified symbols and alerts on anomalies.
    """

    def __init__(
        self,
        symbols: list,
        interval: str = "15m",
        z_threshold: float = 2.5,
    ):
        """
        Initialize monitor.

        Args:
            symbols: List of trading pairs to monitor
            interval: Candle interval
            z_threshold: Z-score threshold for anomalies
        """
        self.symbols = symbols
        self.interval = interval
        self.loader = BybitDataLoader()

        # Create detector for each symbol
        self.detectors: Dict[str, StatisticalAnomalyDetector] = {}
        self.signal_generators: Dict[str, AnomalySignalGenerator] = {}
        self.last_data: Dict[str, Any] = {}

        for symbol in symbols:
            self.detectors[symbol] = StatisticalAnomalyDetector(
                z_threshold=z_threshold,
                methods=["zscore", "isolation_forest"],
            )
            self.signal_generators[symbol] = AnomalySignalGenerator(
                strategy="risk",
                min_anomaly_score=0.6,
            )

    def initialize(self, history_limit: int = 500):
        """
        Initialize detectors with historical data.

        Args:
            history_limit: Number of historical candles to load
        """
        print("Initializing anomaly detectors...")

        for symbol in self.symbols:
            print(f"  Loading history for {symbol}...")
            data = self.loader.get_ohlcv(
                symbol,
                interval=self.interval,
                limit=history_limit,
            )

            if data.empty:
                print(f"    Failed to load {symbol}")
                continue

            # Add features and fit detector
            data = self.loader.compute_features(data)
            self.detectors[symbol].fit(data)
            self.last_data[symbol] = data

            print(f"    Trained on {len(data)} candles")

        print("Initialization complete!")

    def check_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Check a single symbol for anomalies.

        Args:
            symbol: Trading pair

        Returns:
            Dictionary with analysis results
        """
        # Get latest data
        data = self.loader.get_ohlcv(
            symbol,
            interval=self.interval,
            limit=50,
        )

        if data.empty:
            return {"error": "Failed to load data"}

        data = self.loader.compute_features(data)
        latest = data.iloc[-1]

        # Detect anomaly
        detector = self.detectors.get(symbol)
        if detector is None:
            return {"error": "Detector not initialized"}

        result = detector.detect_single(latest)

        # Generate signal
        signal_gen = self.signal_generators.get(symbol)
        signal = signal_gen.generate_signal(result, latest) if signal_gen else None

        # Get additional market data
        ticker = self.loader.get_ticker(symbol)

        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "price": latest.get("close"),
            "anomaly": {
                "is_anomaly": result.is_anomaly,
                "score": result.score,
                "type": result.anomaly_type.value,
                "explanation": result.explanation,
            },
            "signal": {
                "type": signal.signal_type.value if signal else "none",
                "confidence": signal.confidence if signal else 0,
                "reason": signal.reason if signal else "",
            },
            "market": {
                "24h_change": ticker.get("price_change_pct", 0),
                "24h_volume": ticker.get("volume_24h", 0),
            },
        }

    def monitor(self, check_interval: int = 60, max_checks: int = 10):
        """
        Run continuous monitoring loop.

        Args:
            check_interval: Seconds between checks
            max_checks: Maximum number of check cycles (0 for infinite)
        """
        print("\n" + "=" * 60)
        print("STARTING REAL-TIME MONITORING")
        print("=" * 60)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Interval: {self.interval}")
        print(f"Check every: {check_interval} seconds")
        print("=" * 60)

        check_count = 0

        while max_checks == 0 or check_count < max_checks:
            check_count += 1
            print(f"\n--- Check #{check_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            for symbol in self.symbols:
                result = self.check_symbol(symbol)

                if "error" in result:
                    print(f"{symbol}: Error - {result['error']}")
                    continue

                anomaly = result["anomaly"]
                signal = result["signal"]

                # Format output
                status = "ANOMALY" if anomaly["is_anomaly"] else "NORMAL"
                color_start = "\033[91m" if anomaly["is_anomaly"] else "\033[92m"  # Red/Green
                color_end = "\033[0m"

                print(f"\n{symbol}:")
                print(f"  Status: {color_start}{status}{color_end}")
                print(f"  Price: ${result['price']:,.2f}")
                print(f"  24h Change: {result['market']['24h_change']:.2f}%")

                if anomaly["is_anomaly"]:
                    print(f"  Anomaly Score: {anomaly['score']:.3f}")
                    print(f"  Type: {anomaly['type']}")
                    print(f"  Explanation: {anomaly['explanation']}")
                    print(f"  Signal: {signal['type'].upper()}")
                    print(f"  Signal Reason: {signal['reason']}")

            if max_checks == 0 or check_count < max_checks:
                print(f"\nNext check in {check_interval} seconds...")
                time.sleep(check_interval)

        print("\n" + "=" * 60)
        print("Monitoring stopped")


def demo_single_check():
    """Demonstrate a single anomaly check."""
    print("=" * 60)
    print("SINGLE CHECK DEMO")
    print("=" * 60)

    monitor = BybitAnomalyMonitor(
        symbols=["BTCUSDT", "ETHUSDT"],
        interval="1h",
        z_threshold=2.5,
    )

    monitor.initialize(history_limit=300)

    print("\nRunning single check...")
    for symbol in monitor.symbols:
        result = monitor.check_symbol(symbol)

        print(f"\n{symbol} Analysis:")
        print(f"  Current Price: ${result.get('price', 0):,.2f}")

        anomaly = result.get("anomaly", {})
        print(f"  Is Anomaly: {anomaly.get('is_anomaly', False)}")
        print(f"  Anomaly Score: {anomaly.get('score', 0):.3f}")

        if anomaly.get("is_anomaly"):
            print(f"  Type: {anomaly.get('type')}")
            print(f"  Explanation: {anomaly.get('explanation')}")


def demo_monitoring_loop():
    """Demonstrate continuous monitoring."""
    print("\n" + "=" * 60)
    print("MONITORING LOOP DEMO")
    print("=" * 60)

    monitor = BybitAnomalyMonitor(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        interval="15m",
        z_threshold=2.5,
    )

    monitor.initialize(history_limit=200)

    # Run for 3 checks with 30 second intervals
    monitor.monitor(check_interval=30, max_checks=3)


def analyze_orderbook_imbalance():
    """Analyze orderbook for potential manipulation."""
    print("\n" + "=" * 60)
    print("ORDERBOOK ANALYSIS")
    print("=" * 60)

    loader = BybitDataLoader()
    symbols = ["BTCUSDT", "ETHUSDT"]

    for symbol in symbols:
        print(f"\n{symbol} Orderbook Analysis:")

        orderbook = loader.get_orderbook(symbol, limit=50)

        if not orderbook:
            print("  Failed to load orderbook")
            continue

        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            print("  Empty orderbook")
            continue

        # Calculate imbalance
        total_bid_volume = sum(b[1] for b in bids)
        total_ask_volume = sum(a[1] for a in asks)

        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

        # Detect large orders (potential manipulation)
        avg_order_size = (total_bid_volume + total_ask_volume) / (len(bids) + len(asks))
        large_bids = [b for b in bids if b[1] > avg_order_size * 5]
        large_asks = [a for a in asks if a[1] > avg_order_size * 5]

        print(f"  Best Bid: ${bids[0][0]:,.2f} ({bids[0][1]:.4f})")
        print(f"  Best Ask: ${asks[0][0]:,.2f} ({asks[0][1]:.4f})")
        print(f"  Spread: ${asks[0][0] - bids[0][0]:.2f}")
        print(f"  Order Imbalance: {imbalance:.2%} ({'bullish' if imbalance > 0 else 'bearish'})")
        print(f"  Large Bid Orders: {len(large_bids)}")
        print(f"  Large Ask Orders: {len(large_asks)}")

        # Check for potential spoofing (large orders far from mid)
        mid_price = (bids[0][0] + asks[0][0]) / 2
        suspicious_orders = []

        for bid in large_bids:
            if bid[0] < mid_price * 0.99:  # More than 1% below mid
                suspicious_orders.append(("BID", bid[0], bid[1]))

        for ask in large_asks:
            if ask[0] > mid_price * 1.01:  # More than 1% above mid
                suspicious_orders.append(("ASK", ask[0], ask[1]))

        if suspicious_orders:
            print("\n  Potential Spoofing Detected:")
            for side, price, size in suspicious_orders[:5]:
                print(f"    {side}: ${price:,.2f} x {size:.4f}")


if __name__ == "__main__":
    print("Bybit Real-time Anomaly Detection Example")
    print("=" * 60)

    # Run demos
    demo_single_check()
    analyze_orderbook_imbalance()

    # Uncomment to run continuous monitoring:
    # demo_monitoring_loop()

    print("\nExample completed!")
