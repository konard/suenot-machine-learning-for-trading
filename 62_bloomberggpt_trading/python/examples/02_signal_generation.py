"""
Example 02: Trading Signal Generation from LLM Analysis

This script demonstrates how to convert LLM sentiment analysis
into actionable trading signals.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockSentimentAnalyzer:
    """Mock analyzer for demonstration when transformers not available."""

    class MockResult:
        def __init__(self, label, confidence):
            self.label = label
            self.confidence = confidence
            self.positive_prob = 0.8 if label == "POSITIVE" else 0.1
            self.negative_prob = 0.8 if label == "NEGATIVE" else 0.1
            self.neutral_prob = 1 - self.positive_prob - self.negative_prob

    def analyze(self, text):
        text_lower = text.lower()
        if any(w in text_lower for w in ["record", "beat", "surge", "growth", "strong"]):
            return self.MockResult("POSITIVE", 0.85)
        elif any(w in text_lower for w in ["drop", "plunge", "miss", "decline", "warning"]):
            return self.MockResult("NEGATIVE", 0.80)
        else:
            return self.MockResult("NEUTRAL", 0.60)

    def analyze_aspects(self, text, entities):
        return {e: self.analyze(text) for e in entities}


def main():
    """Run signal generation demonstration."""
    print("=" * 60)
    print("Trading Signal Generation Demo")
    print("=" * 60)

    # Sample news items
    news_items = [
        {
            "text": "Apple reports record iPhone sales in Q4, beating expectations by 15%",
            "symbol": "AAPL",
            "source_type": "earnings",
            "timestamp": datetime.now() - timedelta(hours=2)
        },
        {
            "text": "Tesla stock drops 8% after disappointing delivery numbers",
            "symbol": "TSLA",
            "source_type": "news",
            "timestamp": datetime.now() - timedelta(hours=5)
        },
        {
            "text": "Microsoft Azure growth exceeds expectations, cloud revenue up 29%",
            "symbol": "MSFT",
            "source_type": "earnings",
            "timestamp": datetime.now() - timedelta(hours=1)
        },
        {
            "text": "NVIDIA faces supply constraints but demand remains strong",
            "symbol": "NVDA",
            "source_type": "news",
            "timestamp": datetime.now() - timedelta(hours=8)
        },
        {
            "text": "Fed maintains interest rates, signals potential cuts in Q2",
            "symbol": "SPY",
            "source_type": "news",
            "timestamp": datetime.now() - timedelta(hours=3)
        }
    ]

    # Initialize components
    try:
        from sentiment import FinancialSentimentAnalyzer
        analyzer = FinancialSentimentAnalyzer()
        print("\nUsing FinancialSentimentAnalyzer (FinBERT)")
    except ImportError:
        analyzer = MockSentimentAnalyzer()
        print("\nUsing MockSentimentAnalyzer (transformers not installed)")

    from signals import LLMSignalGenerator, TradingSignal

    generator = LLMSignalGenerator(
        sentiment_analyzer=analyzer,
        signal_threshold=0.2,
        confidence_threshold=0.5
    )

    print("\n" + "-" * 60)
    print("Processing News Items")
    print("-" * 60)

    signals = []
    for item in news_items:
        print(f"\n{item['source_type'].upper()}: {item['text'][:60]}...")

        signal = generator.generate_signal(
            text=item["text"],
            symbol=item["symbol"],
            source_type=item["source_type"],
            timestamp=item["timestamp"]
        )

        if signal:
            signals.append(signal)
            print(f"  -> {signal.symbol}: {signal.signal_type.value}")
            print(f"     Signal: {signal.signal:+.2f}, Confidence: {signal.confidence:.2%}")
            print(f"     {signal.reasoning}")
        else:
            print(f"  -> No actionable signal (below threshold)")

    # Aggregate signals
    print("\n" + "-" * 60)
    print("Aggregated Position Recommendations")
    print("-" * 60)

    positions = generator.aggregate_signals(signals, window_hours=24)

    if positions:
        for symbol, position in sorted(positions.items(), key=lambda x: -abs(x[1])):
            direction = "LONG" if position > 0 else "SHORT" if position < 0 else "FLAT"
            strength = "Strong" if abs(position) > 0.6 else "Moderate" if abs(position) > 0.3 else "Weak"
            print(f"  {symbol}: {strength} {direction} ({position:+.2f})")
    else:
        print("  No positions recommended")

    # Show signal summary
    print("\n" + "-" * 60)
    print("Signal Summary")
    print("-" * 60)
    print(f"  Total news items: {len(news_items)}")
    print(f"  Actionable signals: {len(signals)}")
    print(f"  Unique symbols: {len(set(s.symbol for s in signals))}")

    from signals import signals_to_dataframe
    if signals:
        df = signals_to_dataframe(signals)
        print(f"\nSignal DataFrame:")
        print(df[["timestamp", "symbol", "signal", "confidence", "signal_type"]].to_string())

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
