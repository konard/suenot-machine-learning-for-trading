#!/usr/bin/env python3
"""
FinGPT Trading Signal Generation Demo

This example demonstrates how to generate trading signals from
financial news using FinGPT sentiment analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fingpt_sentiment import FinGPTSentimentAnalyzer
from signals import FinGPTTradingEngine, TradingSignal


def main():
    """Demo of trading signal generation from news."""
    # Initialize components
    analyzer = FinGPTSentimentAnalyzer(use_mock=True)
    engine = FinGPTTradingEngine(analyzer)

    # Simulated news feed for different assets
    news_feed = {
        "AAPL": [
            {
                "text": "Apple reports record iPhone 15 sales in Q4, beating analyst expectations by 12%.",
                "source": "reuters"
            },
            {
                "text": "Apple announces major expansion of AI features in upcoming iOS update.",
                "source": "bloomberg"
            },
            {
                "text": "Concerns raised about Apple's China market share declining.",
                "source": "twitter"
            },
        ],
        "TSLA": [
            {
                "text": "Tesla Cybertruck deliveries begin, overwhelming demand reported.",
                "source": "reuters"
            },
            {
                "text": "Elon Musk sells $3 billion worth of Tesla shares.",
                "source": "sec_filing"
            },
            {
                "text": "Tesla faces increased competition from Chinese EV makers.",
                "source": "bloomberg"
            },
        ],
        "BTCUSDT": [
            {
                "text": "Bitcoin ETF approval expected imminently, institutional interest surges.",
                "source": "bloomberg"
            },
            {
                "text": "Major bank announces Bitcoin custody services for institutional clients.",
                "source": "reuters"
            },
            {
                "text": "Regulatory concerns in Europe could impact crypto trading.",
                "source": "twitter"
            },
        ],
    }

    print("=" * 70)
    print("FinGPT Trading Signal Generation Demo")
    print("=" * 70)

    # Process each asset
    all_signals = []

    for symbol, news_items in news_feed.items():
        print(f"\n{'='*70}")
        print(f"Processing: {symbol}")
        print("=" * 70)

        # Show individual news analysis
        print("\nNews Analysis:")
        for i, item in enumerate(news_items, 1):
            result = analyzer.analyze(item["text"])
            print(f"\n  [{i}] Source: {item['source'].upper()}")
            print(f"      Text: {item['text'][:60]}...")
            print(f"      Sentiment: {result.sentiment.upper()} "
                  f"(score: {result.score:+.2f}, confidence: {result.confidence:.1%})")

        # Generate aggregated signal
        signal = engine.generate_signal(symbol, news_items)
        all_signals.append(signal)

        print(f"\n  AGGREGATED SIGNAL:")
        print(f"    Symbol: {signal.symbol}")
        print(f"    Action: {signal.action.upper()}")
        print(f"    Strength: {signal.strength:.2f}")
        print(f"    Confidence: {signal.confidence:.1%}")
        print(f"    Reason: {signal.reason}")

    # Summary of all signals
    print("\n" + "=" * 70)
    print("TRADING SIGNALS SUMMARY")
    print("=" * 70)

    # Sort by strength (absolute value)
    sorted_signals = sorted(all_signals, key=lambda s: abs(s.strength), reverse=True)

    print("\n{:<10} {:<8} {:<10} {:<12} {}".format(
        "Symbol", "Action", "Strength", "Confidence", "Recommendation"
    ))
    print("-" * 70)

    for signal in sorted_signals:
        recommendation = ""
        if signal.action == "buy" and signal.strength > 0.5:
            recommendation = "Strong buy opportunity"
        elif signal.action == "buy":
            recommendation = "Consider buying"
        elif signal.action == "sell" and signal.strength < -0.5:
            recommendation = "Strong sell signal"
        elif signal.action == "sell":
            recommendation = "Consider selling"
        else:
            recommendation = "Monitor closely"

        print(f"{signal.symbol:<10} {signal.action.upper():<8} {signal.strength:+.2f}      "
              f"{signal.confidence:.0%}          {recommendation}")

    # Demonstrate position sizing
    print("\n" + "=" * 70)
    print("POSITION SIZING EXAMPLE")
    print("=" * 70)

    portfolio_value = 100000  # $100,000 portfolio
    max_position_pct = 0.10  # Max 10% per position

    print(f"\nPortfolio Value: ${portfolio_value:,.2f}")
    print(f"Max Position Size: {max_position_pct:.0%}")
    print("\nSuggested Allocations:")

    for signal in sorted_signals:
        if signal.action == "hold":
            continue

        # Scale position by signal strength and confidence
        position_pct = max_position_pct * abs(signal.strength) * signal.confidence
        position_value = portfolio_value * position_pct

        action_str = "BUY" if signal.action == "buy" else "SELL"
        print(f"  {signal.symbol}: {action_str} ${position_value:,.2f} "
              f"({position_pct:.1%} of portfolio)")


if __name__ == "__main__":
    main()
