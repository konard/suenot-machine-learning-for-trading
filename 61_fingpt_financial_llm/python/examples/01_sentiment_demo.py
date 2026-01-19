#!/usr/bin/env python3
"""
FinGPT Sentiment Analysis Demo

This example demonstrates how to use FinGPT for financial sentiment analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fingpt_sentiment import FinGPTSentimentAnalyzer


def main():
    """Demo of FinGPT sentiment analysis."""
    # Initialize analyzer (use_mock=True for demo without model)
    analyzer = FinGPTSentimentAnalyzer(use_mock=True)

    # Test cases covering various financial scenarios
    test_texts = [
        # Positive news
        "Apple reported record quarterly revenue of $123.9 billion, beating analyst expectations by 5%.",
        "NVIDIA stock surged to new all-time highs amid strong AI chip demand.",
        "Microsoft cloud revenue growth exceeded expectations, driving stock higher.",
        "Amazon announces 20% increase in Prime membership, shares rally.",

        # Negative news
        "Tesla shares plunged 12% after disappointing delivery numbers.",
        "Bitcoin dropped below $40,000 as regulatory concerns mount.",
        "Meta faces antitrust lawsuit, stock falls 8% in after-hours trading.",
        "Bank announces major layoffs amid declining profits.",

        # Neutral news
        "The Federal Reserve kept interest rates unchanged at the latest meeting.",
        "Apple is expected to announce new products at next month's event.",
        "Trading volume remains steady as markets await earnings reports.",
    ]

    print("=" * 70)
    print("FinGPT Financial Sentiment Analysis Demo")
    print("=" * 70)

    # Analyze each text
    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze(text)

        print(f"\n[{i}] Text: {text[:65]}{'...' if len(text) > 65 else ''}")
        print(f"    Sentiment: {result.sentiment.upper()}")
        print(f"    Confidence: {result.confidence:.1%}")
        print(f"    Score: {result.score:+.2f}")

    # Demonstrate aspect-based sentiment analysis
    print("\n" + "=" * 70)
    print("Aspect-Based Sentiment Analysis")
    print("=" * 70)

    multi_entity_text = """
    In today's market session, Apple reported strong iPhone sales growth,
    beating expectations. However, Microsoft warned of cloud slowdown,
    causing concern among investors. Meanwhile, Tesla announced
    record deliveries but faces margin pressure from price cuts.
    """

    entities = ["Apple", "Microsoft", "Tesla"]
    aspect_results = analyzer.analyze_aspects(multi_entity_text, entities)

    print(f"\nText: {multi_entity_text.strip()[:100]}...")
    print("\nSentiment by Entity:")
    for entity, result in aspect_results.items():
        print(f"  {entity}: {result.sentiment.upper()} "
              f"(confidence: {result.confidence:.1%}, score: {result.score:+.2f})")


if __name__ == "__main__":
    main()
