#!/usr/bin/env python3
"""
Sentiment Analysis Demo

Demonstrates using LLM-based sentiment analysis for financial news.
Uses mock LLM client for demonstration - replace with real client for production.
"""

import asyncio
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_analysis import FinancialSentimentAnalyzer, Sentiment
from llm_client import MockLLMClient


async def basic_sentiment_demo():
    """Basic sentiment analysis demonstration."""
    print("=" * 60)
    print("Basic Sentiment Analysis Demo")
    print("=" * 60)

    # Create mock responses
    mock_responses = {
        "positive": json.dumps({
            "sentiment": "POSITIVE",
            "confidence": 85,
            "market_impact": "BULLISH",
            "time_horizon": "SHORT_TERM",
            "key_factors": ["earnings beat", "strong guidance", "buyback announcement"],
            "reasoning": "Company exceeded analyst expectations with strong revenue growth."
        }),
        "negative": json.dumps({
            "sentiment": "NEGATIVE",
            "confidence": 78,
            "market_impact": "BEARISH",
            "time_horizon": "MEDIUM_TERM",
            "key_factors": ["revenue miss", "lowered guidance", "increased competition"],
            "reasoning": "Disappointing results and outlook revision signal challenges ahead."
        }),
        "neutral": json.dumps({
            "sentiment": "NEUTRAL",
            "confidence": 65,
            "market_impact": "NEUTRAL",
            "time_horizon": "SHORT_TERM",
            "key_factors": ["inline results", "unchanged outlook"],
            "reasoning": "Results met expectations with no significant surprises."
        })
    }

    # Initialize with mock client
    client = MockLLMClient(responses=mock_responses)
    analyzer = FinancialSentimentAnalyzer(client)

    # Test headlines
    headlines = [
        ("AAPL", "Apple Reports Record Quarter, EPS Beats by 15%"),
        ("TSLA", "Tesla Misses Delivery Targets, Shares Fall 8%"),
        ("MSFT", "Microsoft Azure Revenue Grows 29%, In Line with Estimates")
    ]

    for symbol, headline in headlines:
        print(f"\nAnalyzing: {headline}")
        print("-" * 50)

        result = await analyzer.analyze(headline, symbol)

        print(f"Sentiment: {result.sentiment.value}")
        print(f"Confidence: {result.confidence}%")
        print(f"Market Impact: {result.market_impact}")
        print(f"Time Horizon: {result.time_horizon}")
        print(f"Key Factors: {', '.join(result.key_factors[:3])}")
        print(f"Reasoning: {result.reasoning[:100]}...")


async def aspect_based_demo():
    """Demonstrate aspect-based sentiment analysis."""
    print("\n" + "=" * 60)
    print("Aspect-Based Sentiment Analysis Demo")
    print("=" * 60)

    mock_responses = {
        "aspect": json.dumps({
            "overall_sentiment": "POSITIVE",
            "overall_confidence": 75,
            "aspects": {
                "revenue": {"sentiment": "POSITIVE", "confidence": 90},
                "margins": {"sentiment": "NEGATIVE", "confidence": 70},
                "guidance": {"sentiment": "POSITIVE", "confidence": 80},
                "competitive_position": {"sentiment": "NEUTRAL", "confidence": 65}
            },
            "summary": "Strong revenue offset by margin pressure"
        })
    }

    client = MockLLMClient(responses=mock_responses)
    analyzer = FinancialSentimentAnalyzer(client, prompt_type="aspect_based")

    earnings_text = """
    Nvidia reported Q4 earnings with revenue of $22.1 billion, up 265% year-over-year.
    Data center revenue reached $18.4 billion. However, gross margins contracted
    slightly to 76% from 78% last quarter due to increased production costs.
    The company raised next quarter guidance to $24 billion, ahead of analyst estimates.
    """

    print(f"\nAnalyzing earnings report...")
    result = await analyzer.analyze(earnings_text, "NVDA")

    print(f"\nOverall: {result.sentiment.value} ({result.confidence}% confidence)")
    print(f"Summary: {result.reasoning}")


async def batch_analysis_demo():
    """Demonstrate batch sentiment analysis."""
    print("\n" + "=" * 60)
    print("Batch Analysis Demo")
    print("=" * 60)

    mock_responses = {
        "batch": json.dumps({
            "sentiment": "POSITIVE",
            "confidence": 72,
            "market_impact": "BULLISH",
            "time_horizon": "SHORT_TERM",
            "key_factors": ["market trend"],
            "reasoning": "Positive market momentum"
        })
    }

    client = MockLLMClient(responses=mock_responses)
    analyzer = FinancialSentimentAnalyzer(client)

    news_items = [
        ("AAPL", "Apple announces new AI features for iPhone"),
        ("GOOGL", "Google Cloud wins major enterprise contract"),
        ("META", "Meta's Reality Labs shows smaller losses"),
        ("AMZN", "Amazon Prime membership reaches new high"),
        ("MSFT", "Microsoft Copilot adoption exceeds expectations")
    ]

    print("\nAnalyzing batch of news items...")
    results = await analyzer.analyze_batch(news_items, max_concurrent=3)

    print(f"\nProcessed {len(results)} items:")
    for symbol, result in results.items():
        emoji = "+" if result.sentiment == Sentiment.POSITIVE else \
                ("-" if result.sentiment == Sentiment.NEGATIVE else "=")
        print(f"  [{emoji}] {symbol}: {result.sentiment.value} ({result.confidence}%)")


async def self_consistency_demo():
    """Demonstrate self-consistency validation."""
    print("\n" + "=" * 60)
    print("Self-Consistency Validation Demo")
    print("=" * 60)

    # Mock with slight variations
    mock_responses = {
        "consistency": json.dumps({
            "sentiment": "POSITIVE",
            "confidence": 80,
            "market_impact": "BULLISH",
            "time_horizon": "MEDIUM_TERM",
            "key_factors": ["strong fundamentals"],
            "reasoning": "Solid results"
        })
    }

    client = MockLLMClient(responses=mock_responses)
    analyzer = FinancialSentimentAnalyzer(client)

    headline = "Federal Reserve signals potential rate cut in upcoming meeting"

    print(f"\nAnalyzing with self-consistency (3 samples):")
    print(f"Headline: {headline}")

    result = await analyzer.self_consistent_analyze(headline, num_samples=3)

    print(f"\nConsensus Sentiment: {result.sentiment.value}")
    print(f"Adjusted Confidence: {result.confidence:.1f}%")
    print(f"Agreement noted in reasoning: {'Agreement' in result.reasoning}")


async def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("  PROMPT ENGINEERING FOR TRADING - SENTIMENT DEMO")
    print("#" * 60)

    await basic_sentiment_demo()
    await aspect_based_demo()
    await batch_analysis_demo()
    await self_consistency_demo()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nNote: This demo uses MockLLMClient.")
    print("For production, use OpenAIClient, AnthropicClient, or OllamaClient.")


if __name__ == "__main__":
    asyncio.run(main())
