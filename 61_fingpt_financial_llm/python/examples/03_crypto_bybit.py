#!/usr/bin/env python3
"""
FinGPT Cryptocurrency Analysis with Bybit Demo

This example demonstrates how to combine FinGPT sentiment analysis
with real-time Bybit cryptocurrency market data.
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fingpt_sentiment import FinGPTSentimentAnalyzer
from signals import FinGPTTradingEngine
from bybit_client import BybitClient


async def main():
    """Demo of crypto analysis with Bybit data."""
    # Initialize components (use_mock=True for demo)
    analyzer = FinGPTSentimentAnalyzer(use_mock=True)
    engine = FinGPTTradingEngine(analyzer)
    bybit = BybitClient(use_mock=True)  # Set to False with real API keys

    # Crypto pairs to analyze
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # Simulated crypto news feed
    crypto_news = {
        "BTCUSDT": [
            {
                "text": "Bitcoin spot ETF sees record $500M inflows as institutional adoption accelerates.",
                "source": "bloomberg"
            },
            {
                "text": "Major mining company reports 40% increase in hash rate.",
                "source": "reuters"
            },
        ],
        "ETHUSDT": [
            {
                "text": "Ethereum staking rewards increase after network upgrade.",
                "source": "reuters"
            },
            {
                "text": "DeFi total value locked on Ethereum reaches new all-time high.",
                "source": "bloomberg"
            },
        ],
        "SOLUSDT": [
            {
                "text": "Solana network processes 65,000 TPS in stress test, outperforming competitors.",
                "source": "reuters"
            },
            {
                "text": "Major NFT marketplace announces Solana integration.",
                "source": "twitter"
            },
        ],
    }

    print("=" * 70)
    print("FinGPT Cryptocurrency Analysis with Bybit Data")
    print("=" * 70)

    # Fetch market data and analyze each symbol
    analysis_results = []

    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"Analyzing: {symbol}")
        print("=" * 70)

        # Fetch current market data from Bybit
        ticker = await bybit.get_ticker(symbol)
        klines = await bybit.get_klines(symbol, interval="1h", limit=24)

        # Display market data
        print(f"\nMarket Data (Bybit):")
        print(f"  Last Price: ${float(ticker.last_price):,.2f}")
        print(f"  24h Change: {float(ticker.price_24h_pct) * 100:+.2f}%")
        print(f"  24h Volume: ${float(ticker.volume_24h):,.0f}")
        print(f"  Bid: ${float(ticker.bid_price):,.2f}")
        print(f"  Ask: ${float(ticker.ask_price):,.2f}")

        # Calculate technical indicators from klines
        if klines:
            closes = [float(k.close) for k in klines]
            highs = [float(k.high) for k in klines]
            lows = [float(k.low) for k in klines]

            # Simple moving averages
            sma_12 = sum(closes[-12:]) / min(12, len(closes))
            sma_24 = sum(closes) / len(closes)

            # Price range
            high_24h = max(highs)
            low_24h = min(lows)

            print(f"\nTechnical Indicators (24h):")
            print(f"  SMA-12: ${sma_12:,.2f}")
            print(f"  SMA-24: ${sma_24:,.2f}")
            print(f"  24h High: ${high_24h:,.2f}")
            print(f"  24h Low: ${low_24h:,.2f}")
            print(f"  Price Position: {(float(ticker.last_price) - low_24h) / (high_24h - low_24h) * 100:.1f}% of range")

        # Analyze news sentiment
        print(f"\nNews Sentiment Analysis:")
        news_items = crypto_news.get(symbol, [])

        for i, item in enumerate(news_items, 1):
            result = analyzer.analyze(item["text"])
            print(f"  [{i}] {result.sentiment.upper()} ({result.score:+.2f}): "
                  f"{item['text'][:50]}...")

        # Generate trading signal
        signal = engine.generate_signal(symbol, news_items)

        # Combine technical and sentiment analysis
        technical_bias = "bullish" if float(ticker.price_24h_pct) > 0 else "bearish"
        sentiment_bias = "bullish" if signal.strength > 0 else "bearish" if signal.strength < 0 else "neutral"

        print(f"\nCombined Analysis:")
        print(f"  Technical Bias: {technical_bias.upper()}")
        print(f"  Sentiment Bias: {sentiment_bias.upper()}")
        print(f"  Signal: {signal.action.upper()} (strength: {signal.strength:+.2f})")

        analysis_results.append({
            "symbol": symbol,
            "price": float(ticker.last_price),
            "change_24h": float(ticker.price_24h_pct),
            "signal": signal,
            "technical_bias": technical_bias,
            "sentiment_bias": sentiment_bias,
        })

    # Summary dashboard
    print("\n" + "=" * 70)
    print("CRYPTO TRADING DASHBOARD")
    print("=" * 70)

    print("\n{:<12} {:<12} {:<10} {:<12} {:<10} {:<10}".format(
        "Symbol", "Price", "24h Chg", "Technical", "Sentiment", "Signal"
    ))
    print("-" * 70)

    for result in analysis_results:
        print(f"{result['symbol']:<12} "
              f"${result['price']:<10,.2f} "
              f"{result['change_24h']*100:+.1f}%      "
              f"{result['technical_bias']:<12} "
              f"{result['sentiment_bias']:<10} "
              f"{result['signal'].action.upper():<10}")

    # Trading recommendations
    print("\n" + "=" * 70)
    print("TRADING RECOMMENDATIONS")
    print("=" * 70)

    for result in analysis_results:
        symbol = result["symbol"]
        signal = result["signal"]

        # Strong signal when technical and sentiment align
        if result["technical_bias"] == result["sentiment_bias"] == "bullish":
            print(f"\n{symbol}: STRONG BUY")
            print(f"  Both technical and sentiment indicators are bullish")
            print(f"  Confidence: {signal.confidence:.0%}")
        elif result["technical_bias"] == result["sentiment_bias"] == "bearish":
            print(f"\n{symbol}: STRONG SELL")
            print(f"  Both technical and sentiment indicators are bearish")
            print(f"  Confidence: {signal.confidence:.0%}")
        elif result["technical_bias"] != result["sentiment_bias"]:
            print(f"\n{symbol}: HOLD/WAIT")
            print(f"  Mixed signals - technical is {result['technical_bias']}, "
                  f"sentiment is {result['sentiment_bias']}")
            print(f"  Wait for clearer direction")
        else:
            print(f"\n{symbol}: NEUTRAL")
            print(f"  No clear trading opportunity")


if __name__ == "__main__":
    asyncio.run(main())
