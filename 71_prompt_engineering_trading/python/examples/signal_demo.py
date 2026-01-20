#!/usr/bin/env python3
"""
Trading Signal Generation Demo

Demonstrates LLM-based trading signal generation
using various prompt engineering techniques.
"""

import asyncio
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_generator import PromptBasedSignalGenerator, SignalDirection
from llm_client import MockLLMClient


async def basic_signal_demo():
    """Basic signal generation demo."""
    print("=" * 60)
    print("Basic Trading Signal Generation")
    print("=" * 60)

    mock_responses = {
        "signal": json.dumps({
            "direction": "BUY",
            "confidence": 75,
            "entry_price": 185.50,
            "stop_loss": 180.00,
            "take_profit": 195.00,
            "position_size_pct": 5,
            "timeframe": "1d",
            "reasoning": "Strong momentum with support at 180, targeting resistance at 195",
            "key_factors": ["RSI oversold bounce", "Volume confirmation", "Above 50 SMA"]
        })
    }

    client = MockLLMClient(responses=mock_responses)
    generator = PromptBasedSignalGenerator(client)

    # Technical data for analysis
    symbol = "AAPL"
    technical_data = {
        "price": {"current": 185.50, "change_24h": 1.2, "change_7d": 2.5, "change_30d": 5.0},
        "technicals": {
            "sma_20": 183.20,
            "sma_50": 180.50,
            "sma_200": 175.00,
            "rsi": 35,
            "macd": 0.5,
            "atr": 3.50
        },
        "volume": {"current": 45000000, "average": 50000000},
        "sentiment": "BULLISH",
        "regime": "RISK_ON_TRENDING"
    }

    print(f"\nGenerating signal for {symbol}...")
    print(f"Current Price: ${technical_data['price']['current']}")
    print(f"RSI: {technical_data['technicals']['rsi']}")
    print(f"Sentiment: {technical_data['sentiment']}")

    signal = await generator.generate_signal(symbol, technical_data)

    print(f"\n--- Generated Signal ---")
    print(f"Direction: {signal.direction.value}")
    print(f"Confidence: {signal.confidence}%")
    if signal.entry_price:
        print(f"Entry: ${signal.entry_price:.2f}")
    print(f"Stop Loss: ${signal.stop_loss:.2f}")
    if signal.take_profit:
        tp = signal.take_profit[0] if signal.take_profit else 0
        print(f"Take Profit: ${tp:.2f}")
        if signal.entry_price and signal.stop_loss:
            rr = abs(tp - signal.entry_price) / abs(signal.entry_price - signal.stop_loss)
            print(f"Risk/Reward: {rr:.2f}")
    print(f"Reasoning: {signal.reasoning}")


async def news_signal_demo():
    """News-driven signal generation demo."""
    print("\n" + "=" * 60)
    print("News-Driven Signal Generation")
    print("=" * 60)

    mock_responses = {
        "news_signal": json.dumps({
            "direction": "BUY",
            "confidence": 82,
            "entry_price": 380.00,
            "stop_loss": 365.00,
            "take_profit": 410.00,
            "position_size_pct": 3,
            "timeframe": "1w",
            "reasoning": "Major AI partnership accelerates growth trajectory",
            "catalyst_strength": "HIGH"
        })
    }

    client = MockLLMClient(responses=mock_responses)
    generator = PromptBasedSignalGenerator(client)

    headline = "Microsoft announces $10B AI infrastructure investment with OpenAI"
    symbol = "MSFT"
    current_price = 380.00
    pre_news_price = 375.00  # Price before news
    source = "Reuters"
    category = "technology"

    print(f"\nHeadline: {headline}")
    print(f"Symbol: {symbol}")
    print(f"Current Price: ${current_price}")
    print(f"Pre-News Price: ${pre_news_price}")

    signal = await generator.generate_news_signal(
        symbol, headline, current_price, pre_news_price, source, category
    )

    print(f"\n--- News Signal ---")
    print(f"Direction: {signal.direction.value}")
    print(f"Confidence: {signal.confidence}%")
    if signal.entry_price:
        print(f"Entry: ${signal.entry_price:.2f}")
    if signal.take_profit:
        tp = signal.take_profit[0] if signal.take_profit else 0
        if signal.entry_price:
            print(f"Target: ${tp:.2f} (+{((tp/signal.entry_price)-1)*100:.1f}%)")
            print(f"Stop: ${signal.stop_loss:.2f} ({((signal.stop_loss/signal.entry_price)-1)*100:.1f}%)")


async def multi_timeframe_demo():
    """Multi-timeframe signal generation demo."""
    print("\n" + "=" * 60)
    print("Multi-Timeframe Signal Analysis")
    print("=" * 60)

    mock_responses = {
        "mtf": json.dumps({
            "direction": "BUY",
            "confidence": 70,
            "entry_price": 45500.00,
            "stop_loss": 44000.00,
            "take_profit": 48000.00,
            "position_size_pct": 2,
            "timeframe": "4h",
            "reasoning": "Bullish across timeframes with strong momentum",
            "timeframe_alignment": {
                "1h": "BULLISH",
                "4h": "BULLISH",
                "1d": "NEUTRAL"
            }
        })
    }

    client = MockLLMClient(responses=mock_responses)
    generator = PromptBasedSignalGenerator(client, prompt_type="multi_timeframe")

    # Multi-timeframe data
    symbol = "BTC/USDT"
    current_price = 45500.00
    timeframe_data = {
        "short": {
            "trend": "BULLISH",
            "momentum": "POSITIVE",
            "level": "ABOVE_MA"
        },
        "medium": {
            "trend": "BULLISH",
            "momentum": "POSITIVE",
            "level": "AT_SUPPORT"
        },
        "long": {
            "trend": "NEUTRAL",
            "momentum": "FLAT",
            "level": "MID_RANGE"
        }
    }
    sentiment = "BULLISH"

    print(f"\nSymbol: {symbol}")
    print(f"Current Price: ${current_price:,.2f}")
    print("\nTimeframe Analysis:")
    for tf_name, tf_data in timeframe_data.items():
        print(f"  {tf_name}: {tf_data['trend']} (Momentum: {tf_data['momentum']})")

    signal = await generator.generate_multi_timeframe_signal(
        symbol, timeframe_data, current_price, sentiment
    )

    print(f"\n--- Multi-Timeframe Signal ---")
    print(f"Direction: {signal.direction.value}")
    print(f"Confidence: {signal.confidence}%")
    if signal.entry_price:
        print(f"Entry: ${signal.entry_price:,.2f}")
    print(f"Stop: ${signal.stop_loss:,.2f}")
    if signal.take_profit:
        tp = signal.take_profit[0] if signal.take_profit else 0
        print(f"Target: ${tp:,.2f}")


async def self_consistent_signal_demo():
    """Self-consistency validation for signals."""
    print("\n" + "=" * 60)
    print("Self-Consistent Signal Generation")
    print("=" * 60)

    mock_responses = {
        "consistent": json.dumps({
            "direction": "HOLD",
            "confidence": 55,
            "entry_price": 140.00,
            "stop_loss": 135.00,
            "take_profit": 150.00,
            "position_size_pct": 0,
            "timeframe": "1d",
            "reasoning": "Mixed signals, wait for confirmation"
        })
    }

    client = MockLLMClient(responses=mock_responses)
    generator = PromptBasedSignalGenerator(client)

    symbol = "GOOGL"
    technical_data = {
        "price": {"current": 140.00, "change_24h": 0.5, "change_7d": -1.0, "change_30d": 2.0},
        "technicals": {
            "sma_20": 139.50,
            "sma_50": 141.00,
            "sma_200": 135.00,
            "rsi": 50,
            "macd": 0.1,
            "atr": 2.50
        },
        "volume": {"current": 25000000, "average": 28000000},
        "sentiment": "NEUTRAL",
        "regime": "RANGING"
    }

    print(f"\nGenerating self-consistent signal for {symbol}...")
    print("(Running 3 samples for consensus)")

    signal = await generator.self_consistent_signal(symbol, technical_data, num_samples=3)

    print(f"\n--- Consensus Signal ---")
    print(f"Direction: {signal.direction.value}")
    print(f"Confidence: {signal.confidence:.1f}%")
    print(f"Reasoning: {signal.reasoning}")


async def crypto_signal_demo():
    """Cryptocurrency-specific signal demo."""
    print("\n" + "=" * 60)
    print("Cryptocurrency Signal Generation (Bybit-style)")
    print("=" * 60)

    mock_responses = {
        "crypto": json.dumps({
            "direction": "BUY",
            "confidence": 68,
            "entry_price": 2500.00,
            "stop_loss": 2400.00,
            "take_profit": 2750.00,
            "position_size_pct": 2,
            "timeframe": "4h",
            "reasoning": "ETH showing strength, funding rates neutral",
            "leverage_suggestion": "3x"
        })
    }

    client = MockLLMClient(responses=mock_responses)
    generator = PromptBasedSignalGenerator(client)

    # Crypto-specific data (as would come from Bybit)
    symbol = "ETH/USDT"
    crypto_data = {
        "price": {"current": 2500.00, "change_24h": 3.5, "change_7d": 5.0, "change_30d": 10.0},
        "technicals": {
            "sma_20": 2450.00,
            "sma_50": 2380.00,
            "sma_200": 2200.00,
            "rsi": 58,
            "macd": 15.0,
            "atr": 80.00
        },
        "volume": {"current": 1500000000, "average": 1200000000},
        "sentiment": "BULLISH",
        "regime": "RISK_ON_TRENDING",
        # Crypto-specific metrics
        "funding_rate": 0.0001,  # 0.01%
        "open_interest": 5000000000,
        "long_short_ratio": 1.2
    }

    print(f"\nSymbol: {symbol}")
    print(f"Price: ${crypto_data['price']['current']:,.2f}")
    print(f"Funding Rate: {crypto_data['funding_rate']*100:.4f}%")
    print(f"Open Interest: ${crypto_data['open_interest']/1e9:.2f}B")
    print(f"Long/Short Ratio: {crypto_data['long_short_ratio']:.2f}")

    signal = await generator.generate_signal(symbol, crypto_data)

    print(f"\n--- Crypto Signal ---")
    print(f"Direction: {signal.direction.value}")
    print(f"Confidence: {signal.confidence}%")
    if signal.entry_price:
        print(f"Entry: ${signal.entry_price:,.2f}")
        print(f"Stop: ${signal.stop_loss:,.2f} ({((signal.stop_loss/signal.entry_price)-1)*100:.1f}%)")
        if signal.take_profit:
            tp = signal.take_profit[0] if signal.take_profit else 0
            print(f"Target: ${tp:,.2f} (+{((tp/signal.entry_price)-1)*100:.1f}%)")


async def main():
    """Run all signal demos."""
    print("\n" + "#" * 60)
    print("  PROMPT ENGINEERING FOR TRADING - SIGNAL GENERATION DEMO")
    print("#" * 60)

    await basic_signal_demo()
    await news_signal_demo()
    await multi_timeframe_demo()
    await self_consistent_signal_demo()
    await crypto_signal_demo()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nNote: These demos use MockLLMClient with predefined responses.")
    print("For production, connect to real LLM APIs (OpenAI, Anthropic, Ollama).")
    print("\nIMPORTANT: Always validate signals with your own analysis.")
    print("Never trade based solely on AI-generated signals.")


if __name__ == "__main__":
    asyncio.run(main())
