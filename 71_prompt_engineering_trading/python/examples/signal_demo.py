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
    technical_data = {
        "symbol": "AAPL",
        "current_price": 185.50,
        "sma_20": 183.20,
        "sma_50": 180.50,
        "sma_200": 175.00,
        "rsi": 35,
        "macd": 0.5,
        "macd_signal": 0.2,
        "volume": 45000000,
        "avg_volume": 50000000,
        "support": 180.00,
        "resistance": 195.00,
        "atr": 3.50,
        "trend": "BULLISH"
    }

    print(f"\nGenerating signal for {technical_data['symbol']}...")
    print(f"Current Price: ${technical_data['current_price']}")
    print(f"RSI: {technical_data['rsi']}")
    print(f"Trend: {technical_data['trend']}")

    signal = await generator.generate_signal(technical_data)

    print(f"\n--- Generated Signal ---")
    print(f"Direction: {signal.direction.value}")
    print(f"Confidence: {signal.confidence}%")
    print(f"Entry: ${signal.entry_price:.2f}")
    print(f"Stop Loss: ${signal.stop_loss:.2f}")
    print(f"Take Profit: ${signal.take_profit:.2f}")
    print(f"Risk/Reward: {abs(signal.take_profit - signal.entry_price) / abs(signal.entry_price - signal.stop_loss):.2f}")
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

    print(f"\nHeadline: {headline}")
    print(f"Symbol: {symbol}")
    print(f"Current Price: ${current_price}")

    signal = await generator.generate_news_signal(headline, symbol, current_price)

    print(f"\n--- News Signal ---")
    print(f"Direction: {signal.direction.value}")
    print(f"Confidence: {signal.confidence}%")
    print(f"Entry: ${signal.entry_price:.2f}")
    print(f"Target: ${signal.take_profit:.2f} (+{((signal.take_profit/signal.entry_price)-1)*100:.1f}%)")
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
    timeframe_data = {
        "symbol": "BTC/USDT",
        "short_term": {
            "timeframe": "1h",
            "trend": "BULLISH",
            "rsi": 55,
            "momentum": "POSITIVE"
        },
        "medium_term": {
            "timeframe": "4h",
            "trend": "BULLISH",
            "rsi": 60,
            "momentum": "POSITIVE"
        },
        "long_term": {
            "timeframe": "1d",
            "trend": "NEUTRAL",
            "rsi": 52,
            "momentum": "FLAT"
        },
        "current_price": 45500.00,
        "key_levels": {
            "support_1": 44000,
            "support_2": 42500,
            "resistance_1": 47000,
            "resistance_2": 50000
        }
    }

    print(f"\nSymbol: {timeframe_data['symbol']}")
    print(f"Current Price: ${timeframe_data['current_price']:,.2f}")
    print("\nTimeframe Analysis:")
    for tf in ['short_term', 'medium_term', 'long_term']:
        data = timeframe_data[tf]
        print(f"  {data['timeframe']}: {data['trend']} (RSI: {data['rsi']})")

    signal = await generator.generate_multi_timeframe_signal(timeframe_data)

    print(f"\n--- Multi-Timeframe Signal ---")
    print(f"Direction: {signal.direction.value}")
    print(f"Confidence: {signal.confidence}%")
    print(f"Recommended Timeframe: {signal.timeframe}")
    print(f"Entry: ${signal.entry_price:,.2f}")
    print(f"Stop: ${signal.stop_loss:,.2f}")
    print(f"Target: ${signal.take_profit:,.2f}")


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

    technical_data = {
        "symbol": "GOOGL",
        "current_price": 140.00,
        "sma_20": 139.50,
        "sma_50": 141.00,
        "sma_200": 135.00,
        "rsi": 50,
        "macd": 0.1,
        "macd_signal": 0.15,
        "volume": 25000000,
        "avg_volume": 28000000,
        "support": 135.00,
        "resistance": 145.00,
        "atr": 2.50,
        "trend": "NEUTRAL"
    }

    print(f"\nGenerating self-consistent signal for {technical_data['symbol']}...")
    print("(Running 3 samples for consensus)")

    signal = await generator.self_consistent_signal(technical_data, num_samples=3)

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
    crypto_data = {
        "symbol": "ETH/USDT",
        "current_price": 2500.00,
        "sma_20": 2450.00,
        "sma_50": 2380.00,
        "sma_200": 2200.00,
        "rsi": 58,
        "macd": 15.0,
        "macd_signal": 10.0,
        "volume": 1500000000,  # 24h volume
        "avg_volume": 1200000000,
        "support": 2400.00,
        "resistance": 2700.00,
        "atr": 80.00,
        "trend": "BULLISH",
        # Crypto-specific metrics
        "funding_rate": 0.0001,  # 0.01%
        "open_interest": 5000000000,
        "long_short_ratio": 1.2
    }

    print(f"\nSymbol: {crypto_data['symbol']}")
    print(f"Price: ${crypto_data['current_price']:,.2f}")
    print(f"Funding Rate: {crypto_data['funding_rate']*100:.4f}%")
    print(f"Open Interest: ${crypto_data['open_interest']/1e9:.2f}B")
    print(f"Long/Short Ratio: {crypto_data['long_short_ratio']:.2f}")

    signal = await generator.generate_signal(crypto_data)

    print(f"\n--- Crypto Signal ---")
    print(f"Direction: {signal.direction.value}")
    print(f"Confidence: {signal.confidence}%")
    print(f"Entry: ${signal.entry_price:,.2f}")
    print(f"Stop: ${signal.stop_loss:,.2f} ({((signal.stop_loss/signal.entry_price)-1)*100:.1f}%)")
    print(f"Target: ${signal.take_profit:,.2f} (+{((signal.take_profit/signal.entry_price)-1)*100:.1f}%)")


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
