#!/usr/bin/env python3
"""
Example 3: Cryptocurrency Trading with Multi-Agent System

This example demonstrates how to use the multi-agent system
for cryptocurrency trading, specifically with Bybit data.

Note: This example uses mock data for demonstration.
To use real data, uncomment the Bybit loader sections.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from agents import (
    TechnicalAgent,
    SentimentAgent,
    BullAgent,
    BearAgent,
    RiskManagerAgent,
    TraderAgent,
)
from communication import RoundTable


def create_crypto_mock_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """
    Create mock cryptocurrency data with higher volatility.

    In production, use:
        from data_loader import BybitLoader
        loader = BybitLoader()
        data = loader.load("BTCUSDT", days=30)
    """
    np.random.seed(456)
    dates = pd.date_range(start="2024-01-01", periods=days * 24, freq="H")  # Hourly data

    # Crypto-like price movement (more volatile)
    returns = np.random.randn(len(dates)) * 0.015  # ~1.5% hourly volatility
    returns += np.sin(np.linspace(0, 10 * np.pi, len(dates))) * 0.005  # Some cyclicality

    if symbol == "BTCUSDT":
        initial_price = 65000
    elif symbol == "ETHUSDT":
        initial_price = 3500
    else:
        initial_price = 100

    close = initial_price * np.exp(returns.cumsum())

    return pd.DataFrame({
        "open": close * (1 + np.random.randn(len(dates)) * 0.003),
        "high": close * (1 + abs(np.random.randn(len(dates)) * 0.008)),
        "low": close * (1 - abs(np.random.randn(len(dates)) * 0.008)),
        "close": close,
        "volume": np.random.uniform(100, 1000, len(dates)) * close / 1000
    }, index=dates)


def main():
    print("=" * 60)
    print("Multi-Agent LLM Trading - Cryptocurrency Example")
    print("=" * 60)

    # Load crypto data
    print("\n1. Loading cryptocurrency data...")

    # In production, use:
    # from data_loader import BybitLoader
    # loader = BybitLoader()
    # btc_data = loader.load("BTCUSDT", interval="60", days=7)

    btc_data = create_crypto_mock_data("BTCUSDT", days=7)
    eth_data = create_crypto_mock_data("ETHUSDT", days=7)

    print(f"\n   Bitcoin (BTCUSDT):")
    print(f"   Current: ${btc_data['close'].iloc[-1]:,.2f}")
    print(f"   24h High: ${btc_data['high'].tail(24).max():,.2f}")
    print(f"   24h Low: ${btc_data['low'].tail(24).min():,.2f}")
    print(f"   24h Change: {(btc_data['close'].iloc[-1] / btc_data['close'].iloc[-24] - 1):.2%}")

    print(f"\n   Ethereum (ETHUSDT):")
    print(f"   Current: ${eth_data['close'].iloc[-1]:,.2f}")
    print(f"   24h Change: {(eth_data['close'].iloc[-1] / eth_data['close'].iloc[-24] - 1):.2%}")

    # Create crypto-specialized agents
    print("\n2. Assembling crypto analysis team...")

    # For crypto, we emphasize technical and sentiment analysis
    # (fundamentals are harder to evaluate for crypto)
    agents = [
        TechnicalAgent("Crypto-Tech"),
        SentimentAgent("Crypto-Sentiment"),
        BullAgent("Crypto-Bull"),
        BearAgent("Crypto-Bear"),
        RiskManagerAgent("Crypto-Risk", max_position_pct=0.03, max_drawdown=0.20),
    ]

    trader = TraderAgent(
        "Crypto-Trader",
        weights={
            "technical": 0.35,
            "sentiment": 0.25,
            "bull_researcher": 0.15,
            "bear_researcher": 0.15,
            "risk_manager": 0.10,
        }
    )

    # Crypto-specific context
    crypto_context = {
        "sentiment": {
            "social_score": 0.72,  # Positive crypto Twitter
            "fear_greed": 68,      # Greed
            "mentions_change": 0.15  # 15% increase in mentions
        },
        "market_info": {
            "btc_dominance": 0.52,  # 52% market dominance
            "total_market_cap": 2.5e12,  # $2.5T
            "funding_rate": 0.01,  # Positive funding = bullish
        }
    }

    # Round table analysis for BTC
    print("\n3. Analyzing Bitcoin...")
    print("-" * 60)

    round_table = RoundTable(agents)
    btc_result = round_table.conduct("BTCUSDT", btc_data, crypto_context)

    print(f"\n   Consensus: {btc_result['consensus']}")
    print(f"   Buy Signals: {btc_result['buy_ratio']:.0%}")
    print(f"   Sell Signals: {btc_result['sell_ratio']:.0%}")

    # Final trader decision
    trader_decision = trader.analyze("BTCUSDT", btc_data, {
        "analyses": [
            agent.analyze("BTCUSDT", btc_data, crypto_context)
            for agent in agents
        ]
    })

    print(f"\n   TRADER DECISION: {trader_decision.signal.value}")
    print(f"   Confidence: {trader_decision.confidence:.0%}")
    print(f"   Reasoning: {trader_decision.reasoning}")

    # Analyze ETH
    print("\n4. Analyzing Ethereum...")
    print("-" * 60)

    eth_result = round_table.conduct("ETHUSDT", eth_data, crypto_context)

    print(f"\n   Consensus: {eth_result['consensus']}")

    eth_decision = trader.analyze("ETHUSDT", eth_data, {
        "analyses": [
            agent.analyze("ETHUSDT", eth_data, crypto_context)
            for agent in agents
        ]
    })

    print(f"   TRADER DECISION: {eth_decision.signal.value}")
    print(f"   Confidence: {eth_decision.confidence:.0%}")

    # Portfolio allocation suggestion
    print("\n" + "=" * 60)
    print("PORTFOLIO ALLOCATION SUGGESTION")
    print("=" * 60)

    btc_weight = 0.5 if btc_result['consensus'] == 'BULLISH' else 0.3 if btc_result['consensus'] == 'MIXED' else 0.2
    eth_weight = 0.3 if eth_result['consensus'] == 'BULLISH' else 0.2 if eth_result['consensus'] == 'MIXED' else 0.1
    cash_weight = 1 - btc_weight - eth_weight

    print(f"\n   Based on multi-agent analysis:")
    print(f"   - Bitcoin (BTC): {btc_weight:.0%}")
    print(f"   - Ethereum (ETH): {eth_weight:.0%}")
    print(f"   - Stablecoins/Cash: {cash_weight:.0%}")

    print("\n   Risk Management Notes:")
    print("   - Set stop-losses at 2x ATR below entry")
    print("   - Consider reducing position if Fear & Greed > 80")
    print("   - Monitor funding rates for reversal signals")

    # Display 24/7 trading considerations
    print("\n" + "=" * 60)
    print("24/7 CRYPTO TRADING CONSIDERATIONS")
    print("=" * 60)
    print("""
   Crypto markets never sleep. Multi-agent systems are ideal because:

   1. CONTINUOUS MONITORING
      - Agents can analyze markets around the clock
      - No human fatigue or emotional trading

   2. RAPID RESPONSE
      - LLM agents can process news/sentiment in real-time
      - React to market events within seconds

   3. CROSS-MARKET ANALYSIS
      - Monitor correlations between BTC, ETH, and altcoins
      - Track DeFi metrics, on-chain data, exchange flows

   4. RISK MANAGEMENT
      - Automatic position sizing based on volatility
      - Stop-loss monitoring even while you sleep

   For production deployment:
   - Use Bybit API for real-time data and execution
   - Implement proper API key security
   - Set up webhook notifications for important signals
    """)


if __name__ == "__main__":
    main()
