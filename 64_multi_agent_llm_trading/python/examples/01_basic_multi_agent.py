#!/usr/bin/env python3
"""
Example 1: Basic Multi-Agent Trading System

This example demonstrates how to create a simple multi-agent trading system
with multiple specialized agents working together to analyze a stock.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from agents import (
    TechnicalAgent,
    FundamentalsAgent,
    SentimentAgent,
    NewsAgent,
    BullAgent,
    BearAgent,
    RiskManagerAgent,
    TraderAgent,
)
from data_loader import create_mock_data, calculate_technical_indicators


def main():
    print("=" * 60)
    print("Multi-Agent LLM Trading System - Basic Example")
    print("=" * 60)

    # Step 1: Load or create market data
    print("\n1. Loading market data...")

    # Using mock data for demonstration
    # In production, use: DataLoader().load("AAPL") or BybitLoader().load("BTCUSDT")
    market_data = create_mock_data("AAPL", days=252, initial_price=180, seed=42)
    data = market_data.ohlcv

    print(f"   Symbol: {market_data.symbol}")
    print(f"   Period: {market_data.start_date.date()} to {market_data.end_date.date()}")
    print(f"   Latest price: ${data['close'].iloc[-1]:.2f}")

    # Step 2: Create the agent team
    print("\n2. Creating agent team...")

    agents = {
        "technical": TechnicalAgent("Technical-Analyst"),
        "fundamental": FundamentalsAgent("Fundamental-Analyst"),
        "sentiment": SentimentAgent("Sentiment-Analyst"),
        "news": NewsAgent("News-Analyst"),
        "bull": BullAgent("Bull-Researcher"),
        "bear": BearAgent("Bear-Researcher"),
        "risk": RiskManagerAgent("Risk-Manager", max_position_pct=0.05),
    }

    trader = TraderAgent("Head-Trader")

    print(f"   Created {len(agents)} analysis agents + 1 trader agent")

    # Step 3: Each agent performs analysis
    print("\n3. Running agent analyses...")
    print("-" * 60)

    analyses = []
    context = {
        "fundamentals": {
            "pe_ratio": 28.5,
            "revenue_growth": 0.08,
            "profit_margin": 0.25,
        },
        "sentiment": {
            "social_score": 0.65,
            "fear_greed": 58,
        },
        "news": [
            {"headline": "Apple reports strong iPhone sales growth"},
            {"headline": "New Apple Vision Pro receives positive reviews"},
            {"headline": "Competition in AI market intensifies"},
        ]
    }

    for name, agent in agents.items():
        analysis = agent.analyze(market_data.symbol, data, context)
        analyses.append(analysis)

        print(f"\n   {agent.name} ({agent.agent_type}):")
        print(f"   Signal: {analysis.signal.value}")
        print(f"   Confidence: {analysis.confidence:.0%}")
        print(f"   Reasoning: {analysis.reasoning[:80]}...")

    # Step 4: Trader aggregates all analyses
    print("\n" + "-" * 60)
    print("\n4. Trader aggregating analyses...")

    trader_context = {"analyses": analyses}
    final_decision = trader.analyze(market_data.symbol, data, trader_context)

    print(f"\n   FINAL DECISION:")
    print(f"   Signal: {final_decision.signal.value}")
    print(f"   Confidence: {final_decision.confidence:.0%}")
    print(f"   Reasoning: {final_decision.reasoning}")

    # Step 5: Display agent contributions
    print("\n5. Agent Contributions:")
    print("-" * 60)

    contributions = final_decision.metrics.get("agent_contributions", {})
    for agent_name, contrib in contributions.items():
        print(f"   {agent_name}: {contrib['signal']} ({contrib['confidence']:.0%})")

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Symbol: {market_data.symbol}")
    print(f"Current Price: ${data['close'].iloc[-1]:.2f}")
    print(f"Recommendation: {final_decision.signal.value}")
    print(f"Overall Confidence: {final_decision.confidence:.0%}")
    print(f"Bullish Agents: {final_decision.metrics.get('bullish_count', 0)}")
    print(f"Bearish Agents: {final_decision.metrics.get('bearish_count', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
