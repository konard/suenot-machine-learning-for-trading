#!/usr/bin/env python3
"""
Example 2: Bull vs Bear Debate

This example demonstrates how to use adversarial debate between
a Bull (optimistic) and Bear (pessimistic) agent to reach better
trading decisions through argumentation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from agents import BullAgent, BearAgent
from communication import Debate, DebateModerator
from data_loader import create_mock_data


def main():
    print("=" * 60)
    print("Multi-Agent LLM Trading - Bull vs Bear Debate")
    print("=" * 60)

    # Create market data with a trending scenario
    print("\n1. Setting up market scenario...")

    # Scenario: Stock has been rising but showing signs of exhaustion
    np.random.seed(123)
    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")

    # Create a stock that rallied then started to consolidate
    trend_up = np.linspace(0, 0.4, 200)  # Strong uptrend
    consolidation = np.linspace(0.4, 0.35, 52)  # Pullback/consolidation
    full_trend = np.concatenate([trend_up, consolidation])

    noise = np.random.randn(252).cumsum() * 0.01
    close = 100 * np.exp(full_trend + noise)

    data = pd.DataFrame({
        "open": close * (1 + np.random.randn(252) * 0.005),
        "high": close * (1 + abs(np.random.randn(252) * 0.015)),
        "low": close * (1 - abs(np.random.randn(252) * 0.015)),
        "close": close,
        "volume": np.random.randint(1e6, 1e8, 252)
    }, index=dates)

    print(f"   Symbol: DEMO")
    print(f"   Start: ${data['close'].iloc[0]:.2f}")
    print(f"   Peak: ${data['close'].max():.2f}")
    print(f"   Current: ${data['close'].iloc[-1]:.2f}")
    print(f"   YTD Return: {(data['close'].iloc[-1] / data['close'].iloc[0] - 1):.1%}")

    # Create debaters
    print("\n2. Introducing the debaters...")

    bull = BullAgent("Oliver Optimist")
    bear = BearAgent("Patty Pessimist")

    print(f"   Bull Agent: {bull.name}")
    print(f"   Bear Agent: {bear.name}")

    # Conduct debate
    print("\n3. Conducting debate (3 rounds)...")
    print("=" * 60)

    debate = Debate(bull, bear, num_rounds=3)
    result = debate.conduct("DEMO", data)

    # Display each round
    for round_data in result["rounds"]:
        round_num = round_data["round"]
        print(f"\n--- ROUND {round_num} ---")

        print(f"\n{bull.name} (BULL):")
        print(f"   {round_data['bull_argument']['reasoning']}")
        print(f"   Confidence: {round_data['bull_argument']['confidence']:.0%}")

        print(f"\n{bear.name} (BEAR):")
        print(f"   {round_data['bear_argument']['reasoning']}")
        print(f"   Confidence: {round_data['bear_argument']['confidence']:.0%}")

    # Moderated conclusion
    print("\n" + "=" * 60)
    print("MODERATOR'S EVALUATION")
    print("=" * 60)

    moderator = DebateModerator("Judge Judy")
    conclusion = moderator.evaluate(result)

    print(f"\nWinner: {result['final_scores']['winner'].upper()}")
    print(f"\nScores:")
    print(f"   Bull confidence avg: {result['final_scores']['bull_confidence']:.0%}")
    print(f"   Bear confidence avg: {result['final_scores']['bear_confidence']:.0%}")

    print(f"\nModerated Conclusion:")
    print(f"   Signal: {conclusion['signal']}")
    print(f"   Confidence: {conclusion['confidence']:.0%}")
    print(f"   {conclusion['conclusion']}")

    print(f"\nRecommendation:")
    print(f"   {conclusion['recommendation']}")

    # Show how this could inform trading
    print("\n" + "=" * 60)
    print("TRADING IMPLICATIONS")
    print("=" * 60)

    signal = conclusion["signal"]
    if signal in ["STRONG_BUY", "BUY"]:
        print("""
   The debate favored the BULL case. Suggested actions:
   - Consider opening a long position
   - Use the bear's concerns as risk factors to monitor
   - Set stop-loss based on support levels mentioned
        """)
    elif signal in ["STRONG_SELL", "SELL"]:
        print("""
   The debate favored the BEAR case. Suggested actions:
   - Avoid new long positions or consider shorting
   - Take profits on existing positions
   - Watch for the bull's positive catalysts that could change the situation
        """)
    else:
        print("""
   The debate was balanced - no clear winner. Suggested actions:
   - Stay on the sidelines for now
   - Wait for clearer signals
   - Monitor both bull and bear factors mentioned
        """)


if __name__ == "__main__":
    main()
