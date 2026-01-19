#!/usr/bin/env python3
"""
Bubble Detection Simulation Example

Demonstrates how momentum-heavy markets can form bubbles
and how the simulation can detect bubble formation.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from market import OrderBook, MarketEnvironment
from agents import ValueInvestorAgent, MomentumTraderAgent, MarketMakerAgent
from simulation import SimulationEngine
from simulation.metrics import calculate_performance_metrics, detect_bubble


def main():
    """Run bubble-prone simulation"""
    print("=" * 60)
    print("LLM Market Simulation - Bubble Detection Example")
    print("=" * 60)

    np.random.seed(123)  # For reproducibility

    # Parameters that encourage bubble formation
    initial_price = 100.0
    fundamental_value = 100.0
    num_steps = 300

    print("\nScenario: Momentum-Heavy Market")
    print("Creating market with more momentum traders than value investors...")
    print("This configuration is prone to bubble formation.\n")

    # Create simulation engine with higher volatility
    engine = SimulationEngine(
        initial_price=initial_price,
        fundamental_value=fundamental_value,
        volatility=0.025  # Slightly higher volatility
    )

    initial_cash = 100000.0
    initial_shares = 100

    # Only 1 value investor (weak anchor to fundamentals)
    agent = ValueInvestorAgent(
        agent_id="value_1",
        initial_cash=initial_cash,
        initial_shares=initial_shares,
        fundamental_value=fundamental_value,
        discount_threshold=0.15,  # Very patient, only buys at big discounts
        premium_threshold=0.15
    )
    engine.add_agent(agent)
    print(f"Added 1 Value Investor (threshold: 15%)")

    # Many momentum traders (trend followers create positive feedback)
    for i in range(5):
        agent = MomentumTraderAgent(
            agent_id=f"momentum_{i+1}",
            initial_cash=initial_cash,
            initial_shares=initial_shares,
            short_window=3,  # Fast signals
            long_window=8,
            entry_threshold=0.01  # Low threshold, trades often
        )
        engine.add_agent(agent)
    print(f"Added 5 Momentum Traders (aggressive settings)")

    # Market makers
    for i in range(2):
        agent = MarketMakerAgent(
            agent_id=f"mm_{i+1}",
            initial_cash=initial_cash * 2,
            initial_shares=initial_shares * 2,
            base_spread=0.002
        )
        engine.add_agent(agent)
    print(f"Added 2 Market Makers")

    print(f"\nTotal: 1 Value + 5 Momentum + 2 MM = 8 agents")
    print("Momentum traders outnumber value investors 5:1")

    # Run simulation
    print(f"\nRunning simulation for {num_steps} steps...")
    result = engine.run(num_steps=num_steps, verbose=False)

    # Analyze for bubbles
    print("\n" + "=" * 60)
    print("BUBBLE ANALYSIS")
    print("=" * 60)

    bubble_info = detect_bubble(
        result.price_history,
        result.fundamental_history,
        bubble_threshold=0.30  # 30% above fundamental = bubble
    )

    print(f"\nBubble Threshold: 30% above fundamental value")
    print(f"Bubble Detected: {bubble_info.get('bubble_detected', False)}")

    if bubble_info.get("bubble_detected"):
        print(f"Number of Bubble Periods: {bubble_info.get('num_bubbles', 0)}")
        print(f"Maximum Price Deviation: {bubble_info.get('max_deviation', 0)*100:.1f}%")
        print(f"Time in Bubble: {bubble_info.get('time_in_bubble_pct', 0):.1f}% of simulation")

        print("\nBubble Periods:")
        for i, period in enumerate(bubble_info.get("bubble_periods", [])):
            print(f"  Period {i+1}:")
            print(f"    Start Step: {period['start']}")
            print(f"    Peak Step: {period['peak']}")
            print(f"    End Step: {period['end']}")
            print(f"    Duration: {period['duration']} steps")
            print(f"    Peak Deviation: {period['peak_deviation']*100:.1f}%")
            peak_price = result.price_history[period['peak']]
            peak_fund = result.fundamental_history[period['peak']]
            print(f"    Peak Price: ${peak_price:.2f} vs Fundamental: ${peak_fund:.2f}")
    else:
        print("No significant bubble formed in this simulation run.")
        print("(Try running multiple times - bubbles are probabilistic)")

    # Price statistics
    print("\n" + "=" * 60)
    print("PRICE STATISTICS")
    print("=" * 60)

    prices = np.array(result.price_history)
    fundamentals = np.array(result.fundamental_history)
    deviations = (prices - fundamentals) / fundamentals * 100

    print(f"\nPrice Range:")
    print(f"  Minimum: ${prices.min():.2f} ({deviations.min():.1f}% from fundamental)")
    print(f"  Maximum: ${prices.max():.2f} ({deviations.max():.1f}% from fundamental)")
    print(f"  Final: ${prices[-1]:.2f} ({deviations[-1]:.1f}% from fundamental)")

    print(f"\nDeviation Statistics:")
    print(f"  Mean Deviation: {deviations.mean():.1f}%")
    print(f"  Std Deviation: {deviations.std():.1f}%")
    print(f"  Time Above +20%: {(deviations > 20).mean()*100:.1f}%")
    print(f"  Time Below -20%: {(deviations < -20).mean()*100:.1f}%")

    # Performance metrics
    metrics = calculate_performance_metrics(result.price_history, result.fundamental_history)

    print("\n" + "=" * 60)
    print("MARKET PERFORMANCE")
    print("=" * 60)
    print(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"  Volatility: {metrics.get('volatility_pct', 0):.2f}%")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Fundamental Correlation: {metrics.get('fundamental_correlation', 0):.3f}")

    # Simple ASCII visualization
    print("\n" + "=" * 60)
    print("PRICE CHART (ASCII)")
    print("=" * 60)

    # Sample every nth point for display
    sample_rate = max(1, len(prices) // 50)
    sampled_prices = prices[::sample_rate]
    sampled_funds = fundamentals[::sample_rate]

    min_p = min(min(sampled_prices), min(sampled_funds))
    max_p = max(max(sampled_prices), max(sampled_funds))
    height = 15
    width = len(sampled_prices)

    def scale(val):
        if max_p == min_p:
            return height // 2
        return int((val - min_p) / (max_p - min_p) * (height - 1))

    chart = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot prices
    for x, p in enumerate(sampled_prices):
        y = height - 1 - scale(p)
        chart[y][x] = '*'

    # Plot fundamentals
    for x, f in enumerate(sampled_funds):
        y = height - 1 - scale(f)
        if chart[y][x] == ' ':
            chart[y][x] = '-'

    print(f"${max_p:.0f} |", end="")
    for row in chart:
        print("".join(row))
        print("      |", end="")
    print(f"\n${min_p:.0f} |" + "_" * width)
    print("       " + "Step 0" + " " * (width - 12) + f"Step {num_steps}")
    print("\nLegend: * = Price, - = Fundamental Value")

    print("\nDone!")


if __name__ == "__main__":
    main()
