#!/usr/bin/env python3
"""
Basic LLM Market Simulation Example

Demonstrates a simple multi-agent market simulation with different
agent types interacting in a simulated order book environment.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from market import OrderBook, MarketEnvironment
from agents import ValueInvestorAgent, MomentumTraderAgent, MarketMakerAgent
from simulation import SimulationEngine, calculate_performance_metrics


def main():
    """Run basic simulation example"""
    print("=" * 60)
    print("LLM Market Simulation - Basic Example")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Simulation parameters
    initial_price = 100.0
    fundamental_value = 100.0
    num_steps = 500
    initial_cash = 100000.0
    initial_shares = 100

    # Create simulation engine
    engine = SimulationEngine(
        initial_price=initial_price,
        fundamental_value=fundamental_value,
        volatility=0.02
    )

    # Add diverse agents
    print("\nCreating agents...")

    # Value investors (patient, look for discounts)
    for i in range(3):
        agent = ValueInvestorAgent(
            agent_id=f"value_{i+1}",
            initial_cash=initial_cash,
            initial_shares=initial_shares,
            fundamental_value=fundamental_value,
            discount_threshold=0.05 + i * 0.02,  # Different thresholds
            premium_threshold=0.05 + i * 0.02
        )
        engine.add_agent(agent)
        print(f"  Added {agent.agent_id}: Value Investor (threshold: {agent.discount_threshold*100:.0f}%)")

    # Momentum traders (follow trends)
    for i in range(3):
        agent = MomentumTraderAgent(
            agent_id=f"momentum_{i+1}",
            initial_cash=initial_cash,
            initial_shares=initial_shares,
            short_window=5 + i * 5,  # Different windows
            long_window=20 + i * 10
        )
        engine.add_agent(agent)
        print(f"  Added {agent.agent_id}: Momentum Trader (windows: {agent.short_window}/{agent.long_window})")

    # Market makers (provide liquidity)
    for i in range(2):
        agent = MarketMakerAgent(
            agent_id=f"mm_{i+1}",
            initial_cash=initial_cash * 2,  # More capital for MM
            initial_shares=initial_shares * 2,
            base_spread=0.002 + i * 0.001  # Different spreads
        )
        engine.add_agent(agent)
        print(f"  Added {agent.agent_id}: Market Maker (spread: {agent.base_spread*100:.1f}%)")

    # Run simulation
    print(f"\nRunning simulation for {num_steps} steps...")
    result = engine.run(num_steps=num_steps, verbose=True)

    # Print results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    print(f"\nPrice Evolution:")
    print(f"  Start Price: ${result.price_history[0]:.2f}")
    print(f"  End Price: ${result.price_history[-1]:.2f}")
    print(f"  Min Price: ${min(result.price_history):.2f}")
    print(f"  Max Price: ${max(result.price_history):.2f}")
    print(f"  Fundamental Value: ${fundamental_value:.2f}")

    # Calculate overall metrics
    metrics = calculate_performance_metrics(
        result.price_history,
        result.fundamental_history
    )

    print(f"\nMarket Metrics:")
    print(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"  Volatility (annualized): {metrics.get('volatility_pct', 0):.2f}%")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Tracking Error vs Fundamental: {metrics.get('tracking_error', 0):.3f}")
    print(f"  Final Deviation from Fundamental: {metrics.get('final_deviation_pct', 0):.2f}%")

    print(f"\nTotal Trades Executed: {result.total_trades}")

    # Agent performance
    print("\nAgent Performance:")
    print("-" * 60)
    print(f"{'Agent':<15} {'Type':<12} {'Final Value':>12} {'Return':>10} {'Trades':>8}")
    print("-" * 60)

    for agent_id, agent_result in result.agent_results.items():
        agent = engine.agents[agent_id]
        agent_type = agent.__class__.__name__.replace("Agent", "")[:11]
        final_value = agent_result.get("final_value", 0)
        initial_value = initial_cash + initial_shares * initial_price
        if "mm_" in agent_id:
            initial_value = initial_cash * 2 + initial_shares * 2 * initial_price
        ret = (final_value / initial_value - 1) * 100
        trades = agent_result.get("num_trades", 0)
        print(f"{agent_id:<15} {agent_type:<12} ${final_value:>10,.0f} {ret:>9.2f}% {trades:>8}")

    print("-" * 60)

    print("\nSimulation complete!")
    return result


if __name__ == "__main__":
    main()
