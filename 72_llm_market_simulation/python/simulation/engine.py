"""
Simulation Engine for LLM Market Simulation

Orchestrates the market simulation with multiple agents.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market import MarketEnvironment, Order, OrderType, Side
from agents import BaseAgent, AgentDecision, ActionType


@dataclass
class SimulationResult:
    """
    Results from a simulation run

    Attributes:
        price_history: List of prices over time
        fundamental_history: List of fundamental values over time
        agent_performance: Dict mapping agent_id to performance metrics
        market_metrics: Market quality metrics
        trade_log: Log of all trades
    """
    price_history: List[float]
    fundamental_history: List[float]
    agent_performance: Dict[str, Dict[str, float]]
    market_metrics: Dict[str, float]
    trade_log: List[Dict[str, Any]] = field(default_factory=list)


class SimulationEngine:
    """
    Simulation Engine

    Orchestrates the market simulation by coordinating agents,
    processing orders, and collecting metrics.

    Examples:
        >>> engine = SimulationEngine(initial_price=100.0)
        >>> engine.add_agent(ValueInvestorAgent("value_1", 100000))
        >>> results = engine.run(num_steps=100)
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        fundamental_value: Optional[float] = None,
        volatility: float = 0.02,
        tick_size: float = 0.01,
        random_seed: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize simulation engine

        Args:
            initial_price: Starting price
            fundamental_value: True fundamental value
            volatility: Daily volatility for fundamental value
            tick_size: Minimum price increment
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.market = MarketEnvironment(
            initial_price=initial_price,
            fundamental_value=fundamental_value,
            tick_size=tick_size,
            volatility=volatility,
            random_seed=random_seed
        )
        self.agents: List[BaseAgent] = []
        self.verbose = verbose
        self.rng = np.random.default_rng(random_seed)

    def add_agent(self, agent: BaseAgent):
        """
        Add an agent to the simulation

        Args:
            agent: Agent to add
        """
        self.agents.append(agent)
        self.market.register_agent(agent.agent_id, agent.initial_cash)

    def run(self, num_steps: int = 100) -> SimulationResult:
        """
        Run the simulation

        Args:
            num_steps: Number of time steps to simulate

        Returns:
            SimulationResult with all metrics and history
        """
        trade_log = []

        for step in range(num_steps):
            if self.verbose and step % 50 == 0:
                print(f"Step {step}/{num_steps}, Price: ${self.market.current_price:.2f}")

            # Get current market state
            state = self.market.get_state()

            # Each agent makes a decision
            decisions = []
            for agent in self.agents:
                decision = agent.make_decision(
                    current_price=state.current_price,
                    fundamental_value=state.fundamental_value,
                    price_history=state.price_history,
                    market_state={
                        "best_bid": state.best_bid,
                        "best_ask": state.best_ask,
                        "spread": state.spread,
                        "volume_24h": state.volume_24h,
                        "time_step": state.time_step
                    }
                )
                decisions.append((agent, decision))

            # Shuffle order to avoid first-mover advantage
            self.rng.shuffle(decisions)

            # Process decisions
            for agent, decision in decisions:
                if decision.action == ActionType.HOLD:
                    continue

                # Create order
                side = Side.BUY if decision.action == ActionType.BUY else Side.SELL
                order_type = OrderType.LIMIT if decision.order_type == "limit" else OrderType.MARKET

                order = Order(
                    order_type=order_type,
                    side=side,
                    quantity=decision.quantity,
                    price=decision.limit_price if order_type == OrderType.LIMIT else None,
                    agent_id=agent.agent_id
                )

                # Submit order
                result = self.market.submit_order(agent.agent_id, order)

                # Update agent portfolio
                if result.filled_quantity > 0:
                    agent.update_portfolio(
                        decision.action,
                        result.filled_quantity,
                        result.average_price
                    )

                    trade_log.append({
                        "step": step,
                        "agent_id": agent.agent_id,
                        "action": decision.action.value,
                        "quantity": result.filled_quantity,
                        "price": result.average_price,
                        "reasoning": decision.reasoning[:100]
                    })

            # Advance market
            self.market.step()

        # Collect results
        agent_performance = {}
        for agent in self.agents:
            final_value = agent.get_portfolio_value(self.market.current_price)
            returns = agent.get_return(self.market.current_price)

            agent_performance[agent.agent_id] = {
                "final_value": final_value,
                "return": returns,
                "return_pct": returns * 100,
                "num_trades": len(agent.trade_history),
                "final_holdings": agent.holdings,
                "final_cash": agent.cash
            }

        return SimulationResult(
            price_history=list(self.market.price_history),
            fundamental_history=list(self.market.fundamental_history),
            agent_performance=agent_performance,
            market_metrics=self.market.calculate_metrics(),
            trade_log=trade_log
        )

    def reset(self):
        """Reset simulation to initial state"""
        self.market.reset()
        for agent in self.agents:
            agent.reset()


if __name__ == "__main__":
    from agents import ValueInvestorAgent, MomentumTraderAgent, MarketMakerAgent

    print("Running Market Simulation Demo")
    print("=" * 50)

    # Create simulation
    engine = SimulationEngine(
        initial_price=100.0,
        fundamental_value=100.0,
        volatility=0.02,
        random_seed=42,
        verbose=True
    )

    # Add agents
    engine.add_agent(ValueInvestorAgent("value_1", initial_cash=100000))
    engine.add_agent(MomentumTraderAgent("momentum_1", initial_cash=100000))
    engine.add_agent(MarketMakerAgent("mm_1", initial_cash=100000))

    # Run simulation
    results = engine.run(num_steps=100)

    # Print results
    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)

    print(f"\nFinal Price: ${results.price_history[-1]:.2f}")
    print(f"Final Fundamental: ${results.fundamental_history[-1]:.2f}")

    print("\nAgent Performance:")
    for agent_id, perf in results.agent_performance.items():
        print(f"  {agent_id}:")
        print(f"    Return: {perf['return_pct']:.2f}%")
        print(f"    Final Value: ${perf['final_value']:.2f}")
        print(f"    Trades: {perf['num_trades']}")

    print("\nMarket Metrics:")
    for metric, value in results.market_metrics.items():
        print(f"  {metric}: {value:.4f}")
