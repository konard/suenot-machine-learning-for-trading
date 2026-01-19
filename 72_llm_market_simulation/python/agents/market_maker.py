"""
Market Maker Agent

Implements a market making strategy that provides liquidity
by posting both bid and ask orders.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from .base import BaseAgent, AgentDecision, ActionType


class MarketMakerAgent(BaseAgent):
    """
    Market Maker Agent

    Provides liquidity by quoting both bid and ask prices.
    Earns the bid-ask spread while managing inventory risk.

    Attributes:
        target_spread_bps: Target bid-ask spread in basis points
        max_inventory: Maximum inventory (long or short)
        quote_size: Size of quotes
        volatility_multiplier: How much to widen spread in volatile markets
    """

    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 100000.0,
        target_spread_bps: int = 50,
        max_inventory: int = 100,
        quote_size: int = 10,
        volatility_multiplier: float = 2.0
    ):
        """
        Initialize market maker agent

        Args:
            agent_id: Unique identifier
            initial_cash: Starting cash balance
            target_spread_bps: Target spread in basis points (50 = 0.5%)
            max_inventory: Maximum position size
            quote_size: Size of each quote
            volatility_multiplier: Spread multiplier for volatility
        """
        super().__init__(agent_id, initial_cash, "market_maker")
        self.target_spread_bps = target_spread_bps
        self.max_inventory = max_inventory
        self.quote_size = quote_size
        self.volatility_multiplier = volatility_multiplier

    def calculate_quotes(
        self,
        current_price: float,
        inventory: int,
        volatility: float = 0.0
    ) -> Tuple[float, float, int, int]:
        """
        Calculate bid and ask quotes

        Args:
            current_price: Current market price
            inventory: Current inventory position
            volatility: Recent volatility

        Returns:
            Tuple of (bid_price, ask_price, bid_size, ask_size)
        """
        # Base spread calculation
        base_spread = current_price * (self.target_spread_bps / 10000)

        # Volatility adjustment
        vol_adj = 1 + volatility * self.volatility_multiplier

        # Inventory adjustment (widen spread if inventory is large)
        inv_ratio = abs(inventory) / self.max_inventory if self.max_inventory > 0 else 0
        inv_adj = 1 + inv_ratio * 0.5

        # Effective spread
        effective_spread = base_spread * vol_adj * inv_adj
        half_spread = effective_spread / 2

        # Skew quotes based on inventory
        # If long, lower ask to encourage selling
        # If short, raise bid to encourage buying
        inv_skew = (inventory / self.max_inventory) * half_spread * 0.3 if self.max_inventory > 0 else 0

        bid_price = current_price - half_spread - inv_skew
        ask_price = current_price + half_spread - inv_skew

        # Adjust sizes based on inventory
        if inventory > 0:
            # Long inventory, reduce bid size, increase ask size
            bid_ratio = max(0, 1 - inventory / self.max_inventory)
            ask_ratio = min(2, 1 + inventory / self.max_inventory)
        else:
            # Short inventory, increase bid size, reduce ask size
            bid_ratio = min(2, 1 - inventory / self.max_inventory)
            ask_ratio = max(0, 1 + inventory / self.max_inventory)

        bid_size = int(self.quote_size * bid_ratio)
        ask_size = int(self.quote_size * ask_ratio)

        return bid_price, ask_price, bid_size, ask_size

    def make_decision(
        self,
        current_price: float,
        fundamental_value: float,
        price_history: List[float],
        market_state: Dict[str, Any]
    ) -> AgentDecision:
        """
        Make market making decision

        Market makers typically post both bid and ask quotes.
        This returns the more urgent side based on inventory.

        Args:
            current_price: Current market price
            fundamental_value: Estimated fundamental value
            price_history: List of historical prices
            market_state: Additional market information

        Returns:
            AgentDecision with quote details
        """
        # Calculate volatility from price history
        if len(price_history) >= 5:
            returns = np.diff(price_history[-20:]) / np.array(price_history[-20:-1])
            volatility = np.std(returns) if len(returns) > 0 else 0
        else:
            volatility = 0

        # Current inventory (holdings)
        inventory = self.holdings

        # Calculate quotes
        bid_price, ask_price, bid_size, ask_size = self.calculate_quotes(
            current_price, inventory, volatility
        )

        # Decide which side to prioritize based on inventory
        # If long, prioritize selling (ask)
        # If short, prioritize buying (bid)

        if inventory > self.max_inventory * 0.5:
            # Too long, need to sell
            if ask_size > 0:
                return AgentDecision(
                    action=ActionType.SELL,
                    quantity=ask_size,
                    order_type="limit",
                    limit_price=round(ask_price, 2),
                    reasoning=f"Market making ASK: Inventory too long ({inventory}). "
                              f"Quoting ${ask_price:.2f} x {ask_size} to reduce position. "
                              f"Spread: {(ask_price - bid_price):.2f}"
                )

        elif inventory < -self.max_inventory * 0.5:
            # Too short, need to buy
            if bid_size > 0:
                return AgentDecision(
                    action=ActionType.BUY,
                    quantity=bid_size,
                    order_type="limit",
                    limit_price=round(bid_price, 2),
                    reasoning=f"Market making BID: Inventory too short ({inventory}). "
                              f"Quoting ${bid_price:.2f} x {bid_size} to increase position. "
                              f"Spread: {(ask_price - bid_price):.2f}"
                )

        else:
            # Balanced inventory, post both sides
            # Return the bid side (can be modified to return both)
            if bid_size > 0:
                return AgentDecision(
                    action=ActionType.BUY,
                    quantity=bid_size,
                    order_type="limit",
                    limit_price=round(bid_price, 2),
                    reasoning=f"Market making BID: ${bid_price:.2f} x {bid_size}. "
                              f"Inventory: {inventory}. Volatility: {volatility*100:.2f}%. "
                              f"Full spread: {(ask_price - bid_price):.2f}"
                )

        # Default hold
        return AgentDecision(
            action=ActionType.HOLD,
            quantity=0,
            order_type="market",
            reasoning=f"Market maker at capacity. Inventory: {inventory}, Max: {self.max_inventory}"
        )

    def get_both_quotes(
        self,
        current_price: float,
        price_history: List[float]
    ) -> List[AgentDecision]:
        """
        Get both bid and ask quotes

        Returns:
            List of two AgentDecisions (bid and ask)
        """
        # Calculate volatility
        if len(price_history) >= 5:
            returns = np.diff(price_history[-20:]) / np.array(price_history[-20:-1])
            volatility = np.std(returns) if len(returns) > 0 else 0
        else:
            volatility = 0

        inventory = self.holdings
        bid_price, ask_price, bid_size, ask_size = self.calculate_quotes(
            current_price, inventory, volatility
        )

        quotes = []

        if bid_size > 0:
            quotes.append(AgentDecision(
                action=ActionType.BUY,
                quantity=bid_size,
                order_type="limit",
                limit_price=round(bid_price, 2),
                reasoning=f"MM BID: ${bid_price:.2f} x {bid_size}"
            ))

        if ask_size > 0:
            quotes.append(AgentDecision(
                action=ActionType.SELL,
                quantity=ask_size,
                order_type="limit",
                limit_price=round(ask_price, 2),
                reasoning=f"MM ASK: ${ask_price:.2f} x {ask_size}"
            ))

        return quotes

    def get_system_prompt(self) -> str:
        """Get system prompt for LLM-based decision making"""
        return f"""You are a market maker agent in a simulated stock market.

Your Strategy Parameters:
- Target spread: {self.target_spread_bps} basis points ({self.target_spread_bps/100}%)
- Maximum inventory: {self.max_inventory} shares (long or short)
- Quote size: {self.quote_size} shares per side

Your Market Making Philosophy:
1. Provide liquidity by posting both bid and ask orders
2. Capture the bid-ask spread as profit
3. Manage inventory risk - avoid accumulating large positions
4. Adjust quotes based on market conditions and inventory

Risk Management:
- Keep inventory close to neutral (near zero net position)
- Widen spreads during high volatility
- Reduce size during uncertain markets
- Skew quotes to reduce unwanted inventory

Provide your reasoning before making a decision."""


if __name__ == "__main__":
    # Example usage
    agent = MarketMakerAgent("mm_1", initial_cash=100000)

    # Test quote calculation
    current_price = 100.0
    inventories = [-50, -25, 0, 25, 50]

    print("Market Maker Quote Analysis:")
    print("=" * 60)

    for inv in inventories:
        agent.holdings = inv
        bid_price, ask_price, bid_size, ask_size = agent.calculate_quotes(
            current_price, inv, volatility=0.02
        )
        spread = ask_price - bid_price
        spread_bps = (spread / current_price) * 10000

        print(f"Inventory: {inv:+4d} | "
              f"Bid: ${bid_price:.2f} x {bid_size:2d} | "
              f"Ask: ${ask_price:.2f} x {ask_size:2d} | "
              f"Spread: {spread_bps:.0f}bps")

    # Test decision making
    print("\nDecision with various inventories:")
    for inv in inventories:
        agent.holdings = inv
        decision = agent.make_decision(
            current_price=100.0,
            fundamental_value=100.0,
            price_history=[100.0] * 20,
            market_state={}
        )
        print(f"Inv: {inv:+4d} -> {decision.action.value.upper():4s} "
              f"{decision.quantity} @ ${decision.limit_price:.2f if decision.limit_price else 0:.2f}")
