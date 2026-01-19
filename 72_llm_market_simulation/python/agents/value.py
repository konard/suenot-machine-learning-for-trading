"""
Value Investor Agent

Implements a value investing strategy that buys undervalued assets
and sells overvalued ones based on fundamental value comparison.
"""

from typing import Dict, Any, List
from .base import BaseAgent, AgentDecision, ActionType


class ValueInvestorAgent(BaseAgent):
    """
    Value Investor Agent

    Buys when price is significantly below fundamental value,
    sells when price is significantly above fundamental value.

    Attributes:
        discount_threshold: Minimum discount to trigger buy (e.g., 0.10 for 10%)
        premium_threshold: Minimum premium to trigger sell
        max_position_pct: Maximum percentage of cash to use per trade
        patience: Number of periods to wait between trades
    """

    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 100000.0,
        discount_threshold: float = 0.10,
        premium_threshold: float = 0.10,
        max_position_pct: float = 0.30,
        patience: int = 5
    ):
        """
        Initialize value investor agent

        Args:
            agent_id: Unique identifier
            initial_cash: Starting cash balance
            discount_threshold: Buy when price is this much below value
            premium_threshold: Sell when price is this much above value
            max_position_pct: Max fraction of cash per trade
            patience: Minimum periods between trades
        """
        super().__init__(agent_id, initial_cash, "value_investor")
        self.discount_threshold = discount_threshold
        self.premium_threshold = premium_threshold
        self.max_position_pct = max_position_pct
        self.patience = patience
        self.periods_since_trade = patience  # Allow trading from start

    def make_decision(
        self,
        current_price: float,
        fundamental_value: float,
        price_history: List[float],
        market_state: Dict[str, Any]
    ) -> AgentDecision:
        """
        Make value-based trading decision

        Compares current price to fundamental value and decides
        whether to buy, sell, or hold.

        Args:
            current_price: Current market price
            fundamental_value: Estimated fundamental value
            price_history: List of historical prices
            market_state: Additional market information

        Returns:
            AgentDecision with action details
        """
        self.periods_since_trade += 1

        # Calculate value gap
        if fundamental_value <= 0:
            return AgentDecision(
                action=ActionType.HOLD,
                quantity=0,
                order_type="market",
                reasoning="Invalid fundamental value"
            )

        value_gap = (fundamental_value - current_price) / fundamental_value

        # Check patience constraint
        if self.periods_since_trade < self.patience:
            return AgentDecision(
                action=ActionType.HOLD,
                quantity=0,
                order_type="market",
                reasoning=f"Waiting (patience: {self.patience - self.periods_since_trade} periods left)"
            )

        # BUY SIGNAL: Price below fundamental
        if value_gap > self.discount_threshold:
            max_spend = self.cash * self.max_position_pct
            quantity = int(max_spend / current_price)

            if quantity > 0 and self.cash >= quantity * current_price:
                self.periods_since_trade = 0
                return AgentDecision(
                    action=ActionType.BUY,
                    quantity=quantity,
                    order_type="limit",
                    limit_price=current_price * 0.995,  # Slight discount
                    reasoning=f"Price ${current_price:.2f} is {value_gap*100:.1f}% below "
                              f"fundamental ${fundamental_value:.2f}. Buying {quantity} shares."
                )

        # SELL SIGNAL: Price above fundamental
        elif value_gap < -self.premium_threshold and self.holdings > 0:
            # Sell portion of holdings
            sell_quantity = max(1, int(self.holdings * 0.5))

            self.periods_since_trade = 0
            return AgentDecision(
                action=ActionType.SELL,
                quantity=sell_quantity,
                order_type="limit",
                limit_price=current_price * 1.005,  # Slight premium
                reasoning=f"Price ${current_price:.2f} is {-value_gap*100:.1f}% above "
                          f"fundamental ${fundamental_value:.2f}. Selling {sell_quantity} shares."
            )

        # HOLD: Price near fundamental
        return AgentDecision(
            action=ActionType.HOLD,
            quantity=0,
            order_type="market",
            reasoning=f"Price ${current_price:.2f} is within {abs(value_gap)*100:.1f}% of "
                      f"fundamental ${fundamental_value:.2f}. No action needed."
        )

    def get_system_prompt(self) -> str:
        """Get system prompt for LLM-based decision making"""
        return f"""You are a value investor agent in a simulated stock market.

Your Strategy Parameters:
- Discount threshold: {self.discount_threshold*100}% (buy when price is this much below value)
- Premium threshold: {self.premium_threshold*100}% (sell when price is this much above value)
- Maximum position size: {self.max_position_pct*100}% of available cash
- Patience: Wait at least {self.patience} periods between trades

Your Investment Philosophy:
1. Compare current price to fundamental value
2. Buy when price is significantly below fundamental value
3. Sell when price is significantly above fundamental value
4. Be patient - don't trade on small deviations
5. Consider your current portfolio allocation

Risk Management:
- Never invest more than {self.max_position_pct*100}% of cash in a single trade
- Maintain cash reserves for opportunities
- Consider transaction costs in your decisions

Provide your reasoning before making a decision."""


if __name__ == "__main__":
    # Example usage
    agent = ValueInvestorAgent("value_1", initial_cash=100000)

    # Simulate decisions
    test_cases = [
        (90.0, 100.0),   # 10% discount - should buy
        (95.0, 100.0),   # 5% discount - should hold
        (110.0, 100.0),  # 10% premium - should sell (if has holdings)
        (100.0, 100.0),  # Fair value - should hold
    ]

    for price, fundamental in test_cases:
        decision = agent.make_decision(price, fundamental, [price], {})
        print(f"Price: ${price}, Fundamental: ${fundamental}")
        print(f"  Decision: {decision.action.value}")
        print(f"  Reasoning: {decision.reasoning}")
        print()
