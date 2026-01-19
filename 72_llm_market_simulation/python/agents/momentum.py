"""
Momentum Trader Agent

Implements a momentum trading strategy that follows price trends
using moving average crossovers and recent returns.
"""

from typing import Dict, Any, List
import numpy as np
from .base import BaseAgent, AgentDecision, ActionType


class MomentumTraderAgent(BaseAgent):
    """
    Momentum Trader Agent

    Follows price trends by analyzing moving averages and recent returns.
    Buys when momentum is positive, sells when momentum reverses.

    Attributes:
        short_window: Period for short-term moving average
        long_window: Period for long-term moving average
        entry_threshold: Minimum momentum signal to enter trade
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
    """

    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 100000.0,
        short_window: int = 5,
        long_window: int = 20,
        entry_threshold: float = 0.02,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        position_size_pct: float = 0.25
    ):
        """
        Initialize momentum trader agent

        Args:
            agent_id: Unique identifier
            initial_cash: Starting cash balance
            short_window: Short MA period
            long_window: Long MA period
            entry_threshold: Minimum momentum for entry
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            position_size_pct: Position size as fraction of cash
        """
        super().__init__(agent_id, initial_cash, "momentum_trader")
        self.short_window = short_window
        self.long_window = long_window
        self.entry_threshold = entry_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_size_pct = position_size_pct

        # Track entry price for stop loss / take profit
        self.entry_price: float = 0.0
        self.position_open: bool = False

    def make_decision(
        self,
        current_price: float,
        fundamental_value: float,
        price_history: List[float],
        market_state: Dict[str, Any]
    ) -> AgentDecision:
        """
        Make momentum-based trading decision

        Uses moving average crossover and recent returns to identify trends.

        Args:
            current_price: Current market price
            fundamental_value: Estimated fundamental value (ignored by momentum)
            price_history: List of historical prices
            market_state: Additional market information

        Returns:
            AgentDecision with action details
        """
        # Need enough history for analysis
        if len(price_history) < self.long_window:
            return AgentDecision(
                action=ActionType.HOLD,
                quantity=0,
                order_type="market",
                reasoning=f"Insufficient history: need {self.long_window} periods, have {len(price_history)}"
            )

        # Calculate indicators
        prices = np.array(price_history[-self.long_window:])
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices)

        # Momentum signal (MA ratio)
        momentum = (short_ma - long_ma) / long_ma

        # Recent return
        recent_return = (prices[-1] - prices[-min(5, len(prices))]) / prices[-min(5, len(prices))]

        # Check stop loss / take profit if in position
        if self.position_open and self.holdings > 0:
            position_return = (current_price - self.entry_price) / self.entry_price

            # Stop loss
            if position_return < -self.stop_loss_pct:
                self.position_open = False
                return AgentDecision(
                    action=ActionType.SELL,
                    quantity=self.holdings,
                    order_type="market",
                    reasoning=f"STOP LOSS triggered. Position return: {position_return*100:.1f}%. "
                              f"Exiting {self.holdings} shares at ${current_price:.2f}"
                )

            # Take profit
            if position_return > self.take_profit_pct:
                sell_qty = int(self.holdings * 0.5)  # Sell half
                if sell_qty > 0:
                    return AgentDecision(
                        action=ActionType.SELL,
                        quantity=sell_qty,
                        order_type="market",
                        reasoning=f"TAKE PROFIT. Position return: {position_return*100:.1f}%. "
                                  f"Selling {sell_qty} shares at ${current_price:.2f}"
                    )

        # BUY SIGNAL: Bullish momentum
        if momentum > self.entry_threshold and recent_return > 0:
            if not self.position_open and self.cash > 0:
                max_spend = self.cash * self.position_size_pct
                quantity = int(max_spend / current_price)

                if quantity > 0:
                    self.position_open = True
                    self.entry_price = current_price
                    return AgentDecision(
                        action=ActionType.BUY,
                        quantity=quantity,
                        order_type="market",
                        reasoning=f"BULLISH momentum detected. Short MA ({short_ma:.2f}) > "
                                  f"Long MA ({long_ma:.2f}). Momentum: {momentum*100:.2f}%. "
                                  f"Recent return: {recent_return*100:.1f}%. Buying {quantity} shares."
                    )

        # SELL SIGNAL: Bearish momentum
        elif momentum < -self.entry_threshold and self.holdings > 0:
            self.position_open = False
            return AgentDecision(
                action=ActionType.SELL,
                quantity=self.holdings,
                order_type="market",
                reasoning=f"BEARISH momentum detected. Short MA ({short_ma:.2f}) < "
                          f"Long MA ({long_ma:.2f}). Momentum: {momentum*100:.2f}%. "
                          f"Exiting position: {self.holdings} shares."
            )

        # HOLD: No clear signal
        return AgentDecision(
            action=ActionType.HOLD,
            quantity=0,
            order_type="market",
            reasoning=f"No clear momentum signal. Short MA: {short_ma:.2f}, "
                      f"Long MA: {long_ma:.2f}, Momentum: {momentum*100:.2f}%"
        )

    def get_system_prompt(self) -> str:
        """Get system prompt for LLM-based decision making"""
        return f"""You are a momentum trader agent in a simulated stock market.

Your Strategy Parameters:
- Short MA window: {self.short_window} periods
- Long MA window: {self.long_window} periods
- Entry threshold: {self.entry_threshold*100}% momentum
- Stop loss: {self.stop_loss_pct*100}%
- Take profit: {self.take_profit_pct*100}%
- Position size: {self.position_size_pct*100}% of cash

Your Trading Philosophy:
1. Analyze recent price trends (look at last {self.long_window} periods)
2. Buy when you see upward momentum (short MA > long MA)
3. Sell when you see downward momentum or trend reversal
4. Use moving averages to identify trends
5. Cut losses quickly, let winners run

Risk Management:
- Set stop-loss at {self.stop_loss_pct*100}% below entry
- Take profits at {self.take_profit_pct*100}% gains
- Don't fight the trend

Provide your reasoning before making a decision."""

    def reset(self):
        """Reset agent to initial state"""
        super().reset()
        self.entry_price = 0.0
        self.position_open = False


if __name__ == "__main__":
    # Example usage
    agent = MomentumTraderAgent("momentum_1", initial_cash=100000)

    # Generate some price history with uptrend
    np.random.seed(42)
    prices = [100.0]
    for _ in range(30):
        change = np.random.normal(0.001, 0.02)  # Slight upward drift
        prices.append(prices[-1] * (1 + change))

    print("Testing momentum trader with simulated uptrend:")
    for i in range(25, 31):
        decision = agent.make_decision(
            current_price=prices[i],
            fundamental_value=100.0,
            price_history=prices[:i+1],
            market_state={}
        )
        print(f"Period {i}: Price=${prices[i]:.2f}")
        print(f"  Decision: {decision.action.value}")
        print(f"  Reasoning: {decision.reasoning[:100]}...")
        print()
