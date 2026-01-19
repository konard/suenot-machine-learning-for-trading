"""
Base Agent Class for LLM Market Simulation

Abstract base class for all trading agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class ActionType(Enum):
    """Type of trading action"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class AgentDecision:
    """
    Represents an agent's trading decision

    Attributes:
        action: Type of action (buy, sell, hold)
        quantity: Number of shares to trade
        order_type: Type of order (market or limit)
        limit_price: Price for limit orders
        reasoning: Agent's reasoning for the decision
    """
    action: ActionType
    quantity: int
    order_type: str  # "market" or "limit"
    limit_price: Optional[float] = None
    reasoning: str = ""


class BaseAgent(ABC):
    """
    Abstract Base Class for Trading Agents

    All trading agents inherit from this class and implement
    the make_decision method.

    Attributes:
        agent_id: Unique identifier for the agent
        initial_cash: Starting cash balance
        strategy_name: Name of the trading strategy
    """

    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 100000.0,
        strategy_name: str = "base"
    ):
        """
        Initialize agent

        Args:
            agent_id: Unique identifier
            initial_cash: Starting cash balance
            strategy_name: Name of the trading strategy
        """
        self.agent_id = agent_id
        self.initial_cash = initial_cash
        self.strategy_name = strategy_name
        self.cash = initial_cash
        self.holdings = 0
        self.trade_history: List[Dict[str, Any]] = []

    @abstractmethod
    def make_decision(
        self,
        current_price: float,
        fundamental_value: float,
        price_history: List[float],
        market_state: Dict[str, Any]
    ) -> AgentDecision:
        """
        Make a trading decision based on market state

        Args:
            current_price: Current market price
            fundamental_value: Estimated fundamental value
            price_history: List of historical prices
            market_state: Additional market information

        Returns:
            AgentDecision with action details
        """
        pass

    def update_portfolio(self, action: ActionType, quantity: int, price: float):
        """
        Update agent's portfolio after a trade

        Args:
            action: Type of action executed
            quantity: Number of shares traded
            price: Price at which trade executed
        """
        if action == ActionType.BUY:
            self.cash -= quantity * price
            self.holdings += quantity
        elif action == ActionType.SELL:
            self.cash += quantity * price
            self.holdings -= quantity

        self.trade_history.append({
            "action": action.value,
            "quantity": quantity,
            "price": price,
            "cash_after": self.cash,
            "holdings_after": self.holdings
        })

    def get_portfolio_value(self, current_price: float) -> float:
        """
        Calculate total portfolio value

        Args:
            current_price: Current market price

        Returns:
            Total portfolio value (cash + holdings * price)
        """
        return self.cash + self.holdings * current_price

    def get_return(self, current_price: float) -> float:
        """
        Calculate return since inception

        Args:
            current_price: Current market price

        Returns:
            Return as a decimal (e.g., 0.10 for 10%)
        """
        current_value = self.get_portfolio_value(current_price)
        return (current_value - self.initial_cash) / self.initial_cash

    def reset(self):
        """Reset agent to initial state"""
        self.cash = self.initial_cash
        self.holdings = 0
        self.trade_history.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, strategy={self.strategy_name})"
