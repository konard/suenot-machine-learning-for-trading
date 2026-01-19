"""
Market Environment for LLM Market Simulation

Provides the complete market simulation environment including
order book, price history, fundamental value process, and metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
from .order_book import OrderBook, Order, OrderType, Side, OrderResult


@dataclass
class MarketState:
    """
    Current state of the market

    Attributes:
        current_price: Current market price
        fundamental_value: Estimated fundamental value
        best_bid: Best bid price
        best_ask: Best ask price
        spread: Bid-ask spread
        volume_24h: Trading volume in last 24 periods
        price_change_24h: Price change percentage in last 24 periods
        price_history: Recent price history
        time_step: Current time step
    """
    current_price: float
    fundamental_value: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    volume_24h: int
    price_change_24h: float
    price_history: List[float]
    time_step: int


class MarketEnvironment:
    """
    Simulated Market Environment

    A complete market simulation environment with order book,
    price discovery, fundamental value process, and metrics collection.

    Examples:
        >>> market = MarketEnvironment(initial_price=100.0)
        >>> order = Order(OrderType.LIMIT, Side.BUY, 10, price=99.5, agent_id="agent1")
        >>> result = market.submit_order("agent1", order)
        >>> market.step()
        >>> state = market.get_state()
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        fundamental_value: Optional[float] = None,
        tick_size: float = 0.01,
        volatility: float = 0.02,
        dividend_rate: float = 0.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize market environment

        Args:
            initial_price: Starting price for the asset
            fundamental_value: True fundamental value (defaults to initial_price)
            tick_size: Minimum price increment
            volatility: Daily volatility for fundamental value updates
            dividend_rate: Annual dividend rate
            random_seed: Random seed for reproducibility
        """
        self.order_book = OrderBook(tick_size)
        self.current_price = initial_price
        self.fundamental_value = fundamental_value or initial_price
        self.tick_size = tick_size
        self.volatility = volatility
        self.dividend_rate = dividend_rate

        # Random number generator
        self.rng = np.random.default_rng(random_seed)

        # History tracking
        self.price_history: List[float] = [initial_price]
        self.fundamental_history: List[float] = [self.fundamental_value]
        self.volume_history: List[int] = [0]
        self.spread_history: List[Optional[float]] = [None]

        # State
        self.time_step = 0
        self.total_volume = 0

        # Agent portfolios
        self.agent_portfolios: Dict[str, Dict[str, float]] = {}

    def register_agent(self, agent_id: str, initial_cash: float, initial_holdings: int = 0):
        """
        Register an agent with the market

        Args:
            agent_id: Unique identifier for the agent
            initial_cash: Starting cash balance
            initial_holdings: Starting share holdings
        """
        self.agent_portfolios[agent_id] = {
            "cash": initial_cash,
            "holdings": initial_holdings,
            "trades": 0,
            "pnl": 0.0
        }

    def submit_order(self, agent_id: str, order: Order) -> OrderResult:
        """
        Submit an order to the market

        Args:
            agent_id: ID of the agent submitting the order
            order: Order to submit

        Returns:
            OrderResult with fill information
        """
        order.agent_id = agent_id

        # Validate agent has sufficient resources
        if agent_id in self.agent_portfolios:
            portfolio = self.agent_portfolios[agent_id]

            if order.side == Side.BUY:
                max_cost = order.quantity * (order.price or self.current_price * 1.1)
                if portfolio["cash"] < max_cost:
                    return OrderResult(0, 0.0, order.quantity, "rejected", [])
            else:
                if portfolio["holdings"] < order.quantity:
                    return OrderResult(0, 0.0, order.quantity, "rejected", [])

        # Process order
        if order.order_type == OrderType.MARKET:
            result = self.order_book.execute_market_order(order)
        else:
            result = self.order_book.add_limit_order(order)

        # Update agent portfolio if filled
        if result.filled_quantity > 0 and agent_id in self.agent_portfolios:
            self._update_portfolio(agent_id, order.side, result)
            self.total_volume += result.filled_quantity

        # Update current price if trade occurred
        if result.fills:
            self.current_price = result.fills[-1][0]

        return result

    def _update_portfolio(self, agent_id: str, side: Side, result: OrderResult):
        """Update agent portfolio after a fill"""
        portfolio = self.agent_portfolios[agent_id]
        total_value = sum(p * q for p, q in result.fills)

        if side == Side.BUY:
            portfolio["cash"] -= total_value
            portfolio["holdings"] += result.filled_quantity
        else:
            portfolio["cash"] += total_value
            portfolio["holdings"] -= result.filled_quantity

        portfolio["trades"] += 1

    def step(self):
        """
        Advance the market by one time step

        Updates fundamental value, records history, and prepares for next period.
        """
        self.time_step += 1

        # Update fundamental value with random walk
        if self.volatility > 0:
            shock = self.rng.normal(0, self.volatility)
            self.fundamental_value *= (1 + shock)
            self.fundamental_value = max(self.fundamental_value, self.tick_size)

        # Add dividend if applicable
        if self.dividend_rate > 0:
            daily_dividend = self.dividend_rate / 252
            self.fundamental_value *= (1 + daily_dividend)

        # Update current price (use mid price if available, else last trade)
        mid = self.order_book.get_mid_price()
        if mid is not None:
            self.current_price = mid
        elif self.order_book.get_last_trade_price() is not None:
            self.current_price = self.order_book.get_last_trade_price()

        # Record history
        self.price_history.append(self.current_price)
        self.fundamental_history.append(self.fundamental_value)
        self.volume_history.append(self.total_volume)
        self.spread_history.append(self.order_book.get_spread())

        # Reset period volume
        self.total_volume = 0

    def get_state(self, history_length: int = 20) -> MarketState:
        """
        Get current market state

        Args:
            history_length: Number of historical prices to include

        Returns:
            MarketState object with current market information
        """
        # Calculate 24-period metrics
        lookback = min(24, len(self.price_history))
        if lookback > 1:
            price_change = (self.current_price - self.price_history[-lookback]) / self.price_history[-lookback] * 100
        else:
            price_change = 0.0

        volume_24h = sum(self.volume_history[-lookback:])

        return MarketState(
            current_price=self.current_price,
            fundamental_value=self.fundamental_value,
            best_bid=self.order_book.get_best_bid(),
            best_ask=self.order_book.get_best_ask(),
            spread=self.order_book.get_spread(),
            volume_24h=volume_24h,
            price_change_24h=price_change,
            price_history=list(self.price_history[-history_length:]),
            time_step=self.time_step
        )

    def get_agent_portfolio(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio information for an agent"""
        if agent_id not in self.agent_portfolios:
            return None

        portfolio = self.agent_portfolios[agent_id].copy()
        portfolio["portfolio_value"] = portfolio["cash"] + portfolio["holdings"] * self.current_price
        return portfolio

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate market quality metrics

        Returns:
            Dictionary with various market metrics
        """
        prices = np.array(self.price_history)
        fundamentals = np.array(self.fundamental_history)

        metrics = {}

        # Price discovery efficiency
        if len(prices) > 1:
            deviations = prices - fundamentals
            metrics["tracking_error"] = float(np.std(deviations))
            metrics["mean_deviation"] = float(np.mean(np.abs(deviations)))
            metrics["final_deviation_pct"] = float((prices[-1] - fundamentals[-1]) / fundamentals[-1] * 100)

        # Volatility
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            metrics["realized_volatility"] = float(np.std(returns) * np.sqrt(252))

        # Spread statistics
        spreads = [s for s in self.spread_history if s is not None]
        if spreads:
            metrics["avg_spread"] = float(np.mean(spreads))
            metrics["avg_spread_bps"] = float(np.mean(spreads) / self.current_price * 10000)

        # Volume statistics
        metrics["total_volume"] = sum(self.volume_history)
        metrics["avg_daily_volume"] = float(np.mean(self.volume_history))

        return metrics

    def reset(self, initial_price: Optional[float] = None):
        """
        Reset the market to initial state

        Args:
            initial_price: New initial price (optional)
        """
        if initial_price is not None:
            self.current_price = initial_price
            self.fundamental_value = initial_price
        else:
            self.current_price = self.price_history[0]
            self.fundamental_value = self.fundamental_history[0]

        self.order_book.clear()
        self.price_history = [self.current_price]
        self.fundamental_history = [self.fundamental_value]
        self.volume_history = [0]
        self.spread_history = [None]
        self.time_step = 0
        self.total_volume = 0

        # Reset agent portfolios to initial state
        for agent_id in self.agent_portfolios:
            self.agent_portfolios[agent_id] = {
                "cash": 100000.0,  # Default
                "holdings": 0,
                "trades": 0,
                "pnl": 0.0
            }


if __name__ == "__main__":
    # Example usage
    market = MarketEnvironment(
        initial_price=100.0,
        fundamental_value=100.0,
        volatility=0.02,
        random_seed=42
    )

    # Register agents
    market.register_agent("value_investor", 100000.0)
    market.register_agent("momentum_trader", 100000.0)

    # Simulate some trading
    for t in range(10):
        # Value investor buys if below fundamental
        state = market.get_state()
        if state.current_price < state.fundamental_value * 0.95:
            order = Order(OrderType.LIMIT, Side.BUY, 10, price=state.current_price * 1.01)
            result = market.submit_order("value_investor", order)
            print(f"T{t}: Value investor bought {result.filled_quantity} shares")

        # Add some market maker orders
        order = Order(OrderType.LIMIT, Side.BUY, 50, price=state.current_price * 0.99)
        market.submit_order("mm", order)
        order = Order(OrderType.LIMIT, Side.SELL, 50, price=state.current_price * 1.01)
        market.submit_order("mm", order)

        market.step()

    # Print metrics
    metrics = market.calculate_metrics()
    print(f"\nMarket Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
