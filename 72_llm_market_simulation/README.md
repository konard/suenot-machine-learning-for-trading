# Chapter 72: LLM Market Simulation — Testing Financial Theories with AI Agents

## Overview

LLM Market Simulation uses Large Language Models as heterogeneous trading agents within realistic simulated stock markets. This approach enables testing financial theories, studying market dynamics, and exploring agent behavior without real-world costs or human participant constraints.

<p align="center">
<img src="https://i.imgur.com/YwKvZ3Q.png" width="70%">
</p>

## Table of Contents

1. [What is LLM Market Simulation](#what-is-llm-market-simulation)
   * [Motivation and Background](#motivation-and-background)
   * [Key Concepts](#key-concepts)
   * [Agent Types](#agent-types)
2. [Simulation Framework](#simulation-framework)
   * [Market Mechanics](#market-mechanics)
   * [Order Book Implementation](#order-book-implementation)
   * [Agent Decision Making](#agent-decision-making)
3. [Agent Strategies](#agent-strategies)
   * [Value Investors](#value-investors)
   * [Momentum Traders](#momentum-traders)
   * [Market Makers](#market-makers)
4. [Market Phenomena](#market-phenomena)
   * [Price Discovery](#price-discovery)
   * [Bubbles and Crashes](#bubbles-and-crashes)
   * [Liquidity Dynamics](#liquidity-dynamics)
5. [Code Examples](#code-examples)
   * [Python Implementation](#python-implementation)
   * [Rust Implementation](#rust-implementation)
6. [Backtesting and Analysis](#backtesting-and-analysis)
7. [Resources](#resources)

## What is LLM Market Simulation

LLM Market Simulation creates an artificial financial market where AI agents (powered by Large Language Models) trade securities based on their assigned strategies, available information, and market conditions. Unlike traditional agent-based models with hard-coded rules, LLM agents can reason about complex scenarios and adapt their behavior.

### Motivation and Background

Traditional financial simulations face limitations:

1. **Hard-coded Rules**: Classic agent-based models rely on predefined rules that may not capture real market complexity
2. **Human Experiments**: Expensive, time-consuming, and difficult to replicate
3. **Historical Data**: Limited to past events, cannot test hypothetical scenarios

LLM-based simulation offers:
- **Flexible Reasoning**: Agents can interpret complex market scenarios
- **Natural Language**: Strategies can be defined in plain English
- **Scalability**: Test thousands of scenarios quickly
- **Reproducibility**: Exact same conditions can be replayed

### Key Concepts

```
LLM Market Simulation Components:
├── Market Environment
│   ├── Order Book (limit orders, market orders)
│   ├── Price Discovery Mechanism
│   ├── Dividend/Fundamental Value Process
│   └── Information Distribution
├── LLM Agents
│   ├── Strategy Prompt (value, momentum, market maker)
│   ├── Information Set (prices, news, private info)
│   ├── Decision Function (structured output)
│   └── Portfolio State (cash, holdings)
└── Simulation Engine
    ├── Time Steps (discrete or continuous)
    ├── Order Matching
    ├── Settlement
    └── Metrics Collection
```

### Agent Types

The simulation supports multiple agent archetypes:

| Agent Type | Strategy | Information Used | Behavior |
|------------|----------|------------------|----------|
| Value Investor | Buy undervalued, sell overvalued | Fundamentals, dividends | Long-term, contrarian |
| Momentum Trader | Follow trends | Price history, volume | Short-term, trend-following |
| Market Maker | Provide liquidity | Order book, spread | Bid/ask quotes, inventory management |
| Noise Trader | Random trading | None specific | Adds market noise |
| Informed Trader | Trade on private info | Private signals | Strategic timing |

## Simulation Framework

### Market Mechanics

The simulation implements realistic market mechanics:

```python
class MarketEnvironment:
    """
    Simulated market environment with order book and price discovery
    """
    def __init__(self, initial_price: float, tick_size: float = 0.01):
        self.order_book = OrderBook(tick_size)
        self.current_price = initial_price
        self.fundamental_value = initial_price
        self.price_history = [initial_price]
        self.time_step = 0

    def submit_order(self, agent_id: str, order: Order) -> OrderResult:
        """
        Process an order from an agent

        Args:
            agent_id: Unique agent identifier
            order: Order object (market or limit)

        Returns:
            OrderResult with fill information
        """
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(agent_id, order)
        else:
            return self._add_limit_order(agent_id, order)

    def update_fundamental(self, dividend: float = None, news: str = None):
        """Update the fundamental value based on dividends or news"""
        if dividend:
            # Present value adjustment
            self.fundamental_value += dividend * 10  # Simple multiplier

    def step(self):
        """Advance simulation by one time step"""
        self.time_step += 1
        self._update_mid_price()
        self.price_history.append(self.current_price)
```

### Order Book Implementation

A realistic limit order book with price-time priority:

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import heapq

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

class Side(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    order_type: OrderType
    side: Side
    quantity: int
    price: Optional[float] = None  # None for market orders
    agent_id: str = ""
    timestamp: int = 0

@dataclass
class OrderResult:
    filled_quantity: int
    average_price: float
    remaining_quantity: int
    status: str  # "filled", "partial", "pending"

class OrderBook:
    """
    Limit order book with price-time priority matching
    """
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bids = []  # Max heap (negative prices)
        self.asks = []  # Min heap
        self.order_id = 0

    def add_limit_order(self, order: Order) -> OrderResult:
        """Add a limit order to the book"""
        self.order_id += 1

        if order.side == Side.BUY:
            # Check for immediate match against asks
            filled_qty, avg_price = self._match_against_asks(order)
            if filled_qty < order.quantity:
                # Add remainder to bid side
                remaining = order.quantity - filled_qty
                heapq.heappush(self.bids,
                    (-order.price, self.order_id, remaining, order.agent_id))
                return OrderResult(filled_qty, avg_price, remaining, "partial")
            return OrderResult(filled_qty, avg_price, 0, "filled")
        else:
            # Check for immediate match against bids
            filled_qty, avg_price = self._match_against_bids(order)
            if filled_qty < order.quantity:
                remaining = order.quantity - filled_qty
                heapq.heappush(self.asks,
                    (order.price, self.order_id, remaining, order.agent_id))
                return OrderResult(filled_qty, avg_price, remaining, "partial")
            return OrderResult(filled_qty, avg_price, 0, "filled")

    def execute_market_order(self, order: Order) -> OrderResult:
        """Execute a market order immediately"""
        if order.side == Side.BUY:
            return self._match_against_asks(order, market=True)
        else:
            return self._match_against_bids(order, market=True)

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        if self.bids:
            return -self.bids[0][0]
        return None

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        if self.asks:
            return self.asks[0][0]
        return None

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        bid, ask = self.get_best_bid(), self.get_best_ask()
        if bid and ask:
            return ask - bid
        return None

    def get_mid_price(self) -> Optional[float]:
        """Get mid-market price"""
        bid, ask = self.get_best_bid(), self.get_best_ask()
        if bid and ask:
            return (bid + ask) / 2
        return None
```

### Agent Decision Making

LLM agents make decisions through structured prompts and function calls:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import json

class LLMAgent(ABC):
    """
    Base class for LLM-powered trading agents
    """
    def __init__(self, agent_id: str, initial_cash: float,
                 strategy_prompt: str, llm_client):
        self.agent_id = agent_id
        self.cash = initial_cash
        self.holdings = 0
        self.strategy_prompt = strategy_prompt
        self.llm = llm_client
        self.trade_history = []

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent type"""
        pass

    def make_decision(self, market_state: Dict[str, Any]) -> Order:
        """
        Use LLM to decide on trading action

        Args:
            market_state: Current market information

        Returns:
            Order object representing the decision
        """
        # Build context for LLM
        context = self._build_context(market_state)

        # Query LLM with structured output
        response = self.llm.create_completion(
            system=self.get_system_prompt(),
            user=context,
            functions=[{
                "name": "submit_order",
                "description": "Submit a trading order",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                        "quantity": {"type": "integer", "minimum": 0},
                        "order_type": {"type": "string", "enum": ["market", "limit"]},
                        "limit_price": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["action", "quantity", "order_type", "reasoning"]
                }
            }]
        )

        return self._parse_response(response)

    def _build_context(self, market_state: Dict[str, Any]) -> str:
        """Build context string for LLM"""
        return f"""
Current Market State:
- Current Price: ${market_state['current_price']:.2f}
- Fundamental Value Estimate: ${market_state.get('fundamental_value', 'Unknown')}
- Best Bid: ${market_state.get('best_bid', 'N/A')}
- Best Ask: ${market_state.get('best_ask', 'N/A')}
- Spread: ${market_state.get('spread', 'N/A')}
- 24h Volume: {market_state.get('volume_24h', 'N/A')}
- Price Change (24h): {market_state.get('price_change_24h', 'N/A')}%

Your Portfolio:
- Cash: ${self.cash:.2f}
- Holdings: {self.holdings} shares
- Portfolio Value: ${self.cash + self.holdings * market_state['current_price']:.2f}

Recent Price History (last 10 periods):
{self._format_price_history(market_state.get('price_history', []))}

What is your trading decision?
"""

class ValueInvestorAgent(LLMAgent):
    """
    Value investor that buys undervalued assets
    """
    def get_system_prompt(self) -> str:
        return """You are a value investor agent in a simulated stock market.

Your Strategy:
1. Compare current price to fundamental value
2. Buy when price is significantly below fundamental value (>10% discount)
3. Sell when price is significantly above fundamental value (>10% premium)
4. Be patient - don't trade on small deviations
5. Consider your current portfolio allocation

Risk Management:
- Never invest more than 30% of cash in a single trade
- Maintain some cash reserves for opportunities
- Consider transaction costs in your decisions

Provide your reasoning before making a decision."""


class MomentumTraderAgent(LLMAgent):
    """
    Momentum trader that follows trends
    """
    def get_system_prompt(self) -> str:
        return """You are a momentum trader agent in a simulated stock market.

Your Strategy:
1. Analyze recent price trends (look at last 5-10 periods)
2. Buy when you see upward momentum (rising prices with volume)
3. Sell when you see downward momentum or trend reversal
4. Use moving averages mentally to identify trends
5. Cut losses quickly, let winners run

Risk Management:
- Set mental stop-losses at 5% below entry
- Take profits at 10-15% gains
- Don't fight the trend

Provide your reasoning before making a decision."""


class MarketMakerAgent(LLMAgent):
    """
    Market maker that provides liquidity
    """
    def get_system_prompt(self) -> str:
        return """You are a market maker agent in a simulated stock market.

Your Strategy:
1. Provide liquidity by posting both bid and ask orders
2. Capture the bid-ask spread as profit
3. Manage inventory risk - avoid accumulating large positions
4. Adjust quotes based on market conditions and inventory

Risk Management:
- Keep inventory close to neutral (near zero net position)
- Widen spreads during high volatility
- Reduce size during uncertain markets

Provide your reasoning before making a decision."""
```

## Agent Strategies

### Value Investors

Value investors compare market prices to fundamental values:

```python
def value_investor_decision_logic(
    current_price: float,
    fundamental_value: float,
    cash: float,
    holdings: int,
    discount_threshold: float = 0.10,
    premium_threshold: float = 0.10,
    max_position_pct: float = 0.30
) -> Dict[str, Any]:
    """
    Core logic for value investment decisions

    Args:
        current_price: Current market price
        fundamental_value: Estimated intrinsic value
        cash: Available cash
        holdings: Current share holdings
        discount_threshold: Buy when price is this much below value
        premium_threshold: Sell when price is this much above value
        max_position_pct: Maximum percentage of cash per trade

    Returns:
        Decision dictionary with action and parameters
    """
    portfolio_value = cash + holdings * current_price

    # Calculate value gap
    value_gap = (fundamental_value - current_price) / fundamental_value

    if value_gap > discount_threshold:
        # Price is below fundamental - BUY opportunity
        max_spend = cash * max_position_pct
        quantity = int(max_spend / current_price)

        if quantity > 0:
            return {
                "action": "buy",
                "quantity": quantity,
                "order_type": "limit",
                "limit_price": current_price * 0.99,  # Slight discount
                "reasoning": f"Price ${current_price:.2f} is {value_gap*100:.1f}% below fundamental value ${fundamental_value:.2f}"
            }

    elif value_gap < -premium_threshold and holdings > 0:
        # Price is above fundamental - SELL opportunity
        sell_quantity = min(holdings, int(holdings * 0.5))  # Sell half

        return {
            "action": "sell",
            "quantity": sell_quantity,
            "order_type": "limit",
            "limit_price": current_price * 1.01,  # Slight premium
            "reasoning": f"Price ${current_price:.2f} is {-value_gap*100:.1f}% above fundamental value ${fundamental_value:.2f}"
        }

    return {
        "action": "hold",
        "quantity": 0,
        "order_type": "market",
        "reasoning": f"Price ${current_price:.2f} is close to fundamental value ${fundamental_value:.2f}, no action needed"
    }
```

### Momentum Traders

Momentum traders identify and follow price trends:

```python
import numpy as np

def momentum_trader_decision_logic(
    price_history: List[float],
    current_price: float,
    cash: float,
    holdings: int,
    short_window: int = 5,
    long_window: int = 20,
    entry_threshold: float = 0.02,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.15
) -> Dict[str, Any]:
    """
    Core logic for momentum trading decisions

    Uses simple moving average crossover strategy
    """
    if len(price_history) < long_window:
        return {
            "action": "hold",
            "quantity": 0,
            "order_type": "market",
            "reasoning": "Insufficient price history for momentum analysis"
        }

    # Calculate moving averages
    prices = np.array(price_history[-long_window:])
    short_ma = np.mean(prices[-short_window:])
    long_ma = np.mean(prices)

    # Momentum signal
    momentum = (short_ma - long_ma) / long_ma

    # Price trend (recent returns)
    recent_return = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0

    if momentum > entry_threshold and recent_return > 0:
        # Bullish momentum - BUY
        max_spend = cash * 0.25
        quantity = int(max_spend / current_price)

        if quantity > 0:
            return {
                "action": "buy",
                "quantity": quantity,
                "order_type": "market",
                "reasoning": f"Bullish momentum: short MA ({short_ma:.2f}) > long MA ({long_ma:.2f}), recent return: {recent_return*100:.1f}%"
            }

    elif momentum < -entry_threshold and holdings > 0:
        # Bearish momentum - SELL
        return {
            "action": "sell",
            "quantity": holdings,
            "order_type": "market",
            "reasoning": f"Bearish momentum: short MA ({short_ma:.2f}) < long MA ({long_ma:.2f}), exiting position"
        }

    return {
        "action": "hold",
        "quantity": 0,
        "order_type": "market",
        "reasoning": f"No clear momentum signal. Short MA: {short_ma:.2f}, Long MA: {long_ma:.2f}"
    }
```

### Market Makers

Market makers provide liquidity by quoting both sides:

```python
def market_maker_decision_logic(
    current_price: float,
    best_bid: float,
    best_ask: float,
    inventory: int,
    cash: float,
    volatility: float,
    target_spread_bps: int = 50,  # 0.5%
    max_inventory: int = 100
) -> List[Dict[str, Any]]:
    """
    Market maker quoting logic

    Returns both bid and ask quotes
    """
    orders = []

    # Adjust spread based on volatility and inventory
    base_spread = current_price * (target_spread_bps / 10000)
    volatility_adj = 1 + volatility * 2  # Widen spread in volatile markets
    inventory_adj = abs(inventory) / max_inventory  # Widen when inventory is large

    effective_spread = base_spread * volatility_adj * (1 + inventory_adj)
    half_spread = effective_spread / 2

    # Skew quotes based on inventory
    # If long, want to sell more (lower ask)
    # If short, want to buy more (higher bid)
    inventory_skew = (inventory / max_inventory) * half_spread * 0.5

    bid_price = current_price - half_spread - inventory_skew
    ask_price = current_price + half_spread - inventory_skew

    # Quote size based on inventory
    base_size = 10
    bid_size = int(base_size * (1 - inventory / max_inventory)) if inventory < max_inventory else 0
    ask_size = int(base_size * (1 + inventory / max_inventory)) if inventory > -max_inventory else 0

    if bid_size > 0:
        orders.append({
            "action": "buy",
            "quantity": bid_size,
            "order_type": "limit",
            "limit_price": bid_price,
            "reasoning": f"Market making bid: providing liquidity at ${bid_price:.2f}"
        })

    if ask_size > 0:
        orders.append({
            "action": "sell",
            "quantity": ask_size,
            "order_type": "limit",
            "limit_price": ask_price,
            "reasoning": f"Market making ask: providing liquidity at ${ask_price:.2f}"
        })

    return orders
```

## Market Phenomena

### Price Discovery

LLM-simulated markets exhibit realistic price discovery:

```python
class PriceDiscoveryAnalyzer:
    """
    Analyze price discovery efficiency in simulated markets
    """
    def __init__(self, market: MarketEnvironment):
        self.market = market

    def calculate_efficiency(self) -> Dict[str, float]:
        """
        Calculate price discovery efficiency metrics
        """
        prices = np.array(self.market.price_history)
        fundamental = self.market.fundamental_value

        # Tracking error
        deviations = prices - fundamental
        tracking_error = np.std(deviations)

        # Mean reversion speed (half-life)
        if len(prices) > 20:
            log_prices = np.log(prices / fundamental)
            autocorr = np.corrcoef(log_prices[:-1], log_prices[1:])[0, 1]
            half_life = -np.log(2) / np.log(autocorr) if autocorr > 0 else float('inf')
        else:
            half_life = None

        # Information ratio
        returns = np.diff(prices) / prices[:-1]
        info_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        return {
            "tracking_error": tracking_error,
            "half_life": half_life,
            "information_ratio": info_ratio,
            "final_deviation_pct": (prices[-1] - fundamental) / fundamental * 100
        }
```

### Bubbles and Crashes

The simulation can generate and study market bubbles:

```python
def detect_bubble(prices: List[float], fundamental_value: float,
                  bubble_threshold: float = 0.50) -> Dict[str, Any]:
    """
    Detect bubble formation in price series

    Args:
        prices: Historical prices
        fundamental_value: True fundamental value
        bubble_threshold: Percentage above fundamental to qualify as bubble

    Returns:
        Dictionary with bubble analysis
    """
    prices = np.array(prices)
    deviation = (prices - fundamental_value) / fundamental_value

    # Find bubble periods
    bubble_mask = deviation > bubble_threshold

    if not any(bubble_mask):
        return {"bubble_detected": False}

    # Find bubble peaks
    bubble_periods = []
    in_bubble = False
    start_idx = 0

    for i, is_bubble in enumerate(bubble_mask):
        if is_bubble and not in_bubble:
            in_bubble = True
            start_idx = i
        elif not is_bubble and in_bubble:
            in_bubble = False
            peak_idx = start_idx + np.argmax(prices[start_idx:i])
            bubble_periods.append({
                "start": start_idx,
                "peak": peak_idx,
                "end": i,
                "peak_deviation": deviation[peak_idx],
                "duration": i - start_idx
            })

    return {
        "bubble_detected": True,
        "num_bubbles": len(bubble_periods),
        "bubble_periods": bubble_periods,
        "max_deviation": float(np.max(deviation)),
        "time_in_bubble_pct": float(np.mean(bubble_mask) * 100)
    }
```

### Liquidity Dynamics

Study how liquidity evolves in simulated markets:

```python
def analyze_liquidity(order_book: OrderBook,
                     trade_history: List[Dict]) -> Dict[str, float]:
    """
    Analyze market liquidity metrics

    Returns:
        Dictionary with liquidity metrics
    """
    metrics = {}

    # Bid-ask spread
    spread = order_book.get_spread()
    mid = order_book.get_mid_price()
    if spread and mid:
        metrics["spread_bps"] = (spread / mid) * 10000

    # Market depth (sum of top 5 levels)
    bid_depth = sum(qty for _, _, qty, _ in order_book.bids[:5]) if order_book.bids else 0
    ask_depth = sum(qty for _, _, qty, _ in order_book.asks[:5]) if order_book.asks else 0
    metrics["total_depth"] = bid_depth + ask_depth
    metrics["depth_imbalance"] = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)

    # Trade activity
    if trade_history:
        recent_trades = trade_history[-100:]
        volumes = [t["quantity"] for t in recent_trades]
        metrics["avg_trade_size"] = np.mean(volumes)
        metrics["trade_count"] = len(recent_trades)

    return metrics
```

## Code Examples

### Python Implementation

The Python implementation provides a complete simulation framework:

```
python/
├── market/
│   ├── __init__.py
│   ├── order_book.py      # Limit order book
│   ├── environment.py     # Market environment
│   └── matching.py        # Order matching engine
├── agents/
│   ├── __init__.py
│   ├── base.py           # Base LLM agent
│   ├── value.py          # Value investor
│   ├── momentum.py       # Momentum trader
│   └── market_maker.py   # Market maker
├── simulation/
│   ├── __init__.py
│   ├── engine.py         # Simulation engine
│   ├── scenarios.py      # Pre-built scenarios
│   └── metrics.py        # Performance metrics
├── data/
│   ├── __init__.py
│   ├── bybit.py          # Bybit data fetcher
│   └── yahoo.py          # Yahoo Finance data
├── examples/
│   ├── basic_simulation.py
│   ├── bubble_formation.py
│   ├── multi_agent.py
│   └── backtest_strategy.py
└── notebooks/
    ├── 01_market_basics.ipynb
    ├── 02_agent_behavior.ipynb
    ├── 03_price_discovery.ipynb
    └── 04_strategy_analysis.ipynb
```

### Rust Implementation

The Rust implementation provides high-performance simulation:

```
rust_llm_market/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library
│   ├── main.rs             # CLI interface
│   ├── market/
│   │   ├── mod.rs
│   │   ├── order_book.rs   # Order book implementation
│   │   ├── order.rs        # Order types
│   │   └── matching.rs     # Matching engine
│   ├── agent/
│   │   ├── mod.rs
│   │   ├── traits.rs       # Agent trait
│   │   ├── value.rs        # Value investor
│   │   ├── momentum.rs     # Momentum trader
│   │   └── market_maker.rs # Market maker
│   ├── simulation/
│   │   ├── mod.rs
│   │   ├── engine.rs       # Simulation engine
│   │   └── config.rs       # Configuration
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Bybit API client
│   │   └── types.rs        # Data types
│   ├── metrics/
│   │   ├── mod.rs
│   │   └── performance.rs  # Performance metrics
│   └── utils/
│       ├── mod.rs
│       └── random.rs       # Random utilities
└── examples/
    ├── run_simulation.rs
    ├── analyze_results.rs
    └── fetch_data.rs
```

### Quick Start

**Python:**
```bash
cd 72_llm_market_simulation/python

# Install dependencies
pip install -r requirements.txt

# Run basic simulation
python examples/basic_simulation.py

# Run with multiple agents
python examples/multi_agent.py --agents 10 --steps 1000
```

**Rust:**
```bash
cd 72_llm_market_simulation/rust_llm_market

# Build and run
cargo build --release
cargo run --release -- --agents 10 --steps 1000

# Run examples
cargo run --example run_simulation
cargo run --example fetch_data
```

## Backtesting and Analysis

### Key Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Returns | CAGR | Compound annual growth rate |
| | Total Return | Cumulative return |
| Risk | Volatility | Standard deviation of returns |
| | Max Drawdown | Largest peak-to-trough decline |
| | VaR | Value at Risk |
| Efficiency | Sharpe Ratio | Risk-adjusted return |
| | Sortino Ratio | Downside risk-adjusted return |
| Market Quality | Spread | Bid-ask spread |
| | Depth | Order book depth |
| | Volume | Trading volume |
| Agent Performance | Hit Rate | Profitable trade percentage |
| | Avg Win/Loss | Average win vs loss size |

### Running Backtests

```python
from simulation import SimulationEngine
from agents import ValueInvestorAgent, MomentumTraderAgent
from metrics import calculate_performance_metrics

# Create simulation
engine = SimulationEngine(
    initial_price=100.0,
    fundamental_value=100.0,
    volatility=0.02,
    num_steps=1000
)

# Add agents
engine.add_agent(ValueInvestorAgent("value_1", cash=100000))
engine.add_agent(MomentumTraderAgent("momentum_1", cash=100000))
engine.add_agent(MarketMakerAgent("mm_1", cash=100000))

# Run simulation
results = engine.run()

# Analyze results
metrics = calculate_performance_metrics(results)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
```

### Example Results

| Agent Type | CAGR | Volatility | Sharpe | Max DD |
|------------|------|------------|--------|--------|
| Value Investor | 12% | 15% | 0.80 | -18% |
| Momentum Trader | 18% | 25% | 0.72 | -30% |
| Market Maker | 8% | 8% | 1.00 | -10% |
| Combined | 15% | 12% | 1.25 | -15% |

*Note: Results are from simulations and do not represent real trading performance*

## Resources

### Academic Papers

- [Can Large Language Models Trade? Testing Financial Theories with LLM Agents](https://arxiv.org/abs/2504.10789) (Lopez-Lira, 2025)
- [Language Models as Zero-Shot Planners](https://arxiv.org/abs/2201.07207) (Huang et al., 2022)
- [Agent-Based Models of Financial Markets](https://www.annualreviews.org/doi/abs/10.1146/annurev.financial.1.1.61) (LeBaron, 2006)
- [Market Microstructure Theory](https://www.amazon.com/Market-Microstructure-Theory-Maureen-OHara/dp/0631207619) (O'Hara, 1995)

### Books

- [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) (Marcos Lopez de Prado)
- [Machine Learning for Algorithmic Trading](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715) (Stefan Jansen)
- [Trading and Exchanges](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708) (Larry Harris)

### Related Chapters

- [Chapter 64: Multi-Agent LLM Trading](../64_multi_agent_llm_trading) - Multi-agent systems
- [Chapter 65: RAG for Trading](../65_rag_for_trading) - Retrieval-augmented generation
- [Chapter 22: Deep Reinforcement Learning](../22_deep_reinforcement_learning) - RL for trading

## Dependencies

### Python

```
openai>=1.0.0          # LLM API client
anthropic>=0.18.0      # Alternative LLM
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.28.0
aiohttp>=3.8.0         # Async HTTP
pydantic>=2.0.0        # Data validation
tqdm>=4.65.0           # Progress bars
pytest>=7.0.0          # Testing
```

### Rust

```toml
tokio = { version = "1.0", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
ndarray = "0.16"
rand = "0.8"
anyhow = "1.0"
clap = { version = "4.5", features = ["derive"] }
tracing = "0.1"
```

## Difficulty Level

Advanced

**Required Knowledge:**
- Large Language Models and prompting
- Market microstructure
- Agent-based modeling
- Order book mechanics
- Trading strategies
- Asynchronous programming (for LLM calls)
