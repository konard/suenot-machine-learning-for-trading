# Chapter 311: Monte Carlo Tree Search for Trading

## Introduction

Monte Carlo Tree Search (MCTS) is a powerful decision-making algorithm that combines tree search with random sampling to navigate enormous decision spaces. Originally popularized by its success in the game of Go — most notably in DeepMind's AlphaGo and AlphaZero — MCTS has proven itself as a general-purpose planning algorithm capable of handling problems with vast combinatorial complexity.

Trading presents a natural fit for MCTS. At every moment, a trader faces a decision tree: buy, sell, hold, adjust position size, set stop-losses, or choose among multiple assets. Each action leads to a new market state, which in turn presents another set of decisions. The depth of this tree grows exponentially with the trading horizon, making exhaustive search impossible. MCTS addresses this by intelligently sampling the most promising branches while maintaining exploration of alternatives.

Unlike traditional reinforcement learning approaches that learn a single policy, MCTS performs online planning at decision time. It builds a search tree rooted at the current market state, simulates thousands of possible future trajectories, and selects the action that leads to the best expected outcome. This makes it particularly well-suited for:

- **Multi-step order execution**: Planning a sequence of trades to minimize market impact
- **Portfolio rebalancing**: Deciding which assets to trade and in what order
- **Options strategy selection**: Evaluating complex multi-leg strategies
- **Risk-aware decision making**: Balancing expected return against downside risk

The key advantage of MCTS over simpler approaches is its ability to look ahead multiple steps, consider the sequential nature of trading decisions, and adapt its search to focus on the most relevant parts of the decision space.

## Mathematical Foundations

### The UCT Algorithm

The core of modern MCTS is the Upper Confidence bounds applied to Trees (UCT) algorithm. UCT uses the UCB1 formula to balance exploitation (choosing actions with high estimated value) and exploration (trying less-visited actions):

$$UCT(s, a) = \bar{Q}(s, a) + c \cdot \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

Where:
- $\bar{Q}(s, a)$ is the average reward from taking action $a$ in state $s$
- $N(s)$ is the visit count of state $s$
- $N(s, a)$ is the number of times action $a$ has been taken from state $s$
- $c$ is the exploration constant (typically $\sqrt{2}$ for theoretical optimality, tuned empirically in practice)

The first term drives exploitation — favoring actions with historically high rewards. The second term drives exploration — favoring actions that have been tried fewer times, with the logarithmic parent count ensuring exploration diminishes but never stops.

### The Four Phases of MCTS

MCTS operates through four iterative phases:

#### 1. Selection

Starting from the root node, the algorithm traverses the tree by selecting children according to the UCT formula until it reaches a leaf node (a node that has unexpanded actions or has not been visited):

$$a^* = \arg\max_{a \in A(s)} \left[ \bar{Q}(s, a) + c \cdot \sqrt{\frac{\ln N(s)}{N(s, a)}} \right]$$

In trading, this means choosing the action sequence that best balances known profitable paths with unexplored alternatives.

#### 2. Expansion

At the leaf node, one or more child nodes are added to the tree, corresponding to untried actions. In a trading context, this means considering new actions (e.g., "buy 10 units", "sell 5 units", "hold") that haven't been explored from the current state.

#### 3. Simulation (Rollout)

From the newly expanded node, a simulation is run to a terminal state or a fixed horizon using a rollout policy (often random or based on simple heuristics). The simulation produces a reward signal:

$$R = \sum_{t=0}^{T} \gamma^t r_t$$

Where $\gamma$ is a discount factor and $r_t$ is the reward at step $t$. In trading, this could be the cumulative PnL over the simulation horizon, risk-adjusted return (Sharpe ratio), or a custom objective.

#### 4. Backpropagation

The simulation result is propagated back up the tree, updating visit counts and value estimates for every node along the path:

$$N(s, a) \leftarrow N(s, a) + 1$$
$$Q(s, a) \leftarrow Q(s, a) + R$$
$$\bar{Q}(s, a) = \frac{Q(s, a)}{N(s, a)}$$

After many iterations, the visit counts converge and the most-visited action from the root is selected as the best decision.

### Convergence Properties

MCTS with UCT is proven to converge to the optimal policy as the number of iterations approaches infinity. In practice, even a moderate number of iterations (1,000–10,000) often yields strong performance. The exploration term ensures that no part of the tree is permanently ignored, while exploitation focuses computational resources on the most promising branches.

## Applications in Trading

### Multi-Step Order Execution

When executing a large order, a trader must decide how to split it across time steps to minimize market impact. MCTS can plan optimal execution by modeling:

- **State**: Current position, remaining order size, recent price trajectory, order book depth
- **Actions**: Execute portion sizes (e.g., 10%, 20%, 50% of remaining), or wait
- **Reward**: Negative slippage cost, with penalties for unfilled portions

MCTS explores thousands of execution schedules and identifies sequences that balance urgency against market impact, adapting to changing conditions at each decision point.

### Portfolio Rebalancing Decisions

Portfolio rebalancing involves multiple correlated decisions: which assets to trade, in what order, and how much. MCTS handles this naturally:

- **State**: Current portfolio weights, target weights, transaction costs, market conditions
- **Actions**: Trade a specific asset toward target, skip, or adjust target
- **Reward**: Tracking error reduction minus transaction costs

The tree structure captures the sequential nature of rebalancing — trading one asset affects available capital and correlation exposures for subsequent trades.

### Risk-Aware Planning

By modifying the rollout evaluation, MCTS can incorporate risk considerations:

$$R_{risk} = \mathbb{E}[R] - \lambda \cdot \text{CVaR}_\alpha(R)$$

This allows the search to favor action sequences that not only maximize expected return but also limit tail risk.

## AlphaZero-Style MCTS with Learned Networks

The AlphaZero approach enhances MCTS with two neural networks:

1. **Policy Network** $p_\theta(a|s)$: Provides prior probabilities for actions, replacing the uniform prior in standard MCTS. This guides the search toward promising actions from the start.

2. **Value Network** $v_\theta(s)$: Estimates the expected outcome from state $s$, replacing or augmenting random rollouts. This dramatically improves simulation quality.

The modified UCT formula becomes:

$$UCT_{AZ}(s, a) = \bar{Q}(s, a) + c \cdot p_\theta(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

For trading, the policy network can be trained on historical data to capture common trading patterns, while the value network learns to predict future risk-adjusted returns. The networks are trained iteratively: MCTS generates training data through self-play, the networks are updated, and improved networks guide better MCTS searches.

This approach is particularly powerful because:
- The policy network reduces the branching factor by focusing on plausible actions
- The value network eliminates the need for random rollouts, which are noisy in financial markets
- The entire system improves through self-play without requiring labeled data

## Rust Implementation

The implementation in `rust/src/lib.rs` provides a complete MCTS framework for trading:

### Core Components

- **`TradingState`**: Represents the current market state including price history, position, balance, and step count
- **`TradingAction`**: Enumeration of possible actions (Buy, Sell, Hold) with configurable position sizes
- **`MCTSNode`**: Tree node storing visit counts, accumulated value, children, action taken, and parent reference
- **`MCTS`**: The main search engine with configurable exploration constant and iteration count

### Key Features

1. **UCT Selection**: Implements the UCB1 formula for tree traversal with configurable exploration parameter
2. **State-based Expansion**: Generates child nodes for all valid actions from the current trading state
3. **Random Rollout**: Simulates forward from leaf nodes using random action selection
4. **Backpropagation**: Updates value estimates along the entire path from leaf to root
5. **Bybit Integration**: Fetches real BTCUSDT kline data from Bybit's public API

### Usage

```rust
use monte_carlo_tree_search::{MCTS, TradingState};

let prices = vec![50000.0, 50100.0, 49900.0, 50200.0, 50050.0];
let state = TradingState::new(prices, 10000.0);
let mut mcts = MCTS::new(1.414, 1000);
let best_action = mcts.search(&state);
```

The example in `rust/examples/trading_example.rs` demonstrates fetching live data from Bybit and running MCTS to determine optimal trading actions.

## Bybit Data Integration

The implementation connects to Bybit's public REST API to fetch historical kline (candlestick) data:

```
GET https://api.bybit.com/v5/market/kline?category=spot&symbol=BTCUSDT&interval=60&limit=100
```

This provides hourly OHLCV data that feeds into the MCTS trading simulation. The integration:

1. Fetches recent price data for the specified trading pair
2. Extracts closing prices as the state representation
3. Initializes the MCTS search tree with the current market state
4. Runs the search to determine the optimal action

No API key is required for public market data endpoints.

## Key Takeaways

1. **MCTS excels at sequential decisions**: Trading involves chains of dependent decisions where each action affects future opportunities. MCTS naturally captures this sequential structure through its tree representation.

2. **UCT balances exploration and exploitation**: The mathematical foundation of UCT ensures that MCTS efficiently allocates its computational budget, focusing on promising strategies while maintaining awareness of alternatives.

3. **Online planning adapts to changing conditions**: Unlike offline-trained policies, MCTS plans at decision time using the current market state. This makes it inherently adaptive to regime changes and novel market conditions.

4. **AlphaZero-style enhancements are transformative**: Replacing random rollouts with learned value networks and guiding search with policy networks dramatically improves MCTS performance in complex domains like trading.

5. **Computational cost is the main limitation**: MCTS requires running many simulations at decision time. For high-frequency trading, this may be prohibitive, but for medium-frequency strategies (hourly, daily), it is highly practical.

6. **Risk integration is natural**: The rollout evaluation can incorporate any risk measure (VaR, CVaR, maximum drawdown), making MCTS suitable for risk-aware trading strategies.

7. **Scalability through parallelism**: MCTS simulations are largely independent and can be parallelized across CPU cores or even distributed across machines, making it scalable to large decision spaces.

8. **Domain knowledge improves performance**: Custom rollout policies, domain-specific state representations, and learned priors all significantly improve MCTS performance compared to vanilla implementations.
