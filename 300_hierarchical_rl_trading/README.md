# Chapter 300: Hierarchical Reinforcement Learning for Trading

## Introduction

Traditional reinforcement learning (RL) agents for trading operate at a single temporal abstraction level: at each time step they choose an action (buy, sell, hold) based on the current state. While this flat approach can work for simple scenarios, it struggles with the multi-scale nature of real financial markets. Trading decisions naturally decompose into a hierarchy: strategic asset allocation happens over weeks or months, tactical rebalancing over days, and order execution over minutes or seconds.

Hierarchical Reinforcement Learning (HRL) addresses this challenge by introducing multiple levels of abstraction into the decision-making process. A high-level policy (the "manager") sets goals or selects regimes on longer timescales, while a low-level policy (the "worker") executes concrete trading actions to achieve those goals on shorter timescales. This decomposition mirrors how professional trading firms operate: portfolio managers set strategic directions and risk budgets, while execution traders handle the mechanics of entering and exiting positions.

The key advantages of HRL for trading include:

- **Temporal abstraction**: Different policies operate at different frequencies, naturally matching the multi-timescale structure of markets.
- **Transfer and reuse**: Low-level execution skills transfer across different strategic objectives.
- **Exploration efficiency**: The manager explores in a reduced goal space rather than the full action space.
- **Interpretability**: The hierarchy produces human-readable intermediate goals (e.g., "shift to defensive positioning").

This chapter covers the mathematical foundations of HRL — the options framework, feudal networks, and Hierarchical Actor-Critic (HAC) — and demonstrates a complete Rust implementation integrated with Bybit market data.

## Mathematical Foundations

### The Options Framework

The options framework (Sutton, Precup & Singh, 1999) extends the standard MDP with temporally extended actions called **options**. An option $\omega \in \Omega$ is defined as a triple:

$$\omega = (I_\omega, \pi_\omega, \beta_\omega)$$

where:
- $I_\omega \subseteq S$ is the **initiation set** (states where the option can start)
- $\pi_\omega: S \times A \rightarrow [0,1]$ is the **intra-option policy** (how to act while executing)
- $\beta_\omega: S \rightarrow [0,1]$ is the **termination function** (probability of stopping in each state)

In a trading context, an option might represent "accumulate a long position in BTC over the next hour." The initiation set includes states where the portfolio has room for more BTC exposure, the intra-option policy specifies the limit order placement strategy, and the termination function triggers when the target position is reached or market conditions change.

The value function over options follows a semi-MDP structure:

$$Q_\Omega(s, \omega) = \sum_a \pi_\omega(a|s) \left[ r(s,a) + \gamma \sum_{s'} P(s'|s,a) \left[ \beta_\omega(s') \max_{\omega'} Q_\Omega(s', \omega') + (1 - \beta_\omega(s')) Q_\Omega(s', \omega) \right] \right]$$

### Feudal Networks

Feudal Reinforcement Learning (Dayan & Hinton, 1993; Vezhnevets et al., 2017) introduces a **manager-worker** architecture:

**Manager** operates at a lower temporal resolution. Every $c$ steps, it produces a goal vector $g_t$:

$$g_t = f_{\text{manager}}(s_t; \theta_M)$$

The manager's objective is to maximize the extrinsic (environment) reward accumulated over its decision interval.

**Worker** operates at every time step and selects primitive actions to achieve the manager's goal:

$$a_t = f_{\text{worker}}(s_t, g_t; \theta_W)$$

The worker receives an **intrinsic reward** measuring progress toward the manager's goal:

$$r_t^{\text{intrinsic}} = \frac{1}{c} \cos(s_{t+c} - s_t, g_t) = \frac{1}{c} \frac{(s_{t+c} - s_t) \cdot g_t}{\|s_{t+c} - s_t\| \cdot \|g_t\|}$$

This cosine similarity reward encourages the worker to move the state in the direction specified by the manager, without requiring exact goal achievement.

### Hierarchical Actor-Critic (HAC)

HAC (Levy et al., 2019) extends actor-critic methods to multiple hierarchy levels. Each level $k$ has:

- **Actor** $\mu_k(s; \theta_k^{\mu})$: produces a subgoal for level $k-1$ (or a primitive action at level 0)
- **Critic** $Q_k(s, g; \theta_k^{Q})$: evaluates the state-goal pair

The training uses three types of experience transitions:

1. **Hindsight action transitions**: Replace the subgoal proposed by level $k$ with the state actually achieved by level $k-1$, correcting for suboptimal lower-level execution.
2. **Hindsight goal transitions**: Replace the original goal with the achieved state, enabling learning from any trajectory.
3. **Subgoal testing transitions**: Periodically test whether level $k-1$ can achieve the proposed subgoal, penalizing unreachable goals.

The HAC update for level $k$:

$$\mathcal{L}_k = \mathbb{E}\left[(Q_k(s, g) - y_k)^2\right], \quad y_k = r + \gamma Q_k(s', g')$$

$$\nabla_{\theta_k^\mu} J = \mathbb{E}\left[\nabla_g Q_k(s, g)\big|_{g=\mu_k(s)} \cdot \nabla_{\theta_k^\mu} \mu_k(s)\right]$$

## Applications in Trading

### High-Level Regime Selection + Low-Level Execution

The most natural application of HRL in trading decomposes the problem into:

**Manager (High-Level Policy)**:
- Observes: macroeconomic indicators, volatility regimes, cross-asset correlations, order flow imbalances
- Decides every $N$ candles (e.g., daily): the current market regime (bull, bear, sideways) and a target portfolio allocation goal
- Reward: Sharpe ratio over the decision horizon

**Worker (Low-Level Policy)**:
- Observes: tick-level or minute-level price data, order book state, current position
- Decides at each step: specific order placement (market/limit, size, price)
- Intrinsic reward: progress toward the manager's allocation goal
- Extrinsic reward: execution quality (slippage, timing)

This decomposition allows the manager to learn regime detection implicitly through reward maximization, while the worker specializes in efficient execution regardless of the strategic direction.

### Multi-Timescale Portfolio Management

A three-level hierarchy for portfolio management:

1. **Strategic level** (monthly): Asset class allocation weights across equities, crypto, fixed income
2. **Tactical level** (daily): Within each asset class, select specific instruments and position sizes
3. **Execution level** (intraday): Execute the required trades with minimal market impact

Each level receives goals from above and translates them into subgoals for below. The strategic level might output "increase crypto allocation from 10% to 20%"; the tactical level translates this to "buy 5 BTC, 50 ETH over the next 3 days"; the execution level handles the actual order flow.

### Hierarchical Risk Management

HRL can also separate risk management from alpha generation:

- **Risk manager** (high-level): Sets position limits, maximum drawdown thresholds, and hedging requirements
- **Alpha trader** (low-level): Generates trades within the constraints set by the risk manager

The risk manager's intrinsic reward comes from keeping portfolio risk metrics within acceptable bounds, while the alpha trader optimizes for returns within those constraints.

## Rust Implementation

The implementation in `rust/src/lib.rs` provides:

1. **`MarketRegime`** enum: Represents detected market states (Bull, Bear, Sideways)
2. **`HighLevelPolicy`** (Manager): A neural-network-like policy that maps market features to regime classifications and allocation goals. It operates on a coarser timescale, making decisions every `decision_interval` steps.
3. **`LowLevelPolicy`** (Worker): Maps the current state plus the manager's goal to concrete trading actions (position sizing and direction). Receives intrinsic rewards for progressing toward the goal.
4. **`FeudalNetwork`**: Combines manager and worker into a coherent feudal architecture. The manager sets direction vectors in a learned goal space; the worker follows them.
5. **`HierarchicalAgent`**: The top-level agent that orchestrates training and inference across both levels.
6. **`BybitClient`**: Fetches OHLCV kline data from the Bybit v5 public API for backtesting.

Key design decisions:
- The manager goal is a 4-dimensional vector representing target portfolio characteristics (direction, magnitude, volatility tolerance, time horizon).
- The intrinsic reward uses cosine similarity between the state transition and the goal vector.
- Both policies use simple linear layers with ReLU activations for portability and speed.

See `rust/examples/trading_example.rs` for a complete demonstration that fetches BTCUSDT data from Bybit, trains a hierarchical agent, and compares it against a flat (single-level) RL baseline.

## Bybit Data Integration

The implementation connects to the Bybit v5 public API:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

The response provides OHLCV candles that are converted into feature vectors:
- Returns (close-to-close log returns)
- Volatility (rolling standard deviation of returns)
- Momentum (moving average crossover signal)
- Volume ratio (current vs. average volume)

These features feed both the manager (aggregated over longer windows) and the worker (at each step). No authentication is required for public market data endpoints.

## Key Takeaways

1. **Hierarchical RL naturally matches the multi-timescale structure of financial markets.** Strategic decisions (regime, allocation) operate on different frequencies than execution decisions (order placement, sizing).

2. **The options framework provides formal grounding.** Options extend MDPs with temporally extended actions, enabling principled learning of when to initiate and terminate trading strategies.

3. **Feudal networks separate goal-setting from goal-achieving.** The manager sets directional goals; the worker learns to achieve them. This division enables independent improvement of each component.

4. **Intrinsic rewards solve the sparse reward problem.** Rather than waiting for end-of-episode P&L, workers receive continuous feedback on goal progress via cosine similarity.

5. **HAC with hindsight learning accelerates training.** Hindsight action and goal transitions allow every trajectory to contribute useful training signal, even when lower levels execute suboptimally.

6. **The hierarchy improves interpretability.** Manager decisions ("switch to bearish regime") are human-readable, making the system easier to monitor and debug in production.

7. **Rust implementation enables low-latency deployment.** The zero-overhead abstractions and lack of garbage collection make Rust ideal for real-time hierarchical decision-making in trading systems.

## References

- Sutton, R.S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.
- Dayan, P., & Hinton, G.E. (1993). Feudal reinforcement learning.
- Vezhnevets, A.S. et al. (2017). FeUdal Networks for Hierarchical Reinforcement Learning.
- Levy, A., Konidaris, G., Platt, R., & Saenko, K. (2019). Learning Multi-Level Hierarchies with Hindsight.
- Nachum, O. et al. (2018). Data-Efficient Hierarchical Reinforcement Learning.
- Kulkarni, T.D. et al. (2016). Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation.
