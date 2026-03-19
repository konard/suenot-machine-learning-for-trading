# Chapter 317: Goal-Conditioned Reinforcement Learning for Trading

## 1. Introduction

Traditional reinforcement learning (RL) agents in trading environments optimize for a single, fixed reward signal -- typically maximizing cumulative returns or the Sharpe ratio. While this approach has produced impressive results, it suffers from a fundamental limitation: the agent cannot adapt its behavior to different objectives at test time without complete retraining. A portfolio manager may want 5% monthly returns during bull markets but shift to capital preservation during volatile periods. A risk officer may demand strategies that never exceed a 10% drawdown, while a proprietary trading desk targets specific return-per-trade thresholds.

**Goal-Conditioned Reinforcement Learning (GCRL)** addresses this limitation by training agents that accept explicit goals as inputs alongside the market state. Rather than learning a single policy, the agent learns a *universal* policy that can achieve a continuum of objectives on demand. When deployed, the trader specifies the desired outcome -- a target return, a maximum drawdown, a Sharpe ratio threshold -- and the agent adapts its trading behavior accordingly.

The key insight is that a single training run can produce an agent capable of pursuing many different trading objectives, dramatically improving sample efficiency and operational flexibility. This is made possible by three foundational ideas:

1. **Goal-Augmented MDPs**: Extending the state space to include goal specifications
2. **Universal Value Functions (UVFs)**: Value functions that generalize across both states and goals
3. **Hindsight Experience Replay (HER)**: A data augmentation technique that learns from failures by relabeling achieved outcomes as intended goals

In trading, GCRL is particularly powerful because market conditions are non-stationary, and the ability to dynamically adjust objectives without retraining provides a significant operational advantage. This chapter presents the mathematical foundations, practical algorithms, and a complete Rust implementation with Bybit market data integration.

## 2. Mathematical Foundations

### 2.1 Goal-Augmented Markov Decision Process

A standard MDP is defined by the tuple $(S, A, P, R, \gamma)$. A Goal-Conditioned MDP extends this to $(S, A, G, P, R_g, \gamma, \phi)$, where:

- $S$ is the state space (market observations: prices, volumes, indicators)
- $A$ is the action space (buy, sell, hold, position sizes)
- $G$ is the goal space (target returns, drawdown limits, Sharpe targets)
- $P: S \times A \times S \rightarrow [0, 1]$ is the transition function
- $R_g: S \times A \times G \rightarrow \mathbb{R}$ is the goal-conditioned reward function
- $\gamma \in [0, 1)$ is the discount factor
- $\phi: S \rightarrow G$ is the goal mapping function that extracts the achieved goal from a state

The goal-conditioned reward function is typically defined as:

$$R_g(s, a, g) = -\|g - \phi(s')\|_2 + \mathbb{1}[\|\phi(s') - g\| < \epsilon]$$

where $s'$ is the next state, $\phi(s')$ maps the state to the achieved goal (e.g., the current cumulative return), and $\epsilon$ is a tolerance threshold. The first term provides a shaping signal proportional to distance from the goal, while the second term gives a sparse bonus upon goal achievement.

For trading, we define:

$$\phi_{\text{return}}(s_t) = \frac{V_t - V_0}{V_0}$$

$$\phi_{\text{sharpe}}(s_t) = \frac{\bar{r}_t}{\sigma_{r_t}} \cdot \sqrt{252}$$

$$\phi_{\text{drawdown}}(s_t) = \frac{V_t - \max_{\tau \leq t} V_\tau}{\max_{\tau \leq t} V_\tau}$$

where $V_t$ is the portfolio value at time $t$, $\bar{r}_t$ is the mean daily return, and $\sigma_{r_t}$ is the standard deviation of daily returns.

### 2.2 Universal Value Function Approximators (UVFA)

A universal value function $V(s, g; \theta)$ estimates the expected return for being in state $s$ while pursuing goal $g$:

$$V(s, g; \theta) \approx \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R_g(s_t, a_t, g) \mid s_0 = s\right]$$

Similarly, the universal action-value function is:

$$Q(s, a, g; \theta) \approx \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R_g(s_t, a_t, g) \mid s_0 = s, a_0 = a\right]$$

The policy is then goal-conditioned:

$$\pi(a \mid s, g; \theta) = \arg\max_a Q(s, a, g; \theta)$$

In practice, the state and goal vectors are concatenated and fed into a neural network:

$$\text{input} = [s \| g] \in \mathbb{R}^{d_s + d_g}$$

The network architecture typically uses separate encoding branches for states and goals before merging them:

$$h_s = f_s(s; \theta_s), \quad h_g = f_g(g; \theta_g)$$

$$Q(s, a, g) = f_q([h_s \| h_g]; \theta_q)$$

### 2.3 Hindsight Experience Replay (HER)

HER is the key innovation that makes goal-conditioned RL practical. The fundamental problem is that in sparse reward settings, the agent rarely achieves its desired goal, making learning extremely slow. HER addresses this by retroactively relabeling failed episodes with the goals that were actually achieved.

Given a trajectory $\tau = \{(s_0, a_0, g), (s_1, a_1, g), \ldots, (s_T, a_T, g)\}$ where the desired goal $g$ was not achieved, HER creates additional training samples by substituting $g$ with $g' = \phi(s_T)$ (the achieved goal):

$$\tau' = \{(s_0, a_0, g'), (s_1, a_1, g'), \ldots, (s_T, a_T, g')\}$$

The relabeling strategies include:

1. **Final**: Use the goal achieved at the end of the episode: $g' = \phi(s_T)$
2. **Future**: For each transition at time $t$, sample a future state $s_k$ where $k > t$ and set $g' = \phi(s_k)$
3. **Episode**: Sample any state from the episode: $g' = \phi(s_k)$ where $k \sim \text{Uniform}(0, T)$
4. **Random**: Sample goals from the entire replay buffer

In trading terms, if an agent targeted a 5% return but only achieved 2%, HER creates training data as if the agent had intended to achieve 2% all along. This dramatically accelerates learning because every trajectory becomes informative.

### 2.4 Goal-Conditioned Bellman Equation

The Bellman equation for goal-conditioned Q-learning is:

$$Q(s, a, g) = R_g(s, a, g) + \gamma \max_{a'} Q(s', a', g)$$

The loss function for training:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, g, r, s') \sim \mathcal{B}} \left[(r + \gamma \max_{a'} Q(s', a', g; \theta^{-}) - Q(s, a, g; \theta))^2\right]$$

where $\mathcal{B}$ is the replay buffer (augmented with HER relabeled transitions) and $\theta^{-}$ are the target network parameters.

## 3. Applications in Trading

### 3.1 Target-Return Trading

The most direct application of GCRL in trading is specifying target returns. The goal space is simply $G = \mathbb{R}$, representing the desired cumulative return over the trading horizon. The agent learns to modulate position sizes, entry/exit timing, and asset selection to achieve the specified return.

During bullish markets, a target of 10% monthly return may be achievable with moderate leverage. During sideways markets, the same target might require the agent to take on more concentrated positions. The GCRL agent learns these adaptations implicitly.

### 3.2 Drawdown-Constrained Strategies

Goals can encode risk constraints rather than return targets. Setting $g = (\text{max\_drawdown} = -5\%)$ instructs the agent to trade while never allowing the portfolio to decline more than 5% from its peak. The reward function penalizes drawdown violations:

$$R_g(s, a, g) = r_t - \lambda \cdot \max(0, |\text{DD}_t| - |g|)$$

where $\text{DD}_t$ is the current drawdown and $\lambda$ is a penalty coefficient.

### 3.3 Value-at-Risk (VaR) Targets

Institutional trading desks often have VaR limits. GCRL can incorporate VaR targets directly into the goal specification: $g = (\text{VaR}_{95\%} = -2\%)$. The agent learns to construct portfolios where the 95th percentile daily loss does not exceed 2%.

### 3.4 Multi-Goal Composition

Goals can be composed into vectors: $g = (r_{\text{target}}, \text{DD}_{\max}, \text{Sharpe}_{\min})$. This allows the agent to simultaneously target returns while respecting risk constraints, providing a flexible interface for portfolio management.

### 3.5 Dynamic Goal Adjustment

In live trading, goals can be adjusted in real-time based on market regime detection. During high-volatility regimes, the system automatically reduces return targets and tightens drawdown constraints. During low-volatility periods, more aggressive targets can be set. This creates an adaptive trading system that respects changing market conditions without retraining.

## 4. Rust Implementation

The implementation in `rust/src/lib.rs` provides a complete GCRL trading system with the following components:

### 4.1 Core Data Structures

- **`GoalConditionedState`**: Combines market observations with goal specifications into a single augmented state vector
- **`HERReplayBuffer`**: Implements Hindsight Experience Replay with configurable relabeling strategies (Final, Future, Episode)
- **`UniversalValueFunction`**: A neural network approximator that maps (state, goal) pairs to Q-values across actions
- **`GoalConditionedPolicy`**: The policy that selects actions based on the current state and desired goal, with epsilon-greedy exploration

### 4.2 Training Pipeline

The training loop follows the standard GCRL algorithm:

1. For each episode, sample a goal from the goal distribution
2. Execute the episode using the goal-conditioned policy
3. Store transitions in the replay buffer
4. Apply HER to generate additional relabeled transitions
5. Sample minibatches and update the Q-network via gradient descent
6. Periodically update the target network

### 4.3 Goal Specifications

The system supports three types of goals:
- **TargetReturn**: Achieve a specified cumulative return (e.g., 5%)
- **MaxDrawdown**: Keep maximum drawdown within a limit (e.g., -10%)
- **TargetSharpe**: Achieve a target annualized Sharpe ratio (e.g., 1.5)

## 5. Bybit Data Integration

The implementation includes a `BybitClient` that fetches historical kline (candlestick) data from the Bybit V5 API. The client:

1. Connects to `https://api.bybit.com/v5/market/kline`
2. Fetches OHLCV data for any trading pair (default: BTCUSDT)
3. Converts raw API responses into `MarketCandle` structs
4. Computes derived features: returns, volatility, and rolling statistics
5. Constructs state vectors suitable for the GCRL agent

The Bybit integration allows the agent to train on real cryptocurrency market data, providing a realistic testbed for goal-conditioned trading strategies. The example in `rust/examples/trading_example.rs` demonstrates fetching live data, training an agent with multiple goal targets, and evaluating performance across different goal specifications.

## 6. Key Takeaways

1. **Goal-conditioned RL decouples objectives from training**: A single training run produces an agent that can pursue many different trading goals without retraining.

2. **HER is essential for sample efficiency**: Without Hindsight Experience Replay, goal-conditioned agents in sparse-reward trading environments would require impractical amounts of data. HER turns every trajectory into useful training signal.

3. **Universal value functions generalize across goals**: By conditioning on goals, the value function learns shared structure across different objectives, leading to better generalization than training separate agents per goal.

4. **Dynamic goal adjustment enables adaptive trading**: Goals can be changed at deployment time based on market conditions, risk budgets, or client requirements, providing operational flexibility that fixed-objective agents lack.

5. **Multi-dimensional goals capture real trading constraints**: Real trading involves simultaneous return targets, risk limits, and regulatory constraints. GCRL naturally handles this through vector-valued goals.

6. **Rust provides production-grade performance**: The implementation leverages Rust's memory safety and performance characteristics for low-latency trading applications, with zero-cost abstractions for the mathematical operations.

7. **Practical considerations matter**: Goal normalization, reward shaping, and careful goal distribution design are critical for stable training. Goals should be achievable given market conditions to avoid degenerate policies.

8. **GCRL complements traditional risk management**: Rather than replacing VaR limits or position sizing rules, GCRL internalizes these constraints as goals, creating agents that naturally respect risk budgets while optimizing returns.
