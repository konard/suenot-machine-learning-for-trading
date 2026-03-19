# Chapter 320: Multi-Objective Reinforcement Learning for Trading

## Introduction

Traditional reinforcement learning (RL) for trading typically optimizes a single objective -- usually cumulative return or a risk-adjusted metric like the Sharpe ratio. However, real-world trading involves simultaneously balancing multiple, often conflicting objectives: maximizing returns while minimizing drawdown, controlling volatility, limiting turnover costs, and maintaining diversification. A single scalar reward function forces the designer to choose a fixed weighting of these concerns before training, which is brittle and may miss superior trade-off solutions.

**Multi-Objective Reinforcement Learning (MORL)** extends the RL framework to handle vector-valued rewards. Instead of collapsing multiple objectives into one number, MORL algorithms discover a set of policies that represent different trade-offs -- the **Pareto front**. A portfolio manager can then select the policy that best matches their risk appetite without retraining from scratch.

This chapter covers the mathematical foundations of multi-objective optimization in the RL setting, surveys state-of-the-art MORL algorithms (PGMORL, Envelope Q-learning, CAPQL), and demonstrates a complete Rust implementation that trains a multi-objective RL agent on cryptocurrency data from Bybit.

## Mathematical Foundations

### Multi-Objective Markov Decision Process (MOMDP)

A MOMDP extends the standard MDP with a vector-valued reward. Formally, a MOMDP is a tuple $(S, A, P, \mathbf{R}, \gamma)$ where:

- $S$ is the state space
- $A$ is the action space
- $P(s' | s, a)$ is the transition function
- $\mathbf{R}: S \times A \times S \to \mathbb{R}^d$ is a $d$-dimensional reward function
- $\gamma \in [0, 1)$ is the discount factor

Each policy $\pi$ induces a **vector-valued return**:

$$\mathbf{V}^{\pi}(s) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t \mathbf{R}(s_t, a_t, s_{t+1}) \mid s_0 = s\right] \in \mathbb{R}^d$$

In trading, a natural choice is $d = 3$ with components:

1. **Return**: cumulative log-return of the portfolio
2. **Negative drawdown**: $-\max_{t' \le t}(V_{t'} - V_t) / V_{t'}$
3. **Negative volatility**: $-\sigma$ of returns over a rolling window

### Pareto Dominance and the Pareto Front

Given two policies $\pi_1$ and $\pi_2$, we say $\pi_1$ **Pareto dominates** $\pi_2$ (written $\pi_1 \succ \pi_2$) if:

$$\forall i: V_i^{\pi_1} \ge V_i^{\pi_2} \quad \text{and} \quad \exists j: V_j^{\pi_1} > V_j^{\pi_2}$$

The **Pareto front** $\mathcal{P}^*$ is the set of all non-dominated policies:

$$\mathcal{P}^* = \{\pi \mid \nexists \pi' : \pi' \succ \pi\}$$

In the return-risk plane, the Pareto front traces out the efficient frontier -- the best achievable return for each level of risk.

### Scalarization Methods

The simplest approach to MORL is **linear scalarization**, which converts the vector reward into a scalar using a weight vector $\mathbf{w} \in \Delta^{d-1}$ (the simplex):

$$R_{\text{scalar}}(s, a, s') = \mathbf{w}^T \mathbf{R}(s, a, s')$$

By sweeping over different weight vectors, we can recover different points on the Pareto front. However, linear scalarization can only find points on the **convex hull** of the Pareto front. To find non-convex regions, we use **Chebyshev scalarization**:

$$R_{\text{cheb}}(s, a, s') = \min_i \; w_i \cdot (R_i(s, a, s') - z_i^*)$$

where $\mathbf{z}^*$ is a reference point (e.g., the ideal point where each objective is independently maximized).

### MORL Algorithms

#### PGMORL (Prediction-Guided Multi-Objective RL)

PGMORL maintains a population of policies and iteratively:
1. Predicts promising weight vectors using a performance model
2. Trains new policies from the population using the predicted weights
3. Updates the Pareto front approximation

The key insight is using a prediction model to guide exploration of the weight space, avoiding expensive training runs on unpromising regions.

#### CAPQL (Continuous Action Pareto Q-Learning)

CAPQL extends distributional RL to the multi-objective setting by learning a separate Q-function for each objective and using an envelope operator to approximate the Pareto front. It works with continuous action spaces, making it suitable for portfolio allocation.

#### Envelope Q-Learning

For discrete actions, Envelope Q-learning maintains a set of Q-functions $Q_w(s, a)$ indexed by weight vectors. The optimal Q-function satisfies:

$$Q^*_{\mathbf{w}}(s, a) = \mathbb{E}\left[\mathbf{w}^T \mathbf{R} + \gamma \max_{a'} Q^*_{\mathbf{w}}(s', a')\right]$$

By training across a distribution of weight vectors simultaneously, the agent learns to generalize across the preference space.

## Applications in Trading

### Return vs. Risk Trade-off

The most natural application is the classic return-risk trade-off. Instead of choosing a single risk aversion parameter, MORL discovers the full efficient frontier:

- **Aggressive policies**: High expected return, high drawdown tolerance
- **Conservative policies**: Lower return, minimal drawdown
- **Balanced policies**: Moderate return with controlled risk

A portfolio manager can slide along the Pareto front based on market regime. In calm markets, select a more aggressive policy; during high-volatility periods, shift to a conservative one.

### Multi-Asset Portfolio with Constraints

For multi-asset portfolios, objectives might include:
1. Total portfolio return
2. Maximum single-asset drawdown
3. Portfolio turnover (trading costs)
4. Concentration risk (inverse diversification)

The MORL agent learns allocation policies that balance these concerns. Unlike mean-variance optimization, MORL can capture non-linear relationships and path-dependent objectives like drawdown.

### Regime-Adaptive Policy Selection

A meta-policy can select among Pareto-optimal policies based on detected market regime:

```
if volatility_regime == HIGH:
    select policy with lowest drawdown
elif trend_regime == STRONG:
    select policy with highest return
else:
    select balanced policy
```

This provides an adaptive system that responds to changing market conditions without retraining.

## Rust Implementation

The implementation in `rust/src/lib.rs` provides:

1. **`MultiObjectiveReward`**: Computes the 3D reward vector (return, negative drawdown, negative volatility) from price data and trading actions.

2. **`ScalarizedQLearning`**: A tabular Q-learning agent that accepts a weight vector and trains on the scalarized reward. Multiple instances with different weights approximate the Pareto front.

3. **`ParetoFront`**: Maintains a set of non-dominated solutions. Supports insertion with dominance checking and hypervolume computation.

4. **`PolicyEvaluator`**: Evaluates a trained policy on each objective independently, producing the objective vector used for Pareto front construction.

5. **`BybitClient`**: Fetches historical OHLCV data from the Bybit public API for backtesting.

### Key Design Decisions

- **Discretized state space**: Price returns are bucketed into discrete levels (strong down, down, flat, up, strong up) combined with current position (short, neutral, long) for tractable tabular Q-learning.
- **Action space**: Three actions -- sell/short, hold, buy/long.
- **Rolling window**: Volatility and drawdown are computed over a configurable rolling window to keep the reward signal responsive.

### Running the Example

```bash
cd 320_multi_objective_rl_trading/rust
cargo run --example trading_example
```

The example:
1. Fetches BTCUSDT 1-hour klines from Bybit
2. Trains multiple agents with different weight vectors
3. Evaluates each agent on all three objectives
4. Constructs the Pareto front
5. Displays the trade-off between return, drawdown, and volatility
6. Selects the best policy for a given risk preference

## Bybit Data Integration

The implementation uses the Bybit V5 public API to fetch historical kline (candlestick) data:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

No authentication is required for public market data. The response provides OHLCV candles that are converted into returns for the RL environment.

Key considerations:
- **Rate limiting**: The public API allows up to 10 requests per second. The implementation includes appropriate delays.
- **Data quality**: Missing candles are handled by forward-filling the last known price.
- **Time alignment**: Candles are sorted chronologically before processing.

## Key Takeaways

1. **Multi-objective RL preserves the full trade-off structure** between competing objectives like return, drawdown, and volatility, rather than collapsing them into a single number.

2. **The Pareto front is the multi-objective equivalent of the efficient frontier** in portfolio theory, but extends to non-linear, path-dependent objectives that classical optimization cannot handle.

3. **Linear scalarization is simple but limited** -- it can only find convex Pareto front regions. Chebyshev scalarization and envelope methods can discover the full front.

4. **PGMORL and CAPQL are state-of-the-art algorithms** that efficiently explore the objective space. PGMORL uses prediction to guide exploration, while CAPQL extends distributional RL.

5. **Regime-adaptive policy selection** is a powerful application: train once to get the Pareto front, then dynamically select policies based on market conditions.

6. **Practical implementation requires careful discretization** of the state space and reward normalization across objectives with different scales.

7. **The Pareto front enables transparent decision-making** -- portfolio managers can see exactly what they are giving up in one objective to gain in another, rather than trusting an opaque single-metric optimization.

8. **Bybit's public API provides accessible cryptocurrency data** for training and evaluating multi-objective RL agents on real market dynamics.
