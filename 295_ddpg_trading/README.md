# Chapter 295: DDPG for Trading

## 1. Introduction: Deep Deterministic Policy Gradient for Continuous Action Trading

Traditional reinforcement learning approaches to trading, such as Deep Q-Networks (DQN), operate in discrete action spaces: buy, sell, or hold. While this simplification enables tractable learning, it fundamentally misrepresents the nature of real trading decisions. In practice, a trader does not merely decide *whether* to buy --- they decide *how much* to buy. Position sizing is a continuous variable that profoundly affects risk management, portfolio performance, and capital efficiency.

Deep Deterministic Policy Gradient (DDPG) addresses this limitation by operating natively in continuous action spaces. Introduced by Lillicrap et al. (2015), DDPG combines ideas from Deterministic Policy Gradient (DPG) theory with the representational power of deep neural networks. It is an actor-critic, model-free algorithm that learns a deterministic policy mapping states to continuous actions.

In the context of algorithmic trading, DDPG enables an agent to output a continuous signal --- for example, a position size between -1 (full short) and +1 (full long) --- rather than choosing from a finite menu of discrete actions. This continuous control is especially valuable in cryptocurrency markets, where fractional position sizes are the norm, leverage can be applied continuously, and the granularity of order sizing directly impacts slippage and execution quality.

### Why DDPG for Trading?

1. **Continuous position sizing**: Output precise allocation fractions rather than discrete lots.
2. **Smooth policy learning**: The deterministic policy gradient provides lower variance gradients compared to stochastic policy gradient methods.
3. **Sample efficiency**: Off-policy learning with experience replay enables reuse of historical transitions.
4. **Scalability**: The actor-critic architecture decouples policy evaluation from policy improvement.

This chapter develops DDPG from first principles, derives the key mathematical results, implements the algorithm in Rust with a focus on performance and safety, and demonstrates its application to cryptocurrency trading using live Bybit market data.

---

## 2. Mathematical Foundations

### 2.1 Markov Decision Process for Continuous Control

We model the trading environment as a Markov Decision Process (MDP) defined by the tuple (S, A, P, R, gamma), where:

- **S** is the state space (market features: prices, volumes, technical indicators)
- **A** is a continuous action space, A is a subset of R^d (e.g., position size in [-1, 1])
- **P(s'|s, a)** is the transition probability
- **R(s, a, s')** is the reward function (e.g., risk-adjusted PnL)
- **gamma** in [0, 1) is the discount factor

### 2.2 Deterministic Policy Gradient Theorem

Unlike stochastic policies pi(a|s) that output a probability distribution over actions, a deterministic policy mu: S -> A directly maps states to actions:

```
a = mu(s)
```

The objective is to maximize the expected cumulative discounted reward:

```
J(mu) = E[sum_{t=0}^{inf} gamma^t * R(s_t, mu(s_t), s_{t+1})]
```

The Deterministic Policy Gradient (DPG) theorem (Silver et al., 2014) states that the gradient of J with respect to the policy parameters theta is:

```
nabla_theta J(mu_theta) = E_s ~ rho^mu [nabla_theta mu_theta(s) * nabla_a Q^mu(s, a)|_{a=mu_theta(s)}]
```

where rho^mu is the state distribution under policy mu and Q^mu(s, a) is the action-value function. This result is remarkable because it does not require integrating over the action space, unlike the stochastic policy gradient theorem, making it computationally efficient for continuous action spaces.

### 2.3 Actor-Critic Architecture

DDPG employs two neural networks:

**Actor** mu(s; theta^mu): Maps states to deterministic actions.

**Critic** Q(s, a; theta^Q): Estimates the action-value function for any state-action pair.

The critic is updated by minimizing the temporal difference (TD) error:

```
L(theta^Q) = E_{(s,a,r,s') ~ D} [(r + gamma * Q'(s', mu'(s'; theta^{mu'}); theta^{Q'}) - Q(s, a; theta^Q))^2]
```

where D is the replay buffer and the primed networks are target networks.

The actor is updated using the sampled policy gradient:

```
nabla_{theta^mu} J approx (1/N) * sum_i nabla_{theta^mu} mu(s_i; theta^mu) * nabla_a Q(s_i, a; theta^Q)|_{a=mu(s_i)}
```

### 2.4 Target Networks and Soft Updates

Direct bootstrapping with a rapidly changing critic leads to instability. DDPG addresses this through target networks --- slowly updated copies of both actor and critic:

```
theta^{Q'} <- tau * theta^Q + (1 - tau) * theta^{Q'}
theta^{mu'} <- tau * theta^mu + (1 - tau) * theta^{mu'}
```

where tau << 1 (typically tau = 0.005). This Polyak averaging ensures that the target values change slowly, stabilizing learning.

### 2.5 Ornstein-Uhlenbeck Noise for Exploration

Since the policy is deterministic, exploration must be added externally. DDPG uses temporally correlated Ornstein-Uhlenbeck (OU) noise, which is suitable for physical control and trading because it produces smooth, mean-reverting perturbations:

```
dx_t = theta_ou * (mu_ou - x_t) * dt + sigma * dW_t
```

In discrete time:

```
x_{t+1} = x_t + theta_ou * (mu_ou - x_t) + sigma * N(0, 1)
```

Parameters:
- **theta_ou** (mean reversion speed): Controls how quickly noise reverts to mu_ou
- **mu_ou** (long-run mean): Typically 0 for symmetric exploration
- **sigma** (volatility): Controls exploration magnitude

For trading, OU noise is particularly appropriate because it avoids the jerky, uncorrelated exploration of Gaussian noise. Position sizes evolve smoothly, mimicking the behavior of a cautious trader who adjusts positions gradually rather than making wild swings.

### 2.6 Replay Buffer

Experience replay stores transitions (s, a, r, s') in a fixed-size buffer and samples mini-batches uniformly at random for training. This breaks temporal correlations in the data stream and enables each transition to be used for multiple gradient updates, improving sample efficiency.

---

## 3. Applications to Trading

### 3.1 Continuous Position Sizing

The primary application of DDPG in trading is continuous position sizing. The agent observes a state vector constructed from market features and outputs a position signal in [-1, 1]:

- **+1**: Maximum long position (fully invested)
- **0**: No position (flat/cash)
- **-1**: Maximum short position (fully short)

Intermediate values represent fractional positions. For example, an output of +0.3 means allocating 30% of available capital to a long position.

**State features** typically include:
- Normalized price returns over multiple horizons
- Normalized volume
- Technical indicators (RSI, MACD, Bollinger Band width)
- Current position size
- Unrealized PnL
- Time-based features (hour of day, day of week)

**Reward function** design is critical. Common choices include:
- **Simple PnL**: r_t = position_t * return_{t+1}
- **Risk-adjusted**: r_t = position_t * return_{t+1} - lambda * (position_t * return_{t+1})^2
- **Sharpe-inspired**: Running Sharpe ratio over a window
- **Drawdown-penalized**: PnL minus a penalty for drawdowns

### 3.2 Continuous Order Sizing for Crypto Markets

Cryptocurrency markets on exchanges like Bybit offer several properties that make DDPG particularly well-suited:

1. **Fractional sizing**: Crypto assets are infinitely divisible, making continuous position sizing natural.
2. **24/7 markets**: Continuous operation provides abundant training data.
3. **High volatility**: The agent must learn to modulate position sizes with volatility --- a naturally continuous control problem.
4. **Leverage**: Bybit offers up to 100x leverage; DDPG can learn optimal leverage ratios as a continuous variable.

The DDPG agent can be trained to:
- Size positions proportionally to signal confidence
- Reduce exposure during high-volatility regimes
- Gradually build or unwind positions to minimize market impact
- Learn optimal leverage as part of the continuous action space

### 3.3 Multi-Asset Extension

For multi-asset portfolios, the action space extends to R^n, where n is the number of assets. DDPG handles this naturally:

```
a = [w_1, w_2, ..., w_n], where sum |w_i| <= 1
```

Each w_i represents the portfolio weight for asset i, and the constraint can be enforced via normalization or penalty terms in the reward.

---

## 4. Rust Implementation

Our Rust implementation provides a complete, production-grade DDPG system with the following components:

### Architecture Overview

```
ddpg_trading/
  rust/
    src/lib.rs        -- Core DDPG components
    examples/
      trading_example.rs  -- Bybit integration and training loop
    Cargo.toml
```

**Core components in `lib.rs`:**

1. **`ActorNetwork`**: A feed-forward network mapping state vectors to continuous actions in [-1, 1] via tanh activation. Implemented with manual weight matrices and forward propagation using `ndarray`.

2. **`CriticNetwork`**: Takes concatenated state-action vectors and outputs scalar Q-values. Uses ReLU hidden layers and linear output.

3. **`TargetNetwork`**: Wraps both actor and critic with Polyak-averaged soft updates (tau = 0.005).

4. **`OUNoise`**: Ornstein-Uhlenbeck process for temporally correlated exploration noise with configurable mean reversion, volatility, and decay.

5. **`ReplayBuffer`**: Fixed-capacity circular buffer storing (state, action, reward, next_state, done) tuples with uniform random sampling.

6. **`DDPGAgent`**: Orchestrates the full training loop: action selection with noise, experience storage, mini-batch sampling, critic and actor gradient updates, and target network soft updates.

7. **`BybitClient`**: Async HTTP client for fetching OHLCV data from the Bybit v5 API.

### Key Design Decisions

- **No external ML framework dependency**: All neural network operations are implemented from scratch using `ndarray`, ensuring transparency and educational value.
- **Numerical stability**: Careful use of tanh clamping and gradient clipping prevents divergence.
- **Replay buffer efficiency**: Uses a circular buffer with O(1) insertion and O(k) sampling for mini-batches of size k.
- **Separation of concerns**: The trading environment, DDPG algorithm, and data fetching are cleanly separated.

---

## 5. Bybit Data Integration

The implementation fetches live OHLCV (Open, High, Low, Close, Volume) data from Bybit's public API v5:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

This returns hourly candles for BTCUSDT perpetual futures. The data is processed into state features:
- Normalized returns: (close - prev_close) / prev_close
- Normalized volume: volume / rolling_mean_volume
- Simple technical indicators computed from price series

No API keys are required for public market data endpoints, making this example immediately runnable.

### Data Pipeline

1. **Fetch**: HTTP GET to Bybit API for OHLCV data
2. **Parse**: Deserialize JSON response into Rust structs
3. **Transform**: Compute returns, normalize features, construct state vectors
4. **Environment**: Simulate trading with continuous position sizing
5. **Training**: Run DDPG training loop over historical data
6. **Evaluation**: Measure cumulative PnL, Sharpe ratio, max drawdown

---

## 6. Key Takeaways

1. **DDPG enables continuous position sizing**, which is more natural and expressive than discrete buy/sell/hold actions. This is particularly important for crypto trading where fractional positions and variable leverage are standard.

2. **The deterministic policy gradient** avoids integrating over the action space, providing lower variance gradient estimates than stochastic methods like PPO or A2C for continuous control.

3. **Target networks with soft updates** (tau = 0.005) are essential for training stability. Without them, the bootstrapped TD targets change too rapidly, causing divergence.

4. **Ornstein-Uhlenbeck noise** provides temporally correlated exploration that is well-suited to trading, where smooth position adjustments are preferable to random jumps.

5. **Experience replay** breaks temporal correlations and enables sample-efficient off-policy learning from historical market data.

6. **The actor-critic architecture** decouples value estimation (critic) from policy improvement (actor), allowing each to be optimized independently with different learning rates.

7. **Reward engineering** is the most critical design choice. Risk-adjusted rewards (Sharpe-inspired or drawdown-penalized) produce more robust trading strategies than raw PnL.

8. **Rust implementation** provides memory safety, zero-cost abstractions, and performance suitable for production deployment without sacrificing code clarity.

9. **DDPG has known limitations**: overestimation bias in the critic (addressed by TD3), sensitivity to hyperparameters, and potential for premature convergence to suboptimal deterministic policies. These are discussed in the context of trading applications.

10. **For production deployment**, consider TD3 (Twin Delayed DDPG) or SAC (Soft Actor-Critic) as more robust successors, while DDPG remains the foundational algorithm for understanding continuous-action RL in trading.

---

## References

- Lillicrap, T.P., Hunt, J.J., Pritzel, A., et al. (2015). "Continuous control with deep reinforcement learning." arXiv:1509.02971.
- Silver, D., Lever, G., Heess, N., et al. (2014). "Deterministic Policy Gradient Algorithms." ICML.
- Fujimoto, S., van Hoof, H., Meger, D. (2018). "Addressing Function Approximation Error in Actor-Critic Methods." ICML (TD3).
- Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). "Soft Actor-Critic." ICML.
- Bybit API Documentation: https://bybit-exchange.github.io/docs/v5/intro
