# Chapter 304: MuZero Trading

## Introduction: MuZero - Planning Without a Model of the Environment Rules

MuZero, introduced by Schrittwieser et al. (2020) at DeepMind, represents a landmark achievement in artificial intelligence: an agent that masters complex games and decision-making tasks without ever being given the rules of the environment. Unlike its predecessors AlphaGo and AlphaZero, which required a perfect simulator of the game to plan ahead, MuZero learns its own internal model of the environment's dynamics and uses that learned model for planning via Monte Carlo Tree Search (MCTS).

This distinction is profound for trading applications. Financial markets have no known, fixed rules. The "dynamics" of the market are emergent, non-stationary, and partially observable. Traditional model-based reinforcement learning struggles because building an accurate forward model of market behavior is extraordinarily difficult. Model-free methods (DQN, PPO, A2C) sidestep this problem but sacrifice the ability to plan ahead. MuZero offers a middle path: it learns a latent dynamics model that captures the aspects of the environment relevant to decision-making, without needing to reconstruct the full observation space.

In the context of algorithmic trading, MuZero's architecture allows an agent to:

1. **Learn compressed representations** of market states (order book snapshots, OHLCV data, technical indicators) through a representation network.
2. **Simulate future market trajectories** in a learned latent space using a dynamics network, enabling look-ahead planning.
3. **Evaluate positions and generate action distributions** via a prediction network that outputs both value estimates and policy priors.
4. **Plan using MCTS** over the learned model, exploring different sequences of trading actions (buy, sell, hold) without needing a market simulator.

The key insight is that MuZero's learned model does not need to predict future prices or reconstruct candlestick charts. It only needs to predict three things that matter for planning: the reward, the value, and the policy. This makes the learned model both more tractable and more useful than a full generative model of market dynamics.

## Mathematical Foundation

### Core Architecture

MuZero's architecture consists of three neural networks that work together:

#### 1. Representation Function h

The representation function maps a raw observation (market state) to an initial hidden state:

```
s_0 = h_theta(o_1, ..., o_t)
```

Where `o_1, ..., o_t` is a sequence of past observations (e.g., the last T candlesticks with volume, spread, and technical indicators). The hidden state `s_0` is a learned, compact representation — it does not need to be interpretable or correspond to any physical quantity.

For trading, the observation might be a vector containing:
- OHLCV data for the last N candles
- Current portfolio position (long, short, flat)
- Unrealized PnL
- Technical indicators (RSI, MACD, Bollinger Bands)
- Order book imbalance

#### 2. Dynamics Function g

The dynamics function predicts the next hidden state and immediate reward given a hidden state and action:

```
r_k, s_k = g_theta(s_{k-1}, a_k)
```

This is the learned "world model." Given the current latent state and a trading action (buy, sell, hold, or parameterized order sizes), it predicts what the next latent state will look like and what immediate reward (P&L change, risk-adjusted return) will be received.

Crucially, this dynamics function operates entirely in latent space. It does not predict the next candlestick or the next order book snapshot — it predicts the next *planning-relevant* hidden state.

#### 3. Prediction Function f

The prediction function maps a hidden state to a policy and value:

```
p_k, v_k = f_theta(s_k)
```

Where:
- `p_k` is a probability distribution over actions (the policy prior)
- `v_k` is the estimated value (expected future discounted return) from state `s_k`

These outputs serve two purposes:
1. The policy prior guides MCTS to explore promising actions first.
2. The value estimate provides a bootstrap for nodes that haven't been fully expanded.

### Monte Carlo Tree Search with Learned Model

MuZero's MCTS proceeds as follows for each decision step:

**Selection**: Starting from the root (current real state), traverse the tree by selecting actions that maximize the PUCT (Predictor + Upper Confidence bounds applied to Trees) formula:

```
a_k = argmax_a [ Q(s, a) + c(s) * P(s, a) * (sqrt(N(s)) / (1 + N(s, a))) ]
```

Where:
- `Q(s, a)` is the mean value of action `a` from state `s`
- `P(s, a)` is the prior probability from the prediction network
- `N(s)` is the visit count of state `s`
- `N(s, a)` is the visit count of action `a` from state `s`
- `c(s)` is an exploration constant that adapts based on visit counts

**Expansion**: When a leaf node is reached, use the dynamics function to generate the next hidden state and reward, then the prediction function to get the prior and value.

**Backup**: Propagate the value back up the tree, updating Q-values and visit counts.

After a fixed number of simulations (e.g., 50-200), select the action at the root proportional to visit counts.

### Training Objective

MuZero is trained end-to-end by unrolling the model for K steps and minimizing a combined loss:

```
L = sum_{k=0}^{K} [ l_p(pi_t+k, p_k) + l_v(z_t+k, v_k) + l_r(u_t+k, r_k) ] + c * ||theta||^2
```

Where:
- `l_p` is the cross-entropy loss between the MCTS-improved policy `pi` and the predicted policy `p`
- `l_v` is the MSE or cross-entropy loss between the actual returns `z` and predicted values `v`
- `l_r` is the MSE loss between actual rewards `u` and predicted rewards `r`
- `c * ||theta||^2` is L2 regularization

The targets come from actual gameplay/trading experience stored in a replay buffer. **Reanalysis** is a key technique where old trajectories are re-evaluated with the current (improved) model to generate better training targets.

### Value and Reward Scaling

For domains with large value ranges (like trading P&L), MuZero uses an invertible transform to scale targets:

```
h(x) = sign(x) * (sqrt(|x| + 1) - 1) + epsilon * x
```

This compresses large values while preserving sign, making the prediction task easier for the neural network.

## Comparison with AlphaZero and Model-Free RL

### AlphaZero vs. MuZero

| Feature | AlphaZero | MuZero |
|---------|-----------|--------|
| Environment model | Given (perfect simulator) | Learned |
| Observation space | Full game state | Arbitrary observations |
| Planning | MCTS with true dynamics | MCTS with learned dynamics |
| Applicable to trading | No (no perfect market sim) | Yes |
| Training data | Self-play with simulator | Real trajectories + reanalysis |

AlphaZero requires a perfect simulator to plan ahead — it calls the real game engine during MCTS. This is impossible for trading since we cannot simulate the market perfectly. MuZero removes this requirement entirely.

### Model-Free RL (DQN, PPO) vs. MuZero

| Feature | Model-Free RL | MuZero |
|---------|---------------|--------|
| Planning | None (reactive) | Multi-step lookahead |
| Sample efficiency | Low | Higher (reuses data via planning) |
| Computational cost | Low at inference | Higher (MCTS at each step) |
| Adaptation to regime changes | Requires retraining | Can plan through novel states |
| Robustness | Can overfit to patterns | Regularized by planning |

Model-free methods make decisions based solely on the current observation, without any internal simulation. MuZero can "think ahead" by simulating future states in its learned latent space, potentially making more robust decisions in complex market scenarios.

### When MuZero Excels in Trading

1. **Multi-step decision making**: When the optimal action depends on a plan (e.g., scaling into a position over multiple steps).
2. **Regime transitions**: The learned dynamics model can capture different market regimes in its latent space.
3. **Risk management**: Planning ahead allows the agent to evaluate worst-case scenarios before committing to a trade.
4. **Sparse rewards**: When P&L is only realized at position close, MCTS with value estimates helps credit assignment.

## Applications: Trading as a Game

### Framing Trading as a Sequential Decision Problem

We model trading as a single-player game against the market:

- **State**: Market observations (OHLCV, indicators, portfolio status)
- **Actions**: Discrete actions {Strong Buy, Buy, Hold, Sell, Strong Sell} or continuous position sizing
- **Reward**: Risk-adjusted returns (Sharpe-like reward), realized P&L, or a combination
- **Transitions**: Determined by actual market movement + our action's market impact
- **Horizon**: Either fixed (e.g., one trading session) or indefinite with discounting

### Learned Market Dynamics Model

The dynamics network learns an implicit model of how the market evolves in response to our actions. Important properties:

1. **Partial observability**: The representation network can aggregate multiple past observations, effectively learning to infer hidden market state.
2. **Non-stationarity**: By training on a rolling window of recent data, the model adapts to changing market conditions.
3. **Action impact**: For large positions, the dynamics model can learn (implicitly) about market impact and slippage.

### Portfolio Construction with MuZero

MuZero's MCTS naturally handles the sequential nature of portfolio construction:

1. At each time step, the agent observes the market state.
2. MCTS runs N simulations using the learned model to evaluate different action sequences.
3. The best action is selected based on visit counts.
4. The real environment advances one step, and the process repeats.

The planning horizon of MCTS (determined by the number of simulations and tree depth) effectively allows the agent to consider multi-step consequences of its current action.

## Rust Implementation

Our Rust implementation provides a complete MuZero system for trading:

### Architecture Overview

```
muzero_trading/
  rust/
    src/
      lib.rs          # Core MuZero: networks, MCTS, training, Bybit API
    examples/
      trading_example.rs  # End-to-end trading example
    Cargo.toml
```

### Key Components

1. **RepresentationNet**: Transforms raw market observations into hidden states using a multi-layer neural network with ReLU activations.

2. **DynamicsNet**: Takes a hidden state concatenated with a one-hot action encoding and produces the next hidden state plus a scalar reward prediction.

3. **PredictionNet**: Maps hidden states to action probabilities (via softmax) and value estimates (via tanh scaling).

4. **MCTS**: Implements the full MCTS loop — selection with PUCT, expansion using learned dynamics, and backpropagation of values.

5. **Training**: Unrolls the model K steps from sampled trajectories, computes the combined policy + value + reward loss, and updates parameters via gradient descent.

6. **Bybit Integration**: Fetches real OHLCV data from Bybit's public API for BTCUSDT and other trading pairs.

### Usage

```bash
cd 304_muzero_trading/rust
cargo build
cargo run --example trading_example
cargo test
```

## Bybit Data Integration

The implementation includes a complete Bybit API client for fetching historical kline (candlestick) data:

```rust
let candles = fetch_bybit_klines("BTCUSDT", "15", 200).await?;
```

Parameters:
- **symbol**: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
- **interval**: Candlestick interval ("1", "5", "15", "60", "240", "D")
- **limit**: Number of candles to fetch (max 200)

The API returns OHLCV data that is preprocessed into observation vectors for the representation network. Preprocessing includes:
- Log-return transformation of prices
- Volume normalization
- Sliding window creation for temporal context

## Key Takeaways

1. **MuZero learns to plan without knowing the rules**: Unlike AlphaZero, MuZero does not require a perfect environment simulator, making it applicable to domains like trading where no such simulator exists.

2. **The learned model operates in latent space**: MuZero's dynamics model does not predict future prices — it predicts future *planning-relevant* hidden states. This makes the modeling task more tractable.

3. **MCTS provides look-ahead planning**: By simulating future trajectories in the learned model, MuZero can evaluate multi-step consequences of trading actions before committing.

4. **Three networks, one objective**: The representation, dynamics, and prediction networks are trained end-to-end to be useful for planning, not to reconstruct observations.

5. **Reanalysis improves sample efficiency**: By re-evaluating old trajectories with the current model, MuZero extracts more learning signal from limited data — crucial for trading where data is scarce relative to the complexity of the task.

6. **Trading as a game**: Framing trading as a sequential decision problem allows us to apply game-playing techniques. The key difference from board games is non-stationarity, partial observability, and continuous state spaces.

7. **Practical considerations**: MuZero's computational cost at inference (running MCTS) is higher than model-free methods. For high-frequency trading, this may be prohibitive. For lower-frequency strategies (15-minute to daily), the planning overhead is acceptable and potentially beneficial.

8. **Rust for production**: The Rust implementation provides memory safety and performance critical for production trading systems, with zero-cost abstractions enabling efficient MCTS tree operations.

## References

- Schrittwieser, J., et al. (2020). "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." *Nature*, 588, 604-609.
- Silver, D., et al. (2018). "A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go Through Self-Play." *Science*, 362(6419), 1140-1144.
- Ye, W., et al. (2021). "Mastering Atari Games with Limited Data." *NeurIPS*.
- Hubert, T., et al. (2021). "Learning and Planning in Complex Action Spaces." *ICML*.
