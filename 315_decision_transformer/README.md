# Chapter 315: Decision Transformer for Trading

## Overview

The Decision Transformer (DT), introduced by Chen et al. (2021), reframes reinforcement learning as a **sequence modeling** problem. Instead of learning value functions or policy gradients, the Decision Transformer uses a causal transformer architecture to autoregressively generate actions conditioned on desired returns, past states, and past actions. This paradigm shift enables leveraging the power of transformer architectures — which have revolutionized NLP and computer vision — for sequential decision-making.

In the context of algorithmic trading, the Decision Transformer offers a compelling approach to **offline reinforcement learning**: learning trading strategies from historical market data without requiring a simulator or online interaction with live markets. By conditioning on a target return-to-go (RTG), traders can specify the desired performance level and let the model generate appropriate trading actions to achieve that target.

## Table of Contents

1. [Introduction to Decision Transformer](#introduction-to-decision-transformer)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Offline RL Formulation for Trading](#offline-rl-formulation-for-trading)
4. [Applications in Trading](#applications-in-trading)
5. [Implementation in Rust](#implementation-in-rust)
6. [Bybit Data Integration](#bybit-data-integration)
7. [Key Takeaways](#key-takeaways)

---

## Introduction to Decision Transformer

### From RL to Sequence Modeling

Traditional reinforcement learning methods optimize policies through iterative interaction with an environment. Value-based methods (DQN, Q-learning) estimate state-action values; policy gradient methods (PPO, A2C) directly optimize the policy. Both paradigms require either online interaction or careful importance sampling for offline settings.

The Decision Transformer takes a fundamentally different approach: it treats the RL problem as **conditional sequence generation**. Given a trajectory of (return-to-go, state, action) tuples, the transformer learns to predict the next action that is consistent with achieving the specified future return.

### Why Decision Transformer for Trading?

Trading is a natural fit for the Decision Transformer paradigm:

- **Abundant Historical Data**: Financial markets provide massive offline datasets of trajectories (price histories, order flows, executed trades) that can be used for offline RL without a simulator
- **Return Conditioning**: Traders can specify target returns (e.g., "achieve 2% daily return") and let the model generate actions accordingly
- **No Simulator Required**: Unlike model-based RL, the Decision Transformer learns directly from historical data without needing a market simulator that accurately captures slippage, liquidity, and market impact
- **Sequence Modeling Strength**: Financial time series are inherently sequential, making transformer architectures a natural choice
- **Long-Range Dependencies**: Transformers capture long-range temporal patterns that recurrent networks struggle with
- **Multi-Asset Generalization**: A single model can learn across multiple assets and market regimes

### Architecture Overview

The Decision Transformer processes trajectories as sequences of triplets:

```
(R_1, s_1, a_1, R_2, s_2, a_2, ..., R_T, s_T, a_T)
```

Where:
- `R_t` is the return-to-go at timestep `t` (sum of future rewards)
- `s_t` is the market state at timestep `t` (OHLCV features, indicators)
- `a_t` is the trading action at timestep `t` (buy, sell, hold, or position size)

Each element is embedded via a learned linear projection, combined with a learned timestep embedding, and fed into a GPT-style causal transformer. The model predicts the action `a_t` given the sequence up to that point.

---

## Mathematical Foundation

### Markov Decision Process (MDP)

We model trading as an MDP defined by the tuple `(S, A, P, R, gamma)`:

- **State space** `S`: Market features at each timestep — OHLCV data, technical indicators, portfolio state
- **Action space** `A`: Trading decisions — discrete (buy/sell/hold) or continuous (position sizing)
- **Transition function** `P(s' | s, a)`: Market dynamics (unknown and non-stationary)
- **Reward function** `R(s, a)`: Trading PnL, risk-adjusted returns, or custom objectives
- **Discount factor** `gamma in [0, 1]`: Time preference for future rewards

### Return-to-Go (RTG)

The return-to-go at timestep `t` is the sum of future rewards from `t` to the end of the episode:

```
R_t = sum_{t'=t}^{T} r_{t'}
```

where `r_{t'}` is the reward at timestep `t'`. In trading, this represents the cumulative future profit from the current timestep to the end of the trading horizon.

The RTG serves as a **conditioning signal**: by specifying a high RTG, we instruct the model to generate actions consistent with high-return trajectories. By specifying a lower RTG, the model generates more conservative actions.

### Trajectory Representation

A trajectory `tau` is represented as:

```
tau = (R_1, s_1, a_1, R_2, s_2, a_2, ..., R_T, s_T, a_T)
```

The sequence length is `3T` (three tokens per timestep). Each token type has its own embedding:

```
token_embed(R_t) = W_R * R_t + b_R
token_embed(s_t) = W_s * s_t + b_s
token_embed(a_t) = W_a * a_t + b_a
```

A learned timestep embedding `E_t` is added to each token at timestep `t`:

```
h_t^R = token_embed(R_t) + E_t
h_t^s = token_embed(s_t) + E_t
h_t^a = token_embed(a_t) + E_t
```

### Causal Self-Attention

The Decision Transformer uses masked (causal) self-attention to ensure that predictions at timestep `t` only depend on tokens at positions `<= t`. For a sequence of hidden states `H = [h_1, h_2, ..., h_n]`:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k) + M) * V
```

where:
- `Q = H * W_Q`, `K = H * W_K`, `V = H * W_V` are query, key, and value projections
- `d_k` is the dimension of the key vectors
- `M` is the causal mask: `M_{ij} = 0` if `i >= j`, else `M_{ij} = -infinity`

The causal mask ensures that position `i` can only attend to positions `j <= i`, maintaining the autoregressive property.

### Multi-Head Attention

With `h` attention heads:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
head_i = Attention(Q * W_Q^i, K * W_K^i, V * W_V^i)
```

### Transformer Block

Each transformer block consists of:

```
x' = LayerNorm(x + MultiHead(x, x, x))
output = LayerNorm(x' + FFN(x'))
```

where `FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2` is a two-layer feed-forward network with ReLU activation.

### Training Objective

The model is trained to predict actions given the context of returns-to-go and states. The loss function is:

```
L = sum_{t=1}^{T} || a_t - a_hat_t ||^2
```

for continuous actions (MSE loss), or:

```
L = -sum_{t=1}^{T} log P(a_t | R_{<=t}, s_{<=t}, a_{<t})
```

for discrete actions (cross-entropy loss).

### Autoregressive Inference

At inference time, given a target return `R_target` and initial state `s_1`:

1. Set `R_1 = R_target`
2. Feed `(R_1, s_1)` into the transformer
3. Sample action `a_1` from the predicted distribution
4. Observe reward `r_1` and next state `s_2`
5. Update: `R_2 = R_1 - r_1`
6. Feed `(R_1, s_1, a_1, R_2, s_2)` and predict `a_2`
7. Repeat until horizon `T`

---

## Offline RL Formulation for Trading

### Why Offline RL?

Online RL in live financial markets is impractical:
- **Cost of Exploration**: Random exploration means real financial losses
- **Non-Stationarity**: The environment changes as the agent trades
- **Latency**: Real-time interaction requires ultra-low latency infrastructure
- **Regulatory Constraints**: Regulatory frameworks limit algorithmic experimentation

Offline RL solves this by learning from historical data — a **fixed dataset** of previously collected trajectories. The Decision Transformer is particularly well-suited because it sidesteps the distributional shift problem that plagues other offline RL methods (CQL, BCQ, BEAR).

### Constructing the Offline Dataset

From historical OHLCV data, we construct trajectories:

1. **State Definition**: At each timestep `t`, the state `s_t` includes:
   - Normalized OHLCV features (open, high, low, close, volume)
   - Technical indicators (SMA, RSI, MACD, Bollinger Bands)
   - Portfolio state (current position, unrealized PnL)

2. **Action Definition**: Discrete actions `{-1, 0, 1}` representing:
   - `-1`: Short / Sell
   - `0`: Hold / No action
   - `1`: Long / Buy

3. **Reward Definition**: The reward at each step:
   ```
   r_t = position_t * (close_{t+1} - close_t) / close_t
   ```
   This is the percentage return based on the current position.

4. **RTG Calculation**: For each trajectory of length `T`:
   ```
   R_t = sum_{t'=t}^{T} r_{t'}
   ```

### Trajectory Segmentation

Historical data is segmented into episodes (e.g., daily, weekly) to create multiple training trajectories. Each episode captures a complete trading cycle with clear start and end points.

### Data Augmentation

To increase dataset diversity:
- **Multi-timeframe sampling**: Create trajectories from 1m, 5m, 15m, 1h candles
- **Sliding windows**: Overlapping episode boundaries
- **Return scaling**: Normalize RTG values across episodes
- **Noise injection**: Add small perturbations to states for robustness

---

## Applications in Trading

### Return-Conditioned Strategy Generation

The most powerful feature of the Decision Transformer is the ability to generate strategies conditioned on target returns:

- **Conservative Mode** (`R_target = 0.5%`): Generate low-risk strategies that aim for modest returns
- **Aggressive Mode** (`R_target = 5%`): Generate high-risk strategies targeting large returns
- **Adaptive Mode**: Dynamically adjust `R_target` based on market regime

### Multi-Asset Portfolio Management

The Decision Transformer can be extended to multi-asset settings:
- State includes features from multiple assets
- Action space covers position sizing across the portfolio
- RTG reflects portfolio-level returns

### Market Regime Adaptation

By training on data spanning multiple market regimes (bull, bear, sideways, high-volatility), the model learns regime-dependent action distributions. At inference time, the RTG conditioning naturally adapts to the current regime.

### Risk Management Integration

The reward function can incorporate risk metrics:
- Sharpe ratio as reward
- Maximum drawdown penalties
- Position size constraints via action clipping

---

## Implementation in Rust

The Rust implementation provides a from-scratch Decision Transformer with the following components:

### Core Architecture

```rust
// Key structures in lib.rs:

// DecisionTransformer - Main model with embed layers, transformer blocks, prediction heads
// CausalSelfAttention - Masked multi-head attention
// TransformerBlock - Attention + FFN + LayerNorm
// TrajectoryDataset - Offline dataset from OHLCV data
// RTG computation - Return-to-go calculation from rewards
```

### Key Features

1. **Return-to-Go Calculation**: Efficiently computes RTG from reward sequences using reverse cumulative sum
2. **Causal Masking**: Implements proper autoregressive masking for the attention mechanism
3. **State Embedding**: Projects raw OHLCV features into the transformer's hidden dimension
4. **Action Prediction**: Discrete action prediction (buy/sell/hold) from transformer outputs
5. **Bybit Integration**: Fetches live BTCUSDT data for training dataset construction

### Building and Running

```bash
cd 315_decision_transformer/rust
cargo build
cargo test
cargo run --example trading_example
```

### Example Output

```
=== Decision Transformer for Trading ===
Fetching BTCUSDT data from Bybit...
Building offline dataset from 200 candles...
Trajectories created: 10
Training Decision Transformer...
Epoch 1/50: loss = 1.0823
Epoch 10/50: loss = 0.4215
Epoch 50/50: loss = 0.1034
Generating trading actions with target return = 2.0%...
Step 1: State=[0.52, 0.48, ...], Action=BUY, RTG=1.98%
Step 2: State=[0.55, 0.51, ...], Action=HOLD, RTG=1.85%
...
```

---

## Bybit Data Integration

### API Endpoint

The implementation uses Bybit's public V5 API to fetch historical kline (candlestick) data:

```
GET https://api.bybit.com/v5/market/kline
```

Parameters:
- `category`: "spot"
- `symbol`: "BTCUSDT"
- `interval`: "60" (1-hour candles)
- `limit`: 200

### Data Processing Pipeline

1. **Fetch**: HTTP GET request to Bybit API
2. **Parse**: Deserialize JSON response into OHLCV structures
3. **Normalize**: Scale features to [0, 1] range using min-max normalization
4. **Segment**: Split into fixed-length episodes
5. **Label**: Assign actions based on price movement heuristics (for offline dataset)
6. **RTG**: Calculate return-to-go for each timestep

### Error Handling

The implementation includes robust error handling:
- Network timeout and retry logic
- JSON parsing validation
- Missing data interpolation
- API rate limit awareness

---

## Key Takeaways

1. **RL as Sequence Modeling**: The Decision Transformer reframes RL as conditional sequence generation, enabling the use of powerful transformer architectures for sequential decision-making. This paradigm shift eliminates the need for value functions, temporal difference learning, or policy gradients.

2. **Return-to-Go Conditioning**: By conditioning on desired future returns, the Decision Transformer allows traders to specify performance targets and generate strategies accordingly. Higher RTG values produce more aggressive strategies; lower values produce conservative ones.

3. **Offline RL for Trading**: The Decision Transformer is ideal for offline RL in financial markets, where online exploration is costly and dangerous. It learns from historical data without requiring a market simulator.

4. **Causal Transformer Architecture**: The masked self-attention mechanism ensures autoregressive generation — each action prediction only depends on past observations, maintaining temporal causality. This is critical for avoiding look-ahead bias in trading.

5. **Practical Considerations**: While the Decision Transformer is powerful, practitioners must address distribution shift (the model may encounter states not seen during training), non-stationarity (market dynamics change over time), and reward specification (choosing appropriate reward functions that align with trading objectives).

6. **Scalability**: The transformer architecture scales well with data and compute, enabling training on large multi-asset datasets spanning years of market history.

7. **Future Directions**: Extensions include multi-modal state representations (combining OHLCV with order book data and sentiment), hierarchical Decision Transformers for multi-timeframe strategies, and online fine-tuning with conservative exploration.

---

## References

1. Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., & Mordatch, I. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. *NeurIPS 2021*.
2. Janner, M., Li, Q., & Levine, S. (2021). Offline Reinforcement Learning as One Big Sequence Modeling Problem. *NeurIPS 2021*.
3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*.
4. Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. *arXiv:2005.01643*.
5. Zheng, Q., Zhang, A., & Grover, A. (2022). Online Decision Transformer. *ICML 2022*.
