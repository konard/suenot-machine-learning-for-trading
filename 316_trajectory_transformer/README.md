# Chapter 316: Trajectory Transformer for Trading

## 1. Introduction: Treating Reinforcement Learning as Sequence Modeling

Traditional reinforcement learning (RL) approaches to trading rely on iterative value estimation (e.g., Q-learning) or policy gradient methods that optimize actions one step at a time. These methods suffer from well-known problems: distributional shift in offline settings, short-sighted credit assignment, and difficulty with long-horizon planning. The **Trajectory Transformer** (Janner et al., 2021) offers a fundamentally different paradigm: instead of learning a value function or policy, it models the *entire trajectory* of states, actions, and rewards as a sequence, then leverages the powerful sequence modeling capabilities of Transformers to plan.

This insight -- that RL can be recast as a sequence prediction problem -- has profound implications for algorithmic trading. A trading agent's history (market states, executed orders, realized PnL) forms a natural sequence. By training a Transformer to model the joint distribution over these trajectories, we can:

- **Plan multi-step trading strategies** by generating entire trade sequences conditioned on a desired return target.
- **Leverage offline data** without the instability of off-policy RL methods.
- **Perform beam search** over possible futures to find robust action plans.

The Trajectory Transformer draws inspiration from language modeling: just as GPT predicts the next token in a sentence, the Trajectory Transformer predicts the next element in a trajectory. The key difference is that trajectories are structured -- they consist of interleaved state dimensions, action dimensions, and reward values -- requiring a careful tokenization scheme.

### Key Differences from Decision Transformer

While the Decision Transformer (Chen et al., 2021) also frames RL as sequence modeling, it conditions on *desired returns* and autoregressively generates actions. The Trajectory Transformer goes further by modeling the full joint distribution over states, actions, and rewards, enabling:

1. **Beam search planning**: exploring multiple possible futures simultaneously.
2. **State prediction**: forecasting where the environment will be, not just what action to take.
3. **Return redistribution**: the model implicitly learns which parts of a trajectory contribute most to total return.

---

## 2. Mathematical Foundations

### 2.1 Trajectory Representation

A trajectory $\tau$ in the trading context consists of a sequence of transitions:

$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$$

where:
- $s_t \in \mathbb{R}^{d_s}$ is the market state at time $t$ (e.g., prices, volumes, technical indicators, portfolio state)
- $a_t \in \mathbb{R}^{d_a}$ is the trading action (position sizing, order type)
- $r_t \in \mathbb{R}$ is the reward (realized PnL, risk-adjusted return)

### 2.2 Trajectory Tokenization

To apply a Transformer, continuous values must be discretized into tokens. Each dimension of $s_t$, $a_t$, and $r_t$ is independently discretized into $V$ bins using quantile-based binning:

$$\text{tokenize}(x, \text{dim}) = \text{bin}_{\text{dim}}\left(\frac{x - \mu_{\text{dim}}}{\sigma_{\text{dim}}}\right)$$

The full trajectory becomes a token sequence of length $T \times (d_s + d_a + 1)$:

$$\hat{\tau} = (\hat{s}_0^1, \hat{s}_0^2, \ldots, \hat{s}_0^{d_s}, \hat{a}_0^1, \ldots, \hat{a}_0^{d_a}, \hat{r}_0, \hat{s}_1^1, \ldots)$$

where $\hat{x}$ denotes the tokenized version of $x$.

### 2.3 Transformer Architecture

The model uses a GPT-style decoder-only Transformer:

$$P(\hat{\tau}) = \prod_{i=1}^{|\hat{\tau}|} P(\hat{\tau}_i \mid \hat{\tau}_{<i})$$

Each token prediction is a categorical distribution over $V$ vocabulary bins. The model is trained with cross-entropy loss:

$$\mathcal{L} = -\sum_{i=1}^{|\hat{\tau}|} \log P(\hat{\tau}_i \mid \hat{\tau}_{<i}; \theta)$$

#### Multi-Head Self-Attention

For a sequence of embeddings $X \in \mathbb{R}^{n \times d}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q = XW_Q$, $K = XW_K$, $V = XW_V$ are linear projections.

Multi-head attention with $h$ heads:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O$$

### 2.4 Beam Search Planning

Given a current state $s_t$ and a desired return $R^*$, beam search generates candidate trajectories:

1. **Initialize** $B$ beams with tokenized current state $\hat{s}_t$.
2. **Expand** each beam by sampling top-$k$ next tokens from $P(\hat{\tau}_i \mid \hat{\tau}_{<i})$.
3. **Score** beams by log-probability: $\text{score}(\hat{\tau}_{1:i}) = \sum_{j=1}^{i} \log P(\hat{\tau}_j \mid \hat{\tau}_{<j})$.
4. **Prune** to top $B$ beams.
5. **Return conditioning**: filter or re-weight beams whose predicted cumulative reward matches $R^*$.

The return-conditioned objective:

$$a_t^* = \arg\max_{a_t} P(a_t \mid s_t, R^* = R_{\text{target}})$$

This is approximated by the beam search procedure, where we prepend or condition on the desired return tokens.

### 2.5 Return-Conditioned Generation

We can condition trajectory generation on a target return by:

1. Setting the reward tokens to reflect the desired cumulative return.
2. Using the model to generate actions that are most likely under this conditioning.

Formally, we seek:

$$\hat{a}_t = \arg\max_{\hat{a}} P(\hat{a} \mid \hat{s}_t, \hat{R}_t = \hat{R}^*)$$

where $\hat{R}_t = \sum_{k=t}^{T} r_k$ is the return-to-go, tokenized and placed at the beginning of each sub-sequence.

---

## 3. Applications to Trading

### 3.1 Offline RL for Trading

Financial markets make online RL dangerous -- you cannot freely explore with real capital. The Trajectory Transformer is ideally suited for **offline RL**, where we train exclusively on historical data:

- **Historical order flow data**: trajectories of market states and executed trades.
- **Backtesting trajectories**: simulated trading episodes with known outcomes.
- **Multi-strategy data**: combining trajectories from different trading strategies provides diverse coverage of the state-action space.

Unlike traditional offline RL methods (CQL, BCQ) that require careful conservatism constraints, the Trajectory Transformer naturally handles distributional shift because it models the data distribution directly as a sequence model.

### 3.2 Multi-Step Trade Planning

A key advantage is the ability to plan multi-step trade sequences:

1. **Entry/exit planning**: Given a market state, generate a full entry-hold-exit sequence optimized for a return target.
2. **Portfolio rebalancing**: Plan a sequence of trades to transition from current allocation to target allocation while minimizing market impact.
3. **Risk-aware planning**: Condition on both return *and* maximum drawdown constraints by incorporating risk metrics into the reward tokenization.

### 3.3 Regime-Aware Trajectories

By including regime indicators (volatility regime, trend strength) in the state representation, the model learns regime-dependent trading patterns:

- Bull market trajectories favor momentum-following actions.
- High-volatility regimes produce more conservative position sizing.
- Range-bound markets generate mean-reversion sequences.

### 3.4 Practical Considerations

**Trajectory length**: For intraday trading, trajectories might span 50-200 steps (e.g., 5-minute bars over a trading day). For swing trading, 20-60 daily bars.

**Vocabulary size**: Typically $V = 100$ bins per dimension provides sufficient resolution. For prices, use relative returns rather than absolute values for stationarity.

**Context window**: The Transformer's context window limits the trajectory horizon. With modern efficient attention, sequences of 1000+ tokens are feasible.

---

## 4. Rust Implementation

The implementation in `rust/src/lib.rs` provides:

- **TrajectoryTokenizer**: Discretizes continuous market data into tokens using quantile-based binning. Each state dimension, action dimension, and reward is independently binned.
- **TrajectoryTransformer**: A simplified Transformer with multi-head self-attention, layer normalization, and feed-forward networks. Supports autoregressive generation.
- **BeamSearch**: Implements beam search over token sequences to find high-probability trajectory completions.
- **ReturnConditionedPlanner**: Wraps the Transformer and beam search to generate trading actions conditioned on a target return.
- **BybitClient**: Fetches historical OHLCV data from the Bybit REST API.

The Transformer uses a simplified architecture suitable for demonstration:
- Embedding dimension: configurable (default 64)
- Number of heads: configurable (default 4)
- Feed-forward hidden dimension: 4x embedding dimension
- Causal masking for autoregressive generation

See `rust/examples/trading_example.rs` for a complete end-to-end workflow.

---

## 5. Bybit Data Integration

The implementation fetches real market data from Bybit's public API:

```
GET https://api.bybit.com/v5/market/kline
```

Parameters:
- `category=linear` (perpetual futures)
- `symbol=BTCUSDT`
- `interval=60` (1-hour candles)
- `limit=200`

The raw OHLCV data is transformed into trading states:
- **Price features**: log returns, moving average ratios, volatility
- **Volume features**: relative volume, volume momentum
- **Technical indicators**: RSI approximation, price momentum

Actions are discretized into: strong sell (-2), sell (-1), hold (0), buy (+1), strong buy (+2).

Rewards are computed as the PnL from each action given the subsequent price movement.

---

## 6. Key Takeaways

1. **Sequence modeling reframes RL**: By treating trajectories as sequences of tokens, we can leverage the full power of Transformer architectures -- attention, parallel training, and scalability.

2. **Beam search enables planning**: Unlike policy-based methods that commit to a single action, beam search explores multiple candidate futures simultaneously, producing more robust trading plans.

3. **Return conditioning is powerful**: The ability to specify a target return and generate actions likely to achieve it is a natural fit for trading, where risk-return targets are explicit.

4. **Offline RL without conservatism penalties**: The Trajectory Transformer avoids the need for explicit conservatism constraints (like CQL's penalty terms) because it directly models the data distribution.

5. **Tokenization matters**: The quality of discretization directly affects model performance. Quantile-based binning ensures uniform token usage, while the vocabulary size trades off precision against sequence length.

6. **Multi-step planning for trading**: Traditional RL methods optimize one action at a time. The Trajectory Transformer can plan entire trade sequences -- entries, position adjustments, and exits -- as a coherent plan.

7. **Regime adaptability**: By encoding market regime information in the state, the model learns to produce different trading patterns for different market conditions without explicit regime-switching logic.

---

## References

- Janner, M., Li, Q., & Levine, S. (2021). *Offline Reinforcement Learning as One Big Sequence Modeling Problem*. NeurIPS.
- Chen, L., Lu, K., Rajeswaran, A., et al. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling*. NeurIPS.
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Zheng, Q., Zhang, A., & Grover, A. (2022). *Online Decision Transformer*. ICML.
