# Chapter 310: Curiosity-Driven Trading

## Introduction: Intrinsic Motivation and Curiosity for Trading Exploration

Traditional reinforcement learning (RL) agents in trading rely exclusively on extrinsic rewards -- profit, Sharpe ratio, or other performance metrics. While effective in well-explored market conditions, these agents often fail to generalize when markets shift into novel regimes. They exploit known patterns but never truly explore the state space. This chapter introduces **curiosity-driven exploration** as a mechanism to address this fundamental limitation.

In cognitive science, intrinsic motivation refers to the drive to engage in an activity for its inherent satisfaction rather than for an external reward. Curiosity -- the desire to learn about something new or unknown -- is one of the most powerful forms of intrinsic motivation. When applied to trading agents, curiosity provides a bonus reward signal that encourages the agent to visit unfamiliar market states, discover new patterns, and build richer internal representations of market dynamics.

The core insight is simple: an agent that is rewarded for encountering states it cannot predict well will naturally gravitate toward under-explored regions of the market. In trading, these under-explored regions often correspond to regime transitions, rare market events, or emerging correlations -- precisely the scenarios where conventional agents fail catastrophically.

Two foundational approaches have emerged for implementing curiosity in RL:

1. **Intrinsic Curiosity Module (ICM)** -- uses forward model prediction error as intrinsic reward
2. **Random Network Distillation (RND)** -- uses distillation error against a fixed random network as a novelty signal

Both approaches share a common principle: the agent receives higher intrinsic reward when it encounters states that are harder to predict or less familiar, driving exploration toward novel market conditions.

### Why Curiosity Matters in Trading

Financial markets present unique challenges for exploration:

- **Non-stationarity**: Market dynamics change over time, meaning previously well-explored regions can become novel again
- **Rare events**: Black swan events, flash crashes, and regime changes are infrequent but critically important
- **High dimensionality**: The combinatorial explosion of market states means most of the state space remains unexplored
- **Sparse rewards**: Profitable opportunities may be separated by long periods of flat or losing trades

A curiosity-driven agent addresses all four challenges by maintaining an internal model of "what it knows" and actively seeking out what it doesn't.

## Mathematical Foundations

### Intrinsic Curiosity Module (ICM)

The ICM, introduced by Pathak et al. (2017), consists of three components operating in a learned feature space:

**Feature Encoder** maps raw state observations to a compact representation:

$$\phi(s_t) = f_\theta(s_t) \in \mathbb{R}^d$$

where $f_\theta$ is a neural network parameterized by $\theta$.

**Forward Model** predicts the next state's feature representation given the current state features and action:

$$\hat{\phi}(s_{t+1}) = g_{\text{fwd}}(\phi(s_t), a_t)$$

The forward model loss is the L2 prediction error:

$$L_{\text{fwd}} = \frac{1}{2} \| \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) \|^2$$

**Inverse Model** predicts the action taken given two consecutive state features:

$$\hat{a}_t = g_{\text{inv}}(\phi(s_t), \phi(s_{t+1}))$$

The inverse model loss is:

$$L_{\text{inv}} = -\sum_k a_t^{(k)} \log \hat{a}_t^{(k)}$$

The inverse model serves a crucial role: it forces the feature encoder to capture only those aspects of the state that are influenced by the agent's actions, filtering out noise and irrelevant state changes (e.g., random market microstructure noise).

**Intrinsic Reward** is defined as the forward model prediction error:

$$r_t^i = \eta \cdot \| \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) \|^2$$

where $\eta$ is a scaling factor. The total reward becomes:

$$r_t = r_t^e + \beta \cdot r_t^i$$

where $r_t^e$ is the extrinsic reward (trading profit) and $\beta$ controls the exploration-exploitation trade-off.

The combined ICM loss is:

$$L_{\text{ICM}} = (1 - \alpha) \cdot L_{\text{inv}} + \alpha \cdot L_{\text{fwd}}$$

where $\alpha \in [0, 1]$ balances the two objectives.

### Random Network Distillation (RND)

RND, introduced by Burda et al. (2019), provides a simpler alternative to ICM:

**Fixed Target Network**: A randomly initialized neural network $f: \mathcal{S} \to \mathbb{R}^k$ with fixed (non-trainable) weights.

**Predictor Network**: A trainable network $\hat{f}: \mathcal{S} \to \mathbb{R}^k$ that attempts to match the target network's output.

The predictor is trained to minimize:

$$L_{\text{RND}} = \| \hat{f}(s_t) - f(s_t) \|^2$$

**Intrinsic Reward** is the prediction error:

$$r_t^i = \| \hat{f}(s_t) - f(s_t) \|^2$$

The key insight is that the predictor will have low error on frequently visited states (which it has been trained on) and high error on novel states. This provides a novelty signal without requiring a forward dynamics model.

**Advantages of RND over ICM for trading:**
- No need to model market dynamics explicitly (which is extremely difficult)
- Robust to stochastic environments -- random noise does not inflate curiosity
- Simpler architecture with fewer hyperparameters
- Naturally handles non-stationary distributions

### Reward Normalization

Both ICM and RND benefit from running reward normalization to prevent the intrinsic reward from dominating:

$$\hat{r}_t^i = \frac{r_t^i - \mu_r}{\sigma_r + \epsilon}$$

where $\mu_r$ and $\sigma_r$ are running statistics of intrinsic rewards.

## Applications in Trading

### Discovering Novel Market Regimes

Curiosity-driven agents excel at detecting regime changes. When the market transitions from a trending phase to a mean-reverting phase (or vice versa), the agent's forward model prediction error spikes, generating a large intrinsic reward. This encourages the agent to:

1. Pay attention to the transition period rather than ignoring it
2. Allocate more exploration budget to understanding the new regime
3. Build internal representations that distinguish between regimes
4. Adapt trading strategies more quickly to the new conditions

A practical regime detection pipeline using curiosity:
- Monitor the running average of intrinsic reward
- A sustained increase in intrinsic reward signals a potential regime change
- Use the intrinsic reward magnitude to estimate the degree of novelty
- Adjust position sizing inversely to intrinsic reward (reduce exposure during uncertain transitions)

### Exploring Rare Trading Opportunities

Markets occasionally present rare but highly profitable opportunities -- arbitrage windows, liquidation cascades, or structural dislocations. Traditional RL agents trained on historical data rarely encounter these scenarios and thus cannot learn to exploit them.

Curiosity-driven agents, by contrast:
- Are naturally drawn to unusual market configurations
- Build experience in rare but important market states
- Develop more robust value functions that cover tail scenarios
- Can identify and act on opportunities that non-curious agents would miss entirely

### Cross-Asset Exploration

When trading multiple assets simultaneously, curiosity drives the agent to explore correlation structures:
- Discover emerging cross-asset relationships
- Detect breakdown of historical correlations
- Identify new factor exposures as market microstructure evolves

## Rust Implementation

The implementation in `rust/src/lib.rs` provides:

- **`ForwardModel`**: Predicts next market state features from current features and action
- **`InverseModel`**: Predicts action from consecutive state features
- **`ICMModule`**: Combines forward and inverse models with intrinsic reward computation
- **`RNDModule`**: Random Network Distillation with fixed target and trainable predictor
- **`CuriosityDrivenAgent`**: Full agent combining extrinsic and intrinsic rewards
- **`BybitClient`**: Fetches real OHLCV data from Bybit API

The architecture is designed for production use with:
- Configurable feature dimensions, action spaces, and learning rates
- Running reward normalization for stable training
- Modular design allowing ICM and RND to be used independently or combined
- Efficient ndarray-based linear algebra operations

### Key Design Decisions

**Feature Space**: Market states are encoded as normalized vectors of OHLCV data plus derived features (returns, volatility, volume ratios). The feature encoder projects these into a lower-dimensional latent space where dynamics are smoother.

**Action Space**: Discrete actions (buy, sell, hold) are used for simplicity, but the architecture supports extension to continuous position sizing.

**Reward Combination**: The total reward uses an adaptive weighting scheme where the intrinsic reward coefficient $\beta$ decays over time, shifting from exploration to exploitation as the agent becomes more familiar with market dynamics.

## Bybit Data Integration

The implementation includes a `BybitClient` that fetches historical kline (candlestick) data from the Bybit V5 API:

```
GET https://api.bybit.com/v5/market/kline
```

Parameters:
- `category`: "linear" for USDT perpetual contracts
- `symbol`: e.g., "BTCUSDT"
- `interval`: candle interval (1, 5, 15, 60, 240, D, W)
- `limit`: number of candles (max 200)

The fetched data is converted into a feature matrix suitable for the curiosity modules:
- Raw OHLCV values are normalized using z-score normalization
- Returns are computed as log-returns: $r_t = \ln(p_t / p_{t-1})$
- Volatility is estimated using a rolling standard deviation of returns
- Volume features are normalized relative to a moving average

This pipeline ensures the curiosity signal reflects genuine market novelty rather than scaling artifacts.

## Key Takeaways

1. **Curiosity provides a principled exploration mechanism** for trading agents, encouraging them to visit unfamiliar market states rather than repeatedly exploiting known patterns.

2. **ICM uses prediction error as reward**: when the agent's forward model fails to predict the next state, that state is considered novel and worth exploring. The inverse model filters out noise by focusing the feature space on agent-controllable aspects.

3. **RND is simpler and more robust**: by measuring how well a predictor can match a fixed random network's output, RND provides a novelty signal that is immune to stochastic noise -- a critical advantage in financial markets.

4. **Regime detection emerges naturally**: spikes in intrinsic reward correspond to market regime changes, giving curiosity-driven agents an inherent regime-detection capability.

5. **Curiosity improves sample efficiency**: by focusing exploration on novel states, the agent learns more per unit of experience, which is crucial when trading data is limited or expensive to obtain.

6. **The exploration-exploitation balance must be managed carefully**: too much curiosity leads to excessive trading and transaction costs; too little reverts to the baseline behavior. The $\beta$ parameter and reward normalization are essential control mechanisms.

7. **Combining ICM and RND yields the best results**: ICM captures action-dependent novelty while RND captures state novelty, providing complementary signals that together drive more effective exploration.

8. **Curiosity-driven agents are more robust to distribution shift**: because they have explored a wider range of market states during training, they generalize better to unseen market conditions at deployment time.

## References

- Pathak, D., et al. (2017). "Curiosity-driven Exploration by Self-Supervised Prediction." ICML.
- Burda, Y., et al. (2019). "Exploration by Random Network Distillation." ICLR.
- Bellemare, M., et al. (2016). "Unifying Count-Based Exploration and Intrinsic Motivation." NeurIPS.
- Oord, A., et al. (2018). "Representation Learning with Contrastive Predictive Coding." arXiv.
- Tang, H., et al. (2017). "#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning." NeurIPS.
