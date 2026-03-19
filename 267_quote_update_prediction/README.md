# Chapter 267: Quote Update Prediction

## Introduction

In modern electronic markets, prices do not change in a smooth continuum; they move in discrete jumps called **quote updates**. Every time a market participant submits, cancels, or modifies a limit order at the best bid or ask, the top-of-book quote changes and a new message propagates through the exchange's matching engine. Predicting the *direction* and *magnitude* of the next quote update -- even a few milliseconds before it materialises -- is one of the highest-value problems in quantitative finance.

This chapter develops a complete framework for **quote update prediction** (QUP). We start from raw Level-2 order-book data, engineer informative features, build a fast linear scoring model, and deploy it against the Bybit cryptocurrency exchange API. The emphasis is on *latency-aware design*: every component is implemented in Rust so that inference can happen in sub-microsecond time, well within the budget of a high-frequency trading (HFT) system.

## 1. What Is a Quote Update?

A **quote** is the current best bid and best ask price--quantity pair published by an exchange. A **quote update** occurs when any of the following events happen:

| Event | Effect on Quote |
|-------|----------------|
| New limit order at best bid/ask | Size increases |
| Cancellation at best bid/ask | Size decreases; price may shift |
| Market order executes against best | Size decreases; price may shift |
| New order improves the best price | Price and size change |

Formally, let $Q_t = (b_t, a_t, s_t^b, s_t^a)$ denote the quote at time $t$, where $b_t$ is the best bid price, $a_t$ the best ask, $s_t^b$ the bid size and $s_t^a$ the ask size. A quote update is any transition $Q_t \to Q_{t+1}$ where at least one component changes.

### 1.1 Mid-Price and Micro-Price

The **mid-price** is the arithmetic mean of the best bid and ask:

$$m_t = \frac{b_t + a_t}{2}$$

The **micro-price** (also called the volume-weighted mid) adjusts for order-book imbalance:

$$\tilde{m}_t = \frac{s_t^b \cdot a_t + s_t^a \cdot b_t}{s_t^b + s_t^a}$$

When $s_t^b \gg s_t^a$, the micro-price shifts toward the ask, reflecting the higher probability that the next trade will lift the ask. The difference $\tilde{m}_t - m_t$ is one of the most predictive features for the next mid-price movement.

### 1.2 Order-Book Imbalance

The **level-1 imbalance** is:

$$I_t = \frac{s_t^b - s_t^a}{s_t^b + s_t^a} \in [-1, 1]$$

Positive imbalance means more resting volume on the bid side, empirically correlated with upward price pressure. We generalise to **depth-weighted imbalance** across $L$ levels:

$$I_t^{(L)} = \frac{\sum_{l=1}^{L} w_l \cdot s_{t,l}^b - \sum_{l=1}^{L} w_l \cdot s_{t,l}^a}{\sum_{l=1}^{L} w_l \cdot (s_{t,l}^b + s_{t,l}^a)}$$

where $w_l = 1/l$ gives higher weight to levels closer to the best quote.

## 2. Feature Engineering for Quote Prediction

Good features for QUP must satisfy two constraints:

1. **Predictive power** -- they must carry information about the next mid-price direction.
2. **Low latency** -- they must be computable in O(1) per update, with no memory allocations on the hot path.

### 2.1 Feature Set

We use the following eight features extracted from pairs of consecutive snapshots $(Q_{t-1}, Q_t)$:

| # | Feature | Formula | Intuition |
|---|---------|---------|-----------|
| 1 | `imbalance` | $I_t$ | Level-1 buy/sell pressure |
| 2 | `depth_imbalance` | $I_t^{(5)}$ | Multi-level pressure |
| 3 | `normalised_spread_bps` | $\frac{a_t - b_t}{m_t} \times 10^4$ | Liquidity cost |
| 4 | `mid_return_bps` | $\frac{m_t - m_{t-1}}{m_{t-1}} \times 10^4$ | Recent momentum |
| 5 | `micro_mid_diff_bps` | $\frac{\tilde{m}_t - m_t}{m_t} \times 10^4$ | Imbalance-based fair value |
| 6 | `bid_size_change_pct` | $\frac{s_t^b - s_{t-1}^b}{s_{t-1}^b} \times 100$ | Bid replenishment |
| 7 | `ask_size_change_pct` | $\frac{s_t^a - s_{t-1}^a}{s_{t-1}^a} \times 100$ | Ask replenishment |
| 8 | `dt_ms` | $t - (t-1)$ in ms | Update arrival rate |

### 2.2 Z-Score Normalisation

Before feeding features to the model we apply column-wise z-score normalisation:

$$\hat{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are estimated from the training window. In production these statistics are computed over a rolling window (e.g. the last 10,000 updates).

## 3. The Linear Scoring Model

For HFT applications, model complexity is the enemy of latency. A **linear scoring function** provides an excellent trade-off:

$$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$$

The predicted direction is $\hat{y} = \text{sign}(f(\mathbf{x}))$:

- $\hat{y} = +1$: next mid-price expected to move **up**
- $\hat{y} = -1$: next mid-price expected to move **down**
- $\hat{y} = 0$: **stable** (score exactly zero, rare in practice)

### 3.1 Hinge Loss and SGD

We train with an online **hinge loss** (the same loss underlying linear SVMs):

$$\mathcal{L}(y, f(\mathbf{x})) = \max(0,\; 1 - y \cdot f(\mathbf{x}))$$

The sub-gradient update for a single sample is:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y \cdot \mathbf{x} \quad \text{if } y \cdot f(\mathbf{x}) < 1$$

$$b \leftarrow b + \eta \cdot y \quad \text{if } y \cdot f(\mathbf{x}) < 1$$

where $\eta$ is the learning rate. This is a single pass (online SGD), making it suitable for streaming data where the model is updated tick-by-tick.

### 3.2 Why Not Deep Learning?

Deep models (LSTMs, Transformers) can capture longer-horizon dependencies, but they incur:

- **Latency**: matrix multiplications over large hidden states take microseconds, not nanoseconds.
- **Overfitting risk**: with noisy tick data, simpler models often generalise better within the very short prediction horizons used in HFT.
- **Deployment complexity**: GPU inference adds jitter; a linear model runs entirely in L1 cache.

For horizons beyond a few seconds, non-linear models become valuable, but for **next-tick** prediction the linear model is hard to beat on a risk-adjusted basis.

## 4. Labelling Strategy

We assign a directional label to each snapshot pair using a **threshold** $\tau$ (in basis points):

$$y_t = \begin{cases} +1 & \text{if } \frac{m_{t+1} - m_t}{m_t} \times 10^4 > \tau \\ -1 & \text{if } \frac{m_{t+1} - m_t}{m_t} \times 10^4 < -\tau \\ 0 & \text{otherwise (stable)} \end{cases}$$

Setting $\tau$ is a modelling choice:

- **$\tau = 0$**: every non-zero tick is labelled, leading to noisy labels.
- **$\tau = 0.5$ to $2.0$ bps**: filters out micro-noise; typical for crypto markets.
- **$\tau > 5$ bps**: only large moves are predicted; fewer samples but cleaner signal.

Stable labels ($y = 0$) are skipped during training to focus the model on directional moves.

## 5. Rust Implementation Walkthrough

### 5.1 Data Structures

The `OrderBookSnapshot` struct holds a full Level-2 snapshot including multiple depth levels. Key methods:

- `mid_price()` -- arithmetic mid.
- `micro_price()` -- volume-weighted mid.
- `imbalance()` -- level-1 imbalance.
- `depth_imbalance(levels)` -- weighted multi-level imbalance.

### 5.2 Feature Extraction

`extract_features(prev, curr)` computes all eight features from two consecutive snapshots. The function is allocation-free on the hot path (all values are stack-allocated `f64`s).

`build_feature_matrix(snapshots)` processes a full sequence into an `ndarray::Array2<f64>` matrix suitable for batch training.

### 5.3 Model Training

`LinearQuotePredictor` implements:

- `new(num_features, learning_rate)` -- random initialisation.
- `score(features)` -- raw linear score (a single dot product).
- `predict(features)` -- direction from sign of score.
- `train_epoch(x, labels)` -- one pass of online SGD with hinge loss.
- `accuracy(x, labels)` -- evaluation metric (excludes stable labels).

### 5.4 Normalisation

`z_normalise(x)` returns the normalised matrix along with the mean and standard deviation vectors, which are needed at inference time to transform live features.

## 6. Bybit Integration

### 6.1 Fetching Order-Book Data

The library provides both async and blocking functions to query the Bybit v5 REST API:

```rust
// Async version (use inside tokio runtime)
let snap = fetch_bybit_orderbook("BTCUSDT", 10).await?;

// Blocking version (use in synchronous contexts)
let snap = fetch_bybit_orderbook_blocking("BTCUSDT", 10)?;
```

The endpoint `GET /v5/market/orderbook` returns up to 200 depth levels. We parse bids and asks into our `OrderBookSnapshot` structure, extracting the top-of-book prices and sizes automatically.

### 6.2 Live Scoring Pipeline

The trading example (`examples/trading_example.rs`) demonstrates the full pipeline:

1. **Train** on synthetic data (or historical snapshots).
2. **Fetch** live snapshots from Bybit at ~500ms intervals.
3. **Extract** features from consecutive snapshots.
4. **Normalise** using training-set statistics.
5. **Score** and predict the next direction.

In production, you would replace the polling loop with a WebSocket subscription to `orderbook.1.BTCUSDT` for true real-time updates.

### 6.3 From Prediction to Execution

A positive score suggests the mid-price will tick up; a negative score suggests it will tick down. A simple execution strategy:

| Score | Action |
|-------|--------|
| $f(\mathbf{x}) > \theta$ | Place a limit buy at best bid |
| $f(\mathbf{x}) < -\theta$ | Place a limit sell at best ask |
| $|f(\mathbf{x})| \leq \theta$ | No action (wait) |

The threshold $\theta$ controls the aggressiveness of the strategy. Higher $\theta$ means fewer trades but higher expected profit per trade.

## 7. Performance Considerations

### 7.1 Latency Budget

| Operation | Typical Time |
|-----------|-------------|
| Feature extraction (8 features) | ~50 ns |
| Z-score normalisation | ~20 ns |
| Linear score (dot product) | ~10 ns |
| **Total inference** | **~80 ns** |

This leaves ample headroom within a typical exchange round-trip of 1-10 ms.

### 7.2 Model Refresh

Market microstructure is non-stationary. The model should be re-trained (or weights incrementally updated) at regular intervals:

- **Intraday**: re-fit every 30-60 minutes using the last N snapshots.
- **Rolling window**: maintain a FIFO buffer of the last 50,000 snapshots and retrain periodically.
- **Online learning**: update weights after every prediction using the realised label (when available).

### 7.3 Risk Controls

- **Position limits**: never accumulate more than a configured maximum position.
- **Loss limits**: halt trading if cumulative PnL drops below a threshold.
- **Spread filter**: do not trade when the spread exceeds a maximum (illiquid conditions).
- **Staleness check**: if the last snapshot is older than T ms, assume connectivity loss and flatten.

## 8. Extensions and Advanced Topics

### 8.1 Non-Linear Models

Replace the linear scorer with a small decision tree or gradient-boosted model for marginal accuracy gains. The key constraint is that inference must remain under ~1 microsecond.

### 8.2 Multi-Horizon Prediction

Instead of predicting only the next tick, predict returns at multiple horizons (1, 5, 25, 100 ticks ahead). Each horizon gets its own model, and signals are combined in a meta-model.

### 8.3 Cross-Asset Features

In crypto, BTC often leads altcoin movements. Including BTC order-book imbalance as a feature when predicting ETH quotes can improve accuracy significantly.

### 8.4 Queue Position Modelling

Beyond direction, model the **queue position** of your resting orders. The fill probability depends on how many shares are ahead of you in the queue, which changes with every quote update.

## 9. Key Takeaways

1. **Quote updates are the atomic unit of price discovery.** Predicting the next quote change is the fundamental problem in market microstructure.

2. **Order-book imbalance is the single most predictive feature.** The ratio of bid-to-ask size at the best levels has been shown across equities, futures, and crypto to forecast short-term price direction.

3. **Linear models dominate at ultra-low latency.** A dot product over 8 features runs in ~80 nanoseconds in Rust, well within HFT latency budgets.

4. **Feature normalisation is essential.** Raw features live on very different scales (imbalance in [-1,1] vs dt_ms in [50, 500]). Z-score normalisation ensures the model treats all features fairly.

5. **The micro-price is a better fair-value estimator than the mid-price.** By weighting by displayed size, the micro-price adjusts for asymmetric liquidity.

6. **Online SGD with hinge loss is natural for streaming tick data.** The model can be updated in constant time per observation, matching the pace of incoming market data.

7. **Bybit's REST API provides easy access to Level-2 data.** For production systems, the WebSocket feed delivers lower-latency, push-based updates.

8. **Risk management is as important as prediction.** Position limits, loss limits, and staleness checks prevent the model from accumulating catastrophic losses during regime changes.

## 10. References

- Cont, R., Kukanov, A., & Stoikov, S. (2014). *The Price Impact of Order Book Events*. Journal of Financial Econometrics, 12(1), 47-88.
- Cartea, A., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
- Stoikov, S. (2018). *The Micro-Price: A High-Frequency Estimator of Future Prices*. Quantitative Finance, 18(12), 1959-1966.
- Bybit API Documentation: https://bybit-exchange.github.io/docs/v5/market/orderbook
