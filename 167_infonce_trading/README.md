# Chapter 167: InfoNCE Trading

## 1. Introduction

**InfoNCE** (Information Noise-Contrastive Estimation) is a contrastive learning objective originally introduced by van den Oord et al. in their seminal 2018 paper *Representation Learning with Contrastive Predictive Coding*. The core idea is elegantly simple: learn representations by teaching a model to distinguish a true (positive) signal from a set of distractors (negatives). In the context of financial markets, this principle turns out to be remarkably powerful.

Traditional supervised approaches to market prediction require explicit labels — future returns, direction flags, regime tags — all of which are noisy, lag-prone, and inherently subjective. InfoNCE sidesteps this problem. Instead of predicting a target value, it learns an **embedding space** where similar market conditions cluster together and dissimilar conditions are pushed apart. The model never needs to know the "right answer"; it only needs to understand which states resemble one another.

Why does this matter for trading?

- **Market regimes are fuzzy.** There is no crisp boundary between a trending market and a mean-reverting one. Contrastive learning discovers these boundaries from data rather than imposing them.
- **Labels are expensive.** Labeling every 5-minute candle as "bullish" or "bearish" is reductive. InfoNCE learns structure without labels.
- **Transfer learning becomes natural.** Once you have a good encoder, you can fine-tune it for multiple downstream tasks — signal generation, risk management, portfolio construction — all sharing the same representation backbone.
- **Robustness to noise.** The contrastive objective is inherently more resistant to noisy inputs because it operates on relative comparisons rather than absolute targets.

In this chapter, we will build a complete InfoNCE-based representation learning system for cryptocurrency market data using Rust. We will fetch real OHLCV data from the Bybit exchange, construct contrastive pairs, train an encoder, and examine the learned embedding space.

## 2. Mathematical Foundation

### 2.1 The InfoNCE Loss

Given a query representation **q**, one positive key **k+**, and *N-1* negative keys **k-**, the InfoNCE loss is defined as:

```
L = -log( exp(sim(q, k+) / τ) / Σᵢ₌₁ᴺ exp(sim(q, kᵢ) / τ) )
```

where:
- `sim(a, b)` is a similarity function (typically cosine similarity)
- `τ` (tau) is a temperature hyperparameter
- The sum in the denominator runs over the positive key **and** all negative keys

This is essentially a **softmax cross-entropy loss** where the "correct class" is the positive pair. The model is trained to assign high similarity to the positive pair and low similarity to all negative pairs.

### 2.2 Cosine Similarity

The similarity function we use is cosine similarity:

```
sim(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

Cosine similarity ranges from -1 (opposite directions) to +1 (same direction) and is invariant to the magnitude of the vectors. This is desirable because we care about the *direction* of market-state representations, not their scale.

### 2.3 Temperature Scaling

The temperature parameter `τ` controls the sharpness of the distribution:

- **Low τ (e.g., 0.05):** The model becomes very selective — it strongly penalizes even small deviations from the positive pair. This leads to tighter clusters but can be harder to train.
- **High τ (e.g., 1.0):** The distribution becomes more uniform, and the model is more tolerant of dissimilarity. Training is smoother but representations may be less discriminative.

In practice, `τ ∈ [0.07, 0.5]` works well. For financial data, we find `τ = 0.1` to be a good starting point because market states have subtle differences that require a reasonably sharp temperature.

### 2.4 Connection to Mutual Information

A remarkable property of InfoNCE is that minimizing it **maximizes a lower bound on mutual information** between the query and the positive key:

```
I(q; k+) ≥ log(N) - L_InfoNCE
```

As the number of negatives *N* grows, this bound becomes tighter. This means the encoder is learning representations that preserve the maximum amount of information about the relationship between paired market states.

## 3. Trading Application

### 3.1 Contrastive Pair Construction

The key design decision in applying InfoNCE to trading is **how to construct positive and negative pairs**. We define:

**Positive pairs:** Two market windows that share a similar regime. Concretely:
- **Temporal neighbors:** Windows that are close in time (e.g., within a few candles of each other). The assumption is that market conditions evolve slowly relative to our window size.
- **Return-similarity:** Windows with similar realized returns over a forward-looking horizon.
- **Volatility-matching:** Windows with similar realized volatility.

**Negative pairs:** Windows drawn from different time periods or regimes. In practice, we sample random windows from the dataset — with enough data, random samples are almost certainly from different regimes.

In our implementation, we use **temporal proximity** as the primary signal for positive pairs: if two windows are within a configurable distance of each other, they are considered positive. All other windows in the batch serve as negatives.

### 3.2 Feature Engineering for the Encoder

Each market window is represented as a feature vector derived from OHLCV data:

| Feature | Description |
|---------|-------------|
| Normalized returns | `(close - open) / open` for each candle |
| High-low range | `(high - low) / open` — a proxy for volatility |
| Volume ratio | Volume relative to a rolling average |
| Body ratio | `abs(close - open) / (high - low)` — candlestick body proportion |
| Upper shadow | `(high - max(open, close)) / (high - low)` |
| Lower shadow | `(min(open, close) - low) / (high - low)` |

These features are chosen because they are **scale-invariant** — they describe the shape and character of price action rather than absolute levels. This is critical for contrastive learning because we want the model to group market *behaviors*, not price *levels*.

### 3.3 The Encoder Architecture

We use a simple feedforward encoder:

```
Input (window_size × features) → Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32) → L2-normalize
```

The final L2 normalization ensures all representations lie on a unit hypersphere, which is standard practice when using cosine similarity. The 32-dimensional output space is compact enough to be interpretable yet expressive enough to capture market structure.

### 3.4 Downstream Usage

Once trained, the encoder can be used for:

1. **Regime detection:** Cluster the embeddings (e.g., K-means) to discover market regimes.
2. **Similarity search:** Given a current market state, find the most similar historical states and examine what happened next.
3. **Feature extraction:** Use the embeddings as input features for a downstream trading model (e.g., a return predictor or risk model).
4. **Anomaly detection:** Flag market states whose embeddings are far from any cluster center.

## 4. Implementation Walkthrough

### 4.1 Core Data Structures

We define the core types in Rust:

```rust
pub struct OhlcvCandle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

pub struct InfoNCEConfig {
    pub temperature: f64,
    pub embedding_dim: usize,
    pub window_size: usize,
    pub positive_range: usize,
    pub num_negatives: usize,
    pub learning_rate: f64,
}
```

### 4.2 InfoNCE Loss Computation

The loss computation follows the formula directly:

```rust
pub fn infonce_loss(
    query: &Array1<f64>,
    positive: &Array1<f64>,
    negatives: &[Array1<f64>],
    temperature: f64,
) -> f64 {
    let pos_sim = cosine_similarity(query, positive) / temperature;
    let neg_sims: Vec<f64> = negatives
        .iter()
        .map(|n| cosine_similarity(query, n) / temperature)
        .collect();

    let max_sim = pos_sim.max(neg_sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let pos_exp = (pos_sim - max_sim).exp();
    let neg_sum: f64 = neg_sims.iter().map(|&s| (s - max_sim).exp()).sum();

    -(pos_exp / (pos_exp + neg_sum)).ln()
}
```

Note the **log-sum-exp trick** (subtracting `max_sim` before exponentiation) for numerical stability.

### 4.3 Bybit Data Fetching

We fetch OHLCV data from the Bybit V5 API:

```rust
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<OhlcvCandle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let response: BybitResponse = reqwest::get(&url).await?.json().await?;
    // Parse and return candles...
}
```

### 4.4 Contrastive Pair Generation

Windows are extracted from the OHLCV series. For each anchor window, we select a nearby window as the positive and random distant windows as negatives:

```rust
pub fn generate_contrastive_pairs(
    candles: &[OhlcvCandle],
    config: &InfoNCEConfig,
) -> Vec<ContrastiveSample> {
    // For each valid anchor position:
    // 1. Extract feature window at position i
    // 2. Pick a positive from [i-range, i+range]
    // 3. Sample num_negatives from distant positions
    // ...
}
```

### 4.5 Training Loop

The training loop iterates over contrastive samples, computes the loss, and updates the encoder via gradient descent. Since we implement a simple feedforward network from scratch with `ndarray`, the gradient computation uses finite differences for simplicity (a production system would use automatic differentiation).

## 5. Bybit Crypto Data Integration

The Bybit V5 API provides free, unauthenticated access to historical kline (candlestick) data. The endpoint we use:

```
GET https://api.bybit.com/v5/market/kline
```

**Parameters:**
- `category`: `"linear"` for USDT perpetual futures
- `symbol`: e.g., `"BTCUSDT"`
- `interval`: `"1"` (1 min), `"5"` (5 min), `"15"`, `"60"`, `"240"`, `"D"`, `"W"`
- `limit`: Number of candles (max 200)

**Response structure:**
```json
{
  "retCode": 0,
  "result": {
    "list": [
      ["timestamp", "open", "high", "low", "close", "volume", "turnover"],
      ...
    ]
  }
}
```

Candles are returned in **reverse chronological order** (newest first), so we reverse them before processing. The data is real-time and requires no API key, making it ideal for educational and research purposes.

**Rate limits:** The public endpoint allows approximately 10 requests per second, which is more than sufficient for our use case.

### Integration Considerations

When integrating live Bybit data into the InfoNCE pipeline:

1. **Data normalization:** Raw OHLCV values vary by orders of magnitude across assets. Our feature engineering (Section 3.2) handles this by using ratios rather than raw values.
2. **Missing data:** If the API returns fewer candles than requested, we adjust our window calculations accordingly.
3. **Timestamp alignment:** Bybit timestamps are in milliseconds. We convert to seconds for consistency.
4. **Caching:** For training, it is advisable to cache fetched data to avoid hitting rate limits and to ensure reproducibility.

## 6. Key Takeaways

1. **InfoNCE provides a label-free learning objective** for financial markets. By teaching a model to distinguish similar market states from dissimilar ones, we learn rich representations without requiring explicit prediction targets.

2. **The temperature parameter is critical.** Too low and the model overfits to trivial distinctions; too high and it fails to learn meaningful structure. Start with `τ = 0.1` for financial data and tune from there.

3. **Contrastive pair design is the most important engineering decision.** Temporal proximity is a reasonable starting point, but incorporating return similarity, volatility matching, or regime labels (when available) can significantly improve representation quality.

4. **Rust provides performance and safety guarantees** that are valuable for production trading systems. The type system catches errors at compile time, and the absence of a garbage collector ensures predictable latency.

5. **The learned representations are versatile.** A single trained encoder can support regime detection, similarity search, feature extraction, and anomaly detection — making it a foundational component in a modern trading system.

6. **Scaling the number of negatives improves the mutual information bound.** Use as many negatives as your hardware allows. Batch sizes of 256-1024 are common in contrastive learning literature.

7. **Combine with other objectives.** InfoNCE works well as a pre-training objective. After pre-training, fine-tune the encoder with a small amount of labeled data for your specific trading task to get the best of both worlds.

---

## References

- van den Oord, A., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding*. arXiv:1807.03748.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020.
- He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning*. CVPR 2020.
