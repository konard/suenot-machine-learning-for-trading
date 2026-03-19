# Chapter 219: DARTS Trading — Differentiable Architecture Search for Financial Time Series

## 1. Introduction

Differentiable Architecture Search (DARTS) represents a paradigm shift in neural architecture design. Rather than evaluating thousands of discrete architectures through reinforcement learning or evolutionary algorithms, DARTS relaxes the search space to be continuous, enabling gradient-based optimization of the architecture itself. This chapter adapts DARTS specifically for trading time series, where the goal is to automatically discover optimal neural network architectures for tasks such as price prediction, volatility forecasting, and regime detection.

Traditional approaches to building trading models involve manually selecting from a menu of architectures: LSTMs, temporal convolutional networks (TCNs), transformer-based models, or various hybrids. Each choice involves implicit assumptions about the temporal structure of financial data. DARTS eliminates this guesswork by searching over a combinatorial space of operations and connectivity patterns, allowing the data itself to dictate the best architecture.

The key insight behind DARTS for trading is that financial time series exhibit heterogeneous temporal dependencies. Short-term microstructure effects may require local convolutional filters, medium-term momentum patterns benefit from dilated convolutions, and long-term regime dependencies call for recurrent cells or attention mechanisms. By including all of these as candidate operations in the DARTS search space, the algorithm can compose hybrid architectures that capture multiple temporal scales simultaneously.

In this chapter we develop a full DARTS implementation in Rust, tailored for trading applications. We define a custom operation space containing 1D convolutions, dilated convolutions, recurrent cells, attention layers, moving average filters, and skip connections. We implement continuous relaxation via softmax-weighted mixtures, Gumbel-Softmax for differentiable discrete selection, and bi-level optimization that alternates between architecture parameter updates and network weight updates. Finally, we integrate live market data from the Bybit exchange to demonstrate the full pipeline from data ingestion through architecture search to trading evaluation.

## 2. Mathematical Foundation

### 2.1 Continuous Relaxation of the Search Space

DARTS formulates architecture search as a continuous optimization problem. Consider a directed acyclic graph (DAG) where each edge $(i, j)$ represents a candidate operation from a set $\mathcal{O} = \{o_1, o_2, \ldots, o_K\}$. In a discrete search, each edge selects exactly one operation. DARTS relaxes this by computing a weighted mixture:

$$\bar{o}^{(i,j)}(x) = \sum_{k=1}^{K} \frac{\exp(\alpha_k^{(i,j)})}{\sum_{l=1}^{K} \exp(\alpha_l^{(i,j)})} \cdot o_k(x)$$

where $\alpha_k^{(i,j)}$ are continuous architecture parameters. The softmax ensures the weights form a valid probability distribution. As training progresses, these weights concentrate on the best-performing operations.

### 2.2 Bi-Level Optimization

DARTS frames the search as a bi-level optimization:

$$\min_{\alpha} \quad \mathcal{L}_{val}(w^*(\alpha), \alpha)$$
$$\text{s.t.} \quad w^*(\alpha) = \arg\min_{w} \mathcal{L}_{train}(w, \alpha)$$

The outer objective minimizes validation loss with respect to architecture parameters $\alpha$, while the inner objective trains network weights $w$ on the training set. In practice, we approximate $w^*(\alpha)$ with a single gradient step:

$$w' = w - \xi \nabla_w \mathcal{L}_{train}(w, \alpha)$$

Then the architecture gradient is computed via:

$$\nabla_\alpha \mathcal{L}_{val}(w', \alpha)$$

This alternating scheme — one step on weights, one step on architecture — is the first-order approximation of DARTS. A second-order approximation accounts for how changes in $\alpha$ affect optimal $w^*$, but the first-order version is often sufficient and computationally cheaper.

### 2.3 Gumbel-Softmax for Discrete Selection

While the continuous relaxation enables gradient-based search, we ultimately need a discrete architecture for deployment. The Gumbel-Softmax trick provides a differentiable approximation to categorical sampling:

$$y_k = \frac{\exp((\log \alpha_k + g_k) / \tau)}{\sum_{l=1}^{K} \exp((\log \alpha_l + g_l) / \tau)}$$

where $g_k \sim \text{Gumbel}(0,1)$ are i.i.d. Gumbel noise samples and $\tau > 0$ is a temperature parameter. As $\tau \to 0$, the distribution approaches a one-hot categorical. During training we anneal $\tau$ from a high value (e.g., 1.0) down to a small value (e.g., 0.1), gradually sharpening the architecture decisions.

The Gumbel noise is sampled as $g_k = -\log(-\log(u_k))$ where $u_k \sim \text{Uniform}(0,1)$.

### 2.4 Architecture Discretization

After the search phase completes, we discretize the architecture by selecting the operation with the highest architecture weight on each edge:

$$o^{(i,j)} = o_{k^*} \quad \text{where} \quad k^* = \arg\max_k \alpha_k^{(i,j)}$$

For each intermediate node in the DAG, we retain the top-2 incoming edges (by strength of the selected operation's weight), discarding the rest. This yields a compact, deployable architecture.

## 3. DARTS for Time Series

### 3.1 Custom Operation Space

Standard DARTS for image classification uses 2D convolutions, pooling, and dilated convolutions. For trading time series, we design a domain-specific operation space:

| Operation | Description | Temporal Characteristic |
|-----------|-------------|------------------------|
| Conv1D-3 | 1D convolution, kernel size 3 | Local patterns (2-3 bars) |
| Conv1D-5 | 1D convolution, kernel size 5 | Short-term patterns (4-5 bars) |
| DilConv-3-2 | Dilated conv, kernel 3, dilation 2 | Medium-range (5-6 bars effective) |
| DilConv-3-4 | Dilated conv, kernel 3, dilation 4 | Longer-range (9-10 bars effective) |
| MovingAvg-5 | Moving average, window 5 | Trend smoothing |
| MovingAvg-10 | Moving average, window 10 | Longer trend |
| RecurrentCell | GRU-style recurrent unit | Sequential dependencies |
| Attention | Self-attention over sequence | Global dependencies |
| Skip | Identity connection | Residual pathway |
| Zero | No connection | Pruning |

### 3.2 Cell Structure

A DARTS cell is a DAG with $N$ intermediate nodes. Each node $j$ receives inputs from all previous nodes $\{0, 1, \ldots, j-1\}$ via mixed operations. The cell has two input nodes (outputs from the previous two cells) and one output node (concatenation of all intermediate nodes). For time series, we use $N=4$ intermediate nodes, giving a rich search space while remaining tractable.

The cell is replicated $L$ times in a stack, with each cell operating on the output of the previous one. For trading applications, $L=4$ to $L=8$ cells typically suffice.

### 3.3 Time Series Preprocessing

Financial time series require careful preprocessing before feeding into DARTS. We compute:

- Log returns: $r_t = \log(p_t / p_{t-1})$
- Rolling volatility: $\sigma_t = \text{std}(r_{t-W+1:t})$
- Normalized volume: $v_t / \text{MA}(v, W)$
- Price momentum: $(p_t - p_{t-k}) / p_{t-k}$ for multiple lookbacks $k$

All features are standardized using rolling z-scores to maintain stationarity and prevent look-ahead bias.

## 4. Trading Applications

### 4.1 Price Direction Prediction

The most direct application is predicting the direction of future price movement. DARTS searches for an architecture that maps a window of features to a probability of upward movement. The search loss is binary cross-entropy on a validation set, while the weight training uses the training set.

The discovered architectures often reveal interesting patterns: short-term convolutions on price features combined with attention on volume features, connected through skip connections to preserve raw signal when the learned features are noisy.

### 4.2 Volatility Forecasting

For volatility forecasting, the target is the realized volatility over a future horizon. DARTS optimizes for mean squared error between predicted and realized volatility. Architectures discovered for volatility tend to favor dilated convolutions (capturing clustering effects, consistent with GARCH-like behavior) and moving average operations (capturing the level of recent volatility).

### 4.3 Regime-Adaptive Architecture

A unique application is running DARTS periodically on recent data to adapt the architecture to the current market regime. In trending markets, the search may favor momentum-capturing operations (longer convolutions, recurrent cells). In mean-reverting markets, shorter convolutions and attention mechanisms may dominate. By re-running the architecture search monthly or quarterly, the trading system adapts its structure, not just its parameters, to changing market conditions.

### 4.4 Multi-Asset Architecture Transfer

Once an architecture is discovered on one asset (e.g., BTCUSDT), it can be fine-tuned on other assets. The architecture captures general temporal patterns, while the weights specialize to the particular asset. This transfer is often more effective than searching from scratch on each asset, especially for assets with limited historical data.

## 5. Practical Issues

### 5.1 Performance Collapse

A well-known failure mode of DARTS is performance collapse, where the search converges prematurely to an architecture dominated by skip connections or parameter-free operations. This happens because these operations have lower training loss in the early phases (they don't overfit), but they lack the capacity to learn complex patterns.

**Mitigation strategies:**

- **Early stopping of architecture search:** Monitor the entropy of architecture weights. If entropy drops too quickly, stop the search early before collapse occurs.
- **Operation regularization:** Add an L2 penalty on the architecture weights of parameter-free operations (skip, zero, moving average) to prevent them from dominating.
- **Progressive shrinking:** Start with a larger set of operations and gradually prune the weakest ones, rather than making a single discrete decision at the end.

### 5.2 Search Instability

DARTS can exhibit unstable search dynamics where architecture weights oscillate rather than converging. For trading data, which is inherently noisy, this is particularly problematic.

**Mitigation strategies:**

- **Architecture weight clipping:** Constrain architecture parameters to a bounded range.
- **Slower architecture learning rate:** Use a learning rate for $\alpha$ that is 5-10x smaller than for $w$.
- **Longer warm-up:** Train weights for several epochs before beginning architecture updates, so the operations have reasonable parameters before being compared.

### 5.3 Overfitting the Validation Set

Because DARTS optimizes architecture parameters on the validation set, there is a risk of overfitting the architecture to validation-set quirks. For trading, this means the discovered architecture may exploit specific historical patterns that don't persist.

**Mitigation strategies:**

- **Rolling validation:** Use a rolling window for the validation set, updating it as the search progresses.
- **Architecture ensemble:** Run the search multiple times with different random seeds and data splits, then use the most common architectural choices.
- **Out-of-sample evaluation:** After discretizing the architecture, retrain from scratch and evaluate on a held-out test set that was never used during search.

### 5.4 Computational Cost

While DARTS is much cheaper than RL-based NAS (hours vs. days), the mixed operation computation is still expensive because all operations run in parallel during search. For $K$ operations per edge and $E$ edges, each forward pass computes $K \times E$ operation outputs.

**Mitigation strategies:**

- **Partial channel connections:** Only route a fraction of channels through each operation.
- **Progressive search:** Start with a smaller cell and grow it during search.
- **Proxy tasks:** Search on a simpler version of the task (fewer features, shorter sequences) and transfer the architecture.

## 6. Implementation Walkthrough

Our Rust implementation is organized into the following components:

### 6.1 Operations Module

Each operation implements a common trait that defines the forward pass on a time-series tensor. Operations include:

- `Conv1D`: Standard 1D convolution with configurable kernel size and padding. Captures local temporal patterns.
- `DilatedConv1D`: Dilated convolution that expands the receptive field without increasing parameters. Essential for capturing multi-scale temporal dependencies.
- `MovingAverage`: A parameter-free smoothing operation. Acts as a low-pass filter on the time series.
- `SkipConnection`: Identity mapping that enables residual learning.
- `ZeroOp`: Outputs zeros, effectively pruning the edge.

### 6.2 DARTS Cell

The cell maintains architecture parameters $\alpha$ for every (node pair, operation) combination. During the forward pass, it computes the softmax-weighted mixture of all operations on each edge, then aggregates inputs to each intermediate node by summation. The output is the concatenation (or mean) of all intermediate node outputs.

### 6.3 Search Loop

The search alternates between:

1. **Weight step:** Forward pass on training batch, compute loss, backpropagate to update operation weights $w$.
2. **Architecture step:** Forward pass on validation batch, compute loss, backpropagate to update architecture parameters $\alpha$.

Both steps use separate optimizers (SGD for weights, Adam for architecture parameters). After the search completes, the architecture is discretized.

### 6.4 Gumbel-Softmax Implementation

The Gumbel-Softmax function samples Gumbel noise, adds it to log-architecture-weights, divides by temperature, and applies softmax. We anneal the temperature linearly from 1.0 to 0.1 over the search.

```rust
fn gumbel_softmax(logits: &[f64], temperature: f64, rng: &mut impl Rng) -> Vec<f64> {
    let gumbel_noise: Vec<f64> = logits.iter()
        .map(|_| {
            let u: f64 = rng.gen_range(1e-10..1.0);
            -(-u.ln()).ln()
        })
        .collect();
    let scaled: Vec<f64> = logits.iter().zip(&gumbel_noise)
        .map(|(l, g)| (l + g) / temperature)
        .collect();
    softmax(&scaled)
}
```

### 6.5 Architecture Extraction

After search, we select the top operation per edge and retain the top-2 incoming edges per node:

```rust
fn discretize_architecture(alpha: &AlphaParams) -> DiscreteArchitecture {
    // For each edge, pick argmax operation
    // For each node, keep top-2 edges by weight strength
}
```

## 7. Bybit Data Integration

We fetch OHLCV (Open, High, Low, Close, Volume) kline data from Bybit's public REST API:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=1000
```

The response provides candlestick data which we parse into our feature matrix. We compute:

1. Log returns from close prices
2. High-low range as a volatility proxy
3. Volume ratios
4. Multiple momentum features

Data is split into train/validation/test sets chronologically (60%/20%/20%) to prevent look-ahead bias. The validation set is used for architecture parameter updates, the training set for weight updates, and the test set for final evaluation only.

### Integration Notes

- Bybit's API requires no authentication for public market data
- Rate limits are generous for kline data (10 requests/second)
- We fetch 1000 hourly candles, providing roughly 42 days of data
- For production use, historical data should be accumulated over months

## 8. Key Takeaways

1. **DARTS enables automated architecture discovery for trading.** Instead of manually choosing between LSTMs, CNNs, and transformers, DARTS searches over a combinatorial space of operations and finds data-driven hybrid architectures.

2. **The continuous relaxation is the core innovation.** By replacing discrete operation selection with softmax-weighted mixtures, DARTS converts a combinatorial search into a smooth optimization problem solvable with gradient descent.

3. **Custom operation spaces matter.** The default DARTS operations (designed for image classification) are inappropriate for time series. Domain-specific operations — dilated convolutions, moving averages, recurrent cells — are essential for financial applications.

4. **Bi-level optimization requires care.** The alternating update between architecture and weight parameters can be unstable. Using different learning rates, warm-up phases, and regularization prevents degenerate solutions.

5. **Performance collapse is the primary risk.** Skip connections and parameter-free operations can dominate the search if not properly regularized. Monitoring architecture weight entropy and applying operation-level penalties are effective countermeasures.

6. **Gumbel-Softmax bridges continuous search and discrete deployment.** Temperature annealing gradually sharpens soft decisions into hard architectural choices, enabling smooth transitions from search to deployment.

7. **Regime-adaptive architecture search is a unique trading advantage.** Re-running DARTS periodically on recent data allows the model to adapt its structure to changing market conditions, going beyond simple parameter updates.

8. **Rust provides the performance needed for practical NAS.** Architecture search involves massive parallelism over operations. Rust's zero-cost abstractions and memory safety make it well-suited for implementing the compute-intensive inner loops of DARTS.

9. **Out-of-sample validation is non-negotiable.** The discovered architecture must be retrained from scratch and evaluated on held-out data to confirm that the search did not overfit to the validation set.

10. **Transfer learning amplifies DARTS value.** An architecture discovered on one asset can be fine-tuned across many assets, amortizing the search cost and enabling better models on data-scarce instruments.
