# Chapter 203: Pruning Trading Models

## 1. Introduction

Neural network pruning is one of the most powerful techniques for producing efficient, deployable trading models. In the world of algorithmic trading -- particularly high-frequency trading (HFT) -- every microsecond of inference latency translates directly into profit or loss. A model that is 10% faster at making predictions can capture opportunities that slower competitors miss entirely.

Modern deep learning models used in trading often contain millions of parameters. However, research consistently shows that a large fraction of these parameters contribute minimally to the model's predictive power. Pruning exploits this observation: by systematically removing redundant or low-importance weights, we can produce models that are dramatically smaller and faster while retaining nearly all of their original accuracy.

This chapter provides a comprehensive treatment of neural network pruning techniques as applied to trading models. We cover the mathematical foundations, explore multiple pruning strategies, discuss practical deployment scenarios (FPGA, edge devices, co-located servers), and provide a complete Rust implementation that demonstrates pruning on real market data from the Bybit exchange.

The key insight behind pruning is that overparameterized networks learn sparse solutions -- most of the "knowledge" is concentrated in a relatively small subset of weights. Our goal is to identify and preserve that subset while discarding the rest.

## 2. Mathematical Foundation

### 2.1 Weight Magnitude Pruning

The simplest and most widely used pruning criterion is weight magnitude. Given a neural network with weight tensors $W_l$ for each layer $l$, magnitude pruning removes weights whose absolute values fall below a threshold $\tau$:

$$M_{ij}^{(l)} = \begin{cases} 1 & \text{if } |W_{ij}^{(l)}| \geq \tau \\ 0 & \text{if } |W_{ij}^{(l)}| < \tau \end{cases}$$

where $M^{(l)}$ is the binary pruning mask for layer $l$. The pruned network computes:

$$\hat{W}^{(l)} = W^{(l)} \odot M^{(l)}$$

where $\odot$ denotes element-wise multiplication.

**Global pruning** sets a single threshold $\tau$ across all layers, determined by the desired sparsity level $s \in [0, 1]$. If we want to prune $s \times 100\%$ of all weights, $\tau$ is chosen as the $s$-th percentile of the absolute values of all weights concatenated.

**Layer-wise pruning** applies a separate threshold $\tau_l$ to each layer independently, ensuring each layer achieves exactly the target sparsity. This prevents the pathological case where global pruning removes almost all weights from small layers while barely touching large ones.

### 2.2 Gradient-Based Pruning

Gradient-based methods use the gradient of the loss function to estimate each weight's importance. The intuition is that a weight with a large gradient is actively being used during learning and is therefore important.

The **sensitivity** of weight $W_{ij}^{(l)}$ can be approximated as:

$$S_{ij}^{(l)} = \left| W_{ij}^{(l)} \cdot \frac{\partial \mathcal{L}}{\partial W_{ij}^{(l)}} \right|$$

This is related to the Taylor expansion of the loss change when removing a weight:

$$\Delta \mathcal{L} \approx \frac{\partial \mathcal{L}}{\partial W_{ij}} \cdot W_{ij} + \frac{1}{2} \frac{\partial^2 \mathcal{L}}{\partial (W_{ij})^2} \cdot (W_{ij})^2$$

The first-order term gives the sensitivity criterion above. Including the second-order (Hessian) term provides more accurate importance estimates but is computationally expensive.

### 2.3 Structured vs. Unstructured Pruning

**Unstructured pruning** removes individual weights regardless of their position in the weight matrix. This produces sparse matrices with irregular sparsity patterns. While it achieves the highest compression ratios at a given accuracy level, the irregular memory access patterns make it difficult to achieve actual inference speedup on standard hardware without specialized sparse matrix libraries.

**Structured pruning** removes entire structural units:
- **Neuron pruning**: Removes entire rows/columns from weight matrices, equivalent to removing neurons from a layer.
- **Filter pruning**: In convolutional networks, removes entire filters (output channels).
- **Layer pruning**: Removes entire layers from the network.
- **Block pruning**: Removes contiguous blocks of weights, creating semi-structured sparsity.

The advantage of structured pruning is that the resulting model is a standard dense model (just smaller), so no special sparse computation libraries are needed, and the speedup is realized on any hardware.

### 2.4 Sparsity Metrics

We quantify the degree of pruning using the **sparsity** metric:

$$\text{sparsity} = 1 - \frac{\text{number of non-zero weights}}{\text{total number of weights}}$$

A sparsity of 0.9 means 90% of weights have been removed. We also track:

- **Compression ratio**: $\frac{\text{original size}}{\text{pruned size}}$
- **FLOPs reduction**: The reduction in floating-point operations during inference
- **Accuracy retention**: $\frac{\text{pruned model accuracy}}{\text{original model accuracy}} \times 100\%$

## 3. Pruning Strategies

### 3.1 One-Shot Pruning

One-shot pruning is the simplest approach: train a model to convergence, prune it once to the desired sparsity, and optionally fine-tune the remaining weights. The procedure is:

1. Train dense model until convergence
2. Compute importance scores for all weights
3. Remove the bottom $s\%$ of weights by importance
4. Fine-tune the pruned model for a few epochs

One-shot pruning works well at moderate sparsity levels (up to ~50-70%) but quality degrades rapidly at higher sparsity. It is fast and simple, making it suitable when you need a quick model compression and the target sparsity is modest.

### 3.2 Iterative Magnitude Pruning (IMP)

Iterative Magnitude Pruning achieves much higher sparsity levels by pruning gradually over multiple rounds:

1. Train the dense model to convergence
2. Prune a small fraction $p$ of the remaining weights (e.g., 20%)
3. Retrain the pruned model for $n$ epochs
4. Repeat steps 2-3 until desired sparsity is reached

At each iteration $t$, the effective sparsity is:

$$s_t = 1 - (1 - p)^t$$

For example, with $p = 0.2$ (pruning 20% of remaining weights each round), after 10 rounds the sparsity is $1 - 0.8^{10} \approx 0.893$ or about 89%.

IMP consistently outperforms one-shot pruning at high sparsity levels because the model has the opportunity to adapt its remaining weights after each pruning step. The retraining phase allows the network to redistribute information from pruned weights to surviving ones.

### 3.3 The Lottery Ticket Hypothesis

The Lottery Ticket Hypothesis (Frankle and Carlin, 2019) states that dense networks contain sparse subnetworks ("winning tickets") that, when trained in isolation from the same initialization, can match the full network's performance.

The procedure for finding lottery tickets is:

1. Initialize the network with weights $W_0$
2. Train to convergence, obtaining weights $W_T$
3. Prune the bottom $s\%$ of weights by magnitude, creating mask $M$
4. Reset the remaining weights to their initial values: $W_0 \odot M$
5. Retrain this sparse network from the original initialization

This has profound implications for trading: if we can identify the winning ticket early, we can train and deploy a small, fast model from the start, saving both training compute and inference latency.

In practice, finding exact lottery tickets is expensive. A practical compromise is **late rewinding**: instead of resetting to $W_0$, reset to weights from early in training $W_k$ where $k$ is a small number of epochs. This relaxation makes the technique more robust and is often sufficient.

### 3.4 Pruning Schedule

The pruning schedule determines how sparsity increases over time. Common schedules include:

- **Linear**: $s_t = s_{\text{target}} \cdot \frac{t}{T}$
- **Cubic**: $s_t = s_{\text{target}} \left(1 - \left(1 - \frac{t}{T}\right)^3\right)$
- **Exponential**: $s_t = s_{\text{target}} \cdot (1 - e^{-\alpha t})$

The cubic schedule is generally preferred because it prunes aggressively early (when there is the most redundancy) and slows down later (when each remaining weight is more important).

## 4. Trading Applications

### 4.1 Latency Reduction in HFT

In high-frequency trading, the inference pipeline must complete within microseconds. A typical HFT model pipeline involves:

1. Market data ingestion (~1-5 us)
2. Feature computation (~2-10 us)
3. Model inference (~5-50 us)
4. Order generation (~1-5 us)

Model inference is often the bottleneck. A pruned model with 90% sparsity can reduce inference time by 3-5x when using structured pruning (which translates to smaller dense matrices) or by 2-3x with unstructured pruning using optimized sparse kernels.

For a trading model with 1M parameters, pruning to 10% density yields a 100K parameter model that:
- Fits entirely in L1 cache on modern CPUs
- Requires 10x fewer multiply-accumulate operations
- Produces predictions with nearly identical accuracy

### 4.2 FPGA and Edge Deployment

Trading firms increasingly deploy models on FPGAs for ultra-low-latency inference. FPGAs have limited on-chip memory (typically 1-10 MB of block RAM), making model compression essential.

Structured pruning is particularly valuable for FPGA deployment because:
- Smaller models fit in on-chip BRAM, avoiding slow off-chip memory access
- Regular computation patterns map efficiently to FPGA fabric
- Reduced parameter count means fewer DSP blocks needed
- Power consumption scales with model size

Edge deployment for satellite-based or remote trading systems also benefits from pruning. Models running on ARM processors or specialized AI accelerators must operate within strict power and memory budgets.

### 4.3 Reducing Overfitting

Trading models are notoriously prone to overfitting due to the low signal-to-noise ratio in financial data. Pruning acts as an implicit regularizer:

- Removing parameters reduces model capacity, constraining the hypothesis space
- The surviving weights must encode genuinely predictive features
- Pruned models generalize better to unseen market regimes
- Iterative pruning with retraining forces the model to find robust representations

Empirically, pruned trading models often outperform their dense counterparts on out-of-sample data, even when the dense model has higher in-sample accuracy. This is one of the most compelling arguments for pruning in trading applications.

## 5. Structured Pruning

### 5.1 Filter and Channel Pruning

For convolutional architectures used in processing market data (e.g., treating price series as 1D signals), filter pruning removes entire convolutional filters. The importance of filter $f$ in layer $l$ is typically measured as:

$$\text{Importance}(f) = \|W_f^{(l)}\|_1 = \sum_{i,j,k} |W_{f,i,j,k}^{(l)}|$$

Filters with the smallest L1-norm are removed. This directly reduces the number of output channels, shrinking subsequent layers as well (since they receive fewer input channels).

### 5.2 Neuron Pruning in Fully Connected Layers

For the fully connected layers common in trading models, neuron pruning removes entire neurons by zeroing out all incoming and outgoing weights. A neuron's importance can be measured by:

- **Activation magnitude**: Neurons that consistently produce near-zero activations can be removed
- **Weight norm**: Neurons with small incoming weight norms have minimal impact
- **Redundancy**: Neurons whose activations are highly correlated with other neurons are redundant

Removing a neuron from a layer with $n$ neurons reduces that layer's computation by $\frac{1}{n}$ and removes one row from the weight matrix and one element from the bias vector.

### 5.3 Layer Pruning

For very deep models, entire layers can be removed if they contribute minimally to the output. This is measured by comparing the layer's input and output representations:

$$\text{Importance}(l) = \|h_l - h_{l-1}\|_2$$

If a layer's output is very similar to its input (i.e., it computes approximately the identity function), it can be skipped. This is most applicable in residual architectures where skip connections naturally handle layer removal.

## 6. Implementation Walkthrough

Our Rust implementation provides a complete neural network pruning framework. The key components are:

### 6.1 Network Architecture

We implement a feedforward neural network with pruning mask support. Each layer stores both its weights and a binary mask:

```rust
pub struct PrunableLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub mask: Array2<f64>,       // 1.0 = active, 0.0 = pruned
}
```

Forward pass applies the mask: `effective_weights = weights * mask`.

### 6.2 Magnitude Pruning

The `magnitude_prune` function implements global magnitude pruning:

1. Collect absolute values of all unmasked weights
2. Sort them to find the threshold at the target sparsity percentile
3. Set mask entries to 0.0 where |weight| < threshold

### 6.3 Iterative Magnitude Pruning

The `iterative_prune` function implements IMP:

1. Start with the dense trained model
2. At each step, prune a fraction of remaining weights
3. Retrain for a specified number of epochs
4. Repeat until target sparsity is reached

### 6.4 Structured Pruning

The `structured_prune` function removes entire neurons:

1. Compute L2-norm of each neuron's incoming weights
2. Rank neurons by importance
3. Remove lowest-importance neurons by eliminating rows/columns

### 6.5 Bybit Integration

The implementation fetches real OHLCV data from the Bybit API to train and evaluate models on actual market data. We use the public kline endpoint which requires no authentication.

The complete implementation is provided in `rust/src/lib.rs` with a trading example in `rust/examples/trading_example.rs`.

## 7. Bybit Data Integration

Our implementation connects to the Bybit exchange API to fetch real market data for model training and evaluation. The integration uses the public REST API endpoint for historical klines (candlestick data).

### 7.1 Data Fetching

We fetch OHLCV (Open, High, Low, Close, Volume) data for BTCUSDT using:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=5&limit=200
```

This returns 5-minute candles, providing sufficient granularity for short-term price prediction models.

### 7.2 Feature Engineering

From the raw OHLCV data, we compute features including:
- Price returns: $(close_t - close_{t-1}) / close_{t-1}$
- Normalized volume: $volume_t / \text{mean}(volume)$
- High-low range: $(high_t - low_t) / close_t$
- Open-close difference: $(close_t - open_t) / open_t$

### 7.3 Label Construction

We construct binary labels for price direction prediction:
$$y_t = \begin{cases} 1 & \text{if } close_{t+1} > close_t \\ 0 & \text{otherwise} \end{cases}$$

This provides a straightforward classification target that allows us to clearly measure the impact of pruning on prediction accuracy.

## 8. Key Takeaways

1. **Pruning is essential for production trading models.** The gap between research models and deployed models is often bridged by compression techniques, with pruning being the most effective.

2. **Iterative magnitude pruning outperforms one-shot pruning.** At high sparsity levels (>80%), IMP retains significantly more accuracy by allowing the model to adapt after each pruning step.

3. **Structured pruning provides real speedup.** While unstructured pruning achieves higher compression ratios on paper, structured pruning produces smaller dense models that run faster on all hardware without special library support.

4. **Pruning acts as regularization.** For trading models where overfitting is a primary concern, pruning can actually improve out-of-sample performance. A pruned model that generalizes better is more valuable than a dense model that memorizes noise.

5. **The accuracy-sparsity tradeoff is nonlinear.** Models typically retain >95% accuracy up to 70-80% sparsity, then degrade gradually to 90%, and finally collapse rapidly beyond 95%. The sweet spot for trading models is usually 80-90% sparsity.

6. **Hardware constraints drive pruning strategy.** FPGA deployment favors structured pruning and specific sparsity patterns. GPU deployment can exploit unstructured sparsity with sparse tensor cores. CPU deployment benefits from both approaches.

7. **Rust provides an ideal implementation language.** The combination of zero-cost abstractions, no garbage collection pauses, and predictable performance makes Rust excellent for implementing pruning algorithms that will be used in latency-sensitive trading systems.

8. **Always validate on out-of-sample data.** Pruning decisions based on in-sample performance can be misleading. The true test of a pruned model is its performance on unseen market data, preferably from a different market regime than the training data.
