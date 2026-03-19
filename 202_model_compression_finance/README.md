# Chapter 202: Model Compression Finance

## 1. Introduction

Deploying machine learning models for trading in production environments presents a unique set of engineering challenges. While research teams typically train large, expressive neural networks on powerful GPU clusters, the models that actually execute trades must operate under strict latency, memory, and power constraints. High-frequency trading (HFT) systems demand sub-microsecond inference times. Edge deployments — such as co-located servers near exchange matching engines — impose severe memory budgets. Even cloud-based algorithmic trading platforms benefit from smaller models that reduce infrastructure costs and improve throughput.

Model compression is the family of techniques that bridge the gap between the large models we train and the lean models we deploy. The core insight is that trained neural networks are typically over-parameterized: they contain far more parameters than necessary to represent the learned function. Compression exploits this redundancy to produce smaller, faster models with minimal loss in predictive accuracy.

This chapter provides a comprehensive treatment of model compression techniques as applied to financial trading systems. We cover the theoretical foundations, survey the major compression families, and implement a complete compression pipeline in Rust — a language increasingly favored in quantitative finance for its combination of performance and safety.

## 2. Compression Taxonomy

Model compression techniques fall into five major categories, each attacking redundancy from a different angle:

### 2.1 Pruning

Pruning removes unnecessary weights or entire neurons from a trained network. The fundamental observation is that many weights in a trained network are close to zero and contribute negligibly to the output. By setting these weights to exactly zero (or removing the corresponding connections), we obtain a sparser model that requires less computation and memory.

**Unstructured pruning** removes individual weights regardless of their position in the weight matrix. This produces irregular sparsity patterns that are difficult to accelerate on standard hardware but can achieve very high compression ratios (90%+ sparsity is common).

**Structured pruning** removes entire filters, channels, or layers. The resulting model has a standard dense architecture (just smaller), making it straightforward to accelerate on any hardware. However, structured pruning typically achieves lower compression ratios than unstructured pruning for equivalent accuracy.

### 2.2 Quantization

Quantization reduces the numerical precision of model weights and activations. A standard neural network uses 32-bit floating point (FP32) for all computations. Quantization maps these values to lower-precision representations — typically 8-bit integers (INT8), though 4-bit and even binary (1-bit) quantization are active research areas.

**Post-training quantization (PTQ)** quantizes a pre-trained FP32 model without additional training. It is simple and fast but can suffer accuracy degradation, especially at very low bit widths.

**Quantization-aware training (QAT)** simulates quantization during training, allowing the model to adapt to the reduced precision. QAT typically recovers most or all of the accuracy lost by PTQ, at the cost of a full training run.

### 2.3 Knowledge Distillation

Knowledge distillation trains a small "student" model to mimic the behavior of a large "teacher" model. Rather than training on hard labels (e.g., "buy" or "sell"), the student learns from the teacher's soft probability distributions, which encode richer information about inter-class relationships. For trading applications, the teacher might be a large ensemble model that captures complex market dynamics, while the student is a compact model suitable for real-time execution.

### 2.4 Low-Rank Factorization

Weight matrices in neural networks often have low effective rank — their information content can be captured by matrices of much smaller dimensions. Low-rank factorization decomposes a weight matrix W of shape (m x n) into a product of two smaller matrices: W ≈ U * V, where U is (m x k) and V is (k x n), with k << min(m, n). This reduces both storage (from m*n to k*(m+n) parameters) and computation.

Common decomposition methods include:
- **SVD (Singular Value Decomposition)**: The classical approach, truncating small singular values.
- **Tucker decomposition**: Generalizes SVD to tensors, decomposing convolutional filters.
- **CP decomposition**: Represents a tensor as a sum of rank-one tensors.

### 2.5 Weight Sharing

Weight sharing forces groups of weights to take the same value. The classic approach uses k-means clustering to group weights into clusters, then replaces each weight with its cluster centroid. The model then stores only the cluster indices (which require far fewer bits than full-precision weights) plus a small codebook of centroid values.

## 3. Mathematical Foundation

### 3.1 Compression Ratio

The compression ratio quantifies how much smaller the compressed model is relative to the original:

```
CR = Size(original) / Size(compressed)
```

For pruning with sparsity s (fraction of zero weights), the theoretical compression ratio using sparse storage is approximately 1/(1-s) for large models.

For quantization from b_original bits to b_compressed bits:

```
CR = b_original / b_compressed
```

For example, FP32 to INT8 quantization yields CR = 32/8 = 4x.

### 3.2 Accuracy-Efficiency Tradeoff

Every compression technique trades model quality for efficiency. The relationship is typically convex: initial compression yields large efficiency gains with minimal accuracy loss, but pushing compression further produces diminishing returns and eventually catastrophic accuracy degradation.

We can formalize this as a multi-objective optimization problem:

```
minimize    L(θ_c)           (task loss of compressed model)
subject to  C(θ_c) ≤ budget  (resource constraint: latency, memory, or FLOPs)
```

where θ_c represents the compressed model parameters.

### 3.3 Pareto Frontier

The Pareto frontier is the set of compression configurations where no improvement in one objective (e.g., accuracy) is possible without degrading another (e.g., model size). In practice, we plot accuracy vs. compression ratio and select the operating point that best matches our deployment constraints.

For trading applications, the Pareto frontier is particularly important because the cost of errors is asymmetric: a model that misses a trading signal loses potential profit, but a model that generates a false signal incurs direct losses (transaction costs, adverse price movement). The optimal compression point must account for this asymmetry.

## 4. Trading Requirements

### 4.1 Latency Budgets

Different trading strategies impose vastly different latency requirements:

| Strategy | Typical Latency Budget | Compression Priority |
|----------|----------------------|---------------------|
| High-Frequency Trading | < 1 μs | Extreme: INT4/binary quantization, aggressive pruning |
| Market Making | 1-100 μs | High: INT8 quantization, structured pruning |
| Statistical Arbitrage | 100 μs - 10 ms | Moderate: standard compression sufficient |
| Swing Trading | > 1 s | Low: compression mainly for cost reduction |

### 4.2 Memory Constraints

Co-located trading servers near exchange matching engines operate under strict hardware constraints. FPGA-based systems may have only a few megabytes of on-chip memory. Even GPU-based systems benefit from models that fit entirely in L2 cache (typically 4-40 MB) to avoid memory bandwidth bottlenecks.

### 4.3 Throughput Requirements

A market-making system processing multiple instruments must evaluate its model thousands of times per second across hundreds of symbols. Model compression directly increases the number of instruments that can be processed within a given time window, expanding the strategy's universe and diversification potential.

## 5. Technique Deep Dives

### 5.1 Structured vs. Unstructured Pruning

**Unstructured pruning** applies a mask to individual weights:

```
W_pruned = W ⊙ M
```

where M is a binary mask with M_ij = 0 if |W_ij| < threshold. The threshold is typically set to achieve a target sparsity level.

**Magnitude-based pruning** is the simplest and most common criterion: remove the weights with the smallest absolute values. Despite its simplicity, magnitude pruning is a strong baseline that often matches more sophisticated methods.

**Structured pruning** removes entire rows or columns of weight matrices (corresponding to neurons or filters). For a fully connected layer with weight matrix W of shape (m x n), removing neuron j eliminates column j of W and row j of the next layer's weight matrix. This produces a genuinely smaller dense model.

### 5.2 Post-Training Quantization vs. Quantization-Aware Training

**Post-training quantization** maps FP32 values to INT8 using a linear mapping:

```
x_q = round(x / scale + zero_point)
x_dequant = (x_q - zero_point) * scale
```

where:
- `scale = (x_max - x_min) / (2^b - 1)`
- `zero_point = round(-x_min / scale)`

**Quantization-aware training** inserts "fake quantization" nodes during training that simulate the quantization error in the forward pass while allowing gradients to flow through in the backward pass (using the Straight-Through Estimator). This enables the model to learn weight values that are robust to quantization noise.

For trading models, QAT is generally preferred when feasible, as the quantization error in PTQ can shift predicted probabilities enough to change trading decisions at critical thresholds.

### 5.3 Tucker and CP Decomposition

For convolutional layers (common in models processing order book images or candlestick charts), tensor decompositions provide powerful compression.

**Tucker decomposition** decomposes a 4D convolutional kernel K of shape (C_out x C_in x H x W) into a core tensor G multiplied by factor matrices along each mode:

```
K ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄
```

**CP decomposition** represents K as a sum of R rank-one tensors:

```
K ≈ Σᵣ λᵣ · u₁ᵣ ⊗ u₂ᵣ ⊗ u₃ᵣ ⊗ u₄ᵣ
```

Both decompositions replace a single convolution with a sequence of smaller convolutions, reducing both parameters and FLOPs.

## 6. Implementation Walkthrough

Our Rust implementation provides a complete model compression toolkit for trading applications. The codebase is organized around several core components:

### Neural Network Representation

We represent a neural network as a sequence of dense layers, each with a weight matrix and bias vector stored as `ndarray` arrays. The forward pass applies ReLU activation between layers, with the final layer producing raw logits.

```rust
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
}

pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>,
}
```

### Magnitude Pruning

Our magnitude pruning implementation computes a global threshold from all weights at a target sparsity level, then zeros out every weight below that threshold:

```rust
pub fn magnitude_prune(network: &mut NeuralNetwork, sparsity: f64) {
    let all_weights: Vec<f64> = network.layers.iter()
        .flat_map(|l| l.weights.iter().map(|w| w.abs()))
        .collect();
    // Sort and find threshold at target sparsity percentile
    let threshold = sorted_weights[(sparsity * len) as usize];
    // Zero out weights below threshold
}
```

### INT8 Quantization

The quantization module maps FP32 weights to INT8 values using per-layer min/max calibration. It stores scale and zero-point parameters for dequantization at inference time. The quantized model uses 4x less memory than the original.

### SVD-Based Low-Rank Factorization

For each layer, we compute a truncated SVD of the weight matrix and reconstruct it using only the top-k singular values. The rank k is chosen based on a target compression ratio, balancing accuracy preservation against size reduction.

### Bybit Integration

The implementation includes an HTTP client for the Bybit exchange API, fetching real-time and historical OHLCV data for any trading pair. This data feeds directly into the model training and evaluation pipeline.

## 7. Bybit Data Integration

Our system fetches market data from Bybit's public API v5 endpoint:

```
GET https://api.bybit.com/v5/market/kline
```

Parameters include the trading pair (e.g., BTCUSDT), interval (1m, 5m, 1h, etc.), and limit (number of candles). The response includes timestamp, open, high, low, close, and volume for each candle.

The data pipeline:
1. **Fetch**: HTTP GET request to Bybit API
2. **Parse**: Deserialize JSON response into structured candle data
3. **Feature Engineering**: Compute returns, moving averages, volatility estimates
4. **Normalization**: Scale features to [0, 1] range for neural network input
5. **Train/Test Split**: Chronological split to avoid look-ahead bias

For the compression evaluation, we train the full model on historical data, apply each compression technique, and compare prediction accuracy on held-out test data. This gives us a realistic assessment of how compression affects trading signal quality.

## 8. Key Takeaways

1. **Model compression is essential for production trading systems.** The gap between research model sizes and deployment constraints is large and growing. Compression bridges this gap without requiring fundamental changes to the modeling approach.

2. **Different trading strategies need different compression approaches.** HFT demands extreme compression (quantization to INT4/binary, aggressive pruning). Swing trading may only need mild compression for cost reduction. Match the technique to the latency and memory budget.

3. **Combine multiple techniques for maximum compression.** Pruning, quantization, and low-rank factorization are complementary. A typical production pipeline applies structured pruning first, then quantization, achieving 10-50x total compression.

4. **Always evaluate on realistic trading metrics.** Accuracy alone is insufficient. Measure Sharpe ratio, maximum drawdown, and transaction costs of the compressed model's signals compared to the original. Small accuracy degradation can amplify into significant P&L differences.

5. **The Pareto frontier guides deployment decisions.** Plot accuracy vs. compression ratio for your specific model and data. The optimal operating point depends on your strategy's sensitivity to prediction errors and your infrastructure constraints.

6. **Rust is an excellent choice for compressed model inference.** Its zero-cost abstractions, lack of garbage collection, and deterministic performance make it ideal for latency-sensitive trading systems. The type system catches many errors at compile time that would become runtime bugs in Python.

7. **Quantization provides the best effort-to-compression ratio.** For most trading models, INT8 quantization delivers 4x compression with negligible accuracy loss and requires no retraining (post-training quantization). Start here before exploring more aggressive techniques.

8. **Monitor compressed models in production.** Model behavior can drift differently after compression. Implement ongoing monitoring of prediction distributions and trading performance to detect degradation early.
