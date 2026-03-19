# Chapter 216: One-Shot Neural Architecture Search (NAS)

## 1. Introduction

Neural Architecture Search (NAS) has transformed how we design deep learning models, but traditional NAS methods are prohibitively expensive. Training and evaluating thousands of individual architectures from scratch demands enormous computational resources -- resources that most trading firms cannot justify for model development. One-Shot NAS elegantly resolves this problem by training a single **supernet** that encapsulates all candidate architectures within a shared weight space.

The core insight is deceptively simple: instead of training each architecture independently, we train one overparameterized network whose subgraphs correspond to different architectures. After training this supernet, we evaluate candidate architectures by extracting their corresponding subnetworks and measuring performance using the shared weights. The best-performing architectures are then retrained from scratch to obtain final models.

In the trading domain, One-Shot NAS is particularly compelling. Markets shift between regimes -- trending, mean-reverting, volatile, calm -- and the optimal model architecture may differ across these regimes. One-Shot NAS enables rapid exploration of architectural choices (number of layers, hidden dimensions, activation functions, skip connections) without the cost of training each variant independently. A quant team can explore hundreds of architectures in the time it would normally take to train a handful.

This chapter covers the mathematical foundations of weight sharing and path sampling, walks through the complete One-Shot NAS pipeline, and demonstrates a Rust implementation that fetches live Bybit market data to search for optimal trading model architectures.

## 2. Mathematical Foundation

### 2.1 Weight Sharing in Supernets

A supernet is a directed acyclic graph (DAG) where each edge contains multiple candidate operations. Let the search space be defined over a DAG with N nodes. For each edge (i, j), there is a set of candidate operations:

```
O_{i,j} = {o_1, o_2, ..., o_K}
```

where each operation `o_k` has its own parameters `w_k^{(i,j)}`. A specific architecture `a` is defined by selecting exactly one operation per edge:

```
a = {a_{i,j} | a_{i,j} in {1, ..., K}, for all edges (i,j)}
```

The supernet's total weight set is the union of all operation weights:

```
W_super = Union over all (i,j), k of {w_k^{(i,j)}}
```

The key assumption of One-Shot NAS is that the shared weights provide a reasonable approximation of standalone-trained weights. Formally, if `w_a^*` denotes the optimal weights for architecture `a` trained independently, and `w_a^{shared}` denotes the weights extracted from the trained supernet, then:

```
rank(a | w_a^{shared}) ≈ rank(a | w_a^*)
```

This ranking correlation is what makes One-Shot NAS work. We do not need the shared weights to achieve the same absolute performance -- we only need them to preserve the relative ordering of architectures.

### 2.2 Supernet Training with Path Sampling

Training the supernet involves sampling random paths (subnets) at each training step. At each forward pass, we sample an architecture `a ~ P(A)` uniformly from the search space and only activate the selected operations:

```
L(W_super) = E_{a ~ P(A)} [L(x, y; W_a)]
```

where `W_a` is the subset of supernet weights corresponding to architecture `a`. The gradient update is:

```
W_super <- W_super - lr * gradient of L(x, y; W_a) with respect to W_a
```

Note that only the weights of the sampled path receive gradient updates at each step. Over many iterations, all operations accumulate gradients proportionally to their sampling frequency.

### 2.3 Path Dropout

To improve weight sharing quality, we apply **path dropout**: at each edge, we may activate more than one operation with some probability, or drop operations randomly. This acts as a regularizer and ensures operations are trained in diverse contexts. The dropout probability `p_drop` controls the trade-off between training efficiency and weight sharing quality.

### 2.4 Architecture Ranking

After supernet training converges, we evaluate architectures by:

1. Sampling or enumerating candidate architectures
2. For each architecture, extracting the corresponding weights from the supernet
3. Evaluating on a held-out validation set
4. Ranking architectures by validation performance

The top-K architectures are selected for standalone retraining, which yields the final models.

## 3. One-Shot NAS Pipeline

The complete pipeline consists of four stages:

### Stage 1: Define the Search Space

The search space defines what architectural choices are available. For trading models, typical choices include:

- **Layer types**: Dense, LSTM-like recurrent, convolutional (for pattern detection), skip connections
- **Hidden dimensions**: 16, 32, 64, 128, 256
- **Activation functions**: ReLU, GELU, Tanh, Sigmoid, LeakyReLU
- **Number of layers**: 1 through 6
- **Dropout rates**: 0.0, 0.1, 0.2, 0.3
- **Normalization**: None, LayerNorm, BatchNorm

The total search space is the Cartesian product of all choices. With 5 layer types, 5 hidden dims, 5 activations, and 6 depth options, we already have 750 candidate architectures -- far too many to train individually.

### Stage 2: Train the Supernet

The supernet is trained on historical market data. At each training step:

1. Sample a random architecture from the search space
2. Extract the corresponding subnetwork
3. Forward pass through the subnetwork
4. Compute loss (e.g., MSE for price prediction, cross-entropy for direction prediction)
5. Backpropagate through only the active operations
6. Update only the active weights

Training continues until convergence, typically requiring 2-5x the epochs of a single architecture.

### Stage 3: Evaluate and Rank Subnets

After training, we evaluate a large number of candidate architectures:

1. Sample N random architectures (e.g., N = 50-500)
2. For each, extract weights from the supernet
3. Evaluate on validation data
4. Sort by validation metric (Sharpe ratio, accuracy, profit factor)
5. Select top-K architectures (e.g., K = 3-5)

### Stage 4: Standalone Retraining

The top-K architectures are retrained from scratch with their own independent weights. This eliminates any weight coupling artifacts from the supernet and yields production-ready models. The standalone performance is the true measure of architecture quality.

## 4. Trading Applications

### 4.1 Rapid Architecture Exploration

In quantitative finance, the ability to quickly iterate on model designs is a competitive advantage. One-Shot NAS lets a team explore hundreds of architectures in a single training run. For example:

- Train a supernet on 1 year of BTCUSDT 1-minute data (500K+ candles)
- Evaluate 200 random subnets on 3 months of held-out data
- Identify top 5 architectures in hours instead of days

### 4.2 Regime-Adaptive Architecture Selection

Different market regimes may favor different architectures. A shallow, wide network might excel in trending markets where recent features dominate, while a deeper network with skip connections might better capture the complex dependencies in choppy, mean-reverting markets.

One-Shot NAS enables **regime-conditional architecture selection**:

1. Train the supernet on data spanning multiple regimes
2. Partition validation data by detected regime (trending, mean-reverting, volatile)
3. Rank architectures separately for each regime
4. Deploy regime-specific models with a regime detector as the router

### 4.3 Feature Engineering Search

Beyond model architecture, One-Shot NAS can search over feature engineering choices:

- Which technical indicators to include (RSI, MACD, Bollinger Bands, etc.)
- Lookback window lengths (5, 10, 20, 50, 100 bars)
- Feature transformations (raw, z-score, rank, log-return)

By encoding these choices as operations in the supernet, we simultaneously optimize the feature pipeline and the model architecture.

## 5. Supernet vs. Standalone Training

### 5.1 Weight Coupling Effects

The fundamental challenge of One-Shot NAS is **weight coupling**: operations share the supernet's training dynamics, and an operation's weight quality depends on which other operations were co-activated during training. This manifests as:

- **Co-adaptation**: Operations that are frequently sampled together may develop co-dependent weights that perform poorly when separated.
- **Under-training**: Operations in large search spaces may receive insufficient gradient updates, leading to unreliable evaluations.
- **Interference**: Competing operations on the same edge may push shared upstream weights in conflicting directions.

### 5.2 Ranking Reliability

Empirical studies show that One-Shot NAS rankings have a Kendall tau correlation of 0.5-0.8 with standalone rankings. This means the top-ranked architectures from the supernet are usually good, but not always the absolute best. Practical mitigations include:

- **Larger top-K**: Select more architectures for retraining to increase the probability of including the true best.
- **Multiple supernet runs**: Train several supernets with different random seeds and aggregate rankings.
- **Progressive shrinking**: Gradually reduce the search space during supernet training.

For trading applications, a ranking correlation of 0.6+ is typically sufficient -- we do not need to find the single best architecture, just a set of strong candidates.

## 6. Implementation Walkthrough

Our Rust implementation consists of several key components.

### 6.1 Search Space Definition

We define the search space with enums for each architectural choice:

```rust
enum Activation { ReLU, Tanh, Sigmoid, LeakyReLU, GELU }
enum LayerType { Dense, Conv1D, Residual, GatedDense }
```

An `ArchitectureConfig` specifies the selected operation for each choice point:

```rust
struct ArchitectureConfig {
    layer_type: LayerType,
    hidden_dim: usize,
    num_layers: usize,
    activation: Activation,
    dropout_rate: f64,
}
```

### 6.2 Supernet Structure

The supernet maintains weight matrices for all candidate operations. For each edge (layer position, operation type pair), we store separate weights:

```rust
struct Supernet {
    search_space: SearchSpace,
    // Weights for each (layer, operation) pair
    weights: HashMap<(usize, LayerType), Array2<f64>>,
    biases: HashMap<(usize, LayerType), Array1<f64>>,
}
```

### 6.3 Path Sampling and Training

At each training step, we sample a random architecture and perform a forward pass using only the selected operations' weights. The loss is computed and gradients are applied only to the active weights:

```rust
fn train_step(&mut self, x: &Array2<f64>, y: &Array1<f64>, lr: f64) {
    let arch = self.search_space.sample_random();
    let prediction = self.forward(x, &arch);
    let loss = mse_loss(&prediction, y);
    self.backward(&arch, &loss, lr);
}
```

### 6.4 Subnet Evaluation

After training, we evaluate subnets by extracting their weights and running inference on validation data:

```rust
fn evaluate_subnet(&self, arch: &ArchitectureConfig, val_x: &Array2<f64>, val_y: &Array1<f64>) -> f64 {
    let predictions = self.forward(val_x, arch);
    compute_sharpe_ratio(&predictions, val_y)
}
```

### 6.5 Architecture Selection and Retraining

The top-K architectures are selected and retrained from scratch with freshly initialized weights, ensuring no weight coupling artifacts remain.

## 7. Bybit Data Integration

The implementation fetches real market data from the Bybit public API. We use the `/v5/market/kline` endpoint to retrieve OHLCV candle data:

```rust
async fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Vec<Candle> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    // Parse response into Candle structs
}
```

Features are engineered from the raw candles:
- Log returns over multiple horizons (1, 5, 10 bars)
- Simple moving average ratios (close/SMA_20, close/SMA_50)
- Volatility (rolling standard deviation of returns)
- Volume ratio (volume/SMA_volume_20)
- RSI approximation

The target variable is the next-bar return direction (binary classification) or the next-bar return magnitude (regression).

## 8. Key Takeaways

1. **One-Shot NAS trains a single supernet** containing all candidate architectures, dramatically reducing the computational cost of architecture search from O(N * T) to O(T), where N is the number of candidates and T is the training time.

2. **Weight sharing provides approximate rankings**: The shared weights are not as good as independently trained weights, but they preserve the relative ordering of architectures well enough to identify strong candidates.

3. **The pipeline is four stages**: Define search space, train supernet, evaluate and rank subnets, retrain top-K from scratch.

4. **Trading-specific benefits** include rapid exploration of model designs, regime-adaptive architecture selection, and joint optimization of features and model structure.

5. **Weight coupling is the main limitation**: Co-adaptation, under-training, and interference can degrade ranking quality. Mitigations include larger top-K selection, multiple supernet runs, and progressive shrinking.

6. **Standalone retraining is essential**: Never deploy the supernet directly. Always retrain the selected architectures from scratch to eliminate weight coupling artifacts.

7. **The ranking need not be perfect**: In practice, identifying a set of strong architectures (top 10%) is sufficient. One-Shot NAS excels at quickly eliminating poor architectures rather than precisely identifying the single best one.

8. **Rust implementation** provides the performance needed to train supernets on large market datasets while maintaining type safety and memory efficiency.
