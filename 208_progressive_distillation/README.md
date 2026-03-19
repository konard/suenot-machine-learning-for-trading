# Chapter 208: Progressive Distillation

## 1. Introduction

Knowledge distillation compresses a large, high-capacity "teacher" model into a smaller "student" model that can run faster, consume less memory, and deploy to constrained environments. The classical approach performs this transfer in a single step — one teacher, one student, one training run. **Progressive distillation** takes a fundamentally different path: it compresses the teacher through a sequence of intermediate stages, each only slightly smaller than the one before, so that the final tiny model retains significantly more of the teacher's predictive power.

The intuition is straightforward. Asking a 100-layer network to directly teach a 4-layer network is analogous to asking a university professor to explain quantum field theory to a five-year-old in one conversation. A better strategy is a chain of translations: professor teaches a graduate student, the graduate student teaches an undergraduate, the undergraduate teaches a high-school student, and the high-school student finally explains the core idea to the child. At every link in the chain the knowledge gap is small, so less information is lost.

In the context of algorithmic trading, progressive distillation unlocks a powerful deployment pipeline. A research team can train an enormous model with hundreds of features and millions of parameters, then progressively distill it through staging, production, and edge tiers — each tier optimized for the latency and resource constraints of its target environment. The result is a family of models with a shared lineage, where every model is the best possible approximation of the original given its size budget.

This chapter covers the mathematical foundations, progressive strategies, trading-specific applications, and a full Rust implementation that fetches live Bybit market data and demonstrates multi-stage distillation with accuracy and model-size tracking.

## 2. Mathematical Foundation

### 2.1 Multi-Stage Distillation Pipeline

Let a progressive distillation pipeline consist of $N$ stages. We begin with a teacher model $M_0$ (the original large model) and produce a sequence of progressively smaller models $M_1, M_2, \ldots, M_N$. At stage $k$ the model $M_{k-1}$ serves as teacher and $M_k$ serves as student.

The distillation objective at stage $k$ is:

$$\mathcal{L}_k = \alpha_k \cdot \mathcal{L}_{\text{soft}}(M_{k-1}, M_k; T_k) + (1 - \alpha_k) \cdot \mathcal{L}_{\text{hard}}(M_k, y)$$

where:
- $\mathcal{L}_{\text{soft}}$ is the soft-target loss comparing the softened outputs of teacher $M_{k-1}$ and student $M_k$ at temperature $T_k$
- $\mathcal{L}_{\text{hard}}$ is the standard supervised loss against ground-truth labels $y$
- $\alpha_k \in [0, 1]$ controls the balance, typically decreasing as stages progress

For regression tasks common in trading (predicting returns, volatility, spreads), the soft loss simplifies to mean squared error between teacher and student outputs:

$$\mathcal{L}_{\text{soft}}^{(k)} = \frac{1}{n} \sum_{i=1}^{n} \left( M_{k-1}(x_i) - M_k(x_i) \right)^2$$

### 2.2 Progressive Capacity Reduction

Define the capacity of model $M_k$ as $C_k$ (for example, total parameter count). A progressive schedule satisfies:

$$C_0 > C_1 > C_2 > \cdots > C_N$$

The **compression ratio** at each stage is $r_k = C_{k-1} / C_k$. A key design principle is keeping $r_k$ moderate (typically between 1.5 and 4.0) so that each student can closely match its immediate teacher. The overall compression is the product:

$$R = \prod_{k=1}^{N} r_k = \frac{C_0}{C_N}$$

A progressive pipeline with $N = 3$ stages and per-stage ratio $r = 2$ achieves an overall compression of $R = 8$, but each individual step only needs to bridge a 2x gap.

### 2.3 Curriculum-Based Distillation

Progressive distillation naturally supports a curriculum. Early stages use high temperatures ($T_k$ large) to transfer broad distributional knowledge. Later stages lower the temperature to focus on hard, decision-boundary samples. Formally:

$$T_k = T_{\max} \cdot \left(1 - \frac{k}{N}\right)^{\gamma}$$

where $\gamma$ controls the cooling schedule. Additionally, the mixing coefficient $\alpha_k$ can follow a similar schedule:

$$\alpha_k = \alpha_{\max} \cdot \left(1 - \frac{k}{N}\right)$$

This ensures the final student $M_N$ is anchored more to ground truth than to the (now heavily compressed) teacher signal.

## 3. Progressive Strategies

### 3.1 Layer-Wise Reduction

Remove layers progressively. A 12-layer network becomes 8 layers, then 6, then 4. At each stage the remaining layers absorb the knowledge of the removed ones via distillation. This preserves the width (hidden dimension) of the network, maintaining representational capacity per layer while reducing depth.

### 3.2 Width Reduction

Narrow the hidden dimensions at each stage. A network with 512-unit hidden layers becomes 256, then 128, then 64. Width reduction tends to be gentler than depth reduction because each neuron's contribution can be more smoothly approximated by fewer neurons in the next stage.

### 3.3 Depth Reduction

Reduce the number of processing stages or blocks. This is particularly effective for transformer-based trading models where attention layers can be merged or removed. Each distillation stage teaches the shallower model to replicate the deeper model's intermediate representations.

### 3.4 Combined Progressive Schedules

The most effective approach combines width and depth reduction. For example:

| Stage | Layers | Hidden Dim | Parameters | Compression |
|-------|--------|-----------|------------|-------------|
| 0 (Teacher) | 8 | 256 | ~530K | 1.0x |
| 1 (Medium) | 6 | 192 | ~225K | 2.4x |
| 2 (Small) | 4 | 128 | ~70K | 7.6x |
| 3 (Tiny) | 2 | 64 | ~9K | 59x |

## 4. Trading Applications

### 4.1 Deployment Pipeline: Research to Edge

Progressive distillation maps naturally to the tiered infrastructure of a trading operation:

**Research tier (Model $M_0$):** The full teacher model runs on GPU clusters during overnight batch processing. It uses all available features — hundreds of alternative data signals, order-book microstructure, cross-asset correlations — and can take seconds per prediction. This model establishes the performance ceiling.

**Staging tier (Model $M_1$):** A moderately compressed model runs on standard servers for paper-trading validation. It uses a reduced feature set and runs 3–5x faster than the teacher. This is where strategy PnL is verified before going live.

**Production tier (Model $M_2$):** A small model runs on co-located servers with strict latency budgets (sub-millisecond). It uses only price and volume features available in real-time and produces predictions fast enough for live order management.

**Edge tier (Model $M_3$):** A tiny model runs on embedded hardware or within exchange API gateways. It handles simple risk checks and position adjustments with microsecond latency, using only the most recent price tick.

Each tier's model is the best possible approximation of the research model given its computational budget, because progressive distillation ensures minimal information loss at each transition.

### 4.2 Adaptive Complexity by Market Conditions

Market regimes vary in complexity. During calm, trending markets a tiny model may suffice. During volatile, regime-changing periods the full teacher may be needed. Progressive distillation provides a natural solution: maintain the entire model family and route predictions to the appropriate tier based on detected market conditions.

A regime-detection module can monitor volatility, spread widths, and order-book imbalance to select the active model tier:

- **Low volatility, strong trend:** Use $M_3$ (tiny) — patterns are simple and latency matters.
- **Normal conditions:** Use $M_2$ (small) — balanced speed and accuracy.
- **High volatility, unclear regime:** Use $M_1$ (medium) — more capacity to capture complex dynamics.
- **Crisis / regime change:** Use $M_0$ (teacher) — maximum accuracy, latency is secondary to correctness.

## 5. Comparison with One-Shot Distillation

One-shot distillation trains a single student directly from the teacher. Progressive distillation introduces intermediate stages. The key advantages of progressive distillation are:

### 5.1 Reduced Knowledge Gap

Each stage bridges a small capacity gap. The student at stage $k$ only needs to approximate a model that is slightly larger than itself, rather than one that may be 10–100x larger. This dramatically reduces the difficulty of each individual distillation task.

### 5.2 Regularization Effect

Intermediate models act as implicit regularizers. The soft targets from $M_{k-1}$ are themselves a smoothed version of the original teacher's knowledge, which can prevent the final student from overfitting to noise in the teacher's outputs.

### 5.3 Feature Space Alignment

Progressive stages naturally create a sequence of increasingly abstract feature representations. Each student learns features that are structurally similar to (but simpler than) its teacher's features, making the distillation loss landscape smoother and easier to optimize.

### 5.4 Empirical Evidence

In practice, progressive distillation with $N = 3$ stages typically retains 90–95% of the teacher's accuracy at 50–100x compression, whereas one-shot distillation at the same compression retains only 75–85%. The gap widens as the overall compression ratio increases.

### 5.5 Costs

The primary cost of progressive distillation is training time: $N$ sequential training runs instead of one. However, each run is faster than the one-shot run (because the student is smaller), so the total wall-clock overhead is typically only 1.5–2.5x. Given the accuracy benefits, this is almost always a worthwhile trade-off.

## 6. Implementation Walkthrough (Rust)

The Rust implementation in this chapter provides a complete progressive distillation system with the following components:

### 6.1 FlexibleNetwork

The `FlexibleNetwork` struct represents a feedforward neural network with configurable layer sizes. It stores weight matrices and bias vectors for an arbitrary number of layers, using `ndarray` for linear algebra. The forward pass applies ReLU activations at hidden layers and a linear output at the final layer.

```rust
pub struct FlexibleNetwork {
    pub weights: Vec<Array2<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub layer_sizes: Vec<usize>,
}
```

Key methods:
- `new(layer_sizes)` — initializes with Xavier/He random weights
- `forward(input)` — computes the network output
- `param_count()` — returns total number of trainable parameters
- `train_supervised(X, y, epochs, lr)` — standard gradient-free training (evolutionary)
- `train_distill(teacher, X, epochs, lr)` — trains to match teacher outputs

### 6.2 Progressive Distillation Pipeline

The `ProgressiveDistiller` orchestrates multi-stage compression:

```rust
pub struct ProgressiveDistiller {
    pub stages: Vec<Vec<usize>>,  // layer configs for each stage
    pub models: Vec<FlexibleNetwork>,
    pub stage_metrics: Vec<StageMetrics>,
}
```

It takes a trained teacher and a list of architecture specifications (one per stage), then sequentially distills from each intermediate model to the next.

### 6.3 Loss and Metrics

At each stage, the implementation tracks:
- **Distillation loss** — MSE between teacher and student outputs
- **Parameter count** — total weights + biases
- **Compression ratio** — relative to the original teacher
- **Accuracy retention** — student's R-squared relative to teacher's R-squared on held-out data

### 6.4 One-Shot Comparison

A `one_shot_distill` function trains a student of the same size as the final progressive student, but directly from the original teacher. This allows fair comparison of the two approaches.

## 7. Bybit Data Integration

The implementation fetches real BTCUSDT kline (candlestick) data from the Bybit public API:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=5&limit=200
```

The response provides OHLCV (open, high, low, close, volume) data which is transformed into features:
- **Returns:** $(close_t - close_{t-1}) / close_{t-1}$
- **Volatility:** $(high_t - low_t) / close_t$
- **Volume change:** $(volume_t - volume_{t-1}) / volume_{t-1}$
- **Price momentum:** rolling window return over 5 periods

The target variable is the next-period return, making this a standard return-prediction regression task. The progressive distillation pipeline then compresses the prediction model through multiple stages while preserving as much predictive accuracy as possible.

## 8. Key Takeaways

1. **Progressive distillation compresses models through multiple intermediate stages**, each bridging a small capacity gap, rather than attempting a single large compression step.

2. **The mathematical framework involves a chain of teacher-student pairs**, where each student becomes the teacher for the next stage. Temperature and mixing schedules can follow curriculum-based decay.

3. **Trading systems benefit from the natural tier mapping**: research (full model) → staging (medium) → production (small) → edge (tiny), with each tier running the best model for its constraints.

4. **Adaptive model selection based on market regime** allows the system to trade off latency and accuracy dynamically, using simpler models in calm markets and more complex ones during crises.

5. **Progressive distillation consistently outperforms one-shot distillation** at high compression ratios (>10x), with the gap widening as compression increases. The cost is modest additional training time.

6. **Combined width and depth reduction** across stages provides the best balance of compression and accuracy retention, as verified by the implementation in this chapter.

7. **Live market data integration** (via Bybit API) demonstrates that progressive distillation works on real, noisy financial time series — not just clean academic benchmarks.
