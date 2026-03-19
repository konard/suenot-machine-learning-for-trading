# Chapter 201: Knowledge Distillation Trading

## 1. Introduction

Knowledge distillation, introduced by Geoffrey Hinton, Oriol Vinyals, and Jeff Dean in their seminal 2015 paper "Distilling the Knowledge in a Neural Network," is a model compression technique that transfers learned representations from a large, complex "teacher" model to a smaller, more efficient "student" model. In the context of algorithmic trading, this technique addresses a fundamental tension: the most accurate models are often too large and slow for real-time deployment, while models fast enough for production often lack the sophistication needed to capture complex market dynamics.

Trading environments impose strict latency constraints. In high-frequency trading (HFT), decisions must be made in microseconds. Even in lower-frequency strategies, model inference must complete within the time window of actionable market signals. A deep ensemble of gradient-boosted trees and neural networks might achieve superior alpha, but if it cannot produce a prediction before the opportunity vanishes, its accuracy is worthless.

Knowledge distillation resolves this by allowing us to train an arbitrarily complex teacher model offline — with no latency constraints — and then compress its knowledge into a compact student model suitable for real-time inference. The student learns not just the hard labels (buy, sell, hold) but the teacher's full probability distribution over outcomes, capturing nuanced relationships that the teacher discovered during training.

This chapter covers the mathematical foundations of knowledge distillation, explains why it is particularly valuable for trading systems, presents teacher-student architectures tailored to financial data, discusses trading-specific distillation techniques, and provides a complete Rust implementation with Bybit market data integration.

## 2. Mathematical Foundation

### 2.1 Soft Targets and Temperature Scaling

The core insight of knowledge distillation is that a trained model's output probability distribution contains far more information than hard labels alone. When a teacher model outputs probabilities [0.7, 0.2, 0.1] for [buy, hold, sell], the relative magnitudes of the non-winning classes (hold at 0.2 vs sell at 0.1) encode valuable structural information about the input — what Hinton calls "dark knowledge."

To amplify this dark knowledge, we apply temperature scaling to the softmax function:

```
q_i = exp(z_i / T) / sum_j(exp(z_j / T))
```

where `z_i` are the logits (pre-softmax outputs), `T` is the temperature parameter, and `q_i` are the resulting soft probabilities. When `T = 1`, this is the standard softmax. As `T` increases, the distribution becomes softer (more uniform), revealing more information about the relative magnitudes of the logits. A typical temperature range for distillation is `T` between 2 and 20.

### 2.2 KL Divergence Loss

The student is trained to match the teacher's soft probability distribution using Kullback-Leibler divergence:

```
L_soft = KL(p_teacher || p_student) = sum_i p_teacher_i * log(p_teacher_i / p_student_i)
```

where `p_teacher` and `p_student` are the temperature-scaled probability distributions from teacher and student respectively. KL divergence measures how much information is lost when the student's distribution is used to approximate the teacher's distribution.

### 2.3 Combined Distillation Loss

The total training loss for the student combines the soft-target loss with a standard hard-label loss:

```
L_total = alpha * L_hard(y_true, p_student(T=1)) + (1 - alpha) * T^2 * L_soft(p_teacher(T), p_student(T))
```

Key components:

- **`alpha`**: Balancing parameter between hard and soft losses, typically 0.1 to 0.5. Lower alpha values give more weight to the teacher's soft targets.
- **`T^2` scaling**: The gradients of the soft loss are scaled by `1/T^2`, so we multiply by `T^2` to ensure the soft and hard gradient magnitudes remain comparable as temperature changes.
- **`L_hard`**: Standard cross-entropy loss against ground truth labels, computed with `T=1`.
- **`L_soft`**: KL divergence between teacher and student soft distributions at temperature `T`.

### 2.4 Dark Knowledge

Dark knowledge refers to the information encoded in the non-target class probabilities. In trading, this is especially valuable:

- A model predicting "buy" with soft outputs [0.6, 0.35, 0.05] suggests the situation is close to a "hold" — the trade has marginal conviction.
- The same "buy" prediction with [0.6, 0.05, 0.35] suggests the alternative is a strong "sell" signal — the market is highly directional but uncertain.

These nuances are lost in hard labels but preserved through distillation. The student learns to reproduce these nuanced distributions, effectively inheriting the teacher's uncertainty calibration.

## 3. Why Distillation for Trading

### 3.1 Latency Requirements in HFT

High-frequency trading systems operate under extreme latency constraints:

| Latency Budget | Trading Style | Model Constraint |
|---|---|---|
| < 10 microseconds | Market making, HFT | Lookup tables, linear models |
| 10-100 microseconds | Statistical arbitrage | Small neural networks |
| 100 microseconds - 1 ms | Momentum, mean reversion | Medium neural networks |
| 1-100 ms | Swing trading signals | Large models feasible |

Knowledge distillation allows practitioners to train a teacher model in the "1-100 ms" tier and deploy a student in the "10-100 microseconds" tier, gaining the accuracy benefits of the complex model while meeting the latency requirements of the faster strategy.

### 3.2 Edge Deployment

Trading at co-located servers, on FPGAs, or in embedded systems demands minimal model footprint. A teacher ensemble consuming 500 MB of memory and requiring GPU inference is impractical for edge deployment. A distilled student might require 1 MB and run on a single CPU core.

### 3.3 Model Compression for Real-Time Inference

Beyond latency, distillation reduces:

- **Memory footprint**: Fewer parameters means less memory, reducing cache misses and improving throughput.
- **Power consumption**: Critical for co-located servers where power and cooling are constrained.
- **Operational complexity**: Smaller models are easier to version, deploy, and monitor in production.

Empirical results across domains show that distilled students typically retain 90-98% of the teacher's accuracy while being 10-100x smaller and faster.

## 4. Teacher-Student Architectures

### 4.1 Large Ensemble Teacher to Small Student

The most common setup uses an ensemble teacher:

```
Teacher: Ensemble of N models
  - Model 1: Deep neural network (e.g., 8 layers, 512 units)
  - Model 2: Gradient-boosted tree ensemble
  - Model 3: LSTM with attention
  - ...
  - Model N: Transformer encoder

  Ensemble output: Average of all model probabilities

Student: Compact neural network
  - 2-3 layers, 32-64 units
  - Simple feedforward architecture
  - Designed for target inference latency
```

The ensemble teacher captures diverse perspectives on the data — tree models capture feature interactions, LSTMs capture temporal patterns, transformers capture long-range dependencies. The student learns to approximate all of these behaviors simultaneously.

### 4.2 Cross-Architecture Distillation

A particularly powerful approach for trading is cross-architecture distillation:

- **Transformer teacher to MLP student**: A transformer-based teacher captures complex temporal dependencies in orderbook data. The student MLP, while unable to model such dependencies natively, learns to approximate the transformer's outputs using handcrafted temporal features.
- **Graph neural network teacher to linear student**: A GNN teacher models inter-asset correlations. The student linear model learns which asset pairs matter most, effectively distilling the graph structure into feature weights.
- **Reinforcement learning teacher to supervised student**: An RL agent learns an optimal trading policy through interaction. The student learns to imitate the RL agent's actions via supervised distillation, avoiding the complexity and instability of RL at inference time.

## 5. Trading-Specific Techniques

### 5.1 Distilling Alpha Signals

Rather than distilling raw predictions, we can distill intermediate alpha signals:

1. Train the teacher to output alpha factors (expected returns).
2. The student learns to replicate these alpha factors from raw market data.
3. A separate, simple portfolio optimization layer converts alphas to positions.

This separation allows the student to focus on the hardest part (alpha generation) while keeping the portfolio construction interpretable and auditable.

### 5.2 Distilling from Backtested Strategies

A novel approach for trading is to use backtested strategy returns as the teacher signal:

1. Run a complex strategy in backtest, recording its positions at each timestep.
2. Train a student to predict the strategy's positions from market features.
3. The student effectively learns a compressed version of the strategy.

This technique is particularly useful for strategies that involve complex rule-based logic, multi-timeframe analysis, or external data sources that are unavailable in real-time.

### 5.3 Time-Series Aware Distillation

Standard distillation assumes i.i.d. samples. Financial time series violate this assumption. Time-series aware distillation addresses this through:

- **Temporal weighting**: More recent soft targets receive higher weight, as the teacher's knowledge about recent market regimes is more relevant.
- **Regime-conditional distillation**: Separate distillation for different market regimes (trending, mean-reverting, volatile). The student learns regime-specific behavior from the teacher.
- **Sequential distillation**: The student processes data sequentially (as it would in production), and the distillation loss accounts for the student's hidden state evolution.
- **Rolling window distillation**: The teacher is periodically retrained on new data, and the student is continuously distilled from the updated teacher, ensuring adaptation to evolving market conditions.

## 6. Implementation Walkthrough (Rust)

The implementation is organized into several components in the `rust/` directory. The core library (`src/lib.rs`) provides:

### Neural Network Components

The `TeacherModel` is a multi-layer feedforward neural network with configurable hidden layer sizes (default: 128 -> 64 -> 32 units). It uses ReLU activations and is designed for maximum accuracy without latency constraints.

The `StudentModel` is a compact two-layer network (default: 16 -> 8 units) designed for minimal inference latency while maintaining sufficient capacity to capture the teacher's dark knowledge.

### Temperature-Scaled Softmax

```rust
pub fn softmax_with_temperature(logits: &Array1<f64>, temperature: f64) -> Array1<f64> {
    let scaled = logits / temperature;
    let max_val = scaled.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals = (scaled - max_val).mapv(f64::exp);
    let sum = exp_vals.sum();
    exp_vals / sum
}
```

The temperature parameter controls the softness of the distribution. Higher temperatures produce softer distributions that reveal more dark knowledge.

### Distillation Training Loop

The training loop computes both hard and soft losses:

1. Forward pass through teacher (frozen weights) to get soft targets at temperature T.
2. Forward pass through student to get predictions at both T=1 (for hard loss) and T (for soft loss).
3. Compute combined loss: `alpha * cross_entropy(y_true, student_T1) + (1-alpha) * T^2 * kl_div(teacher_T, student_T)`.
4. Backpropagate through student only (teacher weights are frozen).
5. Update student weights via gradient descent.

### Inference Benchmarking

The implementation includes a benchmarking function that measures inference latency for both teacher and student models across thousands of iterations, reporting mean and percentile latencies.

## 7. Bybit Data Integration

The implementation fetches real market data from the Bybit exchange API. The `BybitClient` struct provides methods to:

- Fetch recent kline (candlestick) data for any trading pair.
- Parse OHLCV data into feature vectors suitable for model input.
- Compute derived features: returns, volatility, momentum indicators.

The API endpoint used is `https://api.bybit.com/v5/market/kline`, which provides public market data without authentication. The data is transformed into feature vectors that capture:

- Price returns over multiple lookback windows (1, 5, 10, 20 periods).
- Realized volatility.
- Volume-weighted price changes.
- High-low range as a fraction of closing price.

These features serve as inputs to both teacher and student models for price direction prediction.

## 8. Key Takeaways

1. **Knowledge distillation bridges the accuracy-latency gap** in trading systems by transferring knowledge from complex teacher models to efficient student models suitable for real-time deployment.

2. **Dark knowledge — the non-target class probabilities — encodes valuable information** about market uncertainty, conviction levels, and regime characteristics that hard labels discard.

3. **Temperature scaling controls the information transfer**: higher temperatures reveal more of the teacher's internal representations, while the `T^2` scaling factor in the combined loss ensures gradient magnitudes remain balanced.

4. **Trading-specific distillation techniques** — alpha signal distillation, backtested strategy distillation, and time-series aware distillation — adapt the general framework to the unique characteristics of financial data.

5. **Cross-architecture distillation is particularly powerful for trading**, enabling practitioners to combine the strengths of diverse model types (transformers, GNNs, RL agents) into a single, deployable student.

6. **The combined loss `L = alpha * L_hard + (1-alpha) * T^2 * L_soft`** balances learning from ground truth and from the teacher. In practice, lower alpha values (more weight on soft targets) often work best when the teacher is highly accurate.

7. **Rust implementation enables microsecond-level inference**, making distilled models viable for HFT and co-located deployment scenarios where Python-based solutions are too slow.

8. **Continuous distillation from periodically retrained teachers** addresses the non-stationarity of financial markets, ensuring the deployed student model adapts to evolving market conditions without requiring direct retraining on new data.
