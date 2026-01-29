# Chapter 87: Task-Agnostic Trading

## Overview

Task-Agnostic Trading addresses a fundamental limitation in ML-driven trading systems: traditional approaches require separate models for each trading objective — one for trend prediction, another for volatility forecasting, yet another for regime detection, and so on. Each model learns its own features from scratch, leading to redundant computation, fragmented insights, and poor generalization.

Task-agnostic representation learning solves this by training a single **universal encoder** that maps raw market data into a shared representation space useful across *all* downstream trading tasks simultaneously. Lightweight task-specific heads then decode these representations for each objective, while gradient harmonization ensures balanced multi-task learning.

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architecture Design](#architecture-design)
4. [Multi-Task Learning for Trading](#multi-task-learning-for-trading)
5. [Gradient Harmonization](#gradient-harmonization)
6. [Decision Fusion](#decision-fusion)
7. [Implementation Strategy](#implementation-strategy)
8. [Bybit Integration](#bybit-integration)
9. [Backtesting Framework](#backtesting-framework)
10. [Performance Metrics](#performance-metrics)
11. [References](#references)

---

## Introduction

In quantitative trading, a model trained for trend prediction learns features like momentum indicators and moving average crossovers. A volatility model learns features like realized variance and ATR. A regime model learns features like autocorrelation decay and distribution shape. But these features overlap significantly — they all describe the same underlying market dynamics from different perspectives.

### The Problem with Task-Specific Models

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Traditional Approach: Separate Models                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Raw Market Data                                                            │
│        │                                                                     │
│        ├──────────────────┐                                                  │
│        │                  │                                                  │
│   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐       ┌────▼────┐        │
│   │ Trend   │       │ Vol     │       │ Regime  │       │ Risk    │        │
│   │ Model   │       │ Model   │       │ Model   │       │ Model   │        │
│   │ (Full)  │       │ (Full)  │       │ (Full)  │       │ (Full)  │        │
│   └────┬────┘       └────┬────┘       └────┬────┘       └────┬────┘        │
│        │                  │                  │                  │            │
│   Up/Down/Side       σ forecast        Trending/MR        Low/Med/High     │
│                                                                              │
│   Problem: 4x redundant feature learning, no shared knowledge               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### The Task-Agnostic Solution

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  Task-Agnostic Approach: Shared Encoder                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Raw Market Data                                                            │
│        │                                                                     │
│   ┌────▼──────────────────────────────────────────────────┐                 │
│   │           Universal Encoder (Shared)                   │                 │
│   │   Input → [64] → BN → ReLU → [32] → BN → ReLU → [16] │                 │
│   └────┬──────────────────────────────────────────────────┘                 │
│        │  Shared Representations                                             │
│        ├──────────────┬──────────────┬──────────────┐                       │
│   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐                   │
│   │ Trend   │   │ Vol     │   │ Regime  │   │ Risk    │                   │
│   │ Head    │   │ Head    │   │ Head    │   │ Head    │                   │
│   │ (Light) │   │ (Light) │   │ (Light) │   │ (Light) │                   │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘                   │
│        │              │              │              │                        │
│        └──────────────┴──────┬───────┴──────────────┘                       │
│                              │                                               │
│                     Decision Fusion                                          │
│                              │                                               │
│                    Unified Trading Signal                                    │
│                                                                              │
│   Advantage: Shared features, cross-task knowledge transfer                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Benefits

| Aspect | Task-Specific | Task-Agnostic |
|--------|---------------|---------------|
| Parameters | 4 × full model | 1 encoder + 4 light heads |
| Feature learning | Redundant | Shared |
| Cross-task transfer | None | Automatic |
| Inference time | 4 × forward pass | 1 encode + 4 head passes |
| Consistency | Independent signals | Coherent decisions |
| New tasks | Full retraining | Add a head |

---

## Theoretical Foundation

### Multi-Task Learning Framework

Given a set of $T$ trading tasks $\{\mathcal{T}_1, \ldots, \mathcal{T}_T\}$, we learn:

1. A shared encoder $f_\theta: \mathbb{R}^d \to \mathbb{R}^k$ mapping input features to representations
2. Task-specific heads $g_{\phi_t}: \mathbb{R}^k \to \mathbb{R}^{c_t}$ for each task $t$

The multi-task objective is:

$$\min_{\theta, \{\phi_t\}} \sum_{t=1}^{T} w_t \cdot \mathcal{L}_t(g_{\phi_t}(f_\theta(X)), Y_t)$$

where $w_t$ are task weights and $\mathcal{L}_t$ is the loss for task $t$.

### Task-Agnostic Representations

A representation is **task-agnostic** if it captures the fundamental structure of the data without bias toward any particular downstream task. Formally:

$$I(Z; Y_t) \approx I(Z; Y_{t'}) \quad \forall t, t' \in \{1, \ldots, T\}$$

where $I(Z; Y_t)$ is the mutual information between representation $Z$ and task labels $Y_t$.

This means the encoder extracts features equally useful for trend prediction, volatility forecasting, regime detection, and risk assessment.

### Trading Tasks

We define four core trading tasks:

1. **Trend Prediction** (Classification: 3 classes)
   - Up / Sideways / Down
   - Loss: Cross-entropy

2. **Volatility Forecast** (Regression: 1 output)
   - Predict next-period realized volatility
   - Loss: Mean Squared Error

3. **Regime Detection** (Classification: 4 classes)
   - Trending / Mean-Reverting / Volatile / Calm
   - Loss: Cross-entropy

4. **Risk Assessment** (Classification: 3 classes)
   - Low / Medium / High risk
   - Loss: Cross-entropy

---

## Architecture Design

### Universal Encoder

The encoder is a feedforward neural network with:

- **Batch Normalization** after each hidden layer for stable training
- **ReLU activations** for non-linearity
- **Residual connections** when dimensions match
- **L2 normalization** of output representations to prevent collapse

```
Input (d=20 features)
    │
    ▼
┌─────────────────────┐
│  Linear(20 → 64)    │
│  BatchNorm1d(64)     │
│  ReLU                │
│  Dropout(0.1)        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Linear(64 → 32)    │
│  BatchNorm1d(32)     │
│  ReLU                │
│  Dropout(0.1)        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Linear(32 → 16)    │
│  L2 Normalize        │
└─────────┬───────────┘
          │
          ▼
  Representation (k=16)
```

### Task Heads

Each task head is a two-layer network:

```python
class TaskHead(nn.Module):
    def __init__(self, repr_dim, hidden_dim, output_dim):
        self.hidden = Linear(repr_dim, hidden_dim)
        self.output = Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = relu(self.hidden(z))
        out = self.output(h)
        return softmax(out)  # or sigmoid for regression
```

### Feature Extraction

We extract 19 task-agnostic features from OHLCV data:

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | return_1d | Returns | 1-day price return |
| 2 | return_5d | Returns | 5-day cumulative return |
| 3 | rsi | Momentum | Relative Strength Index (normalized) |
| 4 | macd_signal | Momentum | MACD histogram |
| 5 | ma_crossover | Momentum | Moving average crossover signal |
| 6 | realized_vol | Volatility | Realized volatility (log returns std) |
| 7 | bb_position | Volatility | Position within Bollinger Bands |
| 8 | atr_ratio | Volatility | Average True Range / Price |
| 9 | volume_ratio | Volume | Current volume / 20-day average |
| 10 | volume_trend | Volume | Short-term vs long-term volume |
| 11 | body_ratio | Candle | Candle body / total range |
| 12 | upper_shadow | Candle | Upper shadow / total range |
| 13 | lower_shadow | Candle | Lower shadow / total range |
| 14 | range_pct | Range | High-low range as percentage |
| 15 | close_position | Range | Close position within range |
| 16 | trend_strength | Trend | Linear regression slope |
| 17 | trend_consistency | Trend | Proportion of up-days |
| 18 | return_skewness | Distribution | Skewness of recent returns |
| 19 | return_kurtosis | Distribution | Excess kurtosis of returns |

---

## Multi-Task Learning for Trading

### Training Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Multi-Task Training Loop                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  For each epoch:                                                         │
│    For each mini-batch:                                                  │
│                                                                          │
│    1. Forward pass through encoder                                       │
│       features → representations                                        │
│                                                                          │
│    2. Forward pass through each task head                                │
│       representations → predictions[task]                                │
│                                                                          │
│    3. Compute losses for each task                                       │
│       L_trend = CrossEntropy(pred_trend, label_trend)                   │
│       L_vol   = MSE(pred_vol, label_vol)                                │
│       L_regime = CrossEntropy(pred_regime, label_regime)                │
│       L_risk  = CrossEntropy(pred_risk, label_risk)                     │
│                                                                          │
│    4. Harmonize gradients                                                │
│       w_t = GradNorm(L_1, ..., L_T)                                    │
│                                                                          │
│    5. Compute total loss                                                 │
│       L_total = Σ w_t * L_t                                             │
│                                                                          │
│    6. Backpropagate and update                                           │
│       θ ← θ - lr * ∇_θ L_total                                         │
│       φ_t ← φ_t - lr * ∇_{φ_t} L_total                                │
│                                                                          │
│  Early stopping on validation loss                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Gradient Harmonization

A key challenge in multi-task learning is **gradient conflict**: gradients from different tasks may point in opposing directions, causing the encoder to oscillate or converge to a solution that favors one task over others.

### GradNorm Algorithm

GradNorm dynamically adjusts task weights so all tasks train at similar rates:

1. Track the loss ratio $r_t = L_t / L_t^{(0)}$ (current / initial)
2. Compute relative training rate: $\tilde{r}_t = r_t / \bar{r}$
3. Update weights: $w_t \propto \tilde{r}_t^{-\alpha}$

Tasks that train slower get higher weights, ensuring balanced progress.

### PCGrad (Projected Conflicting Gradients)

For conflicting gradient pairs:

$$g_i^{PC} = g_i - \frac{g_i \cdot g_j}{\|g_j\|^2} g_j \quad \text{if } g_i \cdot g_j < 0$$

This projects away the conflicting component while preserving the cooperative component.

### Uncertainty Weighting

Weight tasks by inverse prediction uncertainty:

$$w_t = \frac{1}{2\sigma_t^2}$$

where $\sigma_t$ is the learned uncertainty for task $t$. Uncertain tasks get less weight.

---

## Decision Fusion

After obtaining predictions from all task heads, we fuse them into a unified trading decision.

### Weighted Average Fusion

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Decision Fusion Pipeline                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Trend Head (w=0.35):                                                │
│    [Up: 0.6, Side: 0.3, Down: 0.1]                                 │
│    → signal += 0.35 * (0.6 - 0.1) = +0.175                        │
│                                                                      │
│  Volatility Head (w=0.20):                                           │
│    [Predicted: 0.3]                                                  │
│    → confidence *= (1 - 0.3*0.3) = 0.91                            │
│    → risk_level = 0.3                                                │
│                                                                      │
│  Regime Head (w=0.25):                                               │
│    [Trend: 0.5, MR: 0.2, Vol: 0.2, Calm: 0.1]                     │
│    → regime = "Trending"                                             │
│    → signal *= 1.2 (boost for trending regime)                      │
│                                                                      │
│  Risk Head (w=0.20):                                                 │
│    [Low: 0.6, Med: 0.3, High: 0.1]                                 │
│    → signal *= (1 - 0.1 * 0.5) = 0.95                              │
│                                                                      │
│  Final: signal=+0.20, confidence=0.72, regime=Trending, risk=0.30   │
│  → Signal Type: BUY                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Signal Classification

| Signal Value | Type | Position |
|-------------|------|----------|
| ≥ 0.40 | Strong Buy | 100% long |
| ≥ 0.15 | Buy | 50% long |
| -0.15 to 0.15 | Hold | Flat |
| ≤ -0.15 | Sell | 50% short |
| ≤ -0.40 | Strong Sell | 100% short |

---

## Implementation Strategy

### Python (PyTorch)

```python
from task_agnostic_trading import (
    TaskAgnosticModel, EncoderConfig, MultiTaskTrainer,
    TrainerConfig, FeatureExtractor, DecisionFusion,
    SignalGenerator, BacktestEngine, BybitClient,
)
import asyncio

# 1. Fetch data
client = BybitClient()
klines = asyncio.run(client.fetch_klines("BTCUSDT", "60", 250))

# 2. Extract features
extractor = FeatureExtractor()
features = extractor.extract_all(klines)

# 3. Create model
config = EncoderConfig(input_dim=19, hidden_dims=[64, 32], repr_dim=16)
model = TaskAgnosticModel(config)

# 4. Train
trainer = MultiTaskTrainer(TrainerConfig(epochs=100))
result = trainer.train(model, features, labels)

# 5. Predict and fuse
outputs = model.predict_single(features)
fusion = DecisionFusion()
fused = fusion.fuse(outputs)

# 6. Generate signals and backtest
signals = SignalGenerator().generate(fused)
bt = BacktestEngine().run(signals, prices)
```

### Rust

```rust
use task_agnostic_trading::prelude::*;

// Create model
let config = TaskAgnosticConfig::default()
    .with_input_dim(19)
    .with_repr_dim(16);
let model = TaskAgnosticModel::new(config);

// Predict
let fused = model.predict(&features);

// Generate signals
let gen = SignalGenerator::new(SignalConfig::default());
let signals = gen.generate(&fused);

// Backtest
let engine = BacktestEngine::new(BacktestConfig::default());
let result = engine.run(&signals, &prices);
println!("Sharpe: {:.2}", result.metrics.sharpe_ratio);
```

---

## Bybit Integration

The system supports cryptocurrency data via the Bybit exchange API:

```python
client = BybitClient()

# Single symbol
klines = await client.fetch_klines("BTCUSDT", interval="60", limit=200)

# Multiple symbols concurrently
multi_data = await client.fetch_multi_klines(
    ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    interval="60", limit=200
)
```

### Supported Data Types

- **Klines (OHLCV)**: Candlestick data at various intervals
- **Tickers**: 24-hour market summaries
- **Order Books**: Bid/ask depth
- **Funding Rates**: Perpetual contract funding

---

## Backtesting Framework

The backtesting engine simulates trading with:

- **Transaction costs**: Configurable per-trade cost (default 0.1%)
- **Slippage**: Execution price deviation (default 0.05%)
- **Position sizing**: Based on signal strength, confidence, and risk

### Performance Metrics

| Metric | Description |
|--------|-------------|
| Total Return | Cumulative portfolio return |
| Annualized Return | Geometric annual return |
| Sharpe Ratio | Risk-adjusted return (annualized) |
| Sortino Ratio | Downside risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| Calmar Ratio | Return / Max Drawdown |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit / Gross loss |

---

## Performance Metrics

### Model Evaluation

- **Classification tasks**: Cross-entropy loss, accuracy, F1-score
- **Regression tasks**: MSE, MAE, R²
- **Representation quality**: Inter-class / intra-class distance ratio
- **Task balance**: Weight variance across tasks (lower = more balanced)

### Strategy Evaluation

- **Risk-adjusted returns**: Sharpe, Sortino, Calmar ratios
- **Drawdown analysis**: Maximum drawdown, recovery time
- **Trade statistics**: Win rate, profit factor, average trade
- **Benchmark comparison**: Strategy vs buy-and-hold

---

## References

1. **Task-Agnostic Representation Learning**
   - Lan et al., 2019
   - URL: https://arxiv.org/abs/1907.12157

2. **GradNorm: Gradient Normalization for Adaptive Loss Balancing**
   - Chen et al., 2018
   - URL: https://arxiv.org/abs/1711.02257

3. **Multi-Task Learning Using Uncertainty to Weigh Losses**
   - Kendall et al., 2018
   - URL: https://arxiv.org/abs/1705.07115

4. **Gradient Surgery for Multi-Task Learning (PCGrad)**
   - Yu et al., 2020
   - URL: https://arxiv.org/abs/2001.06782

5. **An Overview of Multi-Task Learning in Deep Neural Networks**
   - Ruder, 2017
   - URL: https://arxiv.org/abs/1706.05098

---

## Directory Structure

```
87_task_agnostic_trading/
├── README.md                          # This file
├── README.ru.md                       # Russian translation
├── readme.simple.md                   # Simplified explanation (English)
├── readme.simple.ru.md                # Simplified explanation (Russian)
├── README.specify.md                  # Task specification
├── Cargo.toml                         # Rust project configuration
├── Cargo.lock                         # Dependency lock file
├── src/                               # Rust source code
│   ├── lib.rs                         # Library root
│   ├── model/                         # Model components
│   │   ├── mod.rs                     # Module root
│   │   ├── encoder.rs                 # Universal encoder
│   │   ├── task_heads.rs              # Task-specific heads
│   │   └── fusion.rs                  # Decision fusion
│   ├── data/                          # Data handling
│   │   ├── mod.rs                     # Module root
│   │   ├── bybit.rs                   # Bybit API client
│   │   ├── features.rs                # Feature extraction
│   │   └── types.rs                   # Data types
│   ├── training/                      # Training logic
│   │   ├── mod.rs                     # Module root
│   │   ├── multi_task.rs              # Multi-task trainer
│   │   └── gradient.rs                # Gradient harmonization
│   ├── strategy/                      # Trading strategy
│   │   ├── mod.rs                     # Module root
│   │   ├── regime.rs                  # Regime classification
│   │   └── signals.rs                 # Signal generation
│   └── backtest/                      # Backtesting
│       ├── mod.rs                     # Module root
│       └── engine.rs                  # Backtest engine
├── examples/                          # Rust examples
│   ├── basic_task_agnostic.rs         # Basic usage
│   ├── multi_task_strategy.rs         # Full strategy pipeline
│   └── universal_encoder.rs           # Encoder analysis
└── python/                            # Python implementation
    └── task_agnostic_trading.py       # Complete Python module
```

## Quick Start

### Python

```bash
cd 87_task_agnostic_trading/python
pip install torch numpy aiohttp
python task_agnostic_trading.py
```

### Rust

```bash
cd 87_task_agnostic_trading
cargo test
cargo run --example basic_task_agnostic
cargo run --example multi_task_strategy
cargo run --example universal_encoder
```
