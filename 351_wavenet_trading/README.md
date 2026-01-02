# Chapter 351: WaveNet for Trading

## Overview

WaveNet is a deep generative model originally developed by DeepMind for raw audio generation. Its key innovation - **dilated causal convolutions** - makes it exceptionally well-suited for time series prediction in financial markets. This chapter explores how to adapt WaveNet architecture for cryptocurrency trading using data from the Bybit exchange.

## Table of Contents

1. [Introduction](#introduction)
2. [WaveNet Architecture](#wavenet-architecture)
3. [Dilated Causal Convolutions](#dilated-causal-convolutions)
4. [Adapting WaveNet for Trading](#adapting-wavenet-for-trading)
5. [Implementation](#implementation)
6. [Trading Strategy](#trading-strategy)
7. [Backtesting Results](#backtesting-results)
8. [References](#references)

---

## Introduction

### Why WaveNet for Trading?

Traditional recurrent neural networks (RNNs, LSTMs, GRUs) process sequences step-by-step, which can be:
- Computationally slow
- Difficult to parallelize
- Prone to vanishing gradients for long sequences

WaveNet addresses these issues with **dilated causal convolutions**, which:
- Process sequences in parallel (faster training)
- Capture long-term dependencies efficiently
- Maintain causality (no future information leakage)

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Causal Convolution** | Convolution that only looks at past data |
| **Dilated Convolution** | Convolution with gaps between inputs |
| **Receptive Field** | How far back the network can "see" |
| **Skip Connections** | Direct paths for multi-scale feature extraction |
| **Residual Connections** | Help gradient flow in deep networks |

---

## WaveNet Architecture

### Original Architecture

```
Input → Causal Conv → [Dilated Conv Block] × N → Skip Connections → Output
                              ↓
                    Gated Activation Unit
                    (tanh ⊗ sigmoid)
```

### Core Components

1. **Causal Convolution Layer**
   - Ensures no future information leakage
   - First layer that processes raw input

2. **Dilated Convolution Blocks**
   - Multiple layers with increasing dilation rates
   - Exponentially growing receptive field: 1, 2, 4, 8, 16, 32...

3. **Gated Activation Units**
   ```
   z = tanh(Wf * x) ⊙ sigmoid(Wg * x)
   ```
   - Wf: filter weights (what to learn)
   - Wg: gate weights (what to forget)

4. **Skip and Residual Connections**
   - Skip: aggregate information from all layers
   - Residual: enable gradient flow in deep networks

### Receptive Field Calculation

For dilated convolutions with dilation rates [1, 2, 4, 8, 16, ...]:

```
Receptive Field = (kernel_size - 1) × Σ(dilation_rates) + 1
```

**Example:** With kernel_size=2 and 10 layers:
- Dilation rates: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
- Receptive field = (2-1) × 1023 + 1 = 1024 timesteps

This means with hourly data, the model can "see" ~42 days of history!

---

## Dilated Causal Convolutions

### Why Dilated?

Standard convolutions have a linear relationship between receptive field and depth:
- Receptive field of N requires N layers
- 1000 timesteps would need 1000 layers!

Dilated convolutions provide exponential growth:
- Receptive field of N requires only log₂(N) layers
- 1024 timesteps need only 10 layers!

### Visual Representation

```
Dilation = 1:  ●─●─●─●
                │ │ │ │
               [Conv Layer]

Dilation = 2:  ●───●───●───●
                │   │   │   │
               [Conv Layer]

Dilation = 4:  ●───────●───────●───────●
                │       │       │       │
               [Conv Layer]
```

### Causal Padding

To maintain causality, we use asymmetric padding:
- Pad only on the left side
- Ensures output[t] depends only on input[0:t]

```rust
// Causal padding for 1D convolution
fn causal_pad(input: &[f64], padding: usize) -> Vec<f64> {
    let mut padded = vec![0.0; padding];
    padded.extend_from_slice(input);
    padded
}
```

---

## Adapting WaveNet for Trading

### Modifications for Financial Time Series

1. **Input Features**
   - OHLCV data (Open, High, Low, Close, Volume)
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Market microstructure features

2. **Output Heads**
   - **Regression**: Predict next price/return
   - **Classification**: Predict direction (up/down/neutral)
   - **Probabilistic**: Predict distribution parameters

3. **Loss Functions**
   - MSE/MAE for regression
   - Cross-entropy for classification
   - Custom trading losses (Sharpe-aware)

### Architecture for Trading

```
Input Features (OHLCV + Indicators)
        ↓
  Input Projection (Linear)
        ↓
  ┌─────────────────────────┐
  │   WaveNet Blocks (×N)   │
  │  ├─ Dilated Conv        │
  │  ├─ Gated Activation    │
  │  ├─ Residual Connection │
  │  └─ Skip Connection     │
  └─────────────────────────┘
        ↓
  Skip Aggregation (Sum)
        ↓
  Output Layers (Dense)
        ↓
  Predictions (Price/Direction)
```

---

## Implementation

### Rust Implementation Structure

```
351_wavenet_trading/
├── README.md
├── README.ru.md
├── readme.simple.md
├── readme.simple.ru.md
└── rust/
    ├── Cargo.toml
    └── src/
        ├── lib.rs
        ├── api/           # Bybit API client
        │   ├── mod.rs
        │   ├── bybit.rs
        │   └── storage.rs
        ├── models/        # WaveNet model
        │   ├── mod.rs
        │   ├── wavenet.rs
        │   ├── layers.rs
        │   └── activations.rs
        ├── trading/       # Trading logic
        │   ├── mod.rs
        │   ├── strategy.rs
        │   ├── signals.rs
        │   └── backtest.rs
        ├── analysis/      # Data analysis
        │   ├── mod.rs
        │   ├── features.rs
        │   └── indicators.rs
        └── bin/           # Executables
            ├── fetch_data.rs
            ├── train_wavenet.rs
            ├── predict.rs
            └── backtest.rs
```

### Key Implementation Details

#### 1. Dilated Convolution

```rust
pub struct DilatedConv1D {
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
    kernel_size: usize,
    dilation: usize,
    channels_in: usize,
    channels_out: usize,
}

impl DilatedConv1D {
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = input[0].len();
        let mut output = vec![vec![0.0; seq_len]; self.channels_out];

        for t in 0..seq_len {
            for c_out in 0..self.channels_out {
                let mut sum = self.bias[c_out];

                for k in 0..self.kernel_size {
                    let idx = t as i64 - (k * self.dilation) as i64;
                    if idx >= 0 {
                        for c_in in 0..self.channels_in {
                            sum += self.weights[c_out][c_in * self.kernel_size + k]
                                 * input[c_in][idx as usize];
                        }
                    }
                }

                output[c_out][t] = sum;
            }
        }

        output
    }
}
```

#### 2. Gated Activation

```rust
pub fn gated_activation(filter: &[f64], gate: &[f64]) -> Vec<f64> {
    filter.iter()
        .zip(gate.iter())
        .map(|(f, g)| f.tanh() * sigmoid(*g))
        .collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

#### 3. WaveNet Block

```rust
pub struct WaveNetBlock {
    filter_conv: DilatedConv1D,
    gate_conv: DilatedConv1D,
    residual_conv: Conv1D,
    skip_conv: Conv1D,
}

impl WaveNetBlock {
    pub fn forward(&self, input: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // Dilated convolutions
        let filter = self.filter_conv.forward(input);
        let gate = self.gate_conv.forward(input);

        // Gated activation
        let activated: Vec<Vec<f64>> = filter.iter()
            .zip(gate.iter())
            .map(|(f, g)| gated_activation(f, g))
            .collect();

        // Residual connection
        let residual = self.residual_conv.forward(&activated);
        let residual_out: Vec<Vec<f64>> = input.iter()
            .zip(residual.iter())
            .map(|(i, r)| i.iter().zip(r.iter()).map(|(a, b)| a + b).collect())
            .collect();

        // Skip connection
        let skip = self.skip_conv.forward(&activated);

        (residual_out, skip)
    }
}
```

---

## Trading Strategy

### Signal Generation

The WaveNet model outputs are converted to trading signals:

1. **Regression-based Strategy**
   ```rust
   pub fn generate_signal(predicted_return: f64, threshold: f64) -> Signal {
       if predicted_return > threshold {
           Signal::Buy
       } else if predicted_return < -threshold {
           Signal::Sell
       } else {
           Signal::Hold
       }
   }
   ```

2. **Classification-based Strategy**
   ```rust
   pub fn generate_signal(probabilities: [f64; 3]) -> Signal {
       let (max_idx, _) = probabilities.iter()
           .enumerate()
           .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
           .unwrap();

       match max_idx {
           0 => Signal::Sell,
           1 => Signal::Hold,
           2 => Signal::Buy,
           _ => unreachable!()
       }
   }
   ```

### Position Sizing

Using volatility-adjusted position sizing:

```rust
pub fn calculate_position_size(
    capital: f64,
    volatility: f64,
    risk_per_trade: f64,
    max_position: f64,
) -> f64 {
    let vol_adjusted = risk_per_trade / volatility;
    (capital * vol_adjusted).min(capital * max_position)
}
```

### Risk Management

- **Stop Loss**: Dynamic based on ATR (Average True Range)
- **Take Profit**: Risk-reward ratio of 1:2 or 1:3
- **Maximum Drawdown**: Position reduction at 10% drawdown

---

## Backtesting Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Total Return | - |
| Sharpe Ratio | - |
| Sortino Ratio | - |
| Maximum Drawdown | - |
| Win Rate | - |
| Profit Factor | - |

*Note: Run the backtesting examples to see actual results with current data.*

### Running Backtests

```bash
# Fetch data from Bybit
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1h --days 365

# Train WaveNet model
cargo run --bin train_wavenet -- --data ./data/BTCUSDT_1h.csv --epochs 100

# Run backtest
cargo run --bin backtest -- --model ./models/wavenet.bin --data ./data/BTCUSDT_1h.csv
```

---

## Advantages and Limitations

### Advantages

1. **Parallelizable Training**: Unlike RNNs, can process all timesteps simultaneously
2. **Large Receptive Field**: Captures long-term patterns efficiently
3. **No Vanishing Gradients**: Thanks to residual connections
4. **Flexible Architecture**: Easy to adjust depth and receptive field

### Limitations

1. **Memory Intensive**: Stores all intermediate activations
2. **Fixed Receptive Field**: Cannot dynamically adjust lookback
3. **Inference Speed**: Full convolution needed for each prediction
4. **No Attention**: Cannot focus on specific historical events

### When to Use WaveNet

✅ **Good for:**
- Capturing cyclical patterns (hourly, daily, weekly)
- Markets with strong momentum/mean-reversion
- When parallelization is important

❌ **Not ideal for:**
- Very long sequences (>10,000 timesteps)
- When specific event attention is crucial
- Limited computational resources

---

## References

### Papers

1. **van den Oord, A., et al. (2016)** - "WaveNet: A Generative Model for Raw Audio"
   - ArXiv: [https://arxiv.org/abs/1609.03499](https://arxiv.org/abs/1609.03499)

2. **Borovykh, A., Bohte, S., & Oosterlee, C. W. (2017)** - "Conditional Time Series Forecasting with Convolutional Neural Networks"
   - ArXiv: [https://arxiv.org/abs/1703.04691](https://arxiv.org/abs/1703.04691)

3. **Chen, W., et al. (2020)** - "WaveNet-based Deep Learning for Financial Time Series"
   - Applied to stock price prediction

### Related Architectures

- **Temporal Convolutional Networks (TCN)**: Simplified WaveNet for sequence modeling
- **Temporal Fusion Transformers**: Combines attention with temporal patterns
- **N-BEATS**: Interpretable time series forecasting

---

## Usage

### Prerequisites

- Rust 1.70+
- Internet connection for Bybit API

### Quick Start

```bash
cd 351_wavenet_trading/rust

# Build the project
cargo build --release

# Fetch data
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1h --days 30

# Train model (demo)
cargo run --bin train_wavenet

# Generate predictions
cargo run --bin predict

# Run backtest
cargo run --bin backtest
```

---

## File Structure

| File | Description |
|------|-------------|
| `README.md` | This file - main documentation |
| `README.ru.md` | Russian translation |
| `readme.simple.md` | Simple explanation for beginners |
| `readme.simple.ru.md` | Simple explanation in Russian |
| `rust/` | Rust implementation |

---

*WaveNet brings the power of dilated causal convolutions to financial time series, offering a unique blend of long-term pattern recognition and computational efficiency.*
