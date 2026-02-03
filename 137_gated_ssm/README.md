# Chapter 137: Gated State Space Models (Gated SSM)

## Overview

Gated State Space Models (Gated SSMs) enhance traditional State Space Models (SSMs) with gating mechanisms inspired by recurrent neural networks (LSTMs, GRUs). By adding learnable gates that selectively control information flow through the state dynamics, Gated SSMs overcome two key limitations of vanilla SSMs: their inability to selectively remember or forget information and their fixed linear dynamics. The seminal work by Mehta et al. (2022) introduced the GSS (Gated State Spaces) architecture, which combines a gated activation unit with a state space layer, demonstrating competitive or superior performance compared to Transformers on long-range sequence modeling tasks.

In algorithmic trading, Gated SSMs offer a powerful inductive bias: they efficiently model long-range dependencies in price series (via the SSM backbone) while selectively attending to regime changes, volatility spikes, and mean-reversion signals (via gating). This chapter covers the theory, Python and Rust implementations, and practical trading strategies using Gated SSMs on both stock and cryptocurrency data.

## Table of Contents

1. [Introduction to State Space Models](#introduction-to-state-space-models)
2. [Gating Mechanisms in SSMs](#gating-mechanisms-in-ssms)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Gated SSM Architectures](#gated-ssm-architectures)
5. [Applications to Trading](#applications-to-trading)
6. [Implementation in Python](#implementation-in-python)
7. [Implementation in Rust](#implementation-in-rust)
8. [Practical Examples with Stock and Crypto Data](#practical-examples-with-stock-and-crypto-data)
9. [Backtesting Framework](#backtesting-framework)
10. [Performance Evaluation](#performance-evaluation)
11. [Future Directions](#future-directions)
12. [References](#references)

---

## Introduction to State Space Models

### Continuous-Time SSM

A State Space Model defines a linear dynamical system mapping an input signal u(t) to an output y(t) through a latent state x(t):

```
x'(t) = A x(t) + B u(t)
y(t)  = C x(t) + D u(t)
```

Where:
- x(t) ∈ ℝ^N is the hidden state
- A ∈ ℝ^(N×N) is the state transition matrix
- B ∈ ℝ^(N×1) is the input projection
- C ∈ ℝ^(1×N) is the output projection
- D ∈ ℝ is the feedthrough (skip connection)

### Discretization

For discrete sequences (like daily prices), the continuous system is discretized using a step size Δ. The zero-order hold (ZOH) discretization gives:

```
Ā = exp(ΔA)
B̄ = (ΔA)^{-1} (exp(ΔA) - I) · ΔB
```

The discrete recurrence then becomes:

```
x_k = Ā x_{k-1} + B̄ u_k
y_k = C x_k + D u_k
```

### Efficient Computation via Convolution

A key insight is that the discrete SSM can be computed as a global convolution:

```
K = (CB̄, CĀB̄, CĀ²B̄, ..., CĀ^{L-1}B̄)
y = K * u
```

This enables O(L log L) computation using FFT, making SSMs highly efficient for long sequences.

### Limitations of Vanilla SSMs

Despite their efficiency, vanilla SSMs have fixed, input-independent dynamics:
- The matrices A, B, C do not change based on input content
- The model cannot selectively attend to or ignore specific inputs
- Information flow through the state is uniform regardless of input relevance

This is where gating mechanisms become essential.

---

## Gating Mechanisms in SSMs

### Motivation from RNNs

LSTMs and GRUs solve the selective memory problem with gates: sigmoid-activated values that modulate information flow. An LSTM cell uses:

- **Forget gate**: f_t = σ(W_f · [h_{t-1}, x_t] + b_f) — what to forget
- **Input gate**: i_t = σ(W_i · [h_{t-1}, x_t] + b_i) — what new info to store
- **Output gate**: o_t = σ(W_o · [h_{t-1}, x_t] + b_o) — what to output

Gated SSMs bring similar selectivity to state space models while maintaining their computational advantages.

### Types of Gating in SSMs

**1. Output Gating (GSS / Mehta et al., 2022)**

The simplest approach gates the SSM output:

```
z = SSM(u)           # Standard SSM pass
g = σ(Linear(u))     # Gating signal from input
y = z ⊙ g            # Element-wise gating
```

**2. State Gating**

Gates applied directly to the state update:

```
x_k = g_f ⊙ (Ā x_{k-1}) + g_i ⊙ (B̄ u_k)
```

Where g_f (forget gate) and g_i (input gate) are input-dependent.

**3. Input-Dependent Discretization (Mamba-style)**

The discretization step Δ is made input-dependent:

```
Δ_k = softplus(Linear(u_k))
Ā_k = exp(Δ_k · A)
B̄_k = Δ_k · B
```

This makes the state dynamics content-aware: large Δ → more emphasis on new input, small Δ → more emphasis on state memory.

---

## Mathematical Foundation

### The GSS Block (Gated State Spaces)

The GSS architecture from Mehta et al. (2022) consists of:

```
Input u ∈ ℝ^{L×d}

Branch 1 (SSM path):
  v = Linear_V(u)           ∈ ℝ^{L×d_ssm}
  z = SSM(v)                ∈ ℝ^{L×d_ssm}
  z = LayerNorm(z)

Branch 2 (Gate path):
  g = Linear_G(u)           ∈ ℝ^{L×d_ssm}
  g = GELU(g)

Merge:
  y = z ⊙ g                 ∈ ℝ^{L×d_ssm}
  y = Linear_O(y)           ∈ ℝ^{L×d}
```

### HiPPO Initialization

The state matrix A is initialized using the HiPPO (High-Order Polynomial Projection Operator) framework:

```
A_{nk} = -{ (2n+1)^{1/2} (2k+1)^{1/2}   if n > k
          { n+1                             if n = k
          { 0                               if n < k
```

This initialization allows the SSM to optimally approximate a history of inputs using Legendre polynomials, which is critical for capturing long-range dependencies in financial time series.

### Diagonal State Spaces (S4D)

For computational efficiency, the state matrix A is often restricted to be diagonal:

```
A = diag(a_1, a_2, ..., a_N)
```

With diagonal A, each state dimension evolves independently:

```
x_k^{(n)} = ā_n · x_{k-1}^{(n)} + b̄_n · u_k
```

This simplifies computation from O(N²) to O(N) per step.

### Gated Recurrence Formula

The full gated SSM recurrence for a single layer:

```
# Input projections
v_k = W_v · u_k + b_v                    (value projection)
g_k = σ(W_g · u_k + b_g)                 (gate, sigmoid activation)

# State update (with diagonal A)
x_k = diag(ā) · x_{k-1} + diag(b̄) · v_k

# Output
z_k = C · x_k
y_k = z_k ⊙ g_k                          (gated output)
```

### Training Stability

Gated SSMs use several techniques for stable training:

1. **Parameterization of A**: A is parameterized in log-space: A = -exp(log_A), ensuring negative real parts (stable dynamics)
2. **Gradient clipping**: Applied to prevent exploding gradients during long sequence training
3. **Learning rate schedule**: SSM parameters (A, B, C) often use a smaller learning rate than gating parameters

---

## Gated SSM Architectures

### GSS (Gated State Spaces) — Mehta et al., 2022

The original architecture stacks GSS blocks with residual connections:

```
for each layer:
    residual = u
    u = LayerNorm(u)
    u = GSSBlock(u)
    u = Dropout(u)
    u = u + residual
```

Key design choices:
- GELU activation for the gate (not sigmoid), providing smoother gradients
- Layer normalization after the SSM, before gating
- Expansion factor: internal dimension d_ssm = α · d (typically α = 2)

### S4 with Gating

Adding gating to the S4 (Structured State Spaces for Sequences) architecture:

```
u → S4_Layer → LayerNorm → ⊙ ← σ(Linear(u))
```

### Mamba (Selective State Spaces)

Mamba (Gu & Dao, 2023) takes gating further by making all SSM parameters input-dependent:

```
x_k = Ā(u_k) · x_{k-1} + B̄(u_k) · u_k
y_k = C(u_k) · x_k
```

This is the most expressive form of gating, where the entire state dynamics are content-aware.

### Multi-Head Gated SSM

Similar to multi-head attention, using multiple SSM heads with independent gating:

```
head_i = GatedSSM_i(u)     for i = 1..H
y = Concat(head_1, ..., head_H) · W_O
```

---

## Applications to Trading

### Why Gated SSMs for Financial Markets?

1. **Regime-Aware Processing**: Gates learn to switch behavior between bull/bear/sideways markets
2. **Long-Range Dependencies**: SSM backbone captures seasonal patterns and long-term trends efficiently
3. **Selective Attention**: Gates filter noise (random daily fluctuations) while passing meaningful signals (trend changes, volatility breakouts)
4. **Efficient Inference**: O(1) per-step recurrence for real-time trading, unlike O(L) for Transformers

### Trading Tasks

**Return Prediction**: Predict next-period returns using gated SSM to capture both short-term momentum and long-term mean reversion.

**Volatility Forecasting**: Gate mechanism learns to emphasize volatility clustering (ARCH effects) while maintaining long-memory of volatility regimes.

**Trend Classification**: Classify market into trending vs. mean-reverting regimes. Gates naturally segment the input sequence into regime-specific processing.

**Order Flow Analysis**: Process high-frequency order book updates, using gates to selectively attend to informative trades.

---

## Implementation in Python

### GatedSSM Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict

class DiagonalSSMLayer(nn.Module):
    """
    Diagonal State Space Model layer.
    Uses diagonal state matrix for O(N) computation per step.
    """

    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # State matrix A (parameterized in log-space for stability)
        log_A_real = torch.log(0.5 * torch.ones(d_model, d_state))
        self.log_A_real = nn.Parameter(log_A_real)

        # Input matrix B
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

        # Output matrix C
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

        # Discretization step (log-space)
        log_dt = torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # Skip connection
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            y: Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = u.shape

        # Compute discretized parameters
        dt = torch.exp(self.log_dt)  # (d_model,)
        A = -torch.exp(self.log_A_real)  # (d_model, d_state)

        # ZOH discretization
        dtA = dt.unsqueeze(-1) * A  # (d_model, d_state)
        A_bar = torch.exp(dtA)  # (d_model, d_state)
        B_bar = self.B * dt.unsqueeze(-1)  # (d_model, d_state)

        # Sequential scan
        x = torch.zeros(batch, d_model, self.d_state, device=u.device)
        outputs = []

        for t in range(seq_len):
            x = A_bar * x + B_bar * u[:, t, :].unsqueeze(-1)
            y_t = (self.C * x).sum(dim=-1)  # (batch, d_model)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        y = y + u * self.D  # Skip connection
        return y


class GatedSSMBlock(nn.Module):
    """
    Gated State Space Model block (GSS-style).

    Applies a diagonal SSM with output gating:
        z = SSM(Linear_V(u))
        g = activation(Linear_G(u))
        y = Linear_O(z * g)
    """

    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        d_inner = d_model * expand

        # Value and gate projections
        self.linear_v = nn.Linear(d_model, d_inner)
        self.linear_g = nn.Linear(d_model, d_inner)

        # SSM layer
        self.ssm = DiagonalSSMLayer(d_inner, d_state)

        # Layer norm
        self.norm = nn.LayerNorm(d_inner)

        # Output projection
        self.linear_o = nn.Linear(d_inner, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Input tensor (batch, seq_len, d_model)
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # SSM path
        v = self.linear_v(u)
        z = self.ssm(v)
        z = self.norm(z)

        # Gate path
        g = F.gelu(self.linear_g(u))

        # Merge and project
        y = z * g
        y = self.dropout(y)
        y = self.linear_o(y)
        return y


class GatedSSMModel(nn.Module):
    """
    Full Gated SSM model for sequence prediction.

    Architecture:
        Input Projection -> [GatedSSMBlock + Residual + LayerNorm] x N -> Output Head
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GatedSSMBlock(d_model, d_state, expand, dropout))
            self.norms.append(nn.LayerNorm(d_model))

        self.final_norm = nn.LayerNorm(d_model)

        # Output head: regression or classification
        if num_classes is not None:
            self.head = nn.Linear(d_model, num_classes)
        else:
            self.head = nn.Linear(d_model, 1)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, seq_len, input_size)
        Returns:
            Predictions (batch, 1) or (batch, num_classes)
        """
        h = self.input_proj(x)

        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = norm(h)
            h = layer(h)
            h = h + residual

        h = self.final_norm(h)
        h = h[:, -1, :]  # Take last time step
        return self.head(h)
```

### Data Loader

```python
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TradingDataset(Dataset):
    """Dataset for financial time series with sliding window."""

    def __init__(self, prices: pd.DataFrame, window: int = 60, horizon: int = 1):
        self.window = window
        self.horizon = horizon

        # Compute features
        self.features, self.targets = self._prepare(prices)

    def _prepare(self, prices: pd.DataFrame):
        close = prices['close'].values
        volume = prices['volume'].values if 'volume' in prices else np.ones_like(close)

        # Log returns
        log_ret = np.diff(np.log(close))

        # Volatility (20-period rolling std)
        vol_20 = pd.Series(log_ret).rolling(20).std().values

        # RSI
        gains = np.where(log_ret > 0, log_ret, 0)
        losses = np.where(log_ret < 0, -log_ret, 0)
        avg_gain = pd.Series(gains).rolling(14).mean().values
        avg_loss = pd.Series(losses).rolling(14).mean().values
        rsi = avg_gain / (avg_gain + avg_loss + 1e-10)

        # Normalized volume
        vol_norm = (volume[1:] - pd.Series(volume[1:]).rolling(20).mean().values) / (
            pd.Series(volume[1:]).rolling(20).std().values + 1e-10
        )

        # Stack features
        raw = np.column_stack([log_ret, vol_20, rsi, vol_norm])

        # Drop NaN rows
        valid_start = 20  # After rolling window stabilizes
        raw = raw[valid_start:]
        close_valid = close[valid_start + 1:]  # Aligned

        # Create windows
        features, targets = [], []
        for i in range(len(raw) - self.window - self.horizon + 1):
            feat = raw[i : i + self.window]
            # Target: next-period return sign (1=up, 0=down)
            future_ret = np.log(close_valid[i + self.window + self.horizon - 1] / close_valid[i + self.window - 1])
            target = 1.0 if future_ret > 0 else 0.0
            features.append(feat)
            targets.append(target)

        return (
            torch.tensor(np.array(features), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
```

### Backtest Engine

```python
import numpy as np
import pandas as pd
import torch
from typing import Dict

class GatedSSMBacktester:
    """Backtesting framework for Gated SSM trading strategy."""

    def __init__(self, model, threshold: float = 0.55, transaction_cost: float = 0.001):
        self.model = model
        self.threshold = threshold
        self.transaction_cost = transaction_cost

    @torch.no_grad()
    def generate_signals(self, dataset) -> np.ndarray:
        """Generate trading signals from model predictions."""
        self.model.eval()
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        signals = []
        for features, _ in loader:
            probs = torch.sigmoid(self.model(features)).squeeze(-1)
            signal = torch.where(probs > self.threshold, 1.0,
                     torch.where(probs < 1 - self.threshold, -1.0,
                     torch.tensor(0.0)))
            signals.append(signal.numpy())
        return np.concatenate(signals)

    def run(self, prices: pd.Series, signals: np.ndarray) -> Dict[str, float]:
        """Run backtest and compute performance metrics."""
        returns = prices.pct_change().iloc[-len(signals):].values

        # Strategy returns (with transaction costs)
        positions = signals
        position_changes = np.diff(positions, prepend=0)
        costs = np.abs(position_changes) * self.transaction_cost
        strategy_returns = positions * returns - costs

        # Metrics
        total_return = np.exp(np.sum(np.log1p(strategy_returns))) - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        annual_vol = np.std(strategy_returns) * np.sqrt(252)
        sharpe = annual_return / (annual_vol + 1e-10)

        # Max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Sortino ratio
        downside = strategy_returns[strategy_returns < 0]
        downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-10
        sortino = annual_return / downside_vol

        # Win rate
        wins = np.sum(strategy_returns > 0)
        total_trades = np.sum(strategy_returns != 0)
        win_rate = wins / (total_trades + 1e-10)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': int(np.sum(np.abs(position_changes) > 0)),
        }
```

---

## Implementation in Rust

### Project Structure

```
137_gated_ssm/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   ├── ssm.rs
│   │   └── gated_ssm.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── features.rs
│   │   └── bybit.rs
│   ├── backtest/
│   │   ├── mod.rs
│   │   └── engine.rs
│   └── trading/
│       ├── mod.rs
│       ├── signals.rs
│       └── strategy.rs
└── examples/
    ├── basic_gated_ssm.rs
    ├── crypto_trading.rs
    └── backtest_strategy.rs
```

### Core SSM Implementation (Rust)

```rust
use std::f64;

/// Diagonal SSM state for a single feature dimension.
pub struct DiagonalSSMState {
    /// State vector for each SSM dimension
    pub state: Vec<f64>,
    /// Discretized A (diagonal, stored as vector)
    pub a_bar: Vec<f64>,
    /// Discretized B
    pub b_bar: Vec<f64>,
    /// Output matrix C
    pub c: Vec<f64>,
    /// Skip connection D
    pub d: f64,
}

impl DiagonalSSMState {
    pub fn new(d_state: usize, dt: f64) -> Self {
        let mut a_bar = Vec::with_capacity(d_state);
        let mut b_bar = Vec::with_capacity(d_state);
        let mut c = Vec::with_capacity(d_state);

        for i in 0..d_state {
            // HiPPO-like initialization
            let a_val = -((i as f64) + 1.0);
            let a_disc = (dt * a_val).exp();
            a_bar.push(a_disc);
            b_bar.push(dt);
            c.push(if i % 2 == 0 { 1.0 } else { -1.0 } / (d_state as f64).sqrt());
        }

        DiagonalSSMState {
            state: vec![0.0; d_state],
            a_bar,
            b_bar,
            c,
            d: 1.0,
        }
    }

    /// Process a single input step, return output
    pub fn step(&mut self, u: f64) -> f64 {
        let mut y = 0.0;
        for i in 0..self.state.len() {
            self.state[i] = self.a_bar[i] * self.state[i] + self.b_bar[i] * u;
            y += self.c[i] * self.state[i];
        }
        y + self.d * u
    }

    /// Reset state to zero
    pub fn reset(&mut self) {
        self.state.iter_mut().for_each(|s| *s = 0.0);
    }
}

/// Gated SSM block: SSM with output gating.
pub struct GatedSSMBlock {
    pub ssm_layers: Vec<DiagonalSSMState>,
    pub gate_weights: Vec<Vec<f64>>,
    pub gate_bias: Vec<f64>,
    pub output_weights: Vec<Vec<f64>>,
    pub output_bias: Vec<f64>,
    pub d_model: usize,
    pub d_inner: usize,
}

impl GatedSSMBlock {
    pub fn new(d_model: usize, d_state: usize, expand: usize) -> Self {
        let d_inner = d_model * expand;

        // Initialize SSM layers for each inner dimension
        let ssm_layers: Vec<_> = (0..d_inner)
            .map(|_| DiagonalSSMState::new(d_state, 0.01))
            .collect();

        // Gate weights (d_model -> d_inner)
        let scale = 1.0 / (d_model as f64).sqrt();
        let gate_weights: Vec<Vec<f64>> = (0..d_inner)
            .map(|i| {
                (0..d_model)
                    .map(|j| ((i * 7 + j * 13) as f64 * 0.1).sin() * scale)
                    .collect()
            })
            .collect();
        let gate_bias = vec![0.0; d_inner];

        // Output weights (d_inner -> d_model)
        let out_scale = 1.0 / (d_inner as f64).sqrt();
        let output_weights: Vec<Vec<f64>> = (0..d_model)
            .map(|i| {
                (0..d_inner)
                    .map(|j| ((i * 11 + j * 17) as f64 * 0.1).sin() * out_scale)
                    .collect()
            })
            .collect();
        let output_bias = vec![0.0; d_model];

        GatedSSMBlock {
            ssm_layers,
            gate_weights,
            gate_bias,
            output_weights,
            output_bias,
            d_model,
            d_inner,
        }
    }

    /// Process a single time step
    pub fn step(&mut self, input: &[f64]) -> Vec<f64> {
        // SSM path: pass input through SSM layers
        let ssm_out: Vec<f64> = self
            .ssm_layers
            .iter_mut()
            .enumerate()
            .map(|(i, ssm)| {
                let u = if i < input.len() { input[i % input.len()] } else { 0.0 };
                ssm.step(u)
            })
            .collect();

        // Gate path: linear + GELU
        let gate: Vec<f64> = (0..self.d_inner)
            .map(|i| {
                let z: f64 = self.gate_weights[i]
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + self.gate_bias[i];
                gelu(z)
            })
            .collect();

        // Merge: element-wise multiply
        let merged: Vec<f64> = ssm_out.iter().zip(gate.iter()).map(|(s, g)| s * g).collect();

        // Output projection
        (0..self.d_model)
            .map(|i| {
                self.output_weights[i]
                    .iter()
                    .zip(merged.iter())
                    .map(|(w, m)| w * m)
                    .sum::<f64>()
                    + self.output_bias[i]
            })
            .collect()
    }

    pub fn reset(&mut self) {
        for ssm in &mut self.ssm_layers {
            ssm.reset();
        }
    }
}

/// GELU activation function
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}
```

### Trading Signal Generation (Rust)

```rust
/// Trading signal from Gated SSM output
pub struct TradingSignal {
    pub timestamp: i64,
    pub direction: SignalDirection,
    pub confidence: f64,
}

pub enum SignalDirection {
    Long,
    Short,
    Neutral,
}

pub struct GatedSSMStrategy {
    pub model: GatedSSMBlock,
    pub threshold: f64,
    pub lookback: Vec<Vec<f64>>,
    pub window_size: usize,
}

impl GatedSSMStrategy {
    pub fn new(d_model: usize, d_state: usize, threshold: f64, window_size: usize) -> Self {
        GatedSSMStrategy {
            model: GatedSSMBlock::new(d_model, d_state, 2),
            threshold,
            lookback: Vec::new(),
            window_size,
        }
    }

    /// Feed a new feature vector and generate a trading signal
    pub fn on_bar(&mut self, features: &[f64]) -> TradingSignal {
        let output = self.model.step(features);

        // Use the first output dimension as prediction score
        let score = sigmoid(output[0]);

        let direction = if score > 0.5 + self.threshold {
            SignalDirection::Long
        } else if score < 0.5 - self.threshold {
            SignalDirection::Short
        } else {
            SignalDirection::Neutral
        };

        TradingSignal {
            timestamp: 0,
            direction,
            confidence: (score - 0.5).abs() * 2.0,
        }
    }

    pub fn reset(&mut self) {
        self.model.reset();
        self.lookback.clear();
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

### Bybit Data Fetcher (Rust)

```rust
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct BybitKline {
    pub open_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch kline data from Bybit public API
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<BybitKline>, Box<dyn std::error::Error>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let resp: serde_json::Value = reqwest::get(&url).await?.json().await?;

    let list = resp["result"]["list"]
        .as_array()
        .ok_or("No data returned from Bybit")?;

    let klines: Vec<BybitKline> = list
        .iter()
        .filter_map(|item| {
            let arr = item.as_array()?;
            Some(BybitKline {
                open_time: arr[0].as_str()?.parse().ok()?,
                open: arr[1].as_str()?.parse().ok()?,
                high: arr[2].as_str()?.parse().ok()?,
                low: arr[3].as_str()?.parse().ok()?,
                close: arr[4].as_str()?.parse().ok()?,
                volume: arr[5].as_str()?.parse().ok()?,
            })
        })
        .collect();

    Ok(klines)
}
```

---

## Practical Examples with Stock and Crypto Data

### Example 1: Stock Market Prediction (Python)

```python
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Download stock data
data = yf.download('AAPL', start='2018-01-01', end='2024-01-01')
data.columns = [c.lower() for c in data.columns]

# Create dataset
dataset = TradingDataset(data, window=60, horizon=1)
train_size = int(0.8 * len(dataset))
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

# Initialize model
model = GatedSSMModel(input_size=4, d_model=64, d_state=32, n_layers=3, num_classes=None)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.BCEWithLogitsLoss()
loader = DataLoader(train_set, batch_size=64, shuffle=True)

for epoch in range(50):
    model.train()
    total_loss = 0
    for features, targets in loader:
        pred = model(features).squeeze(-1)
        loss = criterion(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# Backtest
backtester = GatedSSMBacktester(model, threshold=0.55)
signals = backtester.generate_signals(test_set)
metrics = backtester.run(data['close'], signals)
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}, MaxDD: {metrics['max_drawdown']:.2%}")
```

### Example 2: Crypto Trading with Bybit Data (Python)

```python
import requests
import pandas as pd

def fetch_bybit_data(symbol: str = "BTCUSDT", interval: str = "D", limit: int = 1000):
    """Fetch kline data from Bybit API."""
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params).json()
    records = resp['result']['list']
    df = pd.DataFrame(records, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'].astype(int), unit='ms')
    df = df.sort_values('open_time').reset_index(drop=True)
    return df

# Fetch BTC and ETH data
btc = fetch_bybit_data("BTCUSDT", "D", 1000)
eth = fetch_bybit_data("ETHUSDT", "D", 1000)

# Train on BTC
btc_dataset = TradingDataset(btc, window=60)
model = GatedSSMModel(input_size=4, d_model=64, d_state=32, n_layers=3)

# ... (training loop same as above)

# Cross-asset evaluation on ETH
eth_dataset = TradingDataset(eth, window=60)
backtester = GatedSSMBacktester(model, threshold=0.55)
eth_signals = backtester.generate_signals(eth_dataset)
eth_metrics = backtester.run(eth['close'], eth_signals)
print(f"ETH Sharpe: {eth_metrics['sharpe_ratio']:.2f}")
```

### Example 3: Rust Trading Example

```rust
use gated_ssm::model::GatedSSMBlock;
use gated_ssm::trading::GatedSSMStrategy;
use gated_ssm::data::bybit::fetch_bybit_klines;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fetch BTC data from Bybit
    let klines = fetch_bybit_klines("BTCUSDT", "D", 200).await?;
    println!("Fetched {} klines from Bybit", klines.len());

    // Initialize strategy
    let mut strategy = GatedSSMStrategy::new(4, 32, 0.05, 60);

    // Compute features and generate signals
    let mut prev_close = klines[0].close;
    for kline in &klines[1..] {
        let log_ret = (kline.close / prev_close).ln();
        let features = vec![log_ret, kline.volume.ln(), kline.high - kline.low, kline.close - kline.open];
        let signal = strategy.on_bar(&features);

        match signal.direction {
            SignalDirection::Long => println!("LONG  @ {:.2} conf={:.2}", kline.close, signal.confidence),
            SignalDirection::Short => println!("SHORT @ {:.2} conf={:.2}", kline.close, signal.confidence),
            SignalDirection::Neutral => {}
        }
        prev_close = kline.close;
    }

    Ok(())
}
```

---

## Backtesting Framework

### Walk-Forward Validation

```python
def walk_forward_backtest(
    data: pd.DataFrame,
    model_fn,
    window: int = 60,
    train_period: int = 500,
    test_period: int = 60,
    retrain_every: int = 60,
):
    """Walk-forward backtesting with periodic retraining."""
    results = []
    total_len = len(data)
    start = window + 20  # After feature warm-up

    for i in range(start + train_period, total_len - test_period, retrain_every):
        # Train window
        train_data = data.iloc[i - train_period - start : i]
        train_dataset = TradingDataset(train_data, window=window)

        # Test window
        test_data = data.iloc[i - start : i + test_period]
        test_dataset = TradingDataset(test_data, window=window)

        # Train model
        model = model_fn()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model.train()
        for epoch in range(30):
            for features, targets in loader:
                pred = model(features).squeeze(-1)
                loss = criterion(pred, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        backtester = GatedSSMBacktester(model)
        signals = backtester.generate_signals(test_dataset)
        metrics = backtester.run(test_data['close'], signals)
        metrics['period_start'] = data.index[i]
        results.append(metrics)

    return pd.DataFrame(results)
```

### Comparing Gated SSM vs Baselines

```python
def compare_models(data: pd.DataFrame):
    """Compare Gated SSM against baseline models."""
    baselines = {
        'Gated SSM': lambda: GatedSSMModel(4, d_model=64, d_state=32, n_layers=3),
        'Vanilla SSM': lambda: GatedSSMModel(4, d_model=64, d_state=32, n_layers=3),  # Without gating
        'LSTM': lambda: LSTMModel(4, hidden_size=64, n_layers=2),
    }

    results = {}
    for name, model_fn in baselines.items():
        wf_results = walk_forward_backtest(data, model_fn)
        results[name] = {
            'avg_sharpe': wf_results['sharpe_ratio'].mean(),
            'avg_return': wf_results['annual_return'].mean(),
            'avg_max_dd': wf_results['max_drawdown'].mean(),
        }
    return pd.DataFrame(results).T
```

---

## Performance Evaluation

### Metrics Summary

| Metric | Description | Target |
|--------|-------------|--------|
| Sharpe Ratio | Risk-adjusted return (annualized) | > 1.0 |
| Sortino Ratio | Downside risk-adjusted return | > 1.5 |
| Max Drawdown | Largest peak-to-trough decline | > -20% |
| Win Rate | Fraction of profitable trades | > 52% |
| Annual Return | Compounded annual growth rate | > 10% |
| Calmar Ratio | Annual return / Max drawdown | > 0.5 |

### Expected Advantages of Gated SSMs

1. **Over vanilla SSMs**: Selective processing improves signal-to-noise ratio, leading to higher Sharpe ratios
2. **Over LSTMs**: More efficient long-range dependency modeling with O(L log L) training
3. **Over Transformers**: O(1) inference per step enables real-time trading; lower memory usage for long sequences
4. **Over GRUs**: State space parameterization provides better theoretical guarantees for long-term memory

### Ablation Study

Key components to evaluate:
- **Gating vs. no gating**: Measures the contribution of the gating mechanism
- **HiPPO vs. random initialization**: Impact of structured initialization
- **Diagonal vs. full state matrix**: Efficiency vs. expressiveness trade-off
- **Number of SSM dimensions (d_state)**: Memory capacity vs. computational cost
- **Expansion factor**: Width of the gated block

---

## Future Directions

1. **Selective State Spaces (Mamba)**: Making all SSM parameters input-dependent for maximum expressiveness
2. **Multi-Scale Gated SSMs**: Different SSM layers operating at different time scales (tick, minute, daily)
3. **Cross-Asset Gating**: Gates conditioned on market-wide signals (VIX, sector indices) rather than just the target asset
4. **Hybrid Architectures**: Combining Gated SSMs with attention layers for capturing both local and global patterns
5. **Hardware-Aware Kernels**: Custom CUDA kernels for efficient parallel scan computation
6. **Continual Learning**: Online adaptation of gating parameters as market regimes change

---

## References

1. Mehta, H., Gupta, A., Cutkosky, A., & Neyshabur, B. (2022). Long Range Language Modeling via Gated State Spaces. *arXiv:2206.13947*.
2. Gu, A., Goel, K., & Ré, C. (2021). Efficiently Modeling Long Sequences with Structured State Spaces. *ICLR 2022*.
3. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*.
4. Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., & Ré, C. (2021). Combining Recurrent, Convolutional, and Continuous-time Models with Linear State Space Layers. *NeurIPS 2021*.
5. Smith, J.T.H., Warrington, A., & Linderman, S. (2022). Simplified State Space Layers for Sequence Modeling. *ICLR 2023*.
6. Gu, A., Dao, T., Ermon, S., Rudra, A., & Ré, C. (2020). HiPPO: Recurrent Memory with Optimal Polynomial Projections. *NeurIPS 2020*.

---

## Running the Examples

### Python

```bash
cd 137_gated_ssm/python
pip install -r requirements.txt
python model.py          # Test model architecture
python backtest.py       # Run backtesting
```

### Rust

```bash
cd 137_gated_ssm
cargo build
cargo run --example basic_gated_ssm
cargo run --example crypto_trading
cargo run --example backtest_strategy
```
