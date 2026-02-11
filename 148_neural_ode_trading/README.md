# Chapter 148: Neural ODE Trading

## Overview

**Neural Ordinary Differential Equations (Neural ODEs)** represent a paradigm shift in deep learning: instead of stacking discrete layers, we define the network as a continuous dynamical system and solve it with ODE solvers. This chapter explores how Neural ODEs can transform trading by modeling markets as continuous-time systems -- naturally handling irregular timestamps, providing constant-memory training, and enabling smooth interpolation of market dynamics.

The core idea is elegantly simple: replace the discrete residual connection `h_{l+1} = h_l + f(h_l)` with the continuous dynamics `dh/dt = f_theta(h(t), t)`, and solve forward in time to get predictions.

## Why Neural ODEs for Trading?

### The Problem with Discrete Models

Traditional sequence models (LSTMs, GRUs, Transformers) operate on a fixed time grid. Financial data, however, is inherently irregular:

- **Tick data** arrives at unpredictable intervals (milliseconds to seconds)
- **Trading halts** create gaps in observations
- **Different exchanges** report at different frequencies
- **Missing data** from network issues or illiquid markets
- **Multi-timeframe analysis** requires reconciling different sampling rates

Discrete models handle this poorly -- they either require interpolation (introducing artifacts) or padding (wasting computation).

### The Neural ODE Solution

Neural ODEs model the **continuous evolution** of market state:

```
Market state at time t:  h(t)
Dynamics:                dh/dt = f_theta(h(t), t)
Prediction at time T:    h(T) = h(0) + integral_0^T f_theta(h(t), t) dt
```

**Key advantages for trading:**

| Feature | Discrete Models | Neural ODE |
|---------|----------------|------------|
| Irregular timestamps | Requires interpolation | Native support |
| Memory (backprop) | O(L) layers | O(1) via adjoint method |
| Time resolution | Fixed grid | Continuous (any t) |
| Missing data | Needs imputation | Natural handling |
| Multi-step forecast | Autoregressive | Single ODE solve |
| Depth control | Integer layers | Continuous "depth" |

## Mathematical Foundation

### From ResNets to Neural ODEs

A residual network computes:

```
h_{l+1} = h_l + f_theta(h_l, theta_l)      for l = 0, 1, ..., L-1
```

As the number of layers L -> infinity and the step size -> 0, this becomes:

```
dh(t)/dt = f_theta(h(t), t)
```

where:
- `h(t)` is the hidden state at continuous time t
- `f_theta` is a neural network parameterizing the dynamics
- `theta` are the learnable parameters (shared across all "depths")

The output is obtained by solving this Initial Value Problem (IVP):

```
h(T) = h(0) + integral_0^T f_theta(h(t), t) dt
```

### ODE Solvers

The forward pass requires numerically solving the ODE. Common solvers:

#### Euler Method (1st order)
```
y_{n+1} = y_n + h * f(t_n, y_n)
```
Simple but inaccurate. O(h) error. Used mainly for comparison.

#### Runge-Kutta 4 (RK4, 4th order)
```python
k1 = f(t_n, y_n)
k2 = f(t_n + h/2, y_n + h*k1/2)
k3 = f(t_n + h/2, y_n + h*k2/2)
k4 = f(t_n + h, y_n + h*k3)
y_{n+1} = y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4)
```
Good balance of accuracy and efficiency. O(h^4) local error.

#### Dormand-Prince (adaptive, 4th/5th order)
```
- Embedded RK method with 4th and 5th order solutions
- Error estimate = |y5 - y4|
- Step size adaptation: h_new = h * (tol / error)^(1/5)
```
The default solver in `torchdiffeq` (`dopri5`). Adapts step size to maintain accuracy -- takes small steps in regions of rapid change and large steps in smooth regions.

### Adjoint Sensitivity Method

The key innovation enabling practical Neural ODE training. Instead of backpropagating through all solver steps (O(L) memory), we solve an **augmented ODE backward in time**:

```
Forward:  h(T) = ODESolve(f_theta, h(0), [0, T])
Loss:     L = L(h(T))

Adjoint:  a(t) = dL/dh(t)      (sensitivity of loss to state at time t)

Backward ODE:
  da/dt = -a(t)^T * (df/dh)    (adjoint dynamics)

Parameter gradient:
  dL/d_theta = integral_T^0 a(t)^T * (df/d_theta) dt
```

**Memory: O(1)** -- we only need to store the current state, not the entire forward trajectory!

This is computed by solving one augmented ODE backward:

```python
# Augmented state: [h(t), a(t), dL/d_theta]
# Solve backward from T to 0
[h(0), a(0), dL/d_theta] = ODESolve(augmented_dynamics, [h(T), dL/dh(T), 0], [T, 0])
```

### Latent ODEs for Irregular Time Series

The **Latent ODE** (Rubanova et al., 2019) combines Neural ODEs with a VAE framework for irregularly-sampled sequences:

```
Architecture:
1. Encoder RNN:  processes observations backward in time
                 x_T, x_{T-1}, ..., x_1 -> q(z_0 | x_{1:T})

2. Latent ODE:   dz/dt = f_theta(z(t), t)
                 z_0 ~ q(z_0 | x_{1:T})
                 z(t_1), z(t_2), ... = ODESolve(f_theta, z_0, [t_1, t_2, ...])

3. Decoder:      x_hat(t_i) = g_phi(z(t_i))

Loss = Reconstruction + KL(q(z_0|x) || p(z_0))
```

**Why this matters for trading:**
- The encoder processes historical market data of any length and irregularity
- The latent ODE captures smooth market dynamics
- The decoder reconstructs/predicts at any desired time point
- The VAE framework provides uncertainty estimates

### ODE-RNN Hybrid

The **ODE-RNN** combines continuous ODE dynamics with discrete RNN updates:

```
For each observation (x_i, t_i):
  1. Between observations: h(t_{i-1}) -> h(t_i^-) via ODE
     dh/dt = f_theta(h, t)    for t in [t_{i-1}, t_i]

  2. At observation:        h(t_i^-) -> h(t_i) via RNN
     h(t_i) = GRU(x_i, h(t_i^-))
```

**Intuition for trading:**
- Between trades: market state evolves smoothly (the ODE captures this)
- At each trade: new information arrives and we update our belief (the RNN captures this)

### Continuous Normalizing Flows

Neural ODEs can define **Continuous Normalizing Flows (CNFs)** for density estimation:

```
Transform: z(0) ~ p_0(z)  (simple base distribution, e.g., Gaussian)
           dz/dt = f_theta(z(t), t)
           z(1) = x  (complex data distribution)

Log-probability:
  log p(x) = log p_0(z(0)) - integral_0^1 tr(df/dz) dt
```

**For trading:** Model the full distribution of returns, not just the mean. Enables:
- VaR (Value at Risk) estimation
- Tail risk analysis
- Option pricing

## Implementation

### Python Implementation

Our Python implementation uses `torchdiffeq` (Chen et al., 2018) for ODE solving with automatic differentiation:

#### Neural ODE Model (`python/neural_ode.py`)

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

class ODEFunc(nn.Module):
    """Dynamics: dh/dt = f_theta(h(t), t)"""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.nfe = 0  # Track function evaluations

    def forward(self, t, h):
        self.nfe += 1
        t_expand = t.expand(h.shape[0], 1)
        h_aug = torch.cat([h, t_expand], dim=-1)
        return self.net(h_aug)

class NeuralODE(nn.Module):
    """Complete Neural ODE for trading."""

    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.ode_func = ODEFunc(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t_pred=None):
        # Encode sequence to initial state
        _, h0 = self.encoder(x)
        h0 = h0.squeeze(0)

        # Solve ODE forward
        if t_pred is None:
            t_pred = torch.tensor([0.0, 1.0])

        h_trajectory = odeint_adjoint(
            self.ode_func, h0, t_pred,
            method='dopri5', rtol=1e-4, atol=1e-5
        )

        # Decode final state
        return self.decoder(h_trajectory[-1])
```

#### Latent ODE for Irregular Data (`python/latent_ode.py`)

```python
class LatentODE(nn.Module):
    """VAE with Neural ODE decoder for irregular time series."""

    def __init__(self, input_dim=5, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.encoder = RecognitionRNN(input_dim, hidden_dim, latent_dim)
        self.ode_func = ODEFunc(latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x, t_obs, mask=None):
        # 1. Encode observations (backward RNN)
        z0_mean, z0_logvar = self.encoder(x, t_obs, mask)

        # 2. Sample initial latent state
        z0 = self.reparameterize(z0_mean, z0_logvar)

        # 3. Solve latent ODE forward
        z_traj = odeint_adjoint(self.ode_func, z0, t_obs)

        # 4. Decode trajectory
        x_pred = self.decoder(z_traj)

        return x_pred, z0_mean, z0_logvar
```

#### ODE-RNN Hybrid (`python/ode_rnn.py`)

```python
class ODERNN(nn.Module):
    """ODE between observations, GRU at observations."""

    def forward(self, x, t, mask=None):
        h = torch.zeros(batch_size, hidden_dim)

        for i in range(n_obs):
            # 1. Continuous evolution via ODE
            h = odeint(self.ode_func, h, [t[i-1], t[i]])[-1]

            # 2. Discrete update with observation
            h = self.gru_cell(x[:, i], h)

        return self.output_net(h)
```

#### Training with Adjoint Method (`python/train.py`)

```python
# Key: use odeint_adjoint for O(1) memory training
from torchdiffeq import odeint_adjoint

# Training loop
for epoch in range(n_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(batch['x'])
        loss = criterion(pred, batch['target'])
        loss.backward()  # Adjoint method computes gradients!
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

### Rust Implementation

Our Rust implementation provides ODE solvers and a Neural ODE inference engine:

#### RK4 Solver (`rust_neural_ode/src/lib.rs`)

```rust
pub struct RK4Solver;

impl ODESolver for RK4Solver {
    fn solve<F>(&self, f: &F, y0: &Array1<f64>, t_start: f64, t_end: f64, dt: f64) -> Array1<f64>
    where
        F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    {
        let mut t = t_start;
        let mut y = y0.clone();

        while t < t_end - 1e-12 {
            let h = (t_end - t).min(dt);

            let k1 = f(t, &y);
            let k2 = f(t + h * 0.5, &(&y + &(&k1 * (h * 0.5))));
            let k3 = f(t + h * 0.5, &(&y + &(&k2 * (h * 0.5))));
            let k4 = f(t + h, &(&y + &(&k3 * h)));

            y = &y + &((&k1 + &(&k2 * 2.0) + &(&k3 * 2.0) + &k4) * (h / 6.0));
            t += h;
        }
        y
    }
}
```

#### Dormand-Prince Adaptive Solver

```rust
pub struct DormandPrinceSolver {
    pub rtol: f64,
    pub atol: f64,
}

impl ODESolver for DormandPrinceSolver {
    fn solve<F>(&self, f: &F, y0: &Array1<f64>, t_start: f64, t_end: f64, dt: f64) -> Array1<f64>
    where F: Fn(f64, &Array1<f64>) -> Array1<f64> {
        // Adaptive step: accept/reject based on error estimate
        // Step size control: h_new = h * (tol / error)^(1/5)
        // See full implementation in rust_neural_ode/src/lib.rs
    }
}
```

## Trading Applications

### 1. Irregularly Sampled Tick Data Modeling

```python
# Fetch tick data from Bybit (irregular timestamps)
loader = BybitDataLoader()
ticks = loader.fetch_recent_trades("BTCUSDT", limit=1000)

# Tick data has irregular time gaps:
# timestamp_ms    price      size    side   time_delta_ms
# 1706000000100   65432.50   0.001   buy    0.0
# 1706000000342   65433.00   0.005   buy    242.0    <- 242ms gap
# 1706000000343   65432.80   0.010   sell   1.0      <- 1ms gap
# 1706000001567   65435.00   0.100   buy    1224.0   <- 1.2s gap

# ODE-RNN handles this naturally:
model = ODERNN(input_dim=3, hidden_dim=64, output_dim=1)
prediction = model(features, timestamps, mask)
# Between ticks: ODE evolves state continuously
# At each tick: GRU updates state with new information
```

### 2. Continuous-Time Portfolio Dynamics

Model how portfolio weights should evolve continuously:

```python
# Portfolio state: [weight_BTC, weight_ETH, weight_SOL, cash]
# dw/dt = f_theta(w(t), market_state(t), t)

model = NeuralODE(input_dim=12, hidden_dim=64, output_dim=4)

# Predict optimal portfolio weights at any future time
t_rebalance = torch.linspace(0, 1, 10)  # 10 rebalancing points
weight_trajectory = model.predict_trajectory(current_state, t_rebalance)
```

### 3. Latent Factor Evolution

Discover and track latent market factors:

```python
# Latent ODE discovers hidden factors from observed prices
latent_ode = LatentODE(input_dim=5, latent_dim=8, hidden_dim=64)

# Train on historical data
result = latent_ode(observations, timestamps, mask)
z_trajectory = result['z_trajectory']  # (time, batch, 8)

# The 8-dimensional latent space may capture:
# z_0: Overall market trend
# z_1: Volatility regime
# z_2: Momentum factor
# z_3: Mean-reversion factor
# z_4-7: Other latent dynamics
```

### 4. Missing Data Handling

```python
# Financial data often has gaps
mask = torch.ones(batch_size, seq_len, n_features)
mask[:, 30:35, :] = 0  # Simulate 5 minutes of missing data

# Latent ODE handles this naturally:
result = latent_ode(observations, timestamps, mask=mask)
# The ODE integrates smoothly through gaps
# No interpolation artifacts
```

### 5. Crypto Trading with Bybit

```python
# Full pipeline for Bybit crypto trading
from python.data_loader import BybitDataLoader, create_irregular_dataset
from python.neural_ode import NeuralODE
from python.backtest import NeuralODEBacktester

# 1. Fetch data
loader = BybitDataLoader()
df = loader.fetch_klines("BTCUSDT", interval="1", limit=1000)

# 2. Create dataset with irregular timestamps
train_ds, test_ds = create_irregular_dataset(
    symbol="BTCUSDT", source="bybit", seq_len=60
)

# 3. Train model
model = NeuralODE(input_dim=5, hidden_dim=64, output_dim=1)
history = train_neural_ode(model, train_loader, test_loader, epochs=100)

# 4. Backtest
backtester = NeuralODEBacktester(model, initial_capital=100_000)
signals = backtester.generate_signals(test_ds)
metrics = backtester.run_backtest(signals, strategy="momentum")
```

## Comparison: Neural ODE vs RNN vs ResNet

### Conceptual Comparison

```
ResNet (discrete, fixed depth):
  h_0 -> [Layer 1] -> h_1 -> [Layer 2] -> h_2 -> ... -> h_L
  Memory: O(L)
  Depth: fixed integer L

RNN (discrete, variable length):
  x_1 -> [RNN] -> h_1 -> x_2 -> [RNN] -> h_2 -> ... -> h_T
  Memory: O(T)
  Requires fixed time steps

Neural ODE (continuous, adaptive):
  h(0) -> |--ODE solver---| -> h(T)
           ^               ^
         t=0        continuous      t=T
  Memory: O(1) with adjoint method
  Evaluates at any time point
```

### Quantitative Comparison

| Metric | LSTM | GRU | Transformer | Neural ODE | ODE-RNN |
|--------|------|-----|-------------|------------|---------|
| Irregular data | Poor | Poor | Moderate | Excellent | Excellent |
| Memory scaling | O(T) | O(T) | O(T^2) | O(1) | O(T) |
| Long sequences | Moderate | Moderate | Good | Good | Good |
| Training speed | Fast | Fast | Fast | Slower | Slower |
| Uncertainty | No | No | No | Via Latent ODE | Via Bayes |
| Continuous time | No | No | No | Yes | Yes |

### When to Use Neural ODEs

**Use Neural ODEs when:**
- Data has irregular timestamps (tick data, sensor data)
- You need continuous-time predictions
- Memory is a constraint (long sequences)
- You want uncertainty quantification
- Missing data is common

**Use discrete models when:**
- Data is regularly sampled
- Speed is the primary concern
- The problem doesn't require continuous dynamics
- You have limited compute for ODE solving

## ODE Solver Selection Guide

```
Decision Tree:
|
|-- Need guaranteed accuracy?
|   |-- Yes: Dormand-Prince (dopri5) - adaptive step
|   |-- No:  RK4 - fixed step, fast
|
|-- Very long time horizons?
|   |-- Yes: dopri5 with loose tolerances (rtol=1e-3)
|   |-- No:  RK4 with dt=0.01
|
|-- Training or Inference?
|   |-- Training: Use adjoint method (odeint_adjoint)
|   |-- Inference: Regular odeint (faster, no adjoint overhead)
|
|-- Memory constrained?
|   |-- Yes: Adjoint method (O(1) memory)
|   |-- No:  Regular backprop through solver (faster but O(L) memory)
```

### Solver Performance Characteristics

| Solver | Order | Adaptive | Steps for tol=1e-4 | Best For |
|--------|-------|----------|-------------------|----------|
| Euler | 1 | No | ~10000 | Debugging only |
| Midpoint | 2 | No | ~1000 | Simple problems |
| RK4 | 4 | No | ~100 | Fixed-step production |
| Dormand-Prince | 4/5 | Yes | ~20-50 | General purpose |
| Adams | Variable | Yes | ~10-30 | Stiff problems |

## Project Structure

```
148_neural_ode_trading/
|
|-- README.md                          # This file (English)
|-- README.ru.md                       # Russian translation
|-- readme.simple.md                   # Simple explanation (English)
|-- readme.simple.ru.md                # Simple explanation (Russian)
|
|-- python/
|   |-- __init__.py                    # Package initialization
|   |-- requirements.txt               # Dependencies
|   |-- neural_ode.py                  # Neural ODE, ODEFunc, CNF
|   |-- latent_ode.py                  # Latent ODE for irregular series
|   |-- ode_rnn.py                     # ODE-RNN hybrid, GRU-ODE-Bayes
|   |-- train.py                       # Training pipeline
|   |-- data_loader.py                 # Bybit + stock data loaders
|   |-- visualize.py                   # Plotting utilities
|   |-- backtest.py                    # Trading strategy backtesting
|
|-- rust_neural_ode/
|   |-- Cargo.toml                     # Rust dependencies
|   |-- src/
|   |   |-- lib.rs                     # Core library (ODE solvers, model, data)
|   |   |-- bin/
|   |       |-- train.rs               # Training binary
|   |       |-- predict.rs             # Prediction + backtest binary
|   |       |-- fetch_data.rs          # Data fetching binary
|   |-- examples/
|       |-- basic_ode.rs               # Basic ODE solver demonstrations
|       |-- trading_demo.rs            # Complete trading pipeline demo
```

## Quick Start

### Python

```bash
cd 148_neural_ode_trading

# Install dependencies
pip install -r python/requirements.txt

# Train a Neural ODE model
python -m python.train --model neural_ode --symbol BTCUSDT --epochs 100

# Train a Latent ODE (for irregular data)
python -m python.train --model latent_ode --symbol ETHUSDT --epochs 200

# Train an ODE-RNN hybrid
python -m python.train --model ode_rnn --source stock --epochs 150

# Run backtest
python -m python.backtest --model neural_ode --symbol BTCUSDT --strategy momentum
```

### Rust

```bash
cd 148_neural_ode_trading/rust_neural_ode

# Fetch data from Bybit
cargo run --bin fetch_data -- --symbol BTCUSDT --data-type klines --limit 1000

# Fetch tick data (irregular timestamps)
cargo run --bin fetch_data -- --symbol BTCUSDT --data-type ticks --limit 500

# Train model
cargo run --bin train -- --symbol BTCUSDT --epochs 50 --hidden-dim 32

# Run predictions and backtest
cargo run --bin predict -- --symbol BTCUSDT --compare-solvers

# Run examples
cargo run --example basic_ode
cargo run --example trading_demo
```

## Key Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| `hidden_dim` | 32-128 | Larger = more expressive but slower ODE |
| `latent_dim` | 8-32 | For Latent ODE; captures latent factors |
| `n_ode_layers` | 2-4 | Layers in f_theta network |
| `solver` | `dopri5` | Adaptive; use `rk4` for fixed step |
| `rtol` | 1e-3 to 1e-5 | Relative tolerance for adaptive solvers |
| `atol` | 1e-4 to 1e-6 | Absolute tolerance |
| `use_adjoint` | True | O(1) memory; set False for small models |
| `kl_weight` | 0.001-0.1 | Latent ODE: KL divergence weight |
| `activation` | `tanh` | Smooth activation for stable ODE dynamics |

## References

1. **Chen et al.** "Neural Ordinary Differential Equations" (NeurIPS 2018) -- The foundational paper introducing Neural ODEs and the adjoint sensitivity method.

2. **Rubanova et al.** "Latent ODEs for Irregularly-Sampled Time Series" (NeurIPS 2019) -- Extends Neural ODEs to handle irregular timestamps via a VAE framework.

3. **De Brouwer et al.** "GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series" (NeurIPS 2019) -- GRU-style ODE dynamics with Bayesian uncertainty.

4. **Kidger et al.** "Neural Controlled Differential Equations for Irregular Time Series" (NeurIPS 2020) -- Further extension using controlled differential equations.

5. **Grathwohl et al.** "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" (ICLR 2019) -- Continuous normalizing flows for density estimation.

6. **Dupont et al.** "Augmented Neural ODEs" (NeurIPS 2019) -- Addresses limitations of standard Neural ODEs by augmenting the state space.

7. **Norcliffe et al.** "On Second Order Behaviour in Augmented Neural ODEs" (NeurIPS 2020) -- Analysis of Neural ODE dynamics and training stability.

8. **Jia & Benson** "Neural Jump Stochastic Differential Equations" (NeurIPS 2019) -- Extends to jump-diffusion processes relevant for financial modeling.

---

*This chapter is part of the "Machine Learning for Trading" series. All code is contained within this directory and can be run independently.*
