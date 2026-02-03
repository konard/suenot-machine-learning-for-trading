# Chapter 136: Hierarchical State Space Models (HiSS) for Trading

## Overview

Hierarchical State Space Models (HiSS) stack deep state space models (such as S4 and Mamba) to capture multi-scale temporal patterns in sequential data. Originally proposed for continuous sequence-to-sequence prediction in robotics and sensor fusion, HiSS is highly applicable to financial markets where price dynamics evolve across multiple time horizons simultaneously — from tick-level microstructure to weekly macro trends.

This chapter implements HiSS for cryptocurrency trading on Bybit and stock market prediction, demonstrating how hierarchical temporal reasoning outperforms flat SSM architectures for financial forecasting.

## Table of Contents

1. [Theoretical Foundations](#theoretical-foundations)
2. [HiSS Architecture](#hiss-architecture)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Implementation](#implementation)
5. [Trading Strategy](#trading-strategy)
6. [Results and Metrics](#results-and-metrics)
7. [References](#references)

## Theoretical Foundations

### State Space Models (SSMs)

A continuous-time state space model maps an input signal `u(t)` to an output `y(t)` through a latent state `x(t)`:

```
x'(t) = A·x(t) + B·u(t)
y(t)  = C·x(t) + D·u(t)
```

Where:
- `A ∈ ℝ^{N×N}` is the state transition matrix
- `B ∈ ℝ^{N×1}` is the input projection matrix
- `C ∈ ℝ^{1×N}` is the output projection matrix
- `D ∈ ℝ` is the direct feedthrough (skip connection)

For discrete sequences, the continuous system is discretized using a step size `Δ`:

```
x_k = Ā·x_{k-1} + B̄·u_k
y_k = C·x_k + D·u_k
```

Where `Ā = exp(ΔA)` and `B̄ = (ΔA)^{-1}(exp(ΔA) - I)·ΔB`.

### Why Hierarchical?

Financial markets exhibit multi-scale dynamics:

| Time Scale | Pattern Type | Example |
|---|---|---|
| Seconds–Minutes | Microstructure noise, order flow | Bid-ask bounce, trade clustering |
| Minutes–Hours | Intraday trends, mean reversion | Session momentum, lunch dip |
| Hours–Days | Swing patterns, event reactions | Earnings drift, news impact |
| Days–Weeks | Trend/momentum regimes | Sector rotation, macro trends |

A flat SSM with a single temporal resolution struggles to simultaneously capture both high-frequency microstructure and low-frequency regime shifts. HiSS addresses this by stacking SSM layers at different temporal granularities.

### HiSS: Hierarchical State Space Models

The key insight of HiSS (Bhirangi et al., 2024) is to create a hierarchy of SSM layers where each level operates at a different temporal resolution:

1. **Level 0 (Finest)**: Processes raw input at full resolution
2. **Level 1**: Downsamples by factor `k₁`, captures medium-scale patterns
3. **Level 2**: Downsamples by factor `k₂`, captures coarse patterns
4. **...and so on**

Each level feeds its output to the next level (after downsampling) and receives context from coarser levels (after upsampling). This creates a bidirectional flow of information across temporal scales.

## HiSS Architecture

```
Input Sequence (T timesteps)
    │
    ▼
┌─────────────────────────┐
│  Level 0: SSM Block     │  Full resolution (T steps)
│  (Fine-grained patterns)│
└─────────┬───────────────┘
          │ Downsample (stride k₁)
          ▼
┌─────────────────────────┐
│  Level 1: SSM Block     │  T/k₁ steps
│  (Medium patterns)      │
└─────────┬───────────────┘
          │ Downsample (stride k₂)
          ▼
┌─────────────────────────┐
│  Level 2: SSM Block     │  T/(k₁·k₂) steps
│  (Coarse patterns)      │
└─────────┬───────────────┘
          │ Upsample + Merge
          ▼
┌─────────────────────────┐
│  Fusion & Prediction    │
│  Head                   │
└─────────────────────────┘
    │
    ▼
Output (predictions)
```

### Components

1. **SSM Block**: Each block contains:
   - Layer normalization
   - S4/Mamba SSM layer with learnable `A`, `B`, `C`, `D` parameters
   - Nonlinear activation (GELU)
   - Residual connection
   - Dropout for regularization

2. **Downsampling**: Average pooling or strided convolution to reduce temporal resolution between levels.

3. **Upsampling**: Linear interpolation or transposed convolution to restore resolution for cross-level fusion.

4. **Fusion Module**: Concatenates or adds features from all hierarchy levels to produce the final representation.

## Mathematical Formulation

### Multi-Scale SSM

At hierarchy level `l`, the SSM operates on a sequence of length `T_l`:

```
T_l = T / (∏_{i=1}^{l} k_i)
```

Each level has its own parameters `(A_l, B_l, C_l, D_l)` and discretization step `Δ_l`.

### Cross-Level Information Flow

The output of level `l` is downsampled and added as context to level `l+1`:

```
h_l = SSM_l(Downsample(h_{l-1}))
```

For the final prediction, features from all levels are upsampled to the original resolution and fused:

```
z = Fusion(h_0, Upsample(h_1), Upsample(h_2), ...)
ŷ = PredictionHead(z)
```

### Loss Function for Trading

For multi-task financial prediction:

```
L = λ_ret · MSE(ŷ_ret, y_ret) + λ_dir · BCE(ŷ_dir, y_dir) + λ_vol · MSE(ŷ_vol, y_vol)
```

Where:
- `ŷ_ret`: Predicted returns
- `ŷ_dir`: Predicted direction (up/down)
- `ŷ_vol`: Predicted volatility
- `λ_*`: Task weights (can be learned via uncertainty weighting)

## Implementation

### Python

The Python implementation uses PyTorch and includes:

- **`python/model.py`**: HiSS model with configurable hierarchy depth and downsampling factors
- **`python/data_loader.py`**: Data loading for stock (yfinance) and crypto (Bybit) markets
- **`python/backtest.py`**: Backtesting engine with Sharpe, Sortino, and drawdown metrics

```python
from python.model import HierarchicalSSM

model = HierarchicalSSM(
    input_dim=8,          # OHLCV + technical indicators
    hidden_dim=64,
    output_dim=3,         # return, direction, volatility
    num_levels=3,         # hierarchy depth
    downsample_factors=[4, 4],  # 4x reduction per level
    ssm_state_dim=16,
    dropout=0.1
)
```

### Rust

The Rust implementation provides a production-ready version using the `ndarray` crate:

- **`src/model/`**: Hierarchical SSM with efficient matrix operations
- **`src/data/`**: Bybit API client and feature engineering
- **`src/trading/`**: Signal generation and strategy execution
- **`src/backtest/`**: Performance evaluation engine

```bash
# Run basic example
cargo run --example basic_hiss

# Run trading strategy
cargo run --example trading_strategy

# Run multi-scale analysis
cargo run --example multi_scale
```

## Trading Strategy

### Signal Generation

The HiSS model predicts three targets:
1. **Expected return** (regression): Used for position sizing
2. **Direction probability** (classification): Used for entry/exit signals
3. **Volatility forecast** (regression): Used for risk management

### Entry Rules

- **Long**: Direction probability > 0.6 AND expected return > threshold
- **Short**: Direction probability < 0.4 AND expected return < -threshold
- **Flat**: Otherwise

### Position Sizing

Kelly-criterion inspired sizing based on predicted return and volatility:

```
position_size = (predicted_return / predicted_volatility²) × scale_factor
```

### Risk Management

- Maximum position size: 20% of portfolio
- Stop-loss: 2× predicted volatility
- Take-profit: 3× predicted volatility
- Maximum drawdown limit: 15%

## Results and Metrics

### Evaluation Metrics

| Metric | Description |
|---|---|
| MSE / MAE | Return prediction accuracy |
| Accuracy / F1 | Direction classification performance |
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk-adjusted return |
| Maximum Drawdown | Worst peak-to-trough decline |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit / Gross loss |

### Comparison with Baselines

The hierarchical approach is compared against:
- Flat S4 model (single-scale SSM)
- Flat Mamba model
- LSTM baseline
- Transformer baseline
- Buy-and-hold benchmark

### Multi-Scale Advantage

HiSS captures patterns that flat models miss:
- **Level 0** (fine): Detects order flow imbalances and microstructure signals
- **Level 1** (medium): Captures intraday momentum and mean reversion
- **Level 2** (coarse): Identifies regime changes and macro trends

## Project Structure

```
136_hierarchical_ssm/
├── README.md                  # This file
├── README.ru.md               # Russian translation
├── readme.simple.md           # Simplified explanation (English)
├── readme.simple.ru.md        # Simplified explanation (Russian)
├── Cargo.toml                 # Rust project configuration
├── python/
│   ├── __init__.py
│   ├── model.py               # HiSS PyTorch model
│   ├── data_loader.py         # Stock & crypto data loading
│   ├── backtest.py            # Backtesting framework
│   └── requirements.txt       # Python dependencies
├── src/
│   ├── lib.rs                 # Rust library root
│   ├── model/
│   │   ├── mod.rs             # Model module
│   │   └── network.rs         # HiSS network implementation
│   ├── data/
│   │   ├── mod.rs             # Data module
│   │   ├── bybit.rs           # Bybit API client
│   │   └── features.rs        # Feature engineering
│   ├── trading/
│   │   ├── mod.rs             # Trading module
│   │   ├── signals.rs         # Signal generation
│   │   └── strategy.rs        # Trading strategy
│   └── backtest/
│       ├── mod.rs             # Backtest module
│       └── engine.rs          # Backtesting engine
└── examples/
    ├── basic_hiss.rs          # Basic HiSS example
    ├── multi_scale.rs         # Multi-scale analysis
    └── trading_strategy.rs    # Full trading strategy
```

## References

1. **Bhirangi, R., Wang, C., Pattabiraman, V., Majidi, C., Gupta, A., Hellebrekers, T., & Pinto, L. (2024)**. Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling. *arXiv:2402.10211*. https://arxiv.org/abs/2402.10211

2. **Gu, A., Goel, K., & Ré, C. (2022)**. Efficiently Modeling Long Sequences with Structured State Spaces. *ICLR 2022*. https://arxiv.org/abs/2111.00396

3. **Gu, A. & Dao, T. (2023)**. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*. https://arxiv.org/abs/2312.00752

4. **Kendall, A., Gal, Y., & Cipolla, R. (2018)**. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. *CVPR 2018*.

5. **De Prado, M. L. (2018)**. Advances in Financial Machine Learning. *Wiley*.

## License

MIT
