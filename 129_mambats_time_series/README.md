# Chapter 129: MambaTS for Time Series Forecasting

MambaTS (Mamba for Time Series) is an innovative architecture that leverages the Mamba selective state space model for long-term time series forecasting. By combining the efficiency of linear-time sequence modeling with adaptive selection mechanisms, MambaTS achieves state-of-the-art performance on financial time series prediction tasks while maintaining computational efficiency.

## Content

1. [Introduction to MambaTS](#introduction-to-mambats)
2. [Theoretical Foundation](#theoretical-foundation)
   - [State Space Models (SSM)](#state-space-models-ssm)
   - [From S4 to Mamba](#from-s4-to-mamba)
   - [MambaTS Architecture](#mambats-architecture)
3. [Key Components](#key-components)
   - [Variable-Aware Scanning](#variable-aware-scanning)
   - [Temporal Resolution Patching](#temporal-resolution-patching)
   - [Channel Mixing](#channel-mixing)
4. [Implementation](#implementation)
   - [Python Implementation](#python-implementation)
   - [Rust Implementation](#rust-implementation)
5. [Applications in Financial Markets](#applications-in-financial-markets)
6. [Backtesting Framework](#backtesting-framework)
7. [References](#references)

---

## Introduction to MambaTS

Traditional transformer-based models for time series forecasting face significant challenges with long sequences due to their quadratic attention complexity O(n²). MambaTS addresses this limitation by utilizing the Mamba selective state space model, which achieves linear complexity O(n) while maintaining strong modeling capabilities.

### Why MambaTS for Trading?

Financial time series exhibit several characteristics that make MambaTS particularly suitable:

1. **Long-range dependencies**: Stock prices and market indicators often show patterns spanning hundreds of time steps
2. **Multi-scale patterns**: Markets exhibit patterns at different temporal resolutions (intraday, daily, weekly)
3. **Multiple variables**: Trading requires modeling correlations between different assets and features
4. **Real-time requirements**: Low latency inference is crucial for algorithmic trading

---

## Theoretical Foundation

### State Space Models (SSM)

State space models represent a dynamical system through a hidden state that evolves over time:

```
h'(t) = Ah(t) + Bx(t)    # State evolution
y(t)  = Ch(t) + Dx(t)    # Output
```

Where:
- `h(t)` is the hidden state
- `x(t)` is the input
- `y(t)` is the output
- `A`, `B`, `C`, `D` are learnable parameters

For discrete sequences, this becomes:

```
h[k] = Āh[k-1] + B̄x[k]
y[k] = Ch[k] + Dx[k]
```

### From S4 to Mamba

The **Structured State Space (S4)** model introduced efficient computation of SSMs through:
- Diagonal structure of matrix A
- Efficient convolution-based training

**Mamba** improves upon S4 by introducing **selective state spaces**:
- Input-dependent parameters B, C, and Δ (discretization step)
- Content-aware reasoning through selection mechanism
- Hardware-efficient parallel scan algorithm

```python
# Simplified Mamba selection mechanism
def selective_ssm(x, A, B, C, delta):
    # B, C, delta are now functions of input x
    B = linear_B(x)
    C = linear_C(x)
    delta = softplus(linear_delta(x))

    # Discretization with input-dependent step
    A_bar = exp(delta * A)
    B_bar = delta * B

    # Selective scan
    h = parallel_scan(A_bar, B_bar, x)
    y = einsum('bln,bln->bl', C, h)
    return y
```

### MambaTS Architecture

MambaTS extends Mamba for multivariate time series through several innovations:

```
Input: X ∈ R^(B×L×C)  [batch, sequence_length, channels]
          ↓
┌─────────────────────────────────────┐
│     Temporal Patch Embedding         │
│  Split sequence into patches         │
│  X_patch ∈ R^(B×N×P×C)              │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│    Variable-Aware Scanning           │
│  Process each variable with SSM      │
│  Capture cross-variable patterns     │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│    Mamba Encoder Layers (×N)         │
│  • Selective SSM                     │
│  • Gated Linear Units                │
│  • Layer Normalization               │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│    Prediction Head                   │
│  Project to forecast horizon         │
│  Y ∈ R^(B×H×C)                      │
└─────────────────────────────────────┘
```

---

## Key Components

### Variable-Aware Scanning

MambaTS processes multivariate time series by considering both temporal and cross-variable dependencies:

```python
class VariableAwareScanning(nn.Module):
    """
    Scans across both time and variable dimensions
    to capture complex correlations
    """
    def __init__(self, d_model, n_variables):
        super().__init__()
        self.temporal_mamba = MambaBlock(d_model)
        self.variable_mamba = MambaBlock(d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # x: [batch, seq_len, n_vars, d_model]
        B, L, V, D = x.shape

        # Temporal scanning
        x_temp = rearrange(x, 'b l v d -> (b v) l d')
        h_temp = self.temporal_mamba(x_temp)
        h_temp = rearrange(h_temp, '(b v) l d -> b l v d', b=B, v=V)

        # Variable scanning
        x_var = rearrange(x, 'b l v d -> (b l) v d')
        h_var = self.variable_mamba(x_var)
        h_var = rearrange(h_var, '(b l) v d -> b l v d', b=B, l=L)

        # Fusion
        return self.fusion(torch.cat([h_temp, h_var], dim=-1))
```

### Temporal Resolution Patching

Instead of processing individual time steps, MambaTS groups time steps into patches:

```python
class TemporalPatchEmbedding(nn.Module):
    """
    Converts time series into patches for efficient processing
    """
    def __init__(self, patch_size, d_model, n_variables):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * n_variables, d_model)
        self.position_embedding = nn.Embedding(512, d_model)

    def forward(self, x):
        # x: [batch, seq_len, n_vars]
        B, L, V = x.shape

        # Create patches
        n_patches = L // self.patch_size
        x = x[:, :n_patches * self.patch_size, :]
        x = x.reshape(B, n_patches, self.patch_size, V)
        x = x.reshape(B, n_patches, self.patch_size * V)

        # Project and add position
        x = self.projection(x)
        positions = torch.arange(n_patches, device=x.device)
        x = x + self.position_embedding(positions)

        return x
```

### Channel Mixing

MambaTS employs a channel-mixing strategy to model dependencies between different financial variables:

```python
class ChannelMixing(nn.Module):
    """
    Mixes information across channels (variables)
    """
    def __init__(self, n_channels, expansion_factor=2):
        super().__init__()
        hidden_dim = n_channels * expansion_factor
        self.fc1 = nn.Linear(n_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [batch, seq_len, n_channels]
        return self.fc2(self.activation(self.fc1(x)))
```

---

## Implementation

### Python Implementation

The Python implementation uses PyTorch and includes:

- `python/model.py` - MambaTS model architecture
- `python/train.py` - Training loop with mixed precision
- `python/backtest.py` - Backtesting framework
- `python/data_loader.py` - Data loading for financial data

**Quick Start:**

```python
from model import MambaTS
from data_loader import FinancialDataLoader

# Initialize model
model = MambaTS(
    n_variables=10,        # OHLCV + technical indicators
    d_model=256,           # Hidden dimension
    n_layers=4,            # Number of Mamba layers
    patch_size=16,         # Temporal patch size
    forecast_horizon=24    # Predict 24 steps ahead
)

# Load data
loader = FinancialDataLoader(
    symbols=['BTCUSDT', 'ETHUSDT'],
    interval='1h',
    lookback=720,          # 30 days of hourly data
    source='bybit'
)

# Train
train_loader, val_loader = loader.get_dataloaders(batch_size=32)
trainer = MambaTSTrainer(model, lr=1e-4)
trainer.fit(train_loader, val_loader, epochs=100)
```

### Rust Implementation

The Rust implementation provides high-performance inference for production trading systems:

- `src/model/` - MambaTS model components
- `src/data/` - Data loading and preprocessing
- `src/api/` - Bybit API integration
- `examples/` - Example usage scripts

**Quick Start:**

```rust
use mambats_time_series::{MambaTS, BybitClient, Interval};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Bybit client
    let client = BybitClient::new();

    // Fetch historical data
    let candles = client
        .get_klines("BTCUSDT", Interval::OneHour, start, end)
        .await?;

    // Load trained model
    let model = MambaTS::load("models/mambats_btc.bin")?;

    // Make prediction
    let features = prepare_features(&candles);
    let prediction = model.predict(&features)?;

    println!("Predicted price change: {:.4}%", prediction * 100.0);
    Ok(())
}
```

---

## Applications in Financial Markets

### 1. Price Direction Prediction

MambaTS excels at predicting whether prices will go up or down:

```python
# Binary classification for price direction
model = MambaTS(
    n_variables=20,
    d_model=128,
    n_layers=3,
    output_dim=2,  # Up/Down
    task='classification'
)
```

### 2. Volatility Forecasting

Predict future volatility for options pricing and risk management:

```python
# Regression for volatility
model = MambaTS(
    n_variables=10,
    d_model=256,
    n_layers=4,
    output_dim=1,  # Single volatility value
    task='regression'
)
```

### 3. Multi-horizon Forecasting

Predict prices at multiple future time steps:

```python
# Multi-step forecasting
model = MambaTS(
    n_variables=5,
    d_model=256,
    n_layers=4,
    forecast_horizons=[1, 6, 12, 24],  # 1h, 6h, 12h, 24h
    task='multi_horizon'
)
```

---

## Backtesting Framework

The chapter includes a comprehensive backtesting framework to evaluate trading strategies:

```python
from backtest import Backtester, Strategy

class MambaTSStrategy(Strategy):
    def __init__(self, model, threshold=0.02):
        self.model = model
        self.threshold = threshold

    def generate_signals(self, data):
        predictions = self.model.predict(data)

        signals = np.zeros(len(predictions))
        signals[predictions > self.threshold] = 1   # Long
        signals[predictions < -self.threshold] = -1  # Short

        return signals

# Run backtest
backtester = Backtester(
    strategy=MambaTSStrategy(model),
    initial_capital=100000,
    commission=0.001
)

results = backtester.run(test_data)
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Total Return: {results.total_return:.2%}")
```

### Performance Metrics

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted return (target > 1.5) |
| Sortino Ratio | Downside risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit / Gross loss |

---

## Directory Structure

```
129_mambats_time_series/
├── README.md                 # This file
├── README.ru.md              # Russian documentation
├── README.specify.md         # Technical specification
├── readme.simple.md          # Simplified English explanation
├── readme.simple.ru.md       # Simplified Russian explanation
├── python/
│   ├── __init__.py
│   ├── model.py              # MambaTS model
│   ├── mamba_block.py        # Core Mamba components
│   ├── train.py              # Training script
│   ├── backtest.py           # Backtesting framework
│   ├── data_loader.py        # Financial data loading
│   └── requirements.txt      # Python dependencies
├── src/                      # Rust source code
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   ├── mamba.rs          # Mamba block
│   │   └── mambats.rs        # Full model
│   ├── data/
│   │   ├── mod.rs
│   │   ├── processor.rs      # Data preprocessing
│   │   └── features.rs       # Technical indicators
│   └── api/
│       ├── mod.rs
│       └── bybit.rs          # Bybit API client
├── examples/
│   ├── train_model.rs        # Training example
│   ├── predict.rs            # Inference example
│   └── backtest.rs           # Backtesting example
└── Cargo.toml                # Rust dependencies
```

---

## References

1. **MambaTS: Improved Selective State Space Models for Long-term Forecasting**
   - URL: https://arxiv.org/abs/2405.16440
   - Year: 2024

2. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**
   - Authors: Albert Gu, Tri Dao
   - URL: https://arxiv.org/abs/2312.00752
   - Year: 2023

3. **Efficiently Modeling Long Sequences with Structured State Spaces (S4)**
   - Authors: Albert Gu et al.
   - URL: https://arxiv.org/abs/2111.00396
   - Year: 2021

4. **Are Transformers Effective for Time Series Forecasting?**
   - URL: https://arxiv.org/abs/2205.13504
   - Year: 2022

5. **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting**
   - URL: https://arxiv.org/abs/2012.07436
   - Year: 2020

---

## Getting Started

### Python Setup

```bash
cd python
pip install -r requirements.txt

# Train model
python train.py --data bybit --symbol BTCUSDT --epochs 100

# Run backtest
python backtest.py --model models/mambats.pt --test-period 2024-01-01:2024-06-01
```

### Rust Setup

```bash
# Build
cargo build --release

# Run examples
cargo run --example train_model
cargo run --example predict
cargo run --example backtest
```

---

## License

This implementation is provided for educational purposes as part of the Machine Learning for Trading book.
