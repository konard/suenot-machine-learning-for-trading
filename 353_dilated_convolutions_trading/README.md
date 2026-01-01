# Chapter 353: Dilated Convolutions for Trading

## Overview

**Dilated Convolutions** (also known as atrous convolutions) are a powerful technique for processing sequential data that allows neural networks to have an exponentially large receptive field without increasing the number of parameters or losing resolution. In trading, this enables models to capture both short-term patterns and long-term dependencies simultaneously.

### Why Dilated Convolutions for Trading?

1. **Multi-scale pattern recognition**: Capture micro-structure (tick-level) and macro-trends (weekly) in one model
2. **Computational efficiency**: Exponentially growing receptive field with linear parameter growth
3. **No information loss**: Unlike pooling, maintains full temporal resolution
4. **Parallelizable**: Unlike RNNs, can process entire sequences in parallel
5. **Causal modeling**: Can be configured for online prediction (no future information leakage)

### Key Advantages Over Traditional Approaches

| Approach | Receptive Field | Parameters | Parallelization | Information Loss |
|----------|-----------------|------------|-----------------|------------------|
| Dense Layers | Limited | O(n²) | Yes | Yes |
| Standard CNN | Linear | O(k) | Yes | Optional |
| RNN/LSTM | Unlimited | O(1) | No | Gradient decay |
| **Dilated CNN** | Exponential | O(log n) | Yes | No |

## Theoretical Foundation

### Standard Convolution vs Dilated Convolution

**Standard 1D Convolution** with kernel size k:
```
y[t] = Σᵢ w[i] · x[t - i]  for i ∈ [0, k-1]
```

**Dilated Convolution** with kernel size k and dilation rate d:
```
y[t] = Σᵢ w[i] · x[t - i·d]  for i ∈ [0, k-1]
```

The dilation rate `d` introduces gaps between kernel elements, allowing the convolution to "skip" over input values.

### Receptive Field Growth

For a stack of L layers with kernel size k and dilation rates d₁, d₂, ..., dₗ:

**Receptive Field = 1 + Σᵢ (k - 1) × dᵢ**

With exponentially increasing dilation (d = 1, 2, 4, 8, ...):
- Layer 1: d=1 → receptive field = k
- Layer 2: d=2 → receptive field = k + 2(k-1) = 3k - 2
- Layer 3: d=4 → receptive field = 3k - 2 + 4(k-1) = 7k - 6
- Layer L: receptive field ≈ 2^L × k

### WaveNet Architecture

The seminal WaveNet architecture uses:
1. **Causal convolutions**: Only uses past information
2. **Dilated convolutions**: Exponentially increasing dilation
3. **Residual connections**: For gradient flow
4. **Gated activations**: tanh(Wf * x) ⊙ σ(Wg * x)

```
                    ┌─────────────────────────────────┐
                    │         Output Layer            │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │   Residual Block (d=8)          │
                    │ ┌─────────┐   ┌─────────┐       │
                    │ │ Dilated │   │ 1×1     │       │
                 ┌──┼─│ Conv    ├───│ Conv    ├───────┼──►
                 │  │ └─────────┘   └─────────┘       │
                 │  └─────────────────────────────────┘
                 │                  │
                 │  ┌─────────────▼───────────────────┐
                 │  │   Residual Block (d=4)          │
                 │  │ ┌─────────┐   ┌─────────┐       │
                 ├──┼─│ Dilated │   │ 1×1     │       │
                 │  │ │ Conv    ├───│ Conv    ├───────┼──►
                 │  │ └─────────┘   └─────────┘       │
                 │  └─────────────────────────────────┘
                 │                  │
                 │  ┌─────────────▼───────────────────┐
                 │  │   Residual Block (d=2)          │
                 │  │ ┌─────────┐   ┌─────────┐       │
                 ├──┼─│ Dilated │   │ 1×1     │       │
                 │  │ │ Conv    ├───│ Conv    ├───────┼──►
                 │  │ └─────────┘   └─────────┘       │
                 │  └─────────────────────────────────┘
                 │                  │
                 │  ┌─────────────▼───────────────────┐
                 │  │   Residual Block (d=1)          │
                 │  │ ┌─────────┐   ┌─────────┐       │
                 └──┼─│ Dilated │   │ 1×1     │       │
                    │ │ Conv    ├───│ Conv    ├───────┼──►
                    │ └─────────┘   └─────────┘       │
                    └─────────────────────────────────┘
                                  ▲
                    ┌─────────────┴───────────────────┐
                    │         Input Layer             │
                    │  (Price, Volume, Features)      │
                    └─────────────────────────────────┘
```

## Trading Strategy

### Strategy Description

Use dilated causal convolutions to predict:
1. **Direction**: Next period price movement (up/down/neutral)
2. **Magnitude**: Expected return
3. **Volatility**: Risk level for position sizing

### Multi-Scale Feature Extraction

The key insight is that different dilation rates capture different time scales:

| Dilation Rate | Kernel=3 Receptive Field | Trading Interpretation |
|---------------|--------------------------|------------------------|
| d=1 | 3 bars | Tick-level patterns |
| d=2 | 7 bars | Short-term momentum |
| d=4 | 15 bars | Intraday trends |
| d=8 | 31 bars | Daily patterns |
| d=16 | 63 bars | Weekly cycles |
| d=32 | 127 bars | Monthly trends |

### Input Features

```
For each timestep t:
- price_returns[t] = (close[t] - close[t-1]) / close[t-1]
- log_volume[t] = log(volume[t] + 1)
- high_low_range[t] = (high[t] - low[t]) / close[t]
- close_position[t] = (close[t] - low[t]) / (high[t] - low[t])
- volume_ma_ratio[t] = volume[t] / SMA(volume, 20)[t]
```

### Architecture for Trading

```python
class DilatedTradingModel:
    def __init__(self,
                 input_channels=5,
                 residual_channels=32,
                 skip_channels=64,
                 n_layers=8,
                 kernel_size=3):

        self.dilation_rates = [2**i for i in range(n_layers)]  # 1,2,4,8,16,32,64,128

        # Input projection
        self.input_conv = CausalConv1d(input_channels, residual_channels, 1)

        # Dilated residual blocks
        self.residual_blocks = [
            DilatedResidualBlock(
                residual_channels,
                skip_channels,
                kernel_size,
                dilation=d
            ) for d in self.dilation_rates
        ]

        # Output layers
        self.output_conv1 = Conv1d(skip_channels, 64, 1)
        self.output_conv2 = Conv1d(64, 3, 1)  # [direction, magnitude, volatility]

    def forward(self, x):
        # x shape: (batch, channels, sequence_length)

        x = self.input_conv(x)

        skip_connections = []
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        # Sum all skip connections
        out = sum(skip_connections)
        out = F.relu(out)
        out = self.output_conv1(out)
        out = F.relu(out)
        out = self.output_conv2(out)

        return out
```

## Implementation Details

### Causal Dilated Convolution

For online prediction, we need **causal** convolutions that only look at past data:

```python
class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Left padding to ensure causality
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding
        )

    def forward(self, x):
        out = self.conv(x)
        # Remove the right padding to make causal
        return out[:, :, :-self.padding] if self.padding > 0 else out
```

### Gated Activation

The gated activation from WaveNet:

```python
class GatedActivation(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.filter_conv = CausalDilatedConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalDilatedConv1d(channels, channels, kernel_size, dilation)

    def forward(self, x):
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        return filter_out * gate_out
```

### Residual Block

```python
class DilatedResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super().__init__()

        self.gated_activation = GatedActivation(residual_channels, kernel_size, dilation)
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(self, x):
        out = self.gated_activation(x)

        skip = self.skip_conv(out)
        residual = self.residual_conv(out) + x

        return residual, skip
```

## Training Pipeline

### Data Preparation

```python
def prepare_training_data(klines, sequence_length=512, forecast_horizon=1):
    """
    Prepare sequences for training.

    Args:
        klines: List of OHLCV candles
        sequence_length: Input sequence length
        forecast_horizon: How many steps ahead to predict

    Returns:
        X: Input sequences (batch, channels, sequence_length)
        y: Target values (batch, 3)  # [direction, magnitude, volatility]
    """
    # Calculate features
    features = calculate_features(klines)

    # Create sequences
    X, y = [], []
    for i in range(len(features) - sequence_length - forecast_horizon):
        X.append(features[i:i+sequence_length])

        # Target: next period return
        future_return = (klines[i+sequence_length+forecast_horizon-1].close -
                        klines[i+sequence_length-1].close) / klines[i+sequence_length-1].close

        direction = 1 if future_return > 0.001 else (-1 if future_return < -0.001 else 0)
        magnitude = abs(future_return)
        volatility = calculate_volatility(klines[i:i+sequence_length])

        y.append([direction, magnitude, volatility])

    return np.array(X), np.array(y)
```

### Loss Function

```python
class TradingLoss(nn.Module):
    def __init__(self, direction_weight=1.0, magnitude_weight=0.5, volatility_weight=0.3):
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.volatility_weight = volatility_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # pred shape: (batch, 3, 1) - last timestep prediction
        # target shape: (batch, 3)

        direction_loss = self.ce_loss(pred[:, :3, -1], target[:, 0].long() + 1)
        magnitude_loss = self.mse_loss(pred[:, 3, -1], target[:, 1])
        volatility_loss = self.mse_loss(pred[:, 4, -1], target[:, 2])

        return (self.direction_weight * direction_loss +
                self.magnitude_weight * magnitude_loss +
                self.volatility_weight * volatility_loss)
```

## Key Metrics

### Model Performance
| Metric | Description |
|--------|-------------|
| **Direction Accuracy** | Percentage of correct direction predictions |
| **Magnitude MAE** | Mean absolute error of return prediction |
| **Sharpe Ratio** | Risk-adjusted return of trading strategy |
| **Max Drawdown** | Maximum peak-to-trough decline |
| **Win Rate** | Percentage of profitable trades |

### Receptive Field Analysis
| Metric | Description |
|--------|-------------|
| **Effective RF** | Actual receptive field size in timesteps |
| **RF Utilization** | How much of RF is actively used |
| **Multi-scale Contribution** | Importance of each dilation level |

## Dependencies

```toml
[dependencies]
# HTTP client for Bybit API
reqwest = { version = "0.11", features = ["json", "rustls-tls"] }

# Async runtime
tokio = { version = "1.0", features = ["full"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Math and statistics
ndarray = "0.15"
ndarray-stats = "0.5"

# Time handling
chrono = { version = "0.4", features = ["serde"] }

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

## Expected Outcomes

1. **Dilated Convolution Module**: Pure Rust implementation with configurable dilation rates
2. **WaveNet-style Architecture**: Complete residual block implementation
3. **Bybit Integration**: Real-time data fetching and feature calculation
4. **Trading Strategy**: Signal generation with position sizing
5. **Backtesting Framework**: Performance evaluation on historical data

## Project Structure

```
353_dilated_convolutions_trading/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Simple explanation
├── readme.simple.ru.md          # Simple explanation (Russian)
└── rust/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               # Library root
    │   ├── api/                 # Bybit API client
    │   │   ├── mod.rs
    │   │   ├── client.rs
    │   │   ├── types.rs
    │   │   └── error.rs
    │   ├── conv/                # Dilated convolutions
    │   │   ├── mod.rs
    │   │   ├── dilated.rs
    │   │   ├── causal.rs
    │   │   └── wavenet.rs
    │   ├── features/            # Feature engineering
    │   │   ├── mod.rs
    │   │   ├── technical.rs
    │   │   └── normalization.rs
    │   ├── strategy/            # Trading strategy
    │   │   ├── mod.rs
    │   │   ├── signals.rs
    │   │   └── position.rs
    │   └── utils/               # Utilities
    │       ├── mod.rs
    │       └── metrics.rs
    └── examples/
        ├── fetch_data.rs        # Fetch Bybit data
        ├── dilated_conv_demo.rs # Demo dilated convolutions
        ├── wavenet_features.rs  # WaveNet feature extraction
        └── trading_backtest.rs  # Backtesting example
```

## References

### Academic Papers
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) - Original WaveNet paper
- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) - Dilated convolutions for semantic segmentation
- [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) - TCN paper

### Trading Applications
- [Deep Learning for Financial Time Series Prediction](https://arxiv.org/abs/2101.09187)
- [WaveNet-based Trading Signal Generation](https://arxiv.org/abs/2003.06503)

### Documentation
- [Bybit API Documentation](https://bybit-exchange.github.io/docs/v5/intro)

## Difficulty Level

⭐⭐⭐⭐ (Advanced)

### Required Knowledge
- **Deep Learning**: CNN architectures, residual connections
- **Signal Processing**: Convolution operations, receptive fields
- **Time Series Analysis**: Feature engineering, stationarity
- **Financial Markets**: OHLCV data, trading signals
- **Rust Programming**: Async programming, error handling
