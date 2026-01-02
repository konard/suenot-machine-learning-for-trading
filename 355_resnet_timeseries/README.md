# Chapter 355: ResNet for Time Series — Deep Residual Networks for Financial Forecasting

## Overview

Residual Networks (ResNet) revolutionized image classification by enabling training of very deep networks through skip connections. This chapter adapts ResNet architecture for time series analysis in trading, allowing models to capture both short-term patterns and long-term dependencies in financial data.

**Key Insight:** Traditional deep networks suffer from vanishing gradients and degradation problems. ResNet's skip connections allow gradients to flow directly through the network, enabling effective training of 50+ layer models for complex financial pattern recognition.

## Trading Strategy

**Core Strategy:** Use deep ResNet architecture to learn hierarchical temporal features from OHLCV data and predict price direction with high accuracy.

**Edge Factors:**
1. Deep feature hierarchies capture patterns at multiple time scales
2. Residual connections preserve fine-grained information through deep layers
3. Skip connections act as ensemble of shallow and deep features
4. Robust to market regime changes through learned feature abstractions

**Target Assets:** Cryptocurrency pairs (BTC/USDT, ETH/USDT) from Bybit exchange

## Technical Foundation

### Why ResNet for Time Series?

Traditional CNNs for time series face challenges:
- Vanishing gradients in deep networks
- Loss of fine-grained temporal information
- Difficulty learning identity mappings

ResNet solves these through:

```
Regular Block:    x → [Conv] → [BN] → [ReLU] → [Conv] → [BN] → y

Residual Block:   x → [Conv] → [BN] → [ReLU] → [Conv] → [BN] → (+) → y
                  |                                              ↑
                  └──────────── Skip Connection ─────────────────┘

Output: y = F(x) + x   (where F is the learned residual function)
```

### Mathematical Foundation

For a residual block:
- Input: x
- Target mapping: H(x)
- Residual function: F(x) = H(x) - x
- Output: y = F(x) + x

**Why this works:**
- If identity mapping is optimal, F(x) → 0 is easier to learn than H(x) → x
- Gradients flow directly through skip connections: ∂L/∂x includes direct path
- Network learns residual refinements rather than complete transformations

### Time Series Adaptations

```
1D ResNet Block for Time Series:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Input: [batch, channels, time_steps]                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Conv1D(kernel=3, padding=1) → BatchNorm1D → ReLU   │   │
│  │  Conv1D(kernel=3, padding=1) → BatchNorm1D          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│                        (add) ←── Skip Connection ←── Input  │
│                          ↓                                  │
│                        ReLU                                 │
│                          ↓                                  │
│  Output: [batch, channels, time_steps]                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Architecture Design

### ResNet-18 for Time Series

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ResNet-18 Time Series Architecture                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input Layer                                                         │
│  ─────────────                                                       │
│  [batch, 5, 256]  ← 5 features (OHLCV), 256 time steps              │
│         ↓                                                            │
│  Conv1D(5→64, k=7, s=2, p=3) → BN → ReLU → MaxPool(k=3, s=2, p=1)  │
│         ↓                                                            │
│  [batch, 64, 64]                                                     │
│                                                                      │
│  Layer 1 (64 channels)                                               │
│  ─────────────────────                                               │
│  ResBlock × 2: [Conv1D(64→64) → BN → ReLU → Conv1D(64→64) → BN] + x │
│         ↓                                                            │
│  [batch, 64, 64]                                                     │
│                                                                      │
│  Layer 2 (128 channels)                                              │
│  ──────────────────────                                              │
│  ResBlock × 2 with downsample: stride=2, projection for skip        │
│         ↓                                                            │
│  [batch, 128, 32]                                                    │
│                                                                      │
│  Layer 3 (256 channels)                                              │
│  ──────────────────────                                              │
│  ResBlock × 2 with downsample: stride=2, projection for skip        │
│         ↓                                                            │
│  [batch, 256, 16]                                                    │
│                                                                      │
│  Layer 4 (512 channels)                                              │
│  ──────────────────────                                              │
│  ResBlock × 2 with downsample: stride=2, projection for skip        │
│         ↓                                                            │
│  [batch, 512, 8]                                                     │
│                                                                      │
│  Output Head                                                         │
│  ───────────                                                         │
│  AdaptiveAvgPool1D(1) → Flatten → Linear(512→3)                     │
│         ↓                                                            │
│  [batch, 3]  ← 3 classes: Down, Neutral, Up                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Bottleneck Architecture (ResNet-50+)

```
Bottleneck Block:
┌────────────────────────────────────────────────────┐
│                                                    │
│  Input: [batch, 256, T]                            │
│         ↓                                          │
│  Conv1D(256→64, k=1) → BN → ReLU   (reduce)       │
│         ↓                                          │
│  Conv1D(64→64, k=3, p=1) → BN → ReLU (process)    │
│         ↓                                          │
│  Conv1D(64→256, k=1) → BN           (expand)      │
│         ↓                                          │
│       (add) ←── Skip Connection                    │
│         ↓                                          │
│       ReLU                                         │
│                                                    │
└────────────────────────────────────────────────────┘

Benefit: 3x fewer parameters for same depth
```

## Feature Engineering for Trading

### Input Features

```
┌────────────────────────────────────────────────────────────────┐
│                     Feature Engineering                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw OHLCV Data:                                                │
│  ───────────────                                                │
│  • Open, High, Low, Close, Volume (5 channels)                  │
│                                                                 │
│  Price-Derived Features:                                        │
│  ───────────────────────                                        │
│  • Returns: (close[t] - close[t-1]) / close[t-1]               │
│  • Log Returns: log(close[t] / close[t-1])                     │
│  • High-Low Range: (high - low) / close                        │
│  • Body Ratio: |close - open| / (high - low)                   │
│                                                                 │
│  Technical Indicators (as channels):                            │
│  ────────────────────────────────────                           │
│  • RSI (14): Relative Strength Index                           │
│  • MACD: Moving Average Convergence Divergence                 │
│  • Bollinger Band %B                                           │
│  • ATR: Average True Range (normalized)                        │
│                                                                 │
│  Volume Features:                                               │
│  ────────────────                                               │
│  • Volume Change: volume[t] / SMA(volume, 20)                  │
│  • OBV Direction: sign of On-Balance Volume change             │
│  • VWAP Distance: (price - VWAP) / VWAP                        │
│                                                                 │
│  Final Input Shape: [batch, 15, 256]                           │
│  (15 feature channels, 256 time steps)                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Label Generation

```python
def generate_labels(prices, forward_window=12, threshold=0.002):
    """
    Generate 3-class labels based on future returns

    Classes:
    - 0: Down (return < -threshold)
    - 1: Neutral (|return| <= threshold)
    - 2: Up (return > threshold)
    """
    future_returns = prices.shift(-forward_window) / prices - 1

    labels = np.where(
        future_returns > threshold, 2,
        np.where(future_returns < -threshold, 0, 1)
    )
    return labels
```

## Training Procedure

### Data Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                    Training Data Pipeline                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Fetch Data from Bybit                                       │
│     └─ BTCUSDT, ETHUSDT: 1-minute candles, 1 year              │
│                                                                 │
│  2. Feature Engineering                                         │
│     └─ Compute 15 feature channels per time step               │
│                                                                 │
│  3. Sequence Creation                                           │
│     └─ Sliding window: 256 steps input → 1 label               │
│                                                                 │
│  4. Train/Val/Test Split                                        │
│     └─ 70% / 15% / 15% (chronological, no leakage)             │
│                                                                 │
│  5. Normalization                                               │
│     └─ Z-score per channel, fitted on training only            │
│                                                                 │
│  6. Data Augmentation                                           │
│     ├─ Gaussian noise (σ=0.01)                                 │
│     ├─ Time warping (slight stretching/compression)            │
│     └─ Magnitude scaling (0.9-1.1×)                            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Training Configuration

```python
config = {
    # Model
    'model': 'ResNet18',
    'input_channels': 15,
    'num_classes': 3,
    'dropout': 0.3,

    # Training
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,

    # Scheduler
    'scheduler': 'CosineAnnealingLR',
    'T_max': 100,
    'eta_min': 1e-6,

    # Early stopping
    'patience': 15,
    'min_delta': 1e-4,

    # Class weights (for imbalanced data)
    'class_weights': [1.2, 0.8, 1.2],  # Up/Down weighted higher
}
```

### Loss Functions

```python
# Standard Cross-Entropy with class weights
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.2, 0.8, 1.2]))

# Alternative: Focal Loss for hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

## Trading Strategy Implementation

### Signal Generation

```python
class ResNetTradingStrategy:
    def __init__(self, model, threshold=0.6, position_sizing='kelly'):
        self.model = model
        self.threshold = threshold
        self.position_sizing = position_sizing

    def generate_signal(self, features):
        """
        Generate trading signal from model prediction
        """
        with torch.no_grad():
            probs = F.softmax(self.model(features), dim=1)

        prob_down, prob_neutral, prob_up = probs[0].numpy()

        # High confidence signals only
        if prob_up > self.threshold:
            return 'LONG', prob_up
        elif prob_down > self.threshold:
            return 'SHORT', prob_down
        else:
            return 'NEUTRAL', prob_neutral

    def calculate_position_size(self, signal, confidence, portfolio_value):
        """
        Kelly Criterion position sizing
        """
        if self.position_sizing == 'kelly':
            # Simplified Kelly: f = p - (1-p)/b
            # where p = win probability, b = win/loss ratio
            edge = confidence - 0.5
            kelly_fraction = max(0, min(0.25, edge))  # Cap at 25%
            return portfolio_value * kelly_fraction
        else:
            return portfolio_value * 0.1  # Fixed 10%
```

### Risk Management

```
┌────────────────────────────────────────────────────────────────┐
│                     Risk Management Rules                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Position Sizing:                                               │
│  ────────────────                                               │
│  • Maximum position: 25% of portfolio                          │
│  • Scale with confidence: size = base × (confidence - 0.5) × 2 │
│                                                                 │
│  Stop Loss:                                                     │
│  ──────────                                                     │
│  • Fixed: -2% from entry                                       │
│  • ATR-based: -2 × ATR(14)                                     │
│  • Trailing: -1.5 × ATR(14) from peak                          │
│                                                                 │
│  Take Profit:                                                   │
│  ────────────                                                   │
│  • Fixed: +3% from entry (1.5:1 risk-reward)                   │
│  • Dynamic: When signal reverses or confidence drops           │
│                                                                 │
│  Time-Based Exits:                                              │
│  ─────────────────                                              │
│  • Maximum hold: 12 hours                                      │
│  • Force exit if new prediction window available               │
│                                                                 │
│  Drawdown Protection:                                           │
│  ────────────────────                                           │
│  • Reduce size by 50% if daily drawdown > 3%                   │
│  • Stop trading if weekly drawdown > 10%                       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Performance Metrics

### Classification Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | > 45% | Overall prediction accuracy (3-class) |
| Precision (Up) | > 55% | True up predictions / All up predictions |
| Precision (Down) | > 55% | True down predictions / All down predictions |
| F1-Score | > 0.50 | Harmonic mean of precision and recall |
| AUC-ROC | > 0.60 | Area under ROC curve |

### Trading Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > 1.5 | Risk-adjusted returns (annualized) |
| Sortino Ratio | > 2.0 | Downside risk-adjusted returns |
| Max Drawdown | < 15% | Maximum peak-to-trough decline |
| Win Rate | > 52% | Percentage of profitable trades |
| Profit Factor | > 1.3 | Gross profit / Gross loss |
| Calmar Ratio | > 1.0 | Annual return / Max drawdown |

## Project Structure

```
355_resnet_timeseries/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Simple explanation (English)
├── readme.simple.ru.md          # Simple explanation (Russian)
└── rust_resnet/                 # Rust implementation
    ├── Cargo.toml
    └── src/
        ├── lib.rs               # Library root
        ├── api/                 # Bybit API client
        │   ├── mod.rs
        │   ├── client.rs
        │   └── types.rs
        ├── model/               # ResNet implementation
        │   ├── mod.rs
        │   ├── resnet.rs
        │   ├── blocks.rs
        │   └── layers.rs
        ├── data/                # Data processing
        │   ├── mod.rs
        │   ├── features.rs
        │   ├── dataset.rs
        │   └── preprocessing.rs
        ├── strategy/            # Trading strategy
        │   ├── mod.rs
        │   ├── signals.rs
        │   └── risk.rs
        ├── utils/               # Utilities
        │   ├── mod.rs
        │   └── metrics.rs
        └── bin/                 # Executable examples
            ├── fetch_data.rs
            ├── train_model.rs
            ├── predict.rs
            └── backtest.rs
```

## Key Innovations

### 1. Multi-Scale Residual Blocks

```rust
// Process at multiple time scales simultaneously
pub struct MultiScaleResBlock {
    branch_3: Conv1d,   // kernel_size = 3
    branch_5: Conv1d,   // kernel_size = 5
    branch_7: Conv1d,   // kernel_size = 7
    fusion: Conv1d,     // Combine branches
}
```

### 2. Attention-Enhanced Residuals

```rust
// Add channel attention to residual blocks
pub struct SEResBlock {
    conv_block: ResidualBlock,
    squeeze: AdaptiveAvgPool1d,
    excitation: Sequential<Linear, ReLU, Linear, Sigmoid>,
}

// Recalibrate feature channels based on importance
fn forward(&self, x: Tensor) -> Tensor {
    let residual = self.conv_block.forward(x);
    let weights = self.excitation.forward(self.squeeze.forward(&residual));
    residual * weights.unsqueeze(-1) + x
}
```

### 3. Temporal Positional Encoding

```rust
// Inject position information for time awareness
pub fn positional_encoding(seq_len: usize, d_model: usize) -> Tensor {
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let dims: Vec<f32> = (0..d_model)
        .map(|i| 1.0 / 10000_f32.powf(i as f32 / d_model as f32))
        .collect();

    // PE(pos, 2i) = sin(pos / 10000^(2i/d))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    // ...
}
```

## References

1. **Deep Residual Learning for Image Recognition** (He et al., 2015)
   - Original ResNet paper: https://arxiv.org/abs/1512.03385

2. **Time Series Classification from Scratch with Deep Neural Networks** (Wang et al., 2017)
   - ResNet for time series: https://arxiv.org/abs/1611.06455

3. **InceptionTime: Finding AlexNet for Time Series Classification** (Fawaz et al., 2019)
   - Comprehensive comparison: https://arxiv.org/abs/1909.04939

4. **Deep Learning for Time Series Forecasting** (Brownlee, 2018)
   - Practical guide for financial applications

5. **ResNet-based Cryptocurrency Price Prediction** (Academic papers 2021-2023)
   - Various applications to crypto markets

## Difficulty Level

**Intermediate to Advanced**

**Prerequisites:**
- Understanding of CNNs and deep learning fundamentals
- Familiarity with time series analysis concepts
- Basic knowledge of trading and backtesting
- Rust programming experience (for implementation)

**Estimated Learning Time:** 15-20 hours

## Next Steps

1. **Start with data collection** - Run `fetch_data` to get Bybit market data
2. **Understand the model** - Study ResNet architecture in `model/resnet.rs`
3. **Train and evaluate** - Use `train_model` and analyze metrics
4. **Backtest strategy** - Run `backtest` to evaluate trading performance
5. **Experiment** - Try different architectures, features, and parameters
