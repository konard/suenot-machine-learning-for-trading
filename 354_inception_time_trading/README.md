# Chapter 354: InceptionTime for Trading — Time Series Classification with Inception Networks

## Overview

InceptionTime is a state-of-the-art deep learning architecture specifically designed for time series classification. Originally introduced by Fawaz et al. (2019), it adapts the Inception module concept from computer vision (GoogLeNet/InceptionNet) to temporal data. In trading, InceptionTime excels at:

- **Market regime classification** (bull, bear, sideways)
- **Trade signal generation** (buy, sell, hold)
- **Volatility state prediction** (high, medium, low)
- **Pattern recognition** (head-and-shoulders, double tops, etc.)

## Key Innovation

Unlike traditional CNNs that use fixed-size kernels, InceptionTime employs **multiple parallel convolutions** with different kernel sizes within each Inception module. This allows the model to capture patterns at multiple temporal scales simultaneously — from short-term microstructure signals to longer-term trends.

## Architecture

### Inception Module for Time Series

```
Input Time Series
       │
       ├──► MaxPooling(3) ──► Conv1D(1) ─────────────┐
       │                                              │
       ├──► Conv1D(kernel=10) ──────────────────────►│
       │                                              │ Concatenate
       ├──► Conv1D(kernel=20) ──────────────────────►│
       │                                              │
       └──► Conv1D(kernel=40) ──────────────────────►│
                                                      │
                                               BatchNorm + ReLU
                                                      │
                                                   Output
```

### Full InceptionTime Network

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Layer                               │
│                    (batch, sequence_length, features)            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Inception Module 1                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │MaxPool+  │ │ Conv1D   │ │ Conv1D   │ │ Conv1D   │            │
│  │ Conv1D   │ │ k=10     │ │ k=20     │ │ k=40     │            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘            │
│       └────────────┴────────────┴────────────┘                   │
│                         │ Concat                                 │
│                         ▼                                        │
│                   BatchNorm + ReLU                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                          (Repeat 5x)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Global Average Pooling                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Dense + Softmax                               │
│                    (num_classes output)                          │
└─────────────────────────────────────────────────────────────────┘
```

### Residual Connections

Every third Inception module includes a residual (skip) connection:

```
Input ─────────────┐
   │               │
   ▼               │
Inception 1        │
   │               │
   ▼               │
Inception 2        │
   │               │
   ▼               │
Inception 3        │
   │               │
   ▼               │
   + ◄─────────────┘  (Residual Connection with 1x1 Conv if needed)
   │
   ▼
Output
```

## Trading Strategy

### Approach: Multi-Timeframe Signal Ensemble

The core trading strategy leverages InceptionTime's multi-scale feature extraction:

1. **Input Features:**
   - OHLCV data (normalized)
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Order book imbalance (if available)
   - Volume profile

2. **Classification Targets:**
   - **Class 0:** Bearish (price decrease > threshold)
   - **Class 1:** Neutral (price within threshold)
   - **Class 2:** Bullish (price increase > threshold)

3. **Signal Generation:**
   - Use an ensemble of 5 InceptionTime models
   - Average predictions for robustness
   - Apply confidence thresholds before trading

### Edge: Why InceptionTime Works for Trading

1. **Multi-scale pattern detection:** Markets exhibit patterns at multiple timeframes (seconds to days)
2. **Robust to noise:** The ensemble approach and depth provide regularization
3. **Fast inference:** Pure convolutional architecture enables low-latency predictions
4. **No recurrence:** Parallelizable computation, unlike LSTMs

## Mathematical Formulation

### Inception Module

For an input tensor $X \in \mathbb{R}^{T \times C_{in}}$:

$$\text{InceptionModule}(X) = \text{BN}\left(\text{ReLU}\left(\text{Concat}\left[
\begin{array}{l}
\text{Conv}_{1 \times 1}(\text{MaxPool}(X)) \\
\text{Conv}_{k_1}(X) \\
\text{Conv}_{k_2}(X) \\
\text{Conv}_{k_3}(X)
\end{array}
\right]\right)\right)$$

Where:
- $k_1, k_2, k_3$ are different kernel sizes (default: 10, 20, 40)
- BN = Batch Normalization
- ReLU = Rectified Linear Unit

### Loss Function

For $N$-class classification with class weights $w_c$:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \sum_{c=1}^{C} w_c \cdot y_{i,c} \cdot \log(\hat{y}_{i,c})$$

### Ensemble Prediction

For $K$ models in the ensemble:

$$\hat{y}_{ensemble} = \frac{1}{K}\sum_{k=1}^{K} \hat{y}_k$$

## Implementation Details

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_filters` | 32 | Base number of filters per Inception module |
| `depth` | 6 | Number of Inception modules |
| `kernel_sizes` | [10, 20, 40] | Convolution kernel sizes |
| `bottleneck_size` | 32 | Bottleneck layer dimension |
| `residual_interval` | 3 | Add residual connection every N modules |
| `ensemble_size` | 5 | Number of models in ensemble |

### Training Configuration

```yaml
optimizer: Adam
learning_rate: 0.001
batch_size: 64
epochs: 1500
early_stopping_patience: 100
lr_scheduler: ReduceLROnPlateau
lr_decay_factor: 0.5
lr_patience: 50
```

### Data Preprocessing

1. **Normalization:** Z-score normalization per feature
2. **Windowing:** Sliding window with stride
3. **Label encoding:** Forward-looking returns classification
4. **Train/Val/Test split:** 70/15/15 with no lookahead

## Backtesting Framework

### Signal-to-Trade Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│ Raw Market  │ ─► │ Feature      │ ─► │ InceptionTime│ ─► │ Position   │
│ Data        │    │ Engineering  │    │ Ensemble    │    │ Sizing     │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Performance │ ◄─ │ Risk         │ ◄─ │ Order       │
│ Analytics   │    │ Management   │    │ Execution   │
└─────────────┘    └──────────────┘    └─────────────┘
```

### Position Sizing

Based on prediction confidence and volatility:

$$\text{PositionSize} = \frac{\text{RiskPerTrade} \times \text{Confidence}}{\text{ATR} \times \text{Multiplier}}$$

### Risk Management

- **Max drawdown limit:** 15%
- **Daily loss limit:** 3%
- **Position limit:** 20% of portfolio per trade
- **Stop-loss:** Dynamic based on ATR

## Performance Metrics

### Classification Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | True positives / (True positives + False positives) |
| Recall | True positives / (True positives + False negatives) |
| F1-Score | Harmonic mean of precision and recall |
| Cohen's Kappa | Agreement beyond chance |

### Trading Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > 1.5 | Risk-adjusted returns |
| Sortino Ratio | > 2.0 | Downside risk-adjusted returns |
| Max Drawdown | < 15% | Maximum peak-to-trough decline |
| Win Rate | > 55% | Percentage of profitable trades |
| Profit Factor | > 1.5 | Gross profit / Gross loss |
| Calmar Ratio | > 1.0 | Annual return / Max drawdown |

## Project Structure

```
354_inception_time_trading/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Simple explanation
├── readme.simple.ru.md          # Simple explanation (Russian)
├── README.specify.md            # Technical specification
│
└── rust/                        # Rust implementation
    ├── Cargo.toml               # Package manifest
    ├── config/
    │   └── default.toml         # Default configuration
    │
    └── src/
        ├── lib.rs               # Library entry point
        ├── main.rs              # Main executable
        │
        ├── data/
        │   ├── mod.rs           # Data module
        │   ├── bybit_client.rs  # Bybit API client
        │   ├── ohlcv.rs         # OHLCV data structures
        │   ├── features.rs      # Feature engineering
        │   └── dataset.rs       # Dataset loading
        │
        ├── model/
        │   ├── mod.rs           # Model module
        │   ├── inception.rs     # Inception module
        │   ├── network.rs       # Full InceptionTime network
        │   └── ensemble.rs      # Ensemble methods
        │
        ├── training/
        │   ├── mod.rs           # Training module
        │   ├── trainer.rs       # Training loop
        │   ├── losses.rs        # Loss functions
        │   └── metrics.rs       # Evaluation metrics
        │
        ├── strategy/
        │   ├── mod.rs           # Strategy module
        │   ├── signals.rs       # Signal generation
        │   ├── position.rs      # Position management
        │   └── risk.rs          # Risk management
        │
        ├── backtest/
        │   ├── mod.rs           # Backtest module
        │   ├── engine.rs        # Backtesting engine
        │   └── analytics.rs     # Performance analytics
        │
        └── utils/
            ├── mod.rs           # Utilities module
            ├── config.rs        # Configuration
            └── logging.rs       # Logging setup
```

## Usage Examples

### Rust CLI

```bash
# Fetch data from Bybit
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 15 --days 90

# Train InceptionTime ensemble
cargo run --bin train -- --config config/default.toml

# Run backtest
cargo run --bin backtest -- --model models/inception_ensemble.pt --data data/btcusdt.csv

# Live prediction (paper trading)
cargo run --bin predict -- --symbol BTCUSDT --interval 15
```

### Example Output

```
InceptionTime Trading System v0.1.0
═══════════════════════════════════════════════════════════════

[DATA] Fetching BTCUSDT 15m data from Bybit...
[DATA] Loaded 8640 candles (90 days)
[FEATURES] Generated 15 technical indicators
[TRAIN] Training ensemble of 5 models...

Model 1/5: Epoch 100/1500, Loss: 0.8234, Val Acc: 0.523
Model 1/5: Epoch 200/1500, Loss: 0.7123, Val Acc: 0.568
...
Model 5/5: Training complete. Best Val Acc: 0.612

[ENSEMBLE] Ensemble accuracy: 0.634
[BACKTEST] Running backtest on test set...

═══════════════════════════════════════════════════════════════
                      BACKTEST RESULTS
═══════════════════════════════════════════════════════════════

Total Return:       +34.21%
Sharpe Ratio:       1.87
Sortino Ratio:      2.43
Max Drawdown:       -8.34%
Win Rate:           58.3%
Profit Factor:      1.72
Total Trades:       234

═══════════════════════════════════════════════════════════════
```

## References

1. **InceptionTime: Finding AlexNet for Time Series Classification**
   - Authors: Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., et al.
   - Year: 2019
   - URL: https://arxiv.org/abs/1909.04939

2. **Deep Learning for Time Series Classification**
   - Authors: Fawaz, H. I., et al.
   - Year: 2019
   - URL: https://arxiv.org/abs/1809.04356

3. **Going Deeper with Convolutions (GoogLeNet)**
   - Authors: Szegedy, C., et al.
   - Year: 2015
   - URL: https://arxiv.org/abs/1409.4842

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests to the main repository.
