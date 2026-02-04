# Chapter 357: EfficientNet Trading

## Overview

EfficientNet is a family of convolutional neural network architectures that achieves state-of-the-art accuracy with significantly fewer parameters and FLOPS compared to previous models. For trading applications, EfficientNet enables the transformation of time series data into 2D image representations (using techniques like Gramian Angular Fields, Markov Transition Fields, and spectrograms), allowing us to leverage powerful computer vision models pre-trained on ImageNet for financial pattern recognition.

## Why EfficientNet for Trading?

### The Problem with Traditional Time Series Approaches

Traditional time series models (LSTMs, GRUs, standard CNNs) process raw numerical sequences directly. However, they often miss complex visual patterns that emerge when data is viewed as images:

- **Pattern Recognition**: Chart patterns (head and shoulders, double tops, flags) are inherently visual
- **Multi-scale Features**: Price movements occur at multiple time scales simultaneously
- **Transfer Learning**: ImageNet pre-trained models capture universal features (edges, textures, shapes) useful for pattern recognition
- **Efficiency**: EfficientNet provides optimal accuracy-to-computation trade-offs

### EfficientNet Solution

EfficientNet applies compound scaling to balance network depth, width, and resolution:

```
Traditional CNN Scaling:
- Scale only depth (more layers) OR
- Scale only width (more channels) OR
- Scale only resolution (larger input)

EfficientNet Compound Scaling:
- depth = alpha^phi
- width = beta^phi
- resolution = gamma^phi
- Constraint: alpha * beta^2 * gamma^2 ≈ 2

where phi controls overall model size (B0 to B7)
```

## Technical Architecture

### 1. Time Series to Image Transformation

```
Time Series Data → Image Conversion Methods:
├── Gramian Angular Field (GAF)
│   ├── GASF (Summation): Captures temporal correlations
│   └── GADF (Difference): Captures momentum patterns
├── Markov Transition Field (MTF)
│   └── Encodes transition probabilities between quantile bins
├── Recurrence Plot (RP)
│   └── Shows recurring patterns in phase space
└── Spectrogram
    └── Time-frequency representation via STFT
```

### 2. Gramian Angular Field (GAF)

The GAF transforms a time series into a polar coordinate system:

```python
# Normalize time series to [-1, 1]
x_normalized = (2 * x - max(x) - min(x)) / (max(x) - min(x))

# Convert to angular representation
phi = arccos(x_normalized)

# GASF: Gramian Angular Summation Field
GASF[i,j] = cos(phi[i] + phi[j])

# GADF: Gramian Angular Difference Field
GADF[i,j] = sin(phi[i] - phi[j])
```

### 3. Markov Transition Field (MTF)

MTF captures transition dynamics between quantile bins:

```python
# Discretize time series into Q quantile bins
bins = quantile_discretize(x, Q)

# Build transition matrix W
W[i,j] = P(bin_j at t+1 | bin_i at t)

# MTF encodes temporal transition probabilities
MTF[i,j] = W[bins[i], bins[j]]
```

### 4. Multi-Timeframe Image Stacking

Combine multiple timeframes into RGB channels:

```
Image Construction:
├── Red Channel: 1-minute GASF (short-term momentum)
├── Green Channel: 5-minute GASF (medium-term trend)
└── Blue Channel: 15-minute GASF (longer-term pattern)

Alternative Stacking:
├── Red Channel: GASF (angular correlation)
├── Green Channel: GADF (angular difference)
└── Blue Channel: MTF (transition dynamics)
```

## EfficientNet Architecture

### Compound Scaling Formula

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPOUND SCALING                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EfficientNet-B0 (Baseline):                                     │
│    - Input resolution: 224x224                                   │
│    - Parameters: 5.3M                                            │
│    - Top-1 Accuracy: 77.1% (ImageNet)                           │
│                                                                  │
│  Scaling coefficients (phi = 1.0 for B0):                        │
│    alpha = 1.2 (depth)                                           │
│    beta = 1.1 (width)                                            │
│    gamma = 1.15 (resolution)                                     │
│                                                                  │
│  For phi = N:                                                    │
│    depth = 1.2^N                                                 │
│    width = 1.1^N                                                 │
│    resolution = 1.15^N * 224                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Variants Comparison

| Model | Resolution | Params | FLOPS | Top-1 Acc | Latency (ms) | Trading Use Case |
|-------|------------|--------|-------|-----------|--------------|------------------|
| B0 | 224 | 5.3M | 0.39B | 77.1% | 5 | Real-time HFT |
| B1 | 240 | 7.8M | 0.70B | 79.1% | 8 | Fast intraday |
| B2 | 260 | 9.2M | 1.0B | 80.1% | 10 | Intraday |
| B3 | 300 | 12M | 1.8B | 81.6% | 15 | Swing trading |
| B4 | 380 | 19M | 4.2B | 82.9% | 25 | Position trading |
| B5 | 456 | 30M | 9.9B | 83.6% | 40 | Research/Analysis |
| B6 | 528 | 43M | 19B | 84.0% | 60 | Batch processing |
| B7 | 600 | 66M | 37B | 84.3% | 100 | Offline analysis |

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EFFICIENTNET TRADING MODEL                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT LAYER                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Multi-timeframe Image Stack (H x W x 3):                  │   │
│  │   - Channel 1: 1-min GASF (momentum patterns)             │   │
│  │   - Channel 2: 5-min GASF (trend patterns)                │   │
│  │   - Channel 3: 15-min GASF (structure patterns)           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  EFFICIENTNET BACKBONE (Pre-trained on ImageNet)                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Stem: Conv3x3 + BN + Swish                          │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ MBConv Blocks (×N):                                 │   │   │
│  │ │   - Expansion: 1x1 Conv → BN → Swish               │   │   │
│  │ │   - Depthwise: 3x3/5x5 DWConv → BN → Swish         │   │   │
│  │ │   - Squeeze-Excitation (SE) Attention              │   │   │
│  │ │   - Projection: 1x1 Conv → BN                      │   │   │
│  │ │   - Skip Connection (if stride=1)                  │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Head: Conv1x1 + BN + Swish + GlobalAvgPool         │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  TRADING HEAD                                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Feature Fusion: Concat(backbone_features, aux_data) │   │   │
│  │ │   - Technical indicators (RSI, MACD, BB position)   │   │   │
│  │ │   - Volume features                                 │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Direction Head: Linear → ReLU → Linear → Softmax    │   │   │
│  │ │   Output: [P(up), P(neutral), P(down)]             │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Magnitude Head: Linear → ReLU → Linear             │   │   │
│  │ │   Output: Expected return magnitude                 │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## MBConv Block (Mobile Inverted Bottleneck)

```python
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block

    Key innovations:
    1. Inverted residuals (expand then project)
    2. Depthwise separable convolutions
    3. Squeeze-and-Excitation attention
    4. Swish activation (x * sigmoid(x))
    """

    def forward(self, x):
        identity = x

        # Expansion phase (increase channels)
        x = self.expand_conv(x)
        x = self.bn1(x)
        x = self.swish(x)

        # Depthwise convolution (spatial mixing)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.swish(x)

        # Squeeze-and-Excitation
        se = self.se_pool(x)
        se = self.se_expand(self.swish(self.se_reduce(se)))
        x = x * torch.sigmoid(se)

        # Projection phase (reduce channels)
        x = self.project_conv(x)
        x = self.bn3(x)

        # Skip connection
        if self.use_residual:
            x = x + identity

        return x
```

## Squeeze-and-Excitation Attention

```
SE Block Operation:
┌─────────────────────────────────────────┐
│ Input: Feature map (H x W x C)          │
│           ↓                             │
│ Global Average Pool → (1 x 1 x C)       │
│           ↓                             │
│ FC (C → C/r) + ReLU  (reduction ratio r)│
│           ↓                             │
│ FC (C/r → C) + Sigmoid                  │
│           ↓                             │
│ Channel-wise multiplication             │
│           ↓                             │
│ Output: Recalibrated features           │
└─────────────────────────────────────────┘

Purpose: Learn to emphasize informative channels
         and suppress less useful ones
```

## Trading Strategy

### Signal Generation

```python
def generate_signals(model, price_data, config):
    """
    Generate trading signals from EfficientNet predictions.

    Args:
        model: Trained EfficientNet model
        price_data: OHLCV data
        config: Strategy configuration

    Returns:
        List of trading signals
    """
    signals = []

    # Convert price data to multi-timeframe images
    images = create_multi_timeframe_images(
        price_data,
        timeframes=['1m', '5m', '15m'],
        image_size=config.image_size
    )

    # Batch inference
    with torch.no_grad():
        direction_probs, magnitude = model(images)

    for i in range(len(images)):
        prob_up = direction_probs[i, 0].item()
        prob_down = direction_probs[i, 2].item()
        expected_return = magnitude[i].item()

        if prob_up > config.buy_threshold and expected_return > config.min_return:
            signals.append(Signal(
                direction='LONG',
                confidence=prob_up,
                expected_return=expected_return
            ))
        elif prob_down > config.sell_threshold and expected_return < -config.min_return:
            signals.append(Signal(
                direction='SHORT',
                confidence=prob_down,
                expected_return=expected_return
            ))
        else:
            signals.append(Signal(direction='HOLD', confidence=0.0))

    return signals
```

### Attention Visualization

```python
def visualize_attention(model, image, target_layer='features.8'):
    """
    Visualize what the model is looking at using Grad-CAM.

    This helps understand which parts of the price chart
    the model considers important for its prediction.
    """
    # Register hooks for gradient extraction
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Attach hooks
    target = dict(model.named_modules())[target_layer]
    target.register_forward_hook(forward_hook)
    target.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image.unsqueeze(0))
    pred_class = output.argmax(dim=1)

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Compute Grad-CAM
    weights = gradients[0].mean(dim=[2, 3], keepdim=True)
    cam = (weights * activations[0]).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, image.shape[-2:], mode='bilinear')

    return cam.squeeze().numpy()
```

## Implementation Details

### Data Requirements

```
Cryptocurrency Market Data:
├── OHLCV data (1-minute resolution minimum)
│   └── Multiple assets (BTC, ETH, SOL, ...)
├── Multi-timeframe data
│   ├── 1-minute (short-term patterns)
│   ├── 5-minute (medium-term trends)
│   └── 15-minute (longer-term structure)
└── Volume data for additional channel

Image Generation Settings:
├── Image size: 224x224 (B0), 380x380 (B4), etc.
├── Lookback window: 60-240 candles per timeframe
├── Quantile bins (MTF): 8-16 bins
└── Normalization: Min-max per channel
```

### Feature Engineering

```python
def create_gasf_image(price_series, image_size=224):
    """Create Gramian Angular Summation Field image."""
    # Normalize to [-1, 1]
    min_val = price_series.min()
    max_val = price_series.max()
    scaled = 2 * (price_series - min_val) / (max_val - min_val + 1e-8) - 1
    scaled = np.clip(scaled, -1, 1)

    # Convert to angular representation
    phi = np.arccos(scaled)

    # Create GASF matrix
    gasf = np.cos(np.add.outer(phi, phi))

    # Resize to target image size
    gasf_resized = cv2.resize(gasf, (image_size, image_size))

    return gasf_resized

def create_mtf_image(price_series, n_bins=8, image_size=224):
    """Create Markov Transition Field image."""
    # Discretize into quantile bins
    bins = np.percentile(price_series, np.linspace(0, 100, n_bins + 1))
    digitized = np.digitize(price_series, bins[:-1]) - 1
    digitized = np.clip(digitized, 0, n_bins - 1)

    # Build transition matrix
    trans_matrix = np.zeros((n_bins, n_bins))
    for i in range(len(digitized) - 1):
        trans_matrix[digitized[i], digitized[i + 1]] += 1

    # Normalize rows
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    trans_matrix = trans_matrix / (row_sums + 1e-8)

    # Create MTF
    n = len(digitized)
    mtf = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mtf[i, j] = trans_matrix[digitized[i], digitized[j]]

    # Resize
    mtf_resized = cv2.resize(mtf, (image_size, image_size))

    return mtf_resized
```

### Training Configuration

```yaml
model:
  variant: "efficientnet_b2"  # B0-B7
  pretrained: true
  num_classes: 3  # up, neutral, down
  dropout: 0.3

image:
  size: 260  # Depends on variant
  channels: 3
  transforms:
    - random_crop: 0.9
    - horizontal_flip: 0.0  # Don't flip time series!
    - color_jitter: 0.1

training:
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_epochs: 5
  max_epochs: 100
  early_stopping_patience: 10
  label_smoothing: 0.1

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  lookback_candles: 120
  prediction_horizon: 5  # 5 candles ahead

augmentation:
  time_warp: true
  magnitude_warp: true
  noise_injection: 0.01
```

## Transfer Learning Strategy

### Fine-tuning Approach

```python
def setup_transfer_learning(model, config):
    """
    Set up transfer learning for EfficientNet.

    Strategy:
    1. Freeze early layers (general features)
    2. Unfreeze later layers (task-specific features)
    3. Use lower learning rate for backbone
    """
    # Freeze early blocks (keep ImageNet features)
    for name, param in model.named_parameters():
        if 'features.0' in name or 'features.1' in name or 'features.2' in name:
            param.requires_grad = False

    # Different learning rates
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name or 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config.lr * 0.1},
        {'params': head_params, 'lr': config.lr}
    ], weight_decay=config.weight_decay)

    return optimizer
```

### Progressive Unfreezing

```
Training Schedule:
├── Epoch 1-5: Only train classification head
│   └── LR: 1e-3
├── Epoch 6-15: Unfreeze last 2 blocks
│   └── LR: 1e-4 (backbone), 1e-3 (head)
├── Epoch 16-30: Unfreeze all layers
│   └── LR: 1e-5 (early), 1e-4 (late), 1e-3 (head)
└── Epoch 31+: Full fine-tuning with low LR
    └── LR: 1e-6 (uniform)
```

## Key Metrics

### Model Performance

- **Classification Accuracy**: Direction prediction accuracy
- **F1 Score**: Balanced measure for imbalanced classes
- **AUC-ROC**: Ranking quality for probability outputs
- **Information Coefficient (IC)**: Correlation with actual returns

### Trading Performance

- **Sharpe Ratio**: Risk-adjusted returns (target > 2.0)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

## Advantages of EfficientNet for Trading

| Aspect | Traditional CNNs | EfficientNet |
|--------|-----------------|--------------|
| Parameter efficiency | High params for accuracy | Optimal params-accuracy trade-off |
| Inference speed | Can be slow | Scalable from real-time to research |
| Transfer learning | Limited | Strong ImageNet initialization |
| Multi-scale patterns | Manual design needed | Automatic via compound scaling |
| Memory usage | Often excessive | Efficient at all scales |
| Deployment | Complex | Easy scaling for any hardware |

## Comparison with Other Approaches

### vs. Raw Time Series Models (LSTM/GRU)

- **LSTM/GRU**: Process sequences directly
- **EfficientNet**: Leverages spatial patterns in image representations

### vs. Vision Transformer (ViT)

- **ViT**: Better on very large datasets
- **EfficientNet**: Better data efficiency, faster inference

### vs. ResNet

- **ResNet**: Fixed scaling
- **EfficientNet**: Optimal scaling for any compute budget

## Production Considerations

```
Inference Pipeline:
├── Data Collection (Bybit WebSocket)
│   └── Real-time OHLCV updates
├── Image Generation
│   ├── GASF computation (~5ms)
│   ├── MTF computation (~10ms)
│   └── Multi-timeframe stacking (~2ms)
├── Model Inference
│   ├── B0: ~5ms (GPU), ~50ms (CPU)
│   ├── B2: ~10ms (GPU), ~100ms (CPU)
│   └── B4: ~25ms (GPU), ~250ms (CPU)
├── Signal Generation
│   └── Threshold-based extraction (~1ms)
└── Order Execution
    └── API call with risk management

Latency Budget (using B2):
├── Data collection: ~10ms (WebSocket)
├── Image generation: ~20ms
├── Model inference: ~10ms (GPU)
├── Signal generation: ~1ms
└── Total: ~40ms (excluding execution)
```

## Directory Structure

```
357_efficientnet_trading/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Beginner-friendly explanation
├── readme.simple.ru.md          # Russian beginner version
├── python/                      # Python implementation
│   ├── __init__.py
│   ├── requirements.txt         # Python dependencies
│   ├── data_loader.py           # Bybit/CCXT data fetching
│   ├── image_transform.py       # GAF, MTF, spectrogram
│   ├── model.py                 # EfficientNet trading model
│   ├── train.py                 # Training script
│   ├── backtest.py              # Backtesting framework
│   └── visualize.py             # Attention visualization
└── rust_efficientnet/           # Rust implementation
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               # Library entry point
    │   ├── api/                 # Bybit API client
    │   ├── image/               # Image transformation
    │   ├── model/               # EfficientNet implementation
    │   ├── strategy/            # Trading strategy
    │   └── backtest/            # Backtesting engine
    └── examples/
        ├── fetch_data.rs
        ├── create_images.rs
        ├── simple_prediction.rs
        └── backtest.rs
```

## References

1. **EfficientNet: Rethinking Model Scaling for CNNs** (Tan & Le, 2019)
   - https://arxiv.org/abs/1905.11946

2. **Encoding Time Series as Images for Visual Inspection** (Wang & Oates, 2015)
   - Gramian Angular Fields and Markov Transition Fields

3. **Imaging Time-Series to Improve Classification and Imputation** (Hatami et al., 2017)
   - Recurrence Plots for time series classification

4. **EfficientNetV2: Smaller Models and Faster Training** (Tan & Le, 2021)
   - https://arxiv.org/abs/2104.00298

5. **Grad-CAM: Visual Explanations from Deep Networks** (Selvaraju et al., 2017)
   - https://arxiv.org/abs/1610.02391

## Difficulty Level

**Advanced** - Requires understanding of:
- Convolutional Neural Networks
- Transfer Learning techniques
- Time series to image transformations
- Financial market microstructure
- PyTorch/computer vision frameworks

## Disclaimer

This chapter is for **educational purposes only**. Cryptocurrency trading involves substantial risk. The strategies described here have not been validated in live trading and should be thoroughly tested before any real-world application. Past performance does not guarantee future results.
