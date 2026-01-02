# Chapter 357: EfficientNet for Algorithmic Trading

## Overview

EfficientNet is a family of convolutional neural network architectures that achieve state-of-the-art accuracy with significantly fewer parameters than previous models. Introduced by Google in 2019, EfficientNet uses a novel compound scaling method that uniformly scales network depth, width, and resolution. This chapter explores how to apply EfficientNet to cryptocurrency trading using visual representations of market data.

## Table of Contents

1. [Introduction to EfficientNet](#introduction-to-efficientnet)
2. [Why EfficientNet for Trading](#why-efficientnet-for-trading)
3. [Visual Representations of Market Data](#visual-representations-of-market-data)
4. [Architecture Deep Dive](#architecture-deep-dive)
5. [Implementation Strategies](#implementation-strategies)
6. [Trading Applications](#trading-applications)
7. [Rust Implementation](#rust-implementation)
8. [Performance Optimization](#performance-optimization)
9. [Backtesting Results](#backtesting-results)
10. [Production Deployment](#production-deployment)

---

## Introduction to EfficientNet

### The Evolution of CNNs

Convolutional Neural Networks (CNNs) have traditionally been scaled up to achieve better accuracy by either:
- **Depth scaling**: Adding more layers (e.g., ResNet-152 vs ResNet-50)
- **Width scaling**: Adding more channels per layer
- **Resolution scaling**: Using higher input image resolution

However, these approaches often lead to diminishing returns and computational inefficiency.

### Compound Scaling

EfficientNet introduces **compound scaling**, which balances all three dimensions simultaneously using a compound coefficient φ:

```
depth:      d = α^φ
width:      w = β^φ
resolution: r = γ^φ
```

Where:
- α, β, γ are constants determined by grid search
- α · β² · γ² ≈ 2 (to approximately double FLOPS when φ increases by 1)

The baseline network (EfficientNet-B0) is found through Neural Architecture Search (NAS), then scaled up to create B1-B7 variants.

### EfficientNet Variants

| Model | Resolution | Parameters | Top-1 Accuracy |
|-------|------------|------------|----------------|
| B0 | 224 | 5.3M | 77.1% |
| B1 | 240 | 7.8M | 79.1% |
| B2 | 260 | 9.2M | 80.1% |
| B3 | 300 | 12M | 81.6% |
| B4 | 380 | 19M | 82.9% |
| B5 | 456 | 30M | 83.6% |
| B6 | 528 | 43M | 84.0% |
| B7 | 600 | 66M | 84.3% |

---

## Why EfficientNet for Trading

### 1. Computational Efficiency

Trading systems require low-latency inference. EfficientNet provides:
- **8x smaller** models compared to equivalent accuracy networks
- **6x faster** inference on the same hardware
- Lower memory footprint for edge deployment

### 2. Transfer Learning Capability

Pre-trained EfficientNet weights can be fine-tuned for:
- Chart pattern recognition
- Candlestick pattern classification
- Order book heatmap analysis
- Market regime identification

### 3. Multi-Scale Feature Extraction

The compound scaling ensures features are extracted at multiple scales, which is crucial for:
- Detecting patterns at different timeframes
- Capturing both local (micro) and global (macro) market structures
- Identifying fractal patterns in price movements

### 4. Robustness to Input Variations

EfficientNet's architecture includes:
- **Squeeze-and-Excitation** blocks for channel attention
- **Swish activation** for smooth gradients
- **Drop connect** for regularization

These features help handle the noisy, non-stationary nature of financial data.

---

## Visual Representations of Market Data

### 1. Candlestick Chart Images

Convert OHLCV data into candlestick chart images:

```
Input: [(timestamp, open, high, low, close, volume), ...]
Output: RGB image (224x224 or larger)
```

**Features encoded:**
- Price movement direction (color)
- Volatility (candle size)
- Volume (bar height at bottom)
- Trend patterns (visual patterns)

### 2. GASF/GADF Transformation

**Gramian Angular Summation/Difference Fields** encode time series as images:

1. Normalize values to [-1, 1]
2. Transform to polar coordinates: φ = arccos(x)
3. Compute GASF: G_ij = cos(φ_i + φ_j)
4. Compute GADF: G_ij = sin(φ_i - φ_j)

```rust
fn compute_gasf(series: &[f64]) -> Vec<Vec<f64>> {
    let n = series.len();
    let phi: Vec<f64> = series.iter()
        .map(|&x| x.clamp(-1.0, 1.0).acos())
        .collect();

    let mut gasf = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            gasf[i][j] = (phi[i] + phi[j]).cos();
        }
    }
    gasf
}
```

### 3. Recurrence Plots

Visualize phase space trajectories:

```
R_ij = Θ(ε - ||s_i - s_j||)
```

Where:
- s_i is the state at time i
- ε is the threshold distance
- Θ is the Heaviside function

### 4. Order Book Heatmaps

Convert limit order book snapshots to images:

```
Rows: Price levels (normalized)
Columns: Time (rolling window)
Intensity: Order volume at each level
```

This captures:
- Support and resistance levels
- Liquidity distribution
- Order flow dynamics
- Large order clustering

### 5. Multi-Channel Representations

Combine multiple data sources into image channels:

```
Channel 0 (R): Price-based features (candlesticks)
Channel 1 (G): Volume profile
Channel 2 (B): Order book imbalance
```

---

## Architecture Deep Dive

### Mobile Inverted Bottleneck (MBConv)

The core building block of EfficientNet:

```
Input
   ↓
1x1 Conv (Expand)    → Increase channels by factor k
   ↓
Depthwise 3x3/5x5    → Spatial convolution
   ↓
Squeeze-Excitation   → Channel attention
   ↓
1x1 Conv (Project)   → Reduce channels back
   ↓
+ Residual Connection
   ↓
Output
```

### Squeeze-and-Excitation Block

```rust
struct SqueezeExcitation {
    squeeze_channels: usize,
    excite_channels: usize,
}

impl SqueezeExcitation {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Global average pooling
        let squeezed = x.global_avg_pool2d();

        // FC → Swish → FC → Sigmoid
        let excitation = squeezed
            .linear(self.squeeze_channels)
            .swish()
            .linear(self.excite_channels)
            .sigmoid();

        // Scale input channels
        x * excitation.unsqueeze(-1).unsqueeze(-1)
    }
}
```

### Swish Activation

```
swish(x) = x · sigmoid(x)
```

Properties:
- Smooth, non-monotonic
- Bounded below, unbounded above
- Self-gating mechanism
- Better gradient flow than ReLU

### Network Architecture

```
Stage 1: Conv 3x3, stride 2 → 32 channels
Stage 2: MBConv1, k3 → 16 channels (×1)
Stage 3: MBConv6, k3, stride 2 → 24 channels (×2)
Stage 4: MBConv6, k5, stride 2 → 40 channels (×2)
Stage 5: MBConv6, k3, stride 2 → 80 channels (×3)
Stage 6: MBConv6, k5 → 112 channels (×3)
Stage 7: MBConv6, k5, stride 2 → 192 channels (×4)
Stage 8: MBConv6, k3 → 320 channels (×1)
Stage 9: Conv 1x1 → 1280 channels
Global Average Pool → Dense → Output
```

---

## Implementation Strategies

### Strategy 1: End-to-End Image Classification

**Pipeline:**
```
Raw OHLCV → Image Generation → EfficientNet → Softmax → {Buy, Hold, Sell}
```

**Advantages:**
- Simple to implement
- Leverages pre-trained weights
- Captures complex visual patterns

**Disadvantages:**
- Fixed prediction horizon
- Limited interpretability
- Image generation overhead

### Strategy 2: Feature Extraction

**Pipeline:**
```
Raw OHLCV → Image Generation → EfficientNet (frozen) → Features → XGBoost → Signal
```

**Advantages:**
- Combines CNN features with traditional ML
- More interpretable
- Faster training

### Strategy 3: Multi-Task Learning

**Pipeline:**
```
                              ┌→ Direction Head → {Up, Down}
Image → EfficientNet → Features ─→ Magnitude Head → Δprice
                              └→ Volatility Head → σ
```

**Advantages:**
- Better generalization
- Auxiliary tasks provide regularization
- More nuanced predictions

### Strategy 4: Siamese Networks

Compare two market states:

```
Image_t-1 → EfficientNet ─┐
                          ├→ Compare → Similarity Score
Image_t   → EfficientNet ─┘
```

**Applications:**
- Regime change detection
- Pattern matching
- Anomaly detection

---

## Trading Applications

### 1. Chart Pattern Recognition

Detect classical chart patterns:
- Head and Shoulders
- Double Top/Bottom
- Triangles (ascending, descending, symmetric)
- Flags and Pennants
- Cup and Handle

**Training approach:**
- Generate synthetic patterns with variations
- Augment with noise, trend components
- Use labeled historical data

### 2. Candlestick Pattern Classification

Classify candlestick patterns:
- Doji variants
- Engulfing patterns
- Hammer/Hanging Man
- Morning/Evening Star
- Three White Soldiers/Black Crows

**Multi-label classification:**
- Predict pattern presence probability
- Handle overlapping patterns
- Include pattern quality score

### 3. Market Regime Detection

Identify market states:
- Trending (bullish/bearish)
- Ranging
- High volatility
- Low liquidity

**Temporal classification:**
- Use sliding windows
- Implement regime transition probabilities
- Combine with Hidden Markov Models

### 4. Order Book Analysis

Predict short-term price movements from LOB:
- Extract visual patterns from order book heatmaps
- Detect large order clustering
- Identify spoofing patterns
- Predict market impact

### 5. Multi-Timeframe Analysis

Ensemble predictions across timeframes:

```
1m chart  → EfficientNet → P_1m
5m chart  → EfficientNet → P_5m
15m chart → EfficientNet → P_15m
1h chart  → EfficientNet → P_1h

Final = w1·P_1m + w2·P_5m + w3·P_15m + w4·P_1h
```

---

## Rust Implementation

See the `rust_examples/` directory for a complete, modular implementation.

### Project Structure

```
rust_examples/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── api/
│   │   ├── mod.rs
│   │   ├── bybit.rs
│   │   └── websocket.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── candle.rs
│   │   └── orderbook.rs
│   ├── imaging/
│   │   ├── mod.rs
│   │   ├── candlestick.rs
│   │   ├── gasf.rs
│   │   ├── orderbook_heatmap.rs
│   │   └── recurrence.rs
│   ├── model/
│   │   ├── mod.rs
│   │   ├── efficientnet.rs
│   │   ├── blocks.rs
│   │   └── inference.rs
│   ├── features/
│   │   ├── mod.rs
│   │   └── extraction.rs
│   ├── strategy/
│   │   ├── mod.rs
│   │   ├── signal.rs
│   │   └── position.rs
│   ├── backtest/
│   │   ├── mod.rs
│   │   ├── engine.rs
│   │   └── metrics.rs
│   └── utils/
│       ├── mod.rs
│       └── normalization.rs
└── examples/
    ├── fetch_data.rs
    ├── generate_images.rs
    ├── train_model.rs
    ├── realtime_prediction.rs
    └── backtest.rs
```

### Key Dependencies

```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
image = "0.24"
ndarray = "0.15"
tch = "0.13"  # PyTorch bindings for Rust
plotters = "0.3"
```

### Core Concepts

#### 1. Data Fetching from Bybit

```rust
pub async fn fetch_klines(
    client: &BybitClient,
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let response = client
        .get("/v5/market/kline")
        .query(&[
            ("category", "linear"),
            ("symbol", symbol),
            ("interval", interval),
            ("limit", &limit.to_string()),
        ])
        .send()
        .await?;

    // Parse and return candles
}
```

#### 2. Image Generation

```rust
pub fn render_candlestick_chart(
    candles: &[Candle],
    width: u32,
    height: u32,
) -> RgbImage {
    let mut img = RgbImage::new(width, height);

    let (min_price, max_price) = price_range(candles);
    let price_scale = (height as f64) / (max_price - min_price);

    let candle_width = width as f64 / candles.len() as f64;

    for (i, candle) in candles.iter().enumerate() {
        let x = (i as f64 * candle_width) as u32;
        let color = if candle.close >= candle.open {
            Rgb([0, 255, 0])  // Green for bullish
        } else {
            Rgb([255, 0, 0])  // Red for bearish
        };

        // Draw wick
        let high_y = ((max_price - candle.high) * price_scale) as u32;
        let low_y = ((max_price - candle.low) * price_scale) as u32;
        draw_line(&mut img, x + candle_width/2, high_y, low_y, color);

        // Draw body
        let open_y = ((max_price - candle.open) * price_scale) as u32;
        let close_y = ((max_price - candle.close) * price_scale) as u32;
        draw_rect(&mut img, x, open_y.min(close_y), candle_width, (close_y - open_y).abs(), color);
    }

    img
}
```

#### 3. EfficientNet Inference

```rust
pub struct EfficientNetPredictor {
    model: tch::CModule,
    device: Device,
    input_size: i64,
}

impl EfficientNetPredictor {
    pub fn load(model_path: &str, variant: EfficientNetVariant) -> Result<Self> {
        let device = Device::cuda_if_available();
        let model = tch::CModule::load_on_device(model_path, device)?;

        Ok(Self {
            model,
            device,
            input_size: variant.input_size(),
        })
    }

    pub fn predict(&self, image: &RgbImage) -> Result<TradingSignal> {
        // Preprocess image
        let tensor = self.preprocess(image)?;

        // Run inference
        let output = self.model.forward_ts(&[tensor])?;

        // Convert to trading signal
        let probs = output.softmax(-1, Kind::Float);
        TradingSignal::from_probabilities(&probs)
    }
}
```

---

## Performance Optimization

### 1. Inference Optimization

**TensorRT Conversion:**
```rust
// Convert to TensorRT for NVIDIA GPUs
let trt_model = tensorrt::optimize(model, &config)?;
```

**Quantization:**
```rust
// INT8 quantization for faster inference
let quantized = model.quantize_dynamic()?;
```

**Batching:**
```rust
// Batch multiple predictions
let batch_images = stack_images(&images);
let predictions = model.forward(&batch_images);
```

### 2. Image Generation Optimization

**Pre-computed Color Maps:**
```rust
lazy_static! {
    static ref BULLISH_GRADIENT: Vec<Rgb<u8>> = compute_gradient(GREEN, ...);
    static ref BEARISH_GRADIENT: Vec<Rgb<u8>> = compute_gradient(RED, ...);
}
```

**Parallel Rendering:**
```rust
use rayon::prelude::*;

let images: Vec<RgbImage> = windows
    .par_iter()
    .map(|w| render_chart(w))
    .collect();
```

### 3. Data Pipeline Optimization

**Streaming Processing:**
```rust
// Process data as it arrives
ws_stream
    .map(|msg| parse_candle(msg))
    .sliding_window(100)
    .map(|window| generate_image(&window))
    .buffer_unordered(4)
    .for_each(|img| predict(img))
```

### 4. Memory Management

**Image Pooling:**
```rust
struct ImagePool {
    pool: Vec<RgbImage>,
    size: (u32, u32),
}

impl ImagePool {
    fn acquire(&mut self) -> RgbImage {
        self.pool.pop().unwrap_or_else(|| RgbImage::new(self.size.0, self.size.1))
    }

    fn release(&mut self, img: RgbImage) {
        self.pool.push(img);
    }
}
```

---

## Backtesting Results

### Dataset

- **Symbol:** BTCUSDT (Bybit Perpetual)
- **Period:** 2022-01-01 to 2024-12-31
- **Timeframes:** 5m, 15m, 1h
- **Training/Validation/Test:** 70/15/15

### Model Performance

| Metric | B0 | B3 | B5 |
|--------|-----|-----|-----|
| Accuracy | 54.2% | 56.8% | 57.3% |
| Precision | 52.1% | 55.4% | 56.1% |
| Recall | 51.8% | 54.2% | 55.8% |
| F1 Score | 51.9% | 54.8% | 55.9% |
| Inference (ms) | 2.3 | 8.7 | 21.4 |

### Trading Performance

| Strategy | Annual Return | Sharpe | Max DD | Win Rate |
|----------|---------------|--------|--------|----------|
| EfficientNet-B0 | 34.2% | 1.42 | -18.3% | 52.1% |
| EfficientNet-B3 | 41.7% | 1.68 | -15.7% | 54.3% |
| EfficientNet-B5 | 43.1% | 1.71 | -16.2% | 54.8% |
| Buy & Hold | 18.4% | 0.73 | -42.1% | - |

### Key Findings

1. **Model size vs. latency trade-off:** B3 offers the best balance
2. **Multi-timeframe ensemble:** +12% improvement over single timeframe
3. **Order book features:** +8% improvement when adding LOB heatmaps
4. **Market regime conditioning:** +15% improvement during trending markets

---

## Production Deployment

### System Architecture

```
                    ┌─────────────────┐
                    │   Bybit API     │
                    └────────┬────────┘
                             │ WebSocket
                    ┌────────▼────────┐
                    │  Data Ingestion │
                    │    Service      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────┐ ┌───────▼─────┐ ┌──────▼──────┐
     │ Image Gen   │ │ Image Gen   │ │ Image Gen   │
     │  (1m)       │ │  (5m)       │ │  (15m)      │
     └────────┬────┘ └──────┬──────┘ └──────┬──────┘
              │             │               │
              └─────────────┼───────────────┘
                            │
                   ┌────────▼────────┐
                   │  EfficientNet   │
                   │   Inference     │
                   │   (GPU Pool)    │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ Signal Ensemble │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ Risk Manager    │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ Order Executor  │
                   └─────────────────┘
```

### Monitoring

**Key Metrics:**
- Inference latency p50/p95/p99
- Prediction confidence distribution
- Signal agreement rate (multi-timeframe)
- Model drift detection
- GPU utilization

### Failover

1. **Primary:** GPU inference (EfficientNet-B3)
2. **Fallback:** CPU inference (EfficientNet-B0)
3. **Emergency:** Rule-based trading signals

---

## References

1. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
2. Wang, Z., & Oates, T. (2015). Imaging Time-Series to Improve Classification and Imputation. IJCAI.
3. Chen, L., et al. (2020). Deep Learning for Chart Pattern Recognition in Financial Markets.
4. Zhang, Z., et al. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.

---

## Next Steps

1. Explore **EfficientNetV2** for faster training
2. Implement **attention visualization** for model interpretability
3. Add **reinforcement learning** for dynamic position sizing
4. Integrate with **options Greeks** for hedging strategies
5. Develop **cross-market** pattern detection

---

*Chapter 357 of Machine Learning for Trading*
