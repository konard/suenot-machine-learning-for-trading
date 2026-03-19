# Chapter 204: Quantization Trading Models

## 1. Introduction

In high-frequency and algorithmic trading, the speed of model inference can be the difference between capturing alpha and missing a trade entirely. Neural network models used for price prediction, signal generation, and risk assessment often contain millions of parameters stored as 32-bit floating-point numbers. While this precision is beneficial during training, it introduces significant computational overhead during inference — overhead that directly translates into latency.

**Quantization** is the process of reducing the numerical precision of a model's weights and activations from high-precision formats (FP32) to lower-precision representations (FP16, INT8, INT4, or even binary). This technique dramatically reduces model size, memory bandwidth requirements, and inference latency while preserving most of the model's predictive accuracy.

For trading applications, quantization enables several critical capabilities:
- **FPGA deployment** with integer-only arithmetic for sub-microsecond inference
- **Mobile and edge trading** on resource-constrained devices
- **Reduced memory bandwidth** allowing larger ensemble models to fit in cache
- **Lower power consumption** for co-located trading servers

This chapter covers the mathematical foundations of quantization, the different quantization strategies available, precision-level tradeoffs, and a complete Rust implementation that fetches live Bybit market data, trains a trading model, and compares inference across FP32, INT8, and INT4 precision levels.

## 2. Mathematical Foundation

### 2.1 Uniform Quantization

Uniform quantization maps a continuous range of floating-point values to a finite set of evenly spaced discrete levels. Given a floating-point value `x`, the quantized value `x_q` is computed as:

```
x_q = clamp(round(x / S + Z), q_min, q_max)
```

where:
- `S` is the **scale factor** (a positive real number)
- `Z` is the **zero-point** (an integer offset)
- `q_min` and `q_max` define the range of the quantized representation
- `round()` maps to the nearest integer
- `clamp()` restricts values to the valid range

For INT8 quantization:
- Unsigned: `q_min = 0`, `q_max = 255`
- Signed: `q_min = -128`, `q_max = 127`

**Dequantization** recovers an approximation of the original value:

```
x_approx = S * (x_q - Z)
```

### 2.2 Scale and Zero-Point Computation

#### Min-Max Method

The simplest approach computes scale and zero-point from the observed range of values:

```
S = (x_max - x_min) / (q_max - q_min)
Z = q_min - round(x_min / S)
```

This method is sensitive to outliers but preserves the full dynamic range.

#### Percentile Method

To improve robustness against outliers, the percentile method clips the range:

```
x_min_clipped = percentile(x, p)
x_max_clipped = percentile(x, 100 - p)
```

where `p` is typically 0.1% to 1%. Values outside this range are clamped, sacrificing outlier accuracy for better resolution of the majority of values.

### 2.3 Symmetric vs Asymmetric Quantization

**Symmetric quantization** forces the zero-point to be zero (`Z = 0`), mapping the range `[-alpha, alpha]` where `alpha = max(|x_min|, |x_max|)`:

```
S = alpha / q_max
x_q = clamp(round(x / S), -q_max, q_max)
```

This simplifies computation (no zero-point offset) and is preferred for weights, which are typically centered around zero.

**Asymmetric quantization** uses the full `[q_min, q_max]` range with a non-zero zero-point, which better captures asymmetric distributions common in activations (e.g., after ReLU where all values are non-negative).

### 2.4 Per-Tensor vs Per-Channel Quantization

**Per-tensor quantization** uses a single scale and zero-point for an entire tensor. This is simple but suboptimal when different channels have vastly different value ranges.

**Per-channel quantization** computes separate scale and zero-point values for each output channel of a weight tensor. This significantly improves accuracy with minimal overhead, as the quantization parameters are computed once and reused across inference.

### 2.5 Quantization Error Analysis

The quantization error for a single value is bounded by half the step size:

```
|error| <= S / 2
```

For a tensor, we measure quantization quality using:

**Mean Squared Error (MSE):**
```
MSE = (1/n) * sum((x_i - x_approx_i)^2)
```

**Signal-to-Noise Ratio (SNR):**
```
SNR = 10 * log10(signal_power / noise_power)
```

where `signal_power = mean(x^2)` and `noise_power = MSE`.

Higher SNR indicates better quantization quality. Typical targets:
- INT8: SNR > 30 dB (excellent for most trading models)
- INT4: SNR > 15 dB (acceptable for less sensitive applications)

### 2.6 Non-Uniform Quantization

Non-uniform quantization uses unevenly spaced quantization levels, which can better match the actual distribution of values. Common approaches include:

- **Logarithmic quantization:** Levels are spaced logarithmically, providing higher resolution near zero where weight values are dense
- **K-means quantization:** Quantization levels are chosen by clustering the weight values
- **Learned quantization:** Step sizes are optimized during training

While non-uniform quantization can achieve better accuracy, it typically requires lookup tables for dequantization, making it less suitable for FPGA and hardware-accelerated inference.

## 3. Quantization Types

### 3.1 Post-Training Quantization (PTQ)

PTQ quantizes a pre-trained model without retraining. It is the simplest approach:

1. **Calibrate:** Run a representative dataset through the model to collect activation statistics (min, max, or histograms)
2. **Compute parameters:** Determine scale and zero-point for each layer
3. **Quantize:** Convert weights to the target precision
4. **Validate:** Measure accuracy degradation on a test set

PTQ is fast and requires no training data beyond a small calibration set. For INT8 quantization, PTQ typically preserves >99% of FP32 accuracy. For INT4, accuracy degradation can be more significant.

**Advantages for trading:**
- Fast turnaround — quantize and deploy within minutes
- No retraining infrastructure needed
- Suitable for models that are frequently retrained on new market data

### 3.2 Quantization-Aware Training (QAT)

QAT simulates quantization during training by inserting "fake quantization" nodes that round values during the forward pass but pass gradients through using the straight-through estimator (STE):

```
Forward: x_q = quantize(x)
Backward: d_loss/d_x = d_loss/d_x_q  (gradient passes through unchanged)
```

QAT allows the model to learn weights that are robust to quantization noise. This is especially important for aggressive quantization (INT4 and below), where PTQ accuracy drops significantly.

**Trading considerations:**
- Longer training time (typically 10-20% overhead)
- Must be integrated into the training pipeline
- Provides the best INT4 and binary quantization results

### 3.3 Dynamic vs Static Quantization

**Static quantization** pre-computes activation quantization parameters during calibration. The same scale and zero-point are used for all inference inputs.

**Dynamic quantization** computes activation quantization parameters at runtime for each input. This adapts to the actual data distribution but adds computational overhead.

For trading models, the choice depends on the use case:
- **Static:** Preferred for FPGA deployment and latency-critical paths where the input distribution is stable
- **Dynamic:** Better for models that process diverse market regimes where activation distributions shift significantly

## 4. Precision Levels

### 4.1 FP32 (32-bit Floating Point)

- **Size:** 4 bytes per parameter
- **Range:** +/-3.4e38 with ~7 decimal digits of precision
- **Use case:** Training, reference inference
- **Trading notes:** Standard training precision; too slow for production HFT

### 4.2 FP16 (16-bit Floating Point)

- **Size:** 2 bytes per parameter (2x compression)
- **Range:** +/-65504 with ~3.3 decimal digits of precision
- **Speedup:** 1.5-2x on GPUs with Tensor Cores
- **Trading notes:** Good balance for GPU-based inference; supported natively on modern GPUs

### 4.3 INT8 (8-bit Integer)

- **Size:** 1 byte per parameter (4x compression)
- **Range:** -128 to 127 (signed) or 0 to 255 (unsigned)
- **Speedup:** 2-4x over FP32; excellent on CPUs and FPGAs
- **Accuracy:** Typically <1% degradation with PTQ
- **Trading notes:** The sweet spot for most trading applications. FPGA-friendly, CPU-efficient, minimal accuracy loss

### 4.4 INT4 (4-bit Integer)

- **Size:** 0.5 bytes per parameter (8x compression)
- **Range:** -8 to 7 (signed) or 0 to 15 (unsigned)
- **Speedup:** 4-8x over FP32
- **Accuracy:** 1-5% degradation; QAT strongly recommended
- **Trading notes:** Useful for ensemble models where memory is the bottleneck; acceptable for signal generation where exact magnitude matters less

### 4.5 Binary (1-bit)

- **Size:** 0.125 bytes per parameter (32x compression)
- **Values:** -1 or +1
- **Operations:** XNOR and popcount replace multiply-add
- **Accuracy:** Significant degradation (10-30%)
- **Trading notes:** Experimental; potentially useful for binary classification tasks (buy/sell signals) where speed matters more than precision

### Precision Comparison Summary

| Precision | Size/Param | Compression | Typical Accuracy Loss | Best For |
|-----------|------------|-------------|----------------------|----------|
| FP32 | 4 bytes | 1x | 0% (baseline) | Training |
| FP16 | 2 bytes | 2x | <0.1% | GPU inference |
| INT8 | 1 byte | 4x | <1% | FPGA, CPU inference |
| INT4 | 0.5 bytes | 8x | 1-5% | Memory-constrained |
| Binary | 1 bit | 32x | 10-30% | Ultra-fast signals |

## 5. Trading Applications

### 5.1 FPGA Deployment with INT8

FPGAs (Field-Programmable Gate Arrays) are widely used in trading infrastructure for their deterministic latency and parallelism. INT8 quantization is the natural precision for FPGA deployment because:

- **Integer ALUs** on FPGAs are smaller and faster than floating-point units
- **Deterministic latency:** No floating-point rounding mode variations
- **Parallelism:** More INT8 multiply-accumulate units fit on the same FPGA fabric
- **Pipeline depth:** Shorter pipelines for integer operations mean lower latency

A typical FPGA trading pipeline:
1. Market data arrives via network interface
2. Feature extraction using fixed-point arithmetic
3. INT8 quantized model inference in < 1 microsecond
4. Order generation and risk checks
5. Order transmission

### 5.2 Mobile Trading Applications

Retail traders increasingly demand sophisticated analytics on mobile devices. Quantized models enable:

- Running prediction models directly on the device (no network latency for inference)
- Battery-efficient model execution
- Privacy-preserving local inference (no need to send portfolio data to servers)
- Offline capability for basic signal generation

INT8 models reduce memory footprint by 4x, making it feasible to run multiple models on a smartphone.

### 5.3 Reducing Memory Bandwidth

In ensemble trading systems that combine dozens of models, memory bandwidth often becomes the bottleneck rather than compute. Quantization directly addresses this:

- **Cache efficiency:** INT8 models are 4x more likely to fit in L2/L3 cache
- **Memory bus utilization:** 4x more parameters transferred per memory access cycle
- **Batch processing:** Run 4x more models in the same memory footprint

For a trading system with 10 models, each 50MB in FP32:
- FP32: 500MB total (exceeds most L3 caches)
- INT8: 125MB total (fits in many L3 caches)
- INT4: 62.5MB total (fits in L2 on some server CPUs)

### 5.4 Latency-Accuracy Tradeoffs in Trading

The optimal quantization level depends on the trading strategy:

- **Market making:** INT8 is ideal — latency matters enormously, models predict spread dynamics where high precision is less critical
- **Statistical arbitrage:** INT8 or FP16 — models must detect small price discrepancies accurately
- **Portfolio optimization:** FP16 or FP32 — numerical precision in covariance matrices matters; inference latency is less critical (decisions at minute/hour frequency)
- **Trend following:** INT4 acceptable — directional signals are robust to quantization noise

## 6. Implementation Walkthrough

The Rust implementation in this chapter provides a complete quantization toolkit for trading models. Here is an overview of the key components:

### 6.1 Quantization Schemes

The implementation supports four quantization schemes:
- **Symmetric per-tensor:** Single scale factor, zero-point fixed at 0
- **Asymmetric per-tensor:** Scale and zero-point computed from min/max
- **Symmetric per-channel:** Per-channel scale factors for weight tensors
- **Asymmetric per-channel:** Per-channel scale and zero-point

### 6.2 Quantization Pipeline

```
Raw FP32 Model
    |
    v
Calibration (collect min/max statistics)
    |
    v
Compute Scale & Zero-Point
    |
    v
Quantize Weights (FP32 → INT8/INT4)
    |
    v
Quantized Inference (integer arithmetic)
    |
    v
Dequantize Output (INT32 → FP32)
```

### 6.3 Quantized Matrix Multiplication

The core operation in neural network inference is matrix multiplication. With quantized weights and activations:

```
Y_fp32 = X_fp32 * W_fp32

becomes:

Y_int32 = (X_int8 - Zx) * (W_int8 - Zw)
Y_fp32 = Sx * Sw * Y_int32
```

This replaces expensive FP32 multiplications with INT8 multiplications followed by a single FP32 scaling at the output — a significant speedup on hardware with integer multiply-accumulate units.

### 6.4 Code Structure

The Rust crate is organized as follows:

- `lib.rs` — Core quantization functions, neural network layers, Bybit API client
- `examples/trading_example.rs` — Complete example fetching live data and comparing precisions

Key types:
- `QuantizationScheme` — Enum of supported quantization strategies
- `QuantizationParams` — Computed scale and zero-point
- `QuantizedTensor` — INT8 data with quantization parameters
- `QuantizedLinearLayer` — A fully-connected layer operating in quantized precision
- `TradingModel` — Multi-layer perceptron for price prediction

## 7. Bybit Data Integration

The implementation connects to the Bybit public API to fetch real-time kline (candlestick) data:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=5&limit=200
```

This returns OHLCV data that is transformed into features for the trading model:
- **Price returns:** `(close - open) / open`
- **High-low range:** `(high - low) / open`
- **Volume change:** Normalized volume relative to moving average

The features are normalized to the `[-1, 1]` range before quantization, which improves INT8 and INT4 accuracy by ensuring the quantization range is well-utilized.

### Integration Notes

- The API requires no authentication for public market data
- Rate limits are generous for kline endpoints (10 requests/second)
- Data is returned as string arrays and parsed into `f32` values
- The example fetches 200 candles of 5-minute data, sufficient for training a simple model and demonstrating quantization effects

## 8. Key Takeaways

1. **Quantization is essential for production trading models.** The 2-4x speedup from INT8 quantization comes with minimal accuracy loss (<1% for most architectures) and directly reduces inference latency.

2. **INT8 is the practical sweet spot.** For most trading applications, INT8 provides the best tradeoff between compression (4x), speedup (2-4x), and accuracy preservation. It is natively supported by CPUs, GPUs, and FPGAs.

3. **Post-training quantization is usually sufficient for INT8.** PTQ with min-max or percentile calibration preserves accuracy well at INT8 precision. QAT is needed primarily for INT4 and below.

4. **Per-channel quantization improves accuracy.** Computing separate scale factors for each output channel of weight tensors significantly reduces quantization error with minimal overhead.

5. **Symmetric quantization is preferred for weights; asymmetric for activations.** Weights are typically centered around zero (symmetric), while activations after ReLU are non-negative (asymmetric).

6. **The choice of precision depends on the trading strategy.** Latency-sensitive strategies (market making) benefit most from aggressive quantization, while precision-sensitive strategies (stat arb) should be more conservative.

7. **Memory bandwidth is often the real bottleneck.** Quantization reduces not just compute requirements but memory traffic, which is frequently the limiting factor in multi-model trading systems.

8. **Always validate quantized models on realistic market data.** Quantization error can compound through layers and interact with specific market regimes. Test across multiple market conditions before deployment.

9. **Rust provides an excellent platform for quantized inference.** Zero-cost abstractions, no garbage collection pauses, and direct hardware access make Rust ideal for implementing quantized trading models in production.

10. **Quantization is complementary to other optimization techniques.** Combine with pruning, knowledge distillation, and architecture search for maximum inference efficiency.
