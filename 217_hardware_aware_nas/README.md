# Chapter 217: Hardware-Aware Neural Architecture Search (NAS)

## 1. Introduction

Neural Architecture Search (NAS) has transformed the way we design deep learning models, automating what was once a painstaking manual process. However, traditional NAS optimizes purely for accuracy, producing architectures that may be impractical to deploy on real hardware. In high-frequency trading (HFT), a model that achieves state-of-the-art prediction accuracy but cannot produce an inference within 10 microseconds is worthless. In edge-based retail trading applications, a model that exceeds available memory is equally useless.

Hardware-Aware NAS addresses this gap by co-optimizing accuracy and hardware efficiency. Rather than searching for the best-performing architecture and then struggling to compress it onto target hardware, hardware-aware NAS incorporates hardware constraints directly into the search objective. The result is an architecture that is born efficient --- one that meets latency SLAs, fits within memory budgets, and maximizes accuracy within those constraints.

For algorithmic trading, this approach is transformative. Different trading strategies deploy on vastly different hardware: FPGA-based systems for sub-microsecond HFT, GPU clusters for batch portfolio optimization, and mobile CPUs for retail trading apps. Each hardware target imposes distinct constraints on model architecture. Hardware-aware NAS allows us to systematically discover the optimal architecture for each deployment scenario.

This chapter covers the mathematical foundations of hardware-aware NAS, explores hardware-specific considerations for trading, walks through implementations adapted from MnasNet, EfficientNet, and FBNet, and provides a complete Rust implementation with Bybit market data integration.

## 2. Mathematical Foundation

### 2.1 Latency as a Differentiable Constraint

The core innovation of hardware-aware NAS is incorporating hardware metrics directly into the optimization objective. Let us define the search space formally.

Given a search space of architectures $\mathcal{A}$, traditional NAS solves:

$$a^* = \arg\max_{a \in \mathcal{A}} \text{ACC}(a)$$

where $\text{ACC}(a)$ is the validation accuracy of architecture $a$. Hardware-aware NAS transforms this into a constrained or multi-objective problem:

$$a^* = \arg\max_{a \in \mathcal{A}} \text{ACC}(a) \quad \text{subject to} \quad \text{LAT}(a) \leq T$$

where $\text{LAT}(a)$ is the latency of architecture $a$ on the target hardware and $T$ is the latency budget.

### 2.2 Latency Lookup Tables

Directly measuring latency during search is prohibitively expensive. Instead, we construct latency lookup tables (LUTs) that map each candidate operation to its measured latency on the target device. For a neural network composed of $L$ layers, each with operation $o_i$ and input dimensions $d_i$, the total latency is approximated as:

$$\text{LAT}(a) = \sum_{i=1}^{L} \text{LUT}(o_i, d_i)$$

This decomposition assumes sequential execution, which is valid for most inference pipelines. For architectures with parallel branches, we take the maximum across parallel paths:

$$\text{LAT}_{\text{parallel}}(a) = \max_{p \in \text{paths}} \sum_{i \in p} \text{LUT}(o_i, d_i)$$

The LUT approach makes latency differentiable with respect to architecture choices. If we parameterize the architecture using a continuous relaxation (as in DARTS), where $\alpha_{i,j}$ is the weight of operation $j$ at layer $i$, the expected latency becomes:

$$\mathbb{E}[\text{LAT}(a)] = \sum_{i=1}^{L} \sum_{j=1}^{|\mathcal{O}|} \frac{\exp(\alpha_{i,j})}{\sum_k \exp(\alpha_{i,k})} \cdot \text{LUT}(o_j, d_i)$$

This is fully differentiable with respect to $\alpha$, enabling gradient-based optimization.

### 2.3 FLOPs Counting

Floating-point operations (FLOPs) provide a hardware-independent proxy for computational cost. For common operations:

- **Convolution**: $\text{FLOPs} = 2 \times H_{out} \times W_{out} \times K^2 \times C_{in} \times C_{out}$
- **Depthwise Convolution**: $\text{FLOPs} = 2 \times H_{out} \times W_{out} \times K^2 \times C$
- **Fully Connected**: $\text{FLOPs} = 2 \times C_{in} \times C_{out}$
- **Attention (self)**: $\text{FLOPs} = 4 \times n^2 \times d + 2 \times n \times d^2$

For time-series trading models, the sequence length $n$ is particularly important. Attention-based models scale quadratically in $n$, making them expensive for long lookback windows.

### 2.4 Multi-Objective Optimization

Rather than imposing a hard constraint, we can search for the entire Pareto front of accuracy vs. latency vs. memory. A solution $a$ dominates $a'$ if $a$ is at least as good as $a'$ in all objectives and strictly better in at least one.

The scalarized objective combines multiple objectives with weights:

$$\mathcal{L}(a) = \text{ACC}(a) \times \left[\frac{\text{LAT}(a)}{T}\right]^{w_1} \times \left[\frac{\text{MEM}(a)}{M}\right]^{w_2}$$

where $T$ and $M$ are target latency and memory, and $w_1, w_2 < 0$ are penalty weights. MnasNet uses $w_1 = -0.07$ as a soft penalty that degrades gracefully rather than imposing a hard cutoff.

### 2.5 Memory Footprint Estimation

Memory consumption during inference has two components:

$$\text{MEM}(a) = \text{MEM}_{\text{params}}(a) + \text{MEM}_{\text{activations}}(a)$$

Parameter memory is straightforward:

$$\text{MEM}_{\text{params}} = \sum_{i=1}^{L} |\theta_i| \times b$$

where $|\theta_i|$ is the parameter count at layer $i$ and $b$ is bytes per parameter (4 for FP32, 2 for FP16, 1 for INT8).

Activation memory depends on the execution schedule:

$$\text{MEM}_{\text{activations}} = \max_{i} \left( \text{size}(x_i) + \text{size}(x_{i+1}) \right) \times b$$

For trading models deployed on memory-constrained devices, this analysis determines whether an architecture is feasible.

## 3. Hardware Targets

### 3.1 FPGA for High-Frequency Trading

FPGAs are the gold standard for HFT, offering sub-microsecond inference latency with deterministic timing. Hardware-aware NAS for FPGA targets must account for:

- **Fixed-point arithmetic**: FPGA implementations typically use INT8 or fixed-point representations. The search space should include quantization-aware operations.
- **Parallelism constraints**: FPGAs have limited DSP slices and BRAM. The architecture must fit within these resource budgets.
- **Pipeline depth**: Deeper pipelines increase throughput but also latency. For HFT, we optimize for latency, not throughput.
- **Operation support**: Not all operations are efficiently implementable on FPGA. Avoid operations like dynamic shapes or complex nonlinearities.

Recommended search space for FPGA: small convolutions (1x1, 3x3), depthwise separable convolutions, ReLU activations (not GELU or Swish), and skip connections with fixed topology.

### 3.2 GPU for Batch Inference

GPU-based trading systems handle batch inference for portfolio optimization, risk management, and medium-frequency strategies. Key considerations:

- **Batch efficiency**: GPU utilization improves dramatically with larger batch sizes. The architecture should have operations that parallelize well.
- **Memory bandwidth**: GPU performance is often memory-bound. Architectures should maximize arithmetic intensity (FLOPs per byte transferred).
- **Tensor Core utilization**: Modern GPUs have specialized hardware for matrix multiplications with specific dimensions (multiples of 8 for FP16, 16 for INT8). Architecture dimensions should align with these.
- **Kernel launch overhead**: Many small operations incur kernel launch overhead. Prefer fewer, larger operations.

### 3.3 CPU/Mobile for Retail Trading

Retail trading apps run on diverse hardware --- from high-end laptops to budget smartphones. Constraints include:

- **Single-thread performance**: Many mobile devices have limited multi-threading. Architectures should not assume high parallelism.
- **Cache hierarchy**: Operations should exhibit good spatial and temporal locality.
- **Power consumption**: Battery-powered devices require energy-efficient architectures.
- **Model size**: App store size limits and download times constrain total model size, typically to under 10 MB.

## 4. Trading Applications

### 4.1 Meeting Strict Latency SLAs

In production trading systems, latency SLAs are non-negotiable. A model that occasionally exceeds its latency budget can cause missed trades, stale signals, or regulatory violations. Hardware-aware NAS provides architectures with predictable, bounded latency.

For example, a market-making system with a 50-microsecond decision budget might allocate 20 microseconds to feature computation, 20 microseconds to model inference, and 10 microseconds to order construction. The NAS search constrains model latency to 20 microseconds on the target FPGA.

### 4.2 FPGA-Deployable Architectures

Traditional NAS might discover an architecture using operations that are impossible or extremely expensive on FPGA (e.g., large matrix multiplications, dynamic routing, softmax over large dimensions). By constraining the search space to FPGA-friendly operations and using FPGA-specific latency tables, we guarantee that discovered architectures are directly synthesizable.

Architecture patterns that work well on FPGA for trading:
- Narrow, deep pipelines with small convolutions
- Binary or ternary weight networks for extreme throughput
- Fixed-topology skip connections (no dynamic routing)
- Lookup-table-based activation functions

### 4.3 Memory-Bounded Models for Edge Trading

Edge trading devices (co-located servers, embedded systems at exchange proximity) often have strict memory constraints. A model consuming 2 GB of memory may be impractical when the device has only 4 GB total, shared with the operating system, data feeds, and order management.

Hardware-aware NAS with memory constraints discovers architectures that:
- Use parameter-efficient operations (depthwise separable convolutions, grouped convolutions)
- Minimize peak activation memory through architecture topology
- Enable weight sharing across time steps
- Support aggressive quantization without accuracy degradation

## 5. Adapted Approaches for Trading

### 5.1 MnasNet Approach

MnasNet uses a reinforcement learning controller to sample architectures, evaluates them on real hardware, and optimizes the scalarized accuracy-latency objective. For trading, we adapt this by:

- Replacing ImageNet accuracy with trading-specific metrics (Sharpe ratio, P&L prediction accuracy)
- Using trading-hardware latency tables instead of mobile phone measurements
- Adding memory as an additional objective
- Constraining the search space to operations supported by the target trading hardware

### 5.2 EfficientNet Compound Scaling

EfficientNet's insight is that width, depth, and resolution should be scaled together using a compound coefficient. For trading time-series models, we adapt the scaling dimensions:

- **Depth**: Number of temporal processing layers
- **Width**: Number of channels (feature dimensions)
- **Resolution**: Input sequence length (lookback window)

The compound scaling rule $d = \alpha^\phi$, $w = \beta^\phi$, $r = \gamma^\phi$ subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ provides a principled way to scale models up or down to meet hardware budgets.

### 5.3 FBNet Differentiable Search

FBNet makes the search fully differentiable using the Gumbel-Softmax trick, enabling gradient-based optimization of the architecture. This is significantly faster than RL-based search. For trading applications:

- The differentiable formulation allows direct gradient flow from latency loss to architecture parameters
- Search completes in hours rather than days
- Temperature annealing gradually sharpens the architecture distribution from soft mixture to hard selection

## 6. Implementation Walkthrough (Rust)

Our Rust implementation provides a complete hardware-aware NAS framework for trading models. The key components are:

### 6.1 Hardware Profiles

We define hardware profiles that capture the latency characteristics of each target device. Each profile contains a lookup table mapping (operation, input_size, output_size) tuples to measured latency values.

```rust
pub struct HardwareProfile {
    pub name: String,
    pub device_type: DeviceType,
    pub latency_table: HashMap<OpSpec, f64>,
    pub memory_budget_bytes: u64,
    pub peak_flops: f64,
}
```

### 6.2 Architecture Representation

Architectures are represented as sequences of layer specifications, each defining the operation type, input/output dimensions, and optional parameters:

```rust
pub struct Architecture {
    pub layers: Vec<LayerSpec>,
}

pub struct LayerSpec {
    pub op: OpType,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub use_skip: bool,
}
```

### 6.3 Multi-Objective Search

The search procedure evaluates candidate architectures on accuracy (estimated via a proxy task), latency (from lookup tables), and memory (computed analytically). It maintains a Pareto front of non-dominated solutions:

```rust
pub fn search(
    search_space: &SearchSpace,
    hardware: &HardwareProfile,
    latency_budget: f64,
    num_iterations: usize,
) -> Vec<(Architecture, Metrics)> { ... }
```

### 6.4 Bybit Integration

We fetch real market data from Bybit's public API to use as training data for evaluating candidate architectures:

```rust
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> { ... }
```

The complete implementation is available in `rust/src/lib.rs` with a trading example in `rust/examples/trading_example.rs`.

## 7. Bybit Data Integration

The implementation fetches OHLCV data from Bybit's V5 API for use in architecture evaluation. The data pipeline:

1. **Fetch**: Retrieve kline data for BTCUSDT (or any symbol) using the public REST API
2. **Preprocess**: Normalize prices using rolling z-scores, compute technical features (returns, volatility, momentum)
3. **Evaluate**: Use the preprocessed data as input to candidate architectures for proxy accuracy estimation
4. **Score**: Combine accuracy estimate with latency and memory metrics for multi-objective ranking

The Bybit integration allows us to evaluate architectures on realistic trading data rather than synthetic benchmarks, ensuring that discovered architectures generalize to actual market conditions.

Data from Bybit is fetched using the endpoint:
```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=15&limit=200
```

This provides 200 candles of 15-minute BTCUSDT perpetual futures data, sufficient for evaluating architecture quality on a proxy task.

## 8. Key Takeaways

1. **Hardware awareness is essential for production trading models.** A model that cannot meet latency SLAs is useless regardless of its accuracy. Hardware-aware NAS builds these constraints into the search process rather than treating them as an afterthought.

2. **Latency lookup tables make hardware metrics differentiable.** By pre-measuring operation latencies on target hardware, we can incorporate latency into gradient-based optimization, dramatically speeding up the search.

3. **Different hardware requires different architectures.** The optimal architecture for FPGA-based HFT looks nothing like the optimal architecture for GPU batch inference. Hardware-aware NAS systematically discovers the right architecture for each target.

4. **Multi-objective optimization reveals the accuracy-efficiency frontier.** Rather than a single "best" architecture, we discover the entire Pareto front, allowing practitioners to choose the right trade-off for their specific deployment constraints.

5. **FLOPs alone are insufficient.** FLOPs correlate with but do not determine latency. Memory access patterns, operation-level parallelism, and hardware-specific optimizations all affect real-world performance. Lookup tables capture these effects.

6. **Trading-specific adaptations are necessary.** Standard NAS approaches designed for computer vision must be adapted for time-series trading data: different input dimensions, different accuracy metrics (Sharpe ratio, prediction accuracy), and different hardware targets (FPGA, co-located servers).

7. **Compound scaling provides principled model sizing.** EfficientNet's compound scaling adapts well to trading models, providing a systematic way to trade off model capacity for hardware efficiency across depth, width, and sequence length dimensions.

8. **Rust provides the performance characteristics needed for production.** The implementation in Rust ensures that the NAS framework itself introduces minimal overhead, and discovered architectures can be efficiently deployed in latency-sensitive trading systems.

9. **Real market data validation is critical.** Evaluating architectures on synthetic data may discover architectures that fail on real market data. Integration with exchanges like Bybit provides realistic evaluation conditions.

10. **The field is rapidly evolving.** Hardware-aware NAS techniques continue to advance, with new approaches for once-for-all networks, zero-cost proxies, and hardware-aware training emerging regularly. Staying current with these developments is essential for maintaining a competitive edge in algorithmic trading.
