# Chapter 139: SSM-GNN Hybrid Models for Trading

## Overview

State Space Model (SSM) and Graph Neural Network (GNN) hybrids combine the strengths of sequential temporal modeling with relational structure learning. SSMs such as S4 and Mamba excel at capturing long-range temporal dependencies in time series, while GNNs model inter-asset relationships, sector correlations, and supply chain linkages. By fusing these two paradigms, SSM-GNN hybrids can simultaneously learn *when* and *how* assets move (temporal dynamics) and *why* they co-move (graph structure).

In financial markets, assets do not exist in isolation. Stock returns are influenced by sector peers, macro factors propagate through supply chains, and cryptocurrency prices co-move through on-chain and exchange-level linkages. An SSM-GNN hybrid captures both the sequential evolution of each asset's features and the cross-asset information flow at every time step.

## Table of Contents

1. [Mathematical Foundation](#mathematical-foundation)
2. [Architecture Design](#architecture-design)
3. [SSM Component](#ssm-component)
4. [GNN Component](#gnn-component)
5. [Fusion Strategies](#fusion-strategies)
6. [Implementation in Python](#implementation-in-python)
7. [Implementation in Rust](#implementation-in-rust)
8. [Trading Application](#trading-application)
9. [Backtesting Framework](#backtesting-framework)
10. [Performance Evaluation](#performance-evaluation)
11. [References](#references)

---

## Mathematical Foundation

### State Space Models (SSM)

A continuous-time linear state space model is defined by:

```
x'(t) = A x(t) + B u(t)
y(t)  = C x(t) + D u(t)
```

where:
- `x(t) ∈ R^N` is the latent state,
- `u(t) ∈ R^1` is the input signal,
- `y(t) ∈ R^1` is the output,
- `A ∈ R^{N×N}`, `B ∈ R^{N×1}`, `C ∈ R^{1×N}`, `D ∈ R^{1×1}` are learnable parameters.

For discrete sequences, these are discretized (e.g., via zero-order hold) to produce:

```
x_k = Ā x_{k-1} + B̄ u_k
y_k = C x_k + D u_k
```

where `Ā = exp(Δ A)` and `B̄ = (Δ A)^{-1}(Ā - I) Δ B` for step size `Δ`.

### Graph Neural Networks (GNN)

Given a graph `G = (V, E)` with node features `h_v` for each node `v ∈ V`, message passing GNNs update node representations by:

```
m_v^{(l)} = AGGREGATE({h_u^{(l-1)} : u ∈ N(v)})
h_v^{(l)} = UPDATE(h_v^{(l-1)}, m_v^{(l)})
```

Common variants include GCN (Kipf & Welling, 2017), GAT (Veličković et al., 2018), and GraphSAGE (Hamilton et al., 2017).

### SSM-GNN Fusion

The hybrid model processes a temporal graph sequence `{G_1, G_2, ..., G_T}` where each graph `G_t = (V, E_t, X_t)` has time-varying node features `X_t ∈ R^{|V|×F}`. Two main fusion strategies exist:

**Sequential Fusion (SSM → GNN):**
1. Apply SSM independently per node to obtain temporal embeddings `z_v = SSM(x_{v,1}, ..., x_{v,T})`
2. Apply GNN on the graph with node features `z_v` to get the final representations

**Interleaved Fusion (SSM ↔ GNN):**
At each time step `t`:
1. Update node features with GNN: `h_v^{(t)} = GNN(x_{v,t}, G_t)`
2. Update temporal state with SSM: `s_v^{(t)} = SSM_step(s_v^{(t-1)}, h_v^{(t)})`

The interleaved approach is more expressive but computationally heavier.

---

## Architecture Design

```
Input: Multi-asset time series + Relationship graph
         │
         ▼
┌─────────────────────┐
│  Feature Extraction  │  Per-asset technical indicators
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   SSM Encoder        │  S4/Mamba per-node temporal encoding
│   (per node)         │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   GNN Layers         │  Cross-asset message passing
│   (GAT / GCN)        │  on correlation / sector graph
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Prediction Heads   │  Return forecast, volatility,
│                      │  trend classification
└─────────┬───────────┘
          │
          ▼
     Trading Signal
```

---

## SSM Component

The SSM component processes each asset's time series independently. We use a simplified S4-style diagonal SSM:

```python
class DiagonalSSM:
    """
    Diagonal State Space Model.
    Uses diagonal A matrix for efficient computation.
    Discretization via zero-order hold.
    """
    def __init__(self, d_model, d_state):
        # A is diagonal (stored as vector)
        self.A = init_hippo_diagonal(d_state)
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.zeros(d_model))
        self.log_dt = nn.Parameter(torch.log(torch.rand(d_model) * 0.1))
```

Key properties:
- **HiPPO initialization**: The `A` matrix is initialized using HiPPO (High-order Polynomial Projection Operators) for optimal long-range dependency capture.
- **Diagonal structure**: Restricting `A` to be diagonal allows `O(N)` computation per step vs. `O(N^2)` for dense matrices.
- **Selective mechanism** (Mamba-style): Input-dependent `Δ`, `B`, `C` allow the model to selectively remember or forget information.

---

## GNN Component

The GNN component captures cross-asset relationships. We support both static and dynamic graphs:

**Static graph construction:**
- Sector/industry membership (binary adjacency)
- Rolling correlation matrix thresholded at a cutoff

**Dynamic graph construction:**
- Attention-based edge weights learned from node embeddings
- Time-varying correlation estimated over sliding windows

The GNN layer uses Graph Attention Networks (GAT):

```
α_{ij} = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
h_i' = σ(Σ_j α_{ij} W h_j)
```

Multi-head attention with `K` heads:

```
h_i' = ||_{k=1}^{K} σ(Σ_j α_{ij}^k W^k h_j)
```

---

## Fusion Strategies

### Strategy 1: Sequential (SSM-then-GNN)

```
For each asset i:
    z_i = SSM(x_{i,1:T})     # temporal encoding

For each GNN layer l:
    z_i = GNN_layer(z_i, G)   # cross-asset aggregation

output_i = MLP(z_i)           # prediction head
```

**Pros**: Simple, each component can be pretrained independently.
**Cons**: Temporal and relational information are not jointly learned at each step.

### Strategy 2: Interleaved (SSM-GNN at each step)

```
For each time step t:
    For each asset i:
        h_i^t = SSM_step(state_i, x_{i,t})

    For each GNN layer l:
        h_i^t = GNN_layer(h_i^t, G_t)

    state_i = update(state_i, h_i^t)

output_i = MLP(state_i)
```

**Pros**: Rich interaction between temporal and relational dynamics.
**Cons**: Slower, harder to parallelize over time.

### Strategy 3: Parallel (SSM + GNN concatenation)

```
z_temporal_i = SSM(x_{i,1:T})
z_graph_i = GNN(mean_pool(x_{i,1:T}), G)
z_i = concat(z_temporal_i, z_graph_i)
output_i = MLP(z_i)
```

**Pros**: Fully parallelizable, captures complementary information.
**Cons**: No deep interaction between temporal and graph features.

---

## Implementation in Python

The Python implementation uses PyTorch with `torch_geometric` for GNN layers. Key files:

- `python/ssm_gnn_model.py` — Core SSM-GNN hybrid model
- `python/data_loader.py` — Data loading with stock and crypto (Bybit) data
- `python/backtest.py` — Backtesting engine with performance metrics

### Quick Start

```python
from python.ssm_gnn_model import SSMGNNHybrid
from python.data_loader import load_stock_data, build_correlation_graph

# Load data
prices, features = load_stock_data(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    period="2y"
)

# Build correlation graph
edge_index, edge_weight = build_correlation_graph(prices, threshold=0.5)

# Create model
model = SSMGNNHybrid(
    n_features=features.shape[-1],
    d_model=64,
    d_state=16,
    n_gnn_layers=2,
    n_heads=4,
    n_classes=3  # up / flat / down
)

# Forward pass
predictions = model(features, edge_index, edge_weight)
```

---

## Implementation in Rust

The Rust implementation provides a high-performance SSM-GNN engine suitable for production trading systems. It uses `ndarray` for numerical computation and implements the core SSM and GNN operations from scratch.

### Key modules:

- `src/ssm.rs` — Diagonal SSM with discretization
- `src/gnn.rs` — GAT-style graph attention layer
- `src/model.rs` — Combined SSM-GNN hybrid
- `src/data/` — Bybit API data fetching
- `src/trading/` — Signal generation and backtesting
- `examples/` — Working examples

### Quick Start

```rust
use ssm_gnn_hybrid::model::SsmGnnHybrid;
use ssm_gnn_hybrid::data::bybit::BybitClient;

#[tokio::main]
async fn main() {
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", "1h", 500).await.unwrap();

    let model = SsmGnnHybrid::new(10, 64, 16, 3);
    let prediction = model.forward(&features, &edge_index);
    println!("Signal: {:?}", prediction);
}
```

---

## Trading Application

### Signal Generation

The SSM-GNN hybrid generates trading signals by:

1. **Feature extraction**: Compute technical indicators (RSI, MACD, Bollinger Bands, OBV) per asset.
2. **Temporal encoding**: SSM processes each asset's feature sequence to produce latent states capturing temporal patterns.
3. **Cross-asset aggregation**: GNN propagates information across the asset graph, allowing the model to incorporate sector momentum, lead-lag effects, and correlation regime shifts.
4. **Classification**: A prediction head maps the fused representation to a 3-class signal: Long (+1), Neutral (0), Short (-1).

### Portfolio Construction

Given signals `s_i ∈ {-1, 0, +1}` for `N` assets:

```
w_i = s_i * confidence_i / Σ_j |s_j * confidence_j|
```

where `confidence_i` is the softmax probability of the predicted class.

### Risk Management

- **Position sizing**: Kelly criterion or fixed fractional
- **Stop-loss**: Trailing stop at 2× ATR
- **Correlation filter**: Reduce exposure when portfolio correlation exceeds threshold

---

## Backtesting Framework

The backtesting framework evaluates the SSM-GNN strategy on historical data:

```python
from python.backtest import Backtester

bt = Backtester(
    initial_capital=100_000,
    commission=0.001,
    slippage=0.0005
)

results = bt.run(signals, prices)
print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
print(f"Max Drawdown: {results.max_drawdown:.3f}")
print(f"Total Return: {results.total_return:.3f}")
```

---

## Performance Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted return (annualized) |
| Sortino Ratio | Downside risk-adjusted return |
| Maximum Drawdown | Largest peak-to-trough decline |
| Win Rate | Fraction of profitable trades |
| Profit Factor | Gross profit / Gross loss |
| Calmar Ratio | Annualized return / Max drawdown |

### Baseline Comparisons

The SSM-GNN hybrid is compared against:
- **SSM-only**: Temporal model without graph structure
- **GNN-only**: Graph model without temporal SSM encoding
- **LSTM**: Standard recurrent baseline
- **Transformer**: Self-attention based temporal model
- **Buy-and-Hold**: Passive benchmark

### Expected Advantages

1. **Better cross-asset signals**: GNN captures lead-lag relationships that SSM alone misses.
2. **Regime awareness**: SSM's long-range memory detects regime changes; GNN propagates regime information across the asset graph.
3. **Robustness**: Joint training regularizes both components, reducing overfitting on noisy financial data.

---

## References

1. Gu, A., Goel, K., & Ré, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces* (S4). ICLR 2022.
2. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
3. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017.
4. Veličković, P., et al. (2018). *Graph Attention Networks*. ICLR 2018.
5. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS 2017.
6. Wang, Y., et al. (2023). *Graph State Space Models*. arXiv:2301.01731.
7. Chen, D., et al. (2022). *Structure-Aware Transformer for Graph Representation Learning*. ICML 2022.
