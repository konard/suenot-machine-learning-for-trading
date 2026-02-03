# Chapter 114: Attention Visualization for Trading

## Overview

Attention Visualization is a powerful interpretability technique for understanding how Transformer-based models make predictions in financial markets. By visualizing the attention weights learned during training, traders and researchers can gain insights into which historical time steps, features, or market events the model considers most important when generating trading signals.

This chapter implements attention visualization methods for interpreting Transformer models applied to cryptocurrency trading on Bybit and stock market prediction, enabling transparent and explainable AI-driven trading strategies.

## Table of Contents

1. [Theoretical Foundations](#theoretical-foundations)
2. [Attention Mechanism Deep Dive](#attention-mechanism-deep-dive)
3. [Visualization Techniques](#visualization-techniques)
4. [Implementation](#implementation)
5. [Trading Strategy](#trading-strategy)
6. [Results and Metrics](#results-and-metrics)
7. [References](#references)

## Theoretical Foundations

### The Attention Mechanism

The self-attention mechanism, introduced in "Attention Is All You Need" (Vaswani et al., 2017), computes a weighted sum of values based on the compatibility between queries and keys:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Where:
- `Q ∈ ℝ^{n×d_k}` is the query matrix
- `K ∈ ℝ^{n×d_k}` is the key matrix
- `V ∈ ℝ^{n×d_v}` is the value matrix
- `d_k` is the dimension of keys (scaling factor)
- `n` is the sequence length

The attention weights `A = softmax(QK^T / sqrt(d_k))` form an `n × n` matrix where `A[i,j]` represents how much position `i` attends to position `j`.

### Multi-Head Attention

Multi-head attention runs `h` parallel attention functions, each with its own learned projections:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

where head_i = Attention(Q*W_Q_i, K*W_K_i, V*W_V_i)
```

Each head can learn to attend to different patterns:
- **Head 1**: Short-term momentum (recent prices)
- **Head 2**: Volatility clusters (high-volume periods)
- **Head 3**: Long-term trends (distant historical context)

### Why Visualization Matters for Trading

| Aspect | Benefit |
|--------|---------|
| Transparency | Understand *why* a trade signal was generated |
| Risk Management | Identify if model relies on spurious correlations |
| Feature Engineering | Discover which inputs drive predictions |
| Regime Detection | See how attention shifts in different market conditions |
| Debugging | Identify model failures (e.g., attention collapse) |

## Attention Mechanism Deep Dive

### Attention Patterns in Financial Data

When applying attention to financial time series, several characteristic patterns emerge:

1. **Local Attention**: Strong weights on recent timesteps (short-term momentum)
2. **Periodic Attention**: Weights on same time-of-day or day-of-week (seasonality)
3. **Event-Driven Attention**: Spikes at high-volume or high-volatility points
4. **Mean-Reverting Attention**: Focus on extreme deviation points

### Attention Score Computation

For a Transformer encoder processing OHLCV data:

```python
# Input: (batch, seq_len, features) where features = [open, high, low, close, volume]
# After embedding: (batch, seq_len, d_model)

# Attention scores for a single head
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

# attention_weights[b, i, j] = how much position i attends to position j
```

### Attention Weight Properties

1. **Row-wise sum to 1**: Each position's attention over all other positions sums to 1
2. **Non-negative**: All weights are ≥ 0 (due to softmax)
3. **Learnable patterns**: Weights are determined by learned Q, K projections

## Visualization Techniques

### 1. Attention Heatmaps

The most direct visualization: plot the attention matrix as a 2D heatmap.

```
Position (Query)
    ↓
    |  t-4   t-3   t-2   t-1   t
----+-----------------------------
t-4 | 0.05  0.10  0.15  0.20  0.50  ← Position (Key)
t-3 | 0.10  0.15  0.20  0.25  0.30
t-2 | 0.15  0.20  0.25  0.20  0.20
t-1 | 0.10  0.15  0.30  0.25  0.20
t   | 0.05  0.10  0.20  0.25  0.40
```

### 2. Attention Flow (BertViz-style)

Visualize attention as connections between input tokens/timesteps:

```
Time:  t-4    t-3    t-2    t-1    t
        │      │      │      │      │
        │      │      ├──────┼──────┤ (strong)
        │      │      │      ├──────┤ (medium)
        │      │      │      │      │
       [P]    [P]    [P]    [P]    [P] ← Predictions
```

### 3. Head-wise Analysis

Compare attention patterns across different heads:

| Head | Pattern | Trading Interpretation |
|------|---------|----------------------|
| 1 | Diagonal (local) | Recent price momentum |
| 2 | Vertical stripes | Specific time-of-day focus |
| 3 | Sparse, high-vol | Event/news reaction |
| 4 | Uniform | Baseline/averaging |

### 4. Layer-wise Aggregation

Combine attention across layers using:
- **Attention Rollout**: Multiply attention matrices across layers
- **Attention Flow**: Account for residual connections

```
A_combined = A_L * A_{L-1} * ... * A_1
```

### 5. Feature Attribution via Attention

Weight input features by aggregated attention:

```python
# attention_weights: (batch, heads, seq_len, seq_len)
# Average over heads
avg_attention = attention_weights.mean(dim=1)  # (batch, seq_len, seq_len)

# Sum attention received by each position
position_importance = avg_attention.sum(dim=1)  # (batch, seq_len)

# Multiply with input features for attribution
feature_attribution = input_features * position_importance.unsqueeze(-1)
```

## Implementation

### Python

The Python implementation uses PyTorch and includes:

- **`python/model.py`**: Transformer model with attention weight extraction
- **`python/visualization.py`**: Attention heatmaps, flow diagrams, and analysis tools
- **`python/data_loader.py`**: Data loading for stock (yfinance) and crypto (Bybit) markets
- **`python/backtest.py`**: Backtesting engine with interpretability metrics

```python
from python.model import AttentionTransformer

model = AttentionTransformer(
    input_dim=8,          # OHLCV + technical indicators
    d_model=64,
    n_heads=4,
    n_layers=3,
    output_dim=3,         # return, direction, volatility
    dropout=0.1,
    return_attention=True  # Enable attention extraction
)

# Forward pass with attention weights
predictions, attention_weights = model(features)
# attention_weights: dict with keys 'layer_0', 'layer_1', ...
# Each value: (batch, heads, seq_len, seq_len)
```

### Rust

The Rust implementation provides a production-ready version:

- **`src/model/`**: Transformer with attention tracking
- **`src/visualization/`**: Efficient attention aggregation
- **`src/data/`**: Bybit API client and feature engineering
- **`src/trading/`**: Signal generation with attention-based confidence
- **`src/backtest/`**: Performance evaluation engine

```bash
# Run basic example
cargo run --example basic_attention

# Run visualization analysis
cargo run --example attention_analysis

# Run trading strategy
cargo run --example trading_strategy
```

## Trading Strategy

### Attention-Based Confidence Scoring

Use attention patterns to assess prediction confidence:

```python
def compute_attention_confidence(attention_weights):
    """
    High confidence: attention is focused (low entropy)
    Low confidence: attention is diffuse (high entropy)
    """
    # Compute entropy of attention distribution
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=-1)
    max_entropy = math.log(attention_weights.shape[-1])

    # Normalize to [0, 1], invert so focused = high confidence
    confidence = 1 - (entropy / max_entropy)
    return confidence.mean()
```

### Signal Generation with Interpretability

```python
class AttentionTradingStrategy:
    def generate_signal(self, model, features):
        pred_return, pred_direction, pred_vol, attention = model(features)

        # Compute attention-based confidence
        confidence = self.compute_attention_confidence(attention)

        # Only trade when attention is focused (confident)
        if confidence < self.confidence_threshold:
            return Signal.HOLD

        # Standard directional signal
        if pred_direction > 0.6 and pred_return > self.return_threshold:
            return Signal.LONG
        elif pred_direction < 0.4 and pred_return < -self.return_threshold:
            return Signal.SHORT
        return Signal.HOLD
```

### Risk Management via Attention Analysis

- **Attention Collapse Detection**: If attention becomes uniform, model may be uncertain
- **Regime Change Detection**: Sudden shift in attention patterns signals new market regime
- **Feature Importance Tracking**: Monitor which inputs drive decisions over time

## Results and Metrics

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MSE / MAE | Return prediction accuracy |
| Accuracy / F1 | Direction classification performance |
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk-adjusted return |
| Maximum Drawdown | Worst peak-to-trough decline |
| Attention Entropy | Measure of attention focus |
| Attention Stability | Consistency of patterns across time |

### Interpretability Analysis

1. **Head Specialization**: Measure how different each head's attention is
2. **Temporal Locality**: Quantify how much attention focuses on recent vs. distant past
3. **Feature Attribution Correlation**: Compare attention-based attribution with SHAP/LIME

### Comparison with Baselines

The attention visualization strategy is compared against:
- LSTM without attention
- Transformer without attention-based confidence filtering
- Random baseline
- Buy-and-hold benchmark

## Project Structure

```
114_attention_visualization_trading/
├── README.md                  # This file
├── README.ru.md               # Russian translation
├── readme.simple.md           # Simplified explanation (English)
├── readme.simple.ru.md        # Simplified explanation (Russian)
├── Cargo.toml                 # Rust project configuration
├── python/
│   ├── __init__.py
│   ├── model.py               # Transformer with attention extraction
│   ├── visualization.py       # Attention visualization tools
│   ├── data_loader.py         # Stock & crypto data loading
│   ├── backtest.py            # Backtesting framework
│   └── requirements.txt       # Python dependencies
├── src/
│   ├── lib.rs                 # Rust library root
│   ├── model/
│   │   ├── mod.rs             # Model module
│   │   └── transformer.rs     # Transformer implementation
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
    ├── basic_attention.rs     # Basic attention visualization
    ├── attention_analysis.rs  # Comprehensive analysis
    └── trading_strategy.rs    # Full trading strategy
```

## References

1. **Vaswani, A., et al. (2017)**. Attention Is All You Need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762

2. **Kobayashi, G., Kuribayashi, T., Yokoi, S., & Inui, K. (2020)**. Attention is Not Only a Weight: Analyzing Transformers with Vector Norms. *EMNLP 2020*. https://arxiv.org/abs/2004.10102

3. **Abnar, S., & Zuidema, W. (2020)**. Quantifying Attention Flow in Transformers. *ACL 2020*. https://arxiv.org/abs/2005.00928

4. **Vig, J. (2019)**. A Multiscale Visualization of Attention in the Transformer Model. *ACL 2019 System Demonstrations*. https://arxiv.org/abs/1906.05714

5. **Clark, K., Khandelwal, U., Levy, O., & Manning, C.D. (2019)**. What Does BERT Look At? An Analysis of BERT's Attention. *BlackboxNLP 2019*. https://arxiv.org/abs/1906.04341

6. **De Prado, M. L. (2018)**. Advances in Financial Machine Learning. *Wiley*.

## License

MIT
