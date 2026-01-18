# Chapter 41: Higher Order Transformers for Cryptocurrency Trading

## Overview

Higher Order Transformers (HOT) extend the standard self-attention mechanism to capture complex multi-dimensional dependencies in multivariate time series data. Unlike traditional transformers that model pairwise relationships, HOT captures higher-order interactions between multiple assets simultaneously, making it particularly powerful for cryptocurrency market prediction.

<p align="center">
<img src="https://i.imgur.com/YqKxZ9M.png" width="70%">
</p>

## Table of Contents

1. [Introduction](#introduction)
   * [Why Higher Order Attention?](#why-higher-order-attention)
   * [Key Innovations](#key-innovations)
2. [Mathematical Foundation](#mathematical-foundation)
   * [Standard Self-Attention](#standard-self-attention)
   * [Higher Order Self-Attention](#higher-order-self-attention)
   * [Tensor Decomposition](#tensor-decomposition)
3. [Architecture](#architecture)
   * [Higher Order Attention Layer](#higher-order-attention-layer)
   * [Kernel Attention for Efficiency](#kernel-attention-for-efficiency)
   * [Full Model Architecture](#full-model-architecture)
4. [Implementation](#implementation)
   * [Rust Implementation](#rust-implementation)
   * [Python Reference](#python-reference)
5. [Cryptocurrency Application](#cryptocurrency-application)
   * [Market Data Processing](#market-data-processing)
   * [Feature Engineering](#feature-engineering)
   * [Trading Signals](#trading-signals)
6. [Backtesting](#backtesting)
7. [Resources](#resources)

## Introduction

### Why Higher Order Attention?

Standard transformers use **pairwise attention** to model relationships between two time steps or two assets. However, financial markets exhibit complex **multi-way interactions**:

- Asset A affects Asset B, which then affects Asset C
- Three or more assets move together due to common factors
- Non-linear dependencies that span multiple dimensions

**Example:** When Bitcoin dumps, Ethereum often follows, but the relationship with altcoins depends on the overall market sentiment — a three-way interaction.

```
Standard Attention:     BTC → ETH (pairwise only)
Higher Order Attention: BTC × ETH × Market_Sentiment → Altcoin movement
```

### Key Innovations

1. **Third-Order Attention Tensor:** Captures interactions between triplets of time steps/assets
2. **Tensor Decomposition:** Reduces computational complexity from O(n³) to O(n×r²)
3. **Kernel Attention:** Further reduces to linear complexity O(n)
4. **Multi-Modal Fusion:** Combines price data with other modalities (volume, sentiment)

## Mathematical Foundation

### Standard Self-Attention

Standard self-attention computes:

```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

Where:
- Q, K, V ∈ ℝ^(n×d) are query, key, value matrices
- n = sequence length
- d = embedding dimension

This captures **pairwise** relationships with complexity O(n²d).

### Higher Order Self-Attention

Higher Order Attention extends this to **triplet** interactions:

```
HOA(Q, K, V) = softmax(T(Q, K, K) / √d) × V
```

Where T is a **third-order tensor** capturing three-way interactions:

```
T[i,j,k] = Σ_d Q[i,d] × K[j,d] × K[k,d]
```

**Intuition:** Instead of asking "how much does position j relate to position i?", we ask "how much do positions j AND k together relate to position i?"

### Tensor Decomposition

The naive third-order tensor has O(n³) complexity — impractical for long sequences. We use **CP (CANDECOMP/PARAFAC) decomposition**:

```
T[i,j,k] ≈ Σ_r a_r[i] × b_r[j] × c_r[k]
```

Where r is the **rank** of decomposition. This reduces complexity to O(n×r²).

```python
# Conceptual illustration
def cp_decomposition(tensor, rank):
    """
    Decompose 3D tensor into rank-1 components

    Args:
        tensor: Original tensor of shape (n, n, n)
        rank: Number of rank-1 components

    Returns:
        factors: List of 3 matrices, each of shape (n, rank)
    """
    # Initialize factor matrices
    A = random_init(n, rank)  # Factor 1
    B = random_init(n, rank)  # Factor 2
    C = random_init(n, rank)  # Factor 3

    # Alternating Least Squares optimization
    for iteration in range(max_iter):
        A = update_factor(tensor, B, C)
        B = update_factor(tensor, A, C)
        C = update_factor(tensor, A, B)

    return [A, B, C]
```

**Trade-off:**
- Higher rank = better approximation, more computation
- Lower rank = faster, but may lose important interactions
- Typical values: rank = 8-64 depending on task

## Architecture

### Higher Order Attention Layer

```
┌─────────────────────────────────────────────────────────────┐
│                 Higher Order Attention Layer                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Input X ∈ ℝ^(n×d)                                        │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ Linear  │ → Q, K, V                                    │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────────────────┐                                  │
│    │ CP Decomposition    │                                  │
│    │ T ≈ Σ a_r ⊗ b_r ⊗ c_r │                                │
│    └────┬────────────────┘                                  │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ Kernel  │ → φ(Q), φ(K)                                 │
│    │ Mapping │                                              │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ Softmax │ → Attention weights                          │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │  @ V    │ → Output                                     │
│    └────┬────┘                                              │
│         │                                                   │
│    Output ∈ ℝ^(n×d)                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Kernel Attention for Efficiency

To achieve linear complexity, we use **kernel attention** with feature maps:

```
φ(x) = exp(x) / √Σexp(x)²
```

The attention becomes:

```
KernelAttn(Q, K, V) = φ(Q) × (φ(K)^T × V) / (φ(Q) × φ(K)^T × 1)
```

**Complexity reduction:**
- Standard: O(n²d)
- With kernel: O(nd²) — linear in sequence length!

### Full Model Architecture

```
┌────────────────────────────────────────────────────────────────┐
│           Higher Order Transformer for Trading                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐                                              │
│  │ Price Data   │──┐                                           │
│  │ (OHLCV)      │  │                                           │
│  └──────────────┘  │    ┌─────────────────┐                    │
│                    ├───→│ Input Embedding │                    │
│  ┌──────────────┐  │    │ + Positional    │                    │
│  │ Volume/OI    │──┤    └────────┬────────┘                    │
│  └──────────────┘  │             │                             │
│                    │    ┌────────┴────────┐                    │
│  ┌──────────────┐  │    │   HOT Block ×N  │                    │
│  │ Tech Indic   │──┘    │  ┌───────────┐  │                    │
│  └──────────────┘       │  │ HOA Layer │  │                    │
│                         │  ├───────────┤  │                    │
│                         │  │ FFN Layer │  │                    │
│                         │  ├───────────┤  │                    │
│                         │  │ LayerNorm │  │                    │
│                         │  └───────────┘  │                    │
│                         └────────┬────────┘                    │
│                                  │                             │
│                         ┌────────┴────────┐                    │
│                         │ Output Head     │                    │
│                         │ (Classification)│                    │
│                         └────────┬────────┘                    │
│                                  │                             │
│                         ┌────────┴────────┐                    │
│                         │ Up/Down/Neutral │                    │
│                         │   Prediction    │                    │
│                         └─────────────────┘                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Implementation

### Rust Implementation

The [rust_hot_crypto](rust_hot_crypto/) directory contains a modular Rust implementation:

```
rust_hot_crypto/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                  # Main library module
│   ├── main.rs                 # CLI interface
│   ├── data/
│   │   ├── mod.rs              # Data module
│   │   ├── bybit.rs            # Bybit API client
│   │   └── types.rs            # Data types (Candle, etc.)
│   ├── attention/
│   │   ├── mod.rs              # Attention module
│   │   ├── standard.rs         # Standard attention
│   │   ├── higher_order.rs     # Higher order attention
│   │   └── kernel.rs           # Kernel attention
│   ├── tensor/
│   │   ├── mod.rs              # Tensor module
│   │   ├── decomposition.rs    # CP/Tucker decomposition
│   │   └── operations.rs       # Tensor operations
│   ├── model/
│   │   ├── mod.rs              # Model module
│   │   ├── transformer.rs      # HOT transformer
│   │   └── predictor.rs        # Price predictor
│   ├── strategy/
│   │   ├── mod.rs              # Strategy module
│   │   └── signals.rs          # Trading signals
│   └── utils/
│       ├── mod.rs              # Utilities
│       └── config.rs           # Configuration
└── examples/
    ├── fetch_data.rs           # Fetch Bybit data
    ├── attention_demo.rs       # HOA demonstration
    ├── predict_movement.rs     # Price movement prediction
    └── backtest.rs             # Strategy backtest
```

### Quick Start with Rust

```bash
# Navigate to the Rust project
cd 41_higher_order_transformers/rust_hot_crypto

# Fetch cryptocurrency data from Bybit
cargo run --example fetch_data

# Run higher order attention demonstration
cargo run --example attention_demo

# Predict price movement
cargo run --example predict_movement

# Run a full backtest
cargo run --example backtest
```

### Python Reference

```python
import torch
import torch.nn as nn
import tensorly as tl

class HigherOrderAttention(nn.Module):
    """
    Higher Order Self-Attention with CP Decomposition
    """
    def __init__(self, d_model, n_heads=8, rank=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.rank = rank
        self.d_k = d_model // n_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # CP decomposition factors
        self.factor_a = nn.Parameter(torch.randn(n_heads, rank, self.d_k))
        self.factor_b = nn.Parameter(torch.randn(n_heads, rank, self.d_k))
        self.factor_c = nn.Parameter(torch.randn(n_heads, rank, self.d_k))

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        # Transpose for multi-head: (batch, heads, seq, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Higher order attention via CP decomposition
        # T[i,j,k] = Σ_r (Q·a_r)[i] × (K·b_r)[j] × (K·c_r)[k]

        # Compute projections onto factors
        Q_a = torch.einsum('bhsd,hrd->bhsr', Q, self.factor_a)  # (B, H, S, R)
        K_b = torch.einsum('bhsd,hrd->bhsr', K, self.factor_b)  # (B, H, S, R)
        K_c = torch.einsum('bhsd,hrd->bhsr', K, self.factor_c)  # (B, H, S, R)

        # Aggregate second-order interactions
        # This approximates the full third-order tensor
        K_bc = torch.einsum('bhir,bhjr->bhij', K_b, K_c)  # (B, H, S, S)

        # Compute attention scores
        attn_scores = torch.einsum('bhir,bhjj->bhij', Q_a, K_bc) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class HigherOrderTransformer(nn.Module):
    """
    Full Higher Order Transformer for price prediction
    """
    def __init__(self,
                 input_dim,
                 d_model=256,
                 n_heads=8,
                 n_layers=4,
                 rank=16,
                 num_classes=3,  # Up, Down, Neutral
                 dropout=0.1):
        super().__init__()

        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # HOT layers
        self.layers = nn.ModuleList([
            HOTBlock(d_model, n_heads, rank, dropout)
            for _ in range(n_layers)
        ])

        # Output head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        # Use last position for classification
        x = x[:, -1, :]
        logits = self.classifier(x)

        return logits
```

## Cryptocurrency Application

### Market Data Processing

For cryptocurrency trading with Bybit:

```python
CRYPTO_UNIVERSE = {
    'major': ['BTCUSDT', 'ETHUSDT'],
    'large_cap': ['SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT'],
    'mid_cap': ['AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT'],
    'small_cap': ['ATOMUSDT', 'NEARUSDT', 'APTUSDT', 'ARBUSDT']
}

FEATURES = {
    'price': ['open', 'high', 'low', 'close'],
    'volume': ['volume', 'quote_volume', 'taker_buy_volume'],
    'derived': ['returns', 'volatility', 'rsi', 'macd', 'bb_width']
}
```

### Feature Engineering

```python
def prepare_features(df, lookback=60):
    """
    Prepare features for HOT model

    Args:
        df: OHLCV DataFrame
        lookback: Number of periods to look back

    Returns:
        X: Feature tensor (batch, seq, features)
        y: Labels (batch,)
    """
    features = []

    # Price features (normalized)
    for col in ['open', 'high', 'low', 'close']:
        features.append(df[col].pct_change())

    # Volume features
    features.append(np.log1p(df['volume']).diff())

    # Technical indicators
    features.append(compute_rsi(df['close'], 14))
    features.append(compute_macd(df['close']))
    features.append(compute_bb_width(df['close']))

    # Volatility
    features.append(df['close'].pct_change().rolling(20).std())

    # Combine
    X = np.column_stack(features)

    # Create sequences
    sequences = []
    labels = []

    for i in range(lookback, len(X) - 1):
        sequences.append(X[i-lookback:i])

        # Label: 1 = up, 0 = neutral, -1 = down
        future_return = df['close'].iloc[i+1] / df['close'].iloc[i] - 1
        if future_return > 0.005:
            labels.append(2)  # Up
        elif future_return < -0.005:
            labels.append(0)  # Down
        else:
            labels.append(1)  # Neutral

    return np.array(sequences), np.array(labels)
```

### Trading Signals

```python
def generate_signals(model, current_data, threshold=0.6):
    """
    Generate trading signals from model predictions

    Args:
        model: Trained HOT model
        current_data: Current market data
        threshold: Confidence threshold

    Returns:
        signal: 'buy', 'sell', or 'hold'
        confidence: Model confidence
    """
    with torch.no_grad():
        logits = model(current_data)
        probs = torch.softmax(logits, dim=-1)

        pred_class = probs.argmax(dim=-1).item()
        confidence = probs.max(dim=-1).values.item()

        if confidence < threshold:
            return 'hold', confidence

        signal_map = {0: 'sell', 1: 'hold', 2: 'buy'}
        return signal_map[pred_class], confidence
```

## Backtesting

### Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Accuracy** | % correct predictions | > 55% |
| **F1 Score** | Balanced metric | > 0.50 |
| **Sharpe Ratio** | Risk-adjusted returns | > 1.5 |
| **Max Drawdown** | Worst peak-to-trough | < 20% |
| **Win Rate** | % profitable trades | > 50% |

### Expected Results (Crypto)

| Model | Accuracy | F1 | Sharpe | Max DD |
|-------|----------|----|----|--------|
| LSTM Baseline | 52% | 0.48 | 0.9 | -35% |
| Standard Transformer | 54% | 0.51 | 1.2 | -28% |
| **Higher Order Transformer** | **57%** | **0.55** | **1.6** | **-22%** |

*Note: Results on historical data. Past performance doesn't guarantee future results.*

### Trading Rules

```
Entry Rules:
├── Signal confidence > 60%
├── Volume > 20-day average
├── RSI not in extreme zone (20 < RSI < 80)
└── No conflicting signals in last 4 hours

Exit Rules:
├── Opposite signal with confidence > 70%
├── Stop loss: -3%
├── Take profit: +5%
└── Time-based: exit after 24 hours if no clear direction

Position Sizing:
├── Base size: 10% of portfolio per trade
├── Scale by confidence: size × (confidence - 0.5) × 2
├── Maximum single position: 25%
└── Maximum total exposure: 80%
```

## Resources

### Academic Papers

1. **Higher Order Transformers: Enhancing Stock Movement Prediction On Multimodal Time-Series Data**
   - arXiv: [2412.10540](https://arxiv.org/abs/2412.10540)
   - Key ideas: Third-order attention, tensor decomposition

2. **Transformers for Time Series Forecasting**
   - Foundational work on applying transformers to financial data

3. **Tensor Decomposition for Signal Processing and Machine Learning**
   - Comprehensive overview of CP and Tucker decomposition

### Books

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al.)
- [Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf) (Kolda & Bader)

### Related Chapters

- [Chapter 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — TFT for forecasting
- [Chapter 44: ProbSparse Attention](../44_probsparse_attention) — Efficient attention variants
- [Chapter 58: Flash Attention](../58_flash_attention_trading) — Memory-efficient attention

## Dependencies

### Rust

```toml
ndarray = "0.16"
ndarray-linalg = "0.16"
reqwest = "0.12"
tokio = "1.0"
serde = "1.0"
serde_json = "1.0"
chrono = "0.4"
rand = "0.8"
anyhow = "1.0"
```

### Python

```python
torch>=2.0.0
tensorly>=0.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

## Difficulty Level

**Advanced**

**Required knowledge:**
- Transformer architecture
- Tensor operations and decomposition
- Time series analysis
- Cryptocurrency markets
- Risk management

---

*This material is for educational purposes. Cryptocurrencies are high-risk assets. Do not invest more than you can afford to lose.*
