# Chapter 44: ProbSparse Attention for Trading

This chapter explores **ProbSparse Attention**, an efficient self-attention mechanism that reduces computational complexity from O(L²) to O(L·log(L)). Originally introduced in the Informer paper for long sequence time-series forecasting, ProbSparse Attention is particularly valuable for financial applications that process extended historical data.

<p align="center">
<img src="https://i.imgur.com/QR7Zk8v.png" width="70%">
</p>

## Contents

1. [Introduction to ProbSparse Attention](#introduction-to-probsparse-attention)
    * [Why Efficient Attention Matters](#why-efficient-attention-matters)
    * [Key Innovations](#key-innovations)
    * [Comparison with Other Methods](#comparison-with-other-methods)
2. [Mathematical Foundations](#mathematical-foundations)
    * [Query Sparsity Measurement](#query-sparsity-measurement)
    * [KL-Divergence Intuition](#kl-divergence-intuition)
    * [Top-u Query Selection](#top-u-query-selection)
3. [Architecture Components](#architecture-components)
    * [ProbSparse Self-Attention](#probsparse-self-attention)
    * [Self-Attention Distilling](#self-attention-distilling)
    * [Encoder Stack](#encoder-stack)
4. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: ProbSparse Attention Implementation](#02-probsparse-attention-implementation)
    * [03: Model Training](#03-model-training)
    * [04: Forecasting](#04-forecasting)
    * [05: Backtesting Strategy](#05-backtesting-strategy)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to ProbSparse Attention

### Why Efficient Attention Matters

Standard self-attention in Transformers has a fundamental limitation: **quadratic complexity O(L²)** where L is the sequence length. For financial time-series forecasting, this creates significant challenges:

```
Standard Attention Memory Usage:
┌─────────────────┬──────────────────┬────────────────┐
│ Sequence Length │ Standard O(L²)   │ ProbSparse     │
├─────────────────┼──────────────────┼────────────────┤
│ L = 96          │ 9,216 operations │ ~640 ops       │
│ L = 720 (month) │ 518,400 ops      │ ~4,700 ops     │
│ L = 8,760 (year)│ 76,737,600 ops   │ ~79,000 ops    │
└─────────────────┴──────────────────┴────────────────┘
```

For trading applications requiring 1+ year of hourly data, ProbSparse Attention makes Transformer models practical.

### Key Innovations

1. **Query Sparsity Measurement**: Not all queries contribute equally to attention. ProbSparse identifies "active" queries that generate diverse attention patterns and focuses computation there.

2. **Top-u Query Selection**: Only the most informative queries (u = c·log(L)) participate in full attention computation.

3. **Self-Attention Distilling**: Progressive reduction of sequence length through encoder layers eliminates redundancy.

```
┌──────────────────────────────────────────────────────────────────┐
│                    PROBSPARSE ATTENTION FLOW                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input Sequence: [q₁, q₂, q₃, ..., qₗ]  (L queries)             │
│                                                                   │
│         │ Calculate Sparsity Measurement M(qᵢ, K)                │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────┐                 │
│  │ M(qᵢ) = max(qᵢKᵀ/√d) - mean(qᵢKᵀ/√d)       │                 │
│  │                                              │                 │
│  │ "Active" queries: High M → Diverse attention │                 │
│  │ "Lazy" queries: Low M → Uniform attention    │                 │
│  └─────────────────────────────────────────────┘                 │
│         │                                                         │
│         │ Select Top-u queries (u = c·log(L))                    │
│         ▼                                                         │
│  Q_reduce = [q₃, q₇, q₁₂, ...]  (only u queries)                │
│                                                                   │
│         │ Compute attention only for Q_reduce                    │
│         ▼                                                         │
│  Output: Sparse attention with O(L·log(L)) complexity            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Comparison with Other Methods

| Method | Complexity | Memory | Long-Range | Trading Use |
|--------|------------|--------|------------|-------------|
| Full Attention | O(L²) | O(L²) | ✓ | Short sequences only |
| Local Attention | O(L·k) | O(L·k) | Limited | Intraday patterns |
| Linformer | O(L·k) | O(L·k) | ✓ | General use |
| Performer | O(L·d) | O(L·d) | ✓ | General use |
| **ProbSparse** | O(L·logL) | O(L·logL) | ✓ | **Long sequences** |
| Flash Attention | O(L²) | O(L) | ✓ | Hardware optimized |

## Mathematical Foundations

### Query Sparsity Measurement

The core insight of ProbSparse is that attention scores follow a **long-tail distribution**. Most queries produce uniform attention (contributing little information), while a few "active" queries focus strongly on specific keys.

The **Query Sparsity Measurement** quantifies how "spiky" a query's attention distribution is:

```
M(qᵢ, K) = max_j(qᵢ · kⱼᵀ / √d) - (1/Lₖ) Σⱼ(qᵢ · kⱼᵀ / √d)
```

Where:
- `qᵢ` is the i-th query vector
- `kⱼ` are key vectors
- `d` is the embedding dimension
- `Lₖ` is the sequence length of keys

**Interpretation**:
- **High M(qᵢ)**: Query has a dominant key → "Active" query
- **Low M(qᵢ)**: Query attends uniformly → "Lazy" query

### KL-Divergence Intuition

The sparsity measurement M approximates the KL-divergence between the actual attention distribution and a uniform distribution:

```
KL(p || q_uniform) ≈ log(Lₖ) + M(qᵢ, K)
```

Active queries have high KL-divergence (far from uniform), while lazy queries have low KL-divergence (close to uniform).

```python
# Intuition: Active vs Lazy Queries
import numpy as np

# Active query: attends strongly to specific keys
active_attention = np.array([0.8, 0.1, 0.05, 0.03, 0.02])  # Spiky
M_active = active_attention.max() - active_attention.mean()  # High

# Lazy query: attends uniformly
lazy_attention = np.array([0.21, 0.20, 0.20, 0.19, 0.20])  # Flat
M_lazy = lazy_attention.max() - lazy_attention.mean()  # Low

print(f"Active M: {M_active:.3f}")  # ~0.6
print(f"Lazy M: {M_lazy:.3f}")      # ~0.01
```

### Top-u Query Selection

Given sparsity measurements, we select only the top-u queries for full attention computation:

```
u = min(c · log(Lq), Lq)
```

Where:
- `c` is the sampling factor (typically 5)
- `Lq` is the query sequence length

For a sequence of 720 time steps:
```
u = 5 × log(720) ≈ 5 × 6.58 ≈ 33 queries
```

This reduces operations from 720² = 518,400 to approximately 720 × 33 = 23,760 — a **22x reduction**.

## Architecture Components

### ProbSparse Self-Attention

```python
class ProbSparseAttention(nn.Module):
    """
    ProbSparse Self-Attention Mechanism

    Achieves O(L·log(L)) complexity by selecting only the most
    informative queries for full attention computation.
    """

    def __init__(self, d_model: int, n_heads: int, sampling_factor: float = 5.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.sampling_factor = sampling_factor
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Calculate number of top queries to select
        u = max(1, min(seq_len, int(self.sampling_factor * math.log(seq_len + 1))))

        # Calculate Query Sparsity Measurement M(q, K)
        # Sample keys for efficiency
        U_part = min(int(self.sampling_factor * seq_len * math.log(seq_len + 1)), seq_len)
        sample_idx = torch.randint(0, seq_len, (U_part,), device=x.device)
        K_sample = K[:, :, sample_idx, :]  # [batch, heads, U_part, head_dim]

        # Q·K_sample^T / sqrt(d)
        scores_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) / self.scale

        # M(q) = max(scores) - mean(scores)
        M = scores_sample.max(dim=-1)[0] - scores_sample.mean(dim=-1)  # [batch, heads, seq_len]

        # Select top-u queries
        M_top_indices = M.topk(u, dim=-1)[1]  # [batch, heads, u]

        # Gather selected queries
        batch_idx = torch.arange(batch, device=x.device)[:, None, None]
        head_idx = torch.arange(self.n_heads, device=x.device)[None, :, None]
        Q_reduce = Q[batch_idx, head_idx, M_top_indices]  # [batch, heads, u, head_dim]

        # Full attention for selected queries only
        attn_scores = torch.matmul(Q_reduce, K.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, V)  # [batch, heads, u, head_dim]

        # Initialize output with mean values, then fill in sparse positions
        output = V.mean(dim=2, keepdim=True).expand(-1, -1, seq_len, -1).clone()
        output.scatter_(2, M_top_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim), context)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(output)
```

### Self-Attention Distilling

The distilling operation progressively reduces sequence length between encoder layers:

```python
class AttentionDistilling(nn.Module):
    """
    Distilling layer that reduces sequence length by half.

    Uses Conv1d + ELU + MaxPool to extract salient features
    while discarding redundant information.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x.transpose(1, 2)  # [batch, seq_len//2, d_model]
```

### Encoder Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                     INFORMER ENCODER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: [batch, L, d_model]                                     │
│                                                                  │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │   Encoder Layer 1       │  ← ProbSparse Attention            │
│  │   [batch, L, d_model]   │                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐                                    │
│  │   Distilling Layer 1    │  ← Conv + MaxPool (L → L/2)        │
│  │   [batch, L/2, d_model] │                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐                                    │
│  │   Encoder Layer 2       │  ← ProbSparse Attention            │
│  │   [batch, L/2, d_model] │                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐                                    │
│  │   Distilling Layer 2    │  ← Conv + MaxPool (L/2 → L/4)      │
│  │   [batch, L/4, d_model] │                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐                                    │
│  │   Encoder Layer 3       │  ← ProbSparse Attention            │
│  │   [batch, L/4, d_model] │                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  Output: [batch, L/4, d_model]                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Practical Examples

### 01: Data Preparation

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import List, Tuple
import torch

def prepare_informer_data(
    df: pd.DataFrame,
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 24,
    features: List[str] = ['close', 'volume', 'high', 'low', 'open']
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for Informer model training.

    Args:
        df: DataFrame with OHLCV data
        seq_len: Input sequence length (encoder)
        label_len: Label sequence length (decoder start)
        pred_len: Prediction horizon
        features: Feature columns to use

    Returns:
        X: Input sequences [n_samples, seq_len, n_features]
        y: Target sequences [n_samples, pred_len]
    """

    # Calculate returns and technical indicators
    df = df.copy()
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(20).std()
    df['volume_ma'] = df['volume'] / df['volume'].rolling(20).mean()

    # Normalize features
    for col in features:
        df[f'{col}_norm'] = (df[col] - df[col].rolling(100).mean()) / df[col].rolling(100).std()

    df = df.dropna()
    data = df[[f'{col}_norm' for col in features]].values
    targets = df['returns'].values

    # Create sequences
    X, y = [], []
    for i in range(seq_len, len(data) - pred_len):
        X.append(data[i-seq_len:i])
        y.append(targets[i:i+pred_len])

    return np.array(X), np.array(y)


class TimeSeriesDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for time series data"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

### 02: ProbSparse Attention Implementation

See [python/model.py](python/model.py) for complete implementation.

### 03: Model Training

```python
# python/03_train.py

import torch
import torch.nn as nn
from model import InformerModel, InformerConfig

def train_informer(
    train_loader,
    val_loader,
    config: InformerConfig,
    epochs: int = 100,
    lr: float = 0.001
):
    """Train Informer model with ProbSparse attention"""

    model = InformerModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    return model
```

### 04: Forecasting

```python
# python/04_forecast.py

def multi_step_forecast(
    model: InformerModel,
    X: torch.Tensor,
    horizon: int = 24
) -> torch.Tensor:
    """
    Generate multi-step forecasts using Informer.

    Args:
        model: Trained Informer model
        X: Input sequence [batch, seq_len, features]
        horizon: Number of steps to forecast

    Returns:
        predictions: [batch, horizon]
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X)

    return predictions[:, :horizon]


def forecast_with_confidence(
    model: InformerModel,
    X: torch.Tensor,
    n_samples: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate forecasts with confidence intervals using MC Dropout.

    Returns:
        mean: Mean prediction
        lower: Lower confidence bound (5th percentile)
        upper: Upper confidence bound (95th percentile)
    """
    model.train()  # Enable dropout

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X)
            predictions.append(pred)

    predictions = torch.stack(predictions)

    mean = predictions.mean(dim=0)
    lower = torch.quantile(predictions, 0.05, dim=0)
    upper = torch.quantile(predictions, 0.95, dim=0)

    return mean, lower, upper
```

### 05: Backtesting Strategy

```python
# python/05_backtest.py

import numpy as np
import pandas as pd
from typing import Dict

def backtest_informer_strategy(
    model,
    test_data: pd.DataFrame,
    seq_len: int = 96,
    pred_len: int = 24,
    threshold: float = 0.0005,  # 0.05% return threshold
    initial_capital: float = 100000,
    transaction_cost: float = 0.001
) -> Dict:
    """
    Backtest trading strategy using Informer predictions.

    Strategy: Go long if predicted return > threshold,
              Go short if predicted return < -threshold,
              Stay flat otherwise.
    """

    capital = initial_capital
    position = 0  # -1: short, 0: flat, 1: long

    results = []

    for i in range(seq_len, len(test_data) - pred_len):
        # Get input sequence
        X = test_data.iloc[i-seq_len:i][['close_norm', 'volume_norm', 'volatility_norm']].values
        X = torch.FloatTensor(X).unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            pred = model(X)[0, 0].item()  # First step prediction

        # Get actual return
        actual_return = np.log(
            test_data.iloc[i+1]['close'] / test_data.iloc[i]['close']
        )

        # Trading logic
        new_position = 0
        if pred > threshold:
            new_position = 1
        elif pred < -threshold:
            new_position = -1

        # Calculate transaction costs if position changes
        if new_position != position:
            capital *= (1 - transaction_cost)

        # Calculate PnL
        pnl = position * actual_return * capital
        capital += pnl

        position = new_position

        results.append({
            'timestamp': test_data.index[i],
            'capital': capital,
            'position': position,
            'predicted_return': pred,
            'actual_return': actual_return,
            'pnl': pnl
        })

    results_df = pd.DataFrame(results)

    # Calculate metrics
    returns = results_df['pnl'] / results_df['capital'].shift(1)
    returns = returns.dropna()

    metrics = {
        'total_return': (capital - initial_capital) / initial_capital,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252 * 24),  # Hourly to annual
        'sortino_ratio': returns.mean() / returns[returns < 0].std() * np.sqrt(252 * 24),
        'max_drawdown': (results_df['capital'].cummax() - results_df['capital']).max() / results_df['capital'].cummax().max(),
        'win_rate': (returns > 0).mean(),
        'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).any() else float('inf'),
        'num_trades': (results_df['position'].diff() != 0).sum()
    }

    return {
        'results': results_df,
        'metrics': metrics
    }
```

## Rust Implementation

See [rust/](rust/) for complete Rust implementation using Bybit data.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── api/                # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP client for Bybit
│   │   └── types.rs        # API response types
│   ├── data/               # Data processing
│   │   ├── mod.rs
│   │   ├── loader.rs       # Data loading utilities
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset for training
│   ├── model/              # Informer architecture
│   │   ├── mod.rs
│   │   ├── attention.rs    # ProbSparse attention
│   │   ├── embedding.rs    # Token embedding
│   │   ├── encoder.rs      # Encoder with distilling
│   │   └── informer.rs     # Complete model
│   └── strategy/           # Trading strategy
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting engine
└── examples/
    ├── fetch_data.rs       # Download Bybit data
    ├── train.rs            # Train model
    └── backtest.rs         # Run backtest
```

### Quick Start (Rust)

```bash
# Navigate to Rust project
cd rust

# Fetch data from Bybit
cargo run --example fetch_data -- --symbol BTCUSDT --interval 1h --limit 10000

# Train model
cargo run --example train -- --epochs 100 --batch-size 32

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── __init__.py
├── model.py                # Informer with ProbSparse attention
├── data.py                 # Data loading and preprocessing
├── train.py                # Training script
├── backtest.py             # Backtesting utilities
├── requirements.txt        # Dependencies
└── examples/
    ├── 01_data_preparation.py
    ├── 02_model_architecture.py
    ├── 03_training.py
    ├── 04_forecasting.py
    └── 05_backtesting.py
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch data
python data.py --symbol BTCUSDT --interval 1h --limit 10000

# Train model
python train.py --config configs/default.yaml

# Run backtest
python backtest.py --model checkpoints/best_model.pt
```

## Best Practices

### When to Use ProbSparse Attention

**Good use cases:**
- Long sequence forecasting (L > 100)
- Multi-horizon predictions
- Resource-constrained environments
- Real-time trading systems

**Not ideal for:**
- Very short sequences (L < 50) — overhead exceeds benefit
- Tasks requiring full attention interpretability
- When maximum accuracy is critical over efficiency

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `seq_len` | 96-720 | Longer for lower frequency data |
| `d_model` | 64-256 | Depends on data complexity |
| `n_heads` | 4-8 | Should divide d_model |
| `sampling_factor` | 5 | Paper default, rarely needs tuning |
| `n_encoder_layers` | 2-4 | More layers, use distilling |
| `dropout` | 0.1-0.2 | Higher for small datasets |

### Common Pitfalls

1. **Sequence length too short**: ProbSparse overhead isn't worth it for L < 50
2. **Missing normalization**: Always normalize inputs for stable training
3. **Ignoring distilling**: For deep encoders, distilling is essential
4. **Over-sampling queries**: Don't set sampling_factor too high (>10)

## Resources

### Papers

- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) — Original ProbSparse paper
- [Autoformer: Decomposition Transformers with Auto-Correlation](https://arxiv.org/abs/2106.13008) — Related efficient Transformer
- [FEDformer: Frequency Enhanced Decomposed Transformer](https://arxiv.org/abs/2201.12740) — Frequency-domain approach
- [Comparing Different Transformer Model Structures for Stock Prediction](https://arxiv.org/abs/2504.16361) — Trading comparison

### Implementations

- [Hugging Face Informer](https://huggingface.co/docs/transformers/en/model_doc/informer) — Official implementation
- [Informer2020 GitHub](https://github.com/zhouhaoyi/Informer2020) — Original authors' code
- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) — Time series library

### Related Chapters

- [Chapter 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — Multi-horizon forecasting
- [Chapter 43: Stockformer Multivariate](../43_stockformer_multivariate) — Cross-asset prediction
- [Chapter 52: Performer Efficient Attention](../52_performer_efficient_attention) — Linear attention
- [Chapter 58: Flash Attention Trading](../58_flash_attention_trading) — Hardware optimization

---

## Difficulty Level

**Intermediate to Advanced**

Prerequisites:
- Transformer architecture fundamentals
- Self-attention mechanism understanding
- Time series forecasting basics
- PyTorch or Rust ML experience

---

Sources:
- [Informer on Hugging Face](https://huggingface.co/blog/informer)
- [Informer Documentation](https://huggingface.co/docs/transformers/en/model_doc/informer)
- [Informer Paper (arXiv)](https://arxiv.org/abs/2012.07436)
