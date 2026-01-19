# Chapter 51: Linformer — Self-Attention with Linear Complexity for Long Sequences

This chapter explores **Linformer**, a breakthrough transformer architecture that reduces self-attention complexity from O(n²) to O(n) using low-rank matrix approximation. This makes it ideal for processing long financial time series sequences efficiently.

<p align="center">
<img src="https://i.imgur.com/JK8m3Qf.png" width="70%">
</p>

## Contents

1. [Introduction to Linformer](#introduction-to-linformer)
    * [The Long Sequence Problem](#the-long-sequence-problem)
    * [Low-Rank Matrix Approximation](#low-rank-matrix-approximation)
    * [Key Advantages](#key-advantages)
    * [Comparison with Other Efficient Transformers](#comparison-with-other-efficient-transformers)
2. [Linformer Architecture](#linformer-architecture)
    * [Standard Self-Attention Recap](#standard-self-attention-recap)
    * [Linear Projection Matrices](#linear-projection-matrices)
    * [Computational Complexity Analysis](#computational-complexity-analysis)
    * [Memory Efficiency](#memory-efficiency)
3. [Mathematical Foundation](#mathematical-foundation)
    * [Johnson-Lindenstrauss Lemma](#johnson-lindenstrauss-lemma)
    * [Low-Rank Property of Self-Attention](#low-rank-property-of-self-attention)
    * [Error Bounds](#error-bounds)
4. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: Linformer Architecture](#02-linformer-architecture)
    * [03: Model Training](#03-model-training)
    * [04: Long Sequence Forecasting](#04-long-sequence-forecasting)
    * [05: Portfolio Backtesting](#05-portfolio-backtesting)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to Linformer

Linformer is a transformer variant developed by Facebook AI that achieves **linear complexity** O(n) in both time and space, compared to the quadratic O(n²) complexity of standard transformers. This is achieved through a clever low-rank approximation of the self-attention matrix.

### The Long Sequence Problem

Standard transformers struggle with long sequences because attention complexity scales quadratically:

```
Sequence Length  │ Standard Attention  │ Memory Usage
─────────────────┼─────────────────────┼──────────────
     512         │   262,144 ops       │   ~0.5 GB
    2,048        │ 4,194,304 ops       │   ~8 GB
    8,192        │ 67,108,864 ops      │  ~128 GB
   32,768        │ 1,073,741,824 ops   │ ~2 TB (!)
```

In financial applications, we often need to process:
- **Tick data**: Thousands of price updates per minute
- **Multi-day patterns**: Weeks or months of hourly data
- **Order book history**: Deep sequences for market microstructure analysis

### Low-Rank Matrix Approximation

The key insight: **self-attention matrices are inherently low-rank**. Most of the information is captured in a much smaller subspace.

```
Standard Attention:
┌─────────────────────────┐
│  Full n × n attention   │  ← O(n²) computation
│  matrix computation     │
└─────────────────────────┘

Linformer:
┌─────────────────────────┐
│  Project to k × n       │  ← k << n
│  Low-rank approximation │  ← O(n × k) computation
└─────────────────────────┘

Where k is typically 128-256, regardless of sequence length n
```

### Key Advantages

1. **Linear Time Complexity**
   - O(n) instead of O(n²)
   - 20x faster for long sequences
   - Enables processing of 10,000+ token sequences

2. **Linear Memory Usage**
   - Memory scales linearly with sequence length
   - Can handle sequences 10x longer with same memory
   - Critical for resource-constrained environments

3. **Preserved Model Quality**
   - Theoretical guarantees via Johnson-Lindenstrauss lemma
   - Empirical results match standard transformers
   - Works well for financial time series

4. **Easy Integration**
   - Drop-in replacement for standard attention
   - Works with existing transformer architectures
   - Compatible with pre-training and fine-tuning

### Comparison with Other Efficient Transformers

| Model | Complexity | Method | Autoregressive | Variable Length |
|-------|------------|--------|----------------|-----------------|
| Standard Transformer | O(n²) | Full attention | ✓ | ✓ |
| **Linformer** | **O(n)** | **Low-rank projection** | **✗** | **Fixed** |
| Performer | O(n) | Random features | ✓ | ✓ |
| Longformer | O(n) | Local + global | ✓ | ✓ |
| BigBird | O(n) | Sparse + random + global | ✓ | ✓ |
| Reformer | O(n log n) | LSH hashing | ✓ | ✓ |

**When to use Linformer:**
- Non-autoregressive tasks (classification, regression, encoding)
- Fixed sequence length scenarios
- Maximum efficiency needed
- Financial time series analysis (often fixed windows)

## Linformer Architecture

### Standard Self-Attention Recap

Standard self-attention computes:

```python
# Standard Transformer Attention
# Input: X of shape [batch, seq_len, d_model]

Q = X @ W_Q  # [batch, n, d_k]
K = X @ W_K  # [batch, n, d_k]
V = X @ W_V  # [batch, n, d_v]

# Attention matrix: O(n²) computation!
Attention = softmax(Q @ K.T / sqrt(d_k))  # [batch, n, n]
Output = Attention @ V  # [batch, n, d_v]
```

The bottleneck is the n × n attention matrix computation.

### Linear Projection Matrices

Linformer introduces projection matrices E and F:

```python
# Linformer Attention
# E: [k, n] - projects keys to k dimensions
# F: [k, n] - projects values to k dimensions
# k << n (typically k = 128 or 256)

Q = X @ W_Q         # [batch, n, d_k]
K_proj = E @ K      # [batch, k, d_k] - compressed!
V_proj = F @ V      # [batch, k, d_v] - compressed!

# Attention matrix: O(n × k) computation
Attention = softmax(Q @ K_proj.T / sqrt(d_k))  # [batch, n, k]
Output = Attention @ V_proj  # [batch, n, d_v]
```

```
┌──────────────────────────────────────────────────────────────────┐
│                         LINFORMER                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Input X: [batch, n, d_model]                                    │
│        │                                                           │
│        ▼                                                           │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                          │
│   │   Q     │  │   K     │  │   V     │                          │
│   │ [n,d_k] │  │ [n,d_k] │  │ [n,d_v] │                          │
│   └────┬────┘  └────┬────┘  └────┬────┘                          │
│        │            │            │                                 │
│        │      ┌─────┴─────┐  ┌───┴─────┐                          │
│        │      │ E @ K     │  │ F @ V   │  ← Linear projections    │
│        │      │ [k, d_k]  │  │ [k,d_v] │    k << n                │
│        │      └─────┬─────┘  └───┬─────┘                          │
│        │            │            │                                 │
│        ▼            ▼            ▼                                 │
│   ┌────────────────────────────────────────┐                      │
│   │  Attention = softmax(Q @ K_proj.T)     │                      │
│   │         [n, k] instead of [n, n]!      │                      │
│   └────────────────────────────────────────┘                      │
│        │                                                           │
│        ▼                                                           │
│   Output: Attention @ V_proj → [batch, n, d_v]                    │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Computational Complexity Analysis

```
Standard Self-Attention:
- Q @ K.T: O(n × n × d_k) = O(n² × d_k)
- Attention @ V: O(n × n × d_v) = O(n² × d_v)
- Total: O(n² × d)

Linformer:
- E @ K: O(k × n × d_k) = O(n × k × d_k)
- F @ V: O(k × n × d_v) = O(n × k × d_v)
- Q @ K_proj.T: O(n × k × d_k) = O(n × k × d_k)
- Attention @ V_proj: O(n × k × d_v) = O(n × k × d_v)
- Total: O(n × k × d)

When k is fixed (e.g., 128), complexity becomes O(n)!
```

### Memory Efficiency

```python
# Memory comparison for batch_size=32, d_model=512

sequence_lengths = [512, 1024, 2048, 4096, 8192]

for n in sequence_lengths:
    # Standard Transformer
    standard_memory = n * n * 4  # float32, attention matrix only

    # Linformer (k=128)
    k = 128
    linformer_memory = n * k * 4

    savings = (1 - linformer_memory / standard_memory) * 100
    print(f"n={n:5d}: Standard={standard_memory/1e6:.1f}MB, "
          f"Linformer={linformer_memory/1e6:.1f}MB, "
          f"Savings={savings:.1f}%")

# Output:
# n=  512: Standard=1.0MB,  Linformer=0.3MB,  Savings=75.0%
# n= 1024: Standard=4.2MB,  Linformer=0.5MB,  Savings=87.5%
# n= 2048: Standard=16.8MB, Linformer=1.0MB,  Savings=93.8%
# n= 4096: Standard=67.1MB, Linformer=2.1MB,  Savings=96.9%
# n= 8192: Standard=268.4MB, Linformer=4.2MB, Savings=98.4%
```

## Mathematical Foundation

### Johnson-Lindenstrauss Lemma

The theoretical foundation of Linformer relies on the **Johnson-Lindenstrauss (JL) lemma**:

> For any ε > 0 and any set of n points in a high-dimensional space,
> there exists a linear projection to a space of dimension k = O(log(n)/ε²)
> such that all pairwise distances are preserved within a factor of (1 ± ε).

**Application to Attention:**
```
If the attention matrix A has effective rank r,
then projecting from n dimensions to k ≥ r dimensions
preserves the essential information with high probability.
```

### Low-Rank Property of Self-Attention

Empirical observation: self-attention matrices are approximately low-rank.

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_attention_rank(attention_matrix):
    """
    Analyze the effective rank of an attention matrix.
    """
    # Singular value decomposition
    U, S, Vh = np.linalg.svd(attention_matrix)

    # Cumulative energy (variance explained)
    total_energy = np.sum(S ** 2)
    cumulative_energy = np.cumsum(S ** 2) / total_energy

    # Effective rank (95% energy)
    effective_rank = np.argmax(cumulative_energy >= 0.95) + 1

    return effective_rank, cumulative_energy

# Typical finding: For n=1024, effective rank is often < 128
# This justifies using k=128 for projection dimension
```

The attention matrix can be decomposed:
```
A = softmax(Q @ K.T / sqrt(d_k))

SVD: A = U @ Σ @ V.T

If Σ has rapid decay (few dominant singular values),
A is effectively low-rank and can be approximated.
```

### Error Bounds

The approximation error is bounded:

```
Given:
- P = softmax(Q @ K.T / sqrt(d_k)) @ V  (standard attention)
- P̂ = softmax(Q @ (E @ K).T / sqrt(d_k)) @ (F @ V)  (Linformer)

Theorem: For k = O(d/ε²), with high probability:
||P - P̂||_F ≤ ε ||P||_F

Meaning: The error is bounded and controllable via k.
```

## Practical Examples

### 01: Data Preparation

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def prepare_long_sequence_data(
    symbols: List[str],
    lookback: int = 2048,  # Long sequences for Linformer!
    horizon: int = 24
) -> Dict:
    """
    Prepare long-sequence data for Linformer training.

    Linformer can efficiently handle much longer sequences
    than standard transformers.

    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        lookback: Number of historical time steps (can be very long!)
        horizon: Prediction horizon

    Returns:
        Dictionary with X (features) and y (targets)
    """
    all_data = []

    for symbol in symbols:
        # Load data from Bybit
        df = load_bybit_data(symbol, interval='1h', limit=lookback + horizon + 100)

        # Calculate features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_change'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility'] = df['log_return'].rolling(20).std()
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['momentum'] = df['close'] / df['close'].shift(20) - 1

        all_data.append(df)

    # Align on timestamp
    aligned = pd.concat(all_data, axis=1, keys=symbols)
    aligned = aligned.dropna()

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(aligned) - horizon):
        X.append(aligned.iloc[i-lookback:i].values)
        y.append(aligned.iloc[i+horizon]['log_return'].values)

    return {
        'X': np.array(X),           # [n_samples, lookback, features]
        'y': np.array(y),           # [n_samples, n_assets]
        'symbols': symbols,
        'lookback': lookback,
        'horizon': horizon
    }


def load_bybit_data(symbol: str, interval: str = '1h', limit: int = 5000) -> pd.DataFrame:
    """
    Load historical data from Bybit.
    """
    import requests

    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': '60' if interval == '1h' else interval,
        'limit': limit
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data['retCode'] != 0:
        raise ValueError(f"API error: {data['retMsg']}")

    df = pd.DataFrame(data['result']['list'], columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    return df.sort_values('timestamp').reset_index(drop=True)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

### 02: Linformer Architecture

```python
# python/linformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LinformerAttention(nn.Module):
    """
    Linformer Self-Attention with Linear Complexity.

    Projects keys and values to a lower dimension k,
    reducing complexity from O(n²) to O(n×k).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        k: int = 128,
        dropout: float = 0.1,
        share_kv: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            seq_len: Fixed sequence length
            k: Projection dimension (k << seq_len)
            dropout: Dropout rate
            share_kv: If True, share projection between K and V
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len
        self.k = k
        self.scale = math.sqrt(self.d_k)

        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Linear projection matrices E and F
        # These project from seq_len to k dimensions
        self.E = nn.Parameter(torch.randn(n_heads, k, seq_len) * 0.02)

        if share_kv:
            # Share projection between K and V (more efficient)
            self.F = self.E
        else:
            # Separate projections for K and V
            self.F = nn.Parameter(torch.randn(n_heads, k, seq_len) * 0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with linear complexity attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections for Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)  # [batch, seq_len, d_model]
        V = self.W_v(x)  # [batch, seq_len, d_model]

        # Reshape for multi-head attention
        # [batch, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Project K and V to lower dimension
        # E: [n_heads, k, seq_len]
        # K: [batch, n_heads, seq_len, d_k]
        # K_proj: [batch, n_heads, k, d_k]
        K_proj = torch.einsum('hkn,bhnd->bhkd', self.E, K)
        V_proj = torch.einsum('hkn,bhnd->bhkd', self.F, V)

        # Compute attention scores
        # Q @ K_proj.T: [batch, n_heads, seq_len, k]
        attention_scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to projected values
        # [batch, n_heads, seq_len, d_k]
        context = torch.matmul(attention_weights, V_proj)

        # Reshape back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # Output projection
        output = self.W_o(context)

        if return_attention:
            return output, attention_weights
        return output, None


class LinformerEncoderLayer(nn.Module):
    """
    Linformer encoder layer with linear attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        k: int = 128,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = LinformerAttention(
            d_model=d_model,
            n_heads=n_heads,
            seq_len=seq_len,
            k=k,
            dropout=dropout
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out

        return x


class Linformer(nn.Module):
    """
    Complete Linformer model for financial time series.

    Handles long sequences efficiently with O(n) complexity.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        k: int = 128,
        d_ff: int = 1024,
        dropout: float = 0.1,
        output_type: str = 'regression',
        n_outputs: int = 1
    ):
        """
        Args:
            n_features: Number of input features
            seq_len: Fixed sequence length
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            k: Projection dimension for linear attention
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            output_type: 'regression', 'classification', or 'allocation'
            n_outputs: Number of output values (e.g., number of assets)
        """
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.output_type = output_type

        # Input embedding
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(seq_len, d_model)

        # Linformer encoder layers
        self.encoder_layers = nn.ModuleList([
            LinformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                seq_len=seq_len,
                k=k,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Output head based on task type
        if output_type == 'regression':
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_outputs)
            )
        elif output_type == 'classification':
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_outputs)
            )
        elif output_type == 'allocation':
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_outputs),
                nn.Tanh()  # Bound allocations to [-1, 1]
            )

    def _create_positional_encoding(
        self,
        seq_len: int,
        d_model: int
    ) -> nn.Parameter:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, n_features]
            mask: Optional attention mask

        Returns:
            Predictions [batch, n_outputs]
        """
        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)

        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)

        x = self.norm(x)

        # Use last token for prediction
        x = x[:, -1, :]

        # Output head
        output = self.output_head(x)

        return output

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss based on output type."""
        if self.output_type == 'regression':
            return F.mse_loss(predictions, targets)
        elif self.output_type == 'classification':
            return F.binary_cross_entropy_with_logits(predictions, targets)
        elif self.output_type == 'allocation':
            # Maximize returns: negative mean of (allocation * returns)
            return -torch.mean(predictions * targets)
```

### 03: Model Training

```python
# python/03_train_model.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from linformer import Linformer
import numpy as np


def train_linformer(
    train_data: dict,
    val_data: dict,
    config: dict,
    device: str = 'cuda'
) -> Linformer:
    """
    Train Linformer model on financial data.

    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        config: Model configuration
        device: Device to train on

    Returns:
        Trained model
    """
    # Initialize model
    model = Linformer(
        n_features=train_data['X'].shape[-1],
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        k=config['k'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        output_type=config['output_type'],
        n_outputs=config['n_outputs']
    ).to(device)

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(train_data['X']),
            torch.FloatTensor(train_data['y'])
        ),
        batch_size=config['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(val_data['X']),
            torch.FloatTensor(val_data['y'])
        ),
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_x)
            loss = model.compute_loss(predictions, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_x)
                loss = model.compute_loss(predictions, batch_y)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"Train Loss={avg_train_loss:.6f}, "
              f"Val Loss={avg_val_loss:.6f}, "
              f"LR={scheduler.get_last_lr()[0]:.2e}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_linformer.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_linformer.pt'))
    return model


# Example configuration
config = {
    'seq_len': 2048,        # Long sequence - Linformer handles this efficiently!
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 4,
    'k': 128,               # Projection dimension
    'd_ff': 1024,
    'dropout': 0.1,
    'output_type': 'regression',
    'n_outputs': 1,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 100,
    'patience': 10
}
```

### 04: Long Sequence Forecasting

```python
# python/04_long_sequence_forecasting.py

import torch
import numpy as np
import pandas as pd
from typing import List, Dict


def forecast_with_long_context(
    model: torch.nn.Module,
    data: pd.DataFrame,
    lookback: int = 2048,
    device: str = 'cuda'
) -> Dict:
    """
    Make predictions using long historical context.

    Linformer can efficiently use much longer context than
    standard transformers, potentially capturing longer-term patterns.

    Args:
        model: Trained Linformer model
        data: Historical data
        lookback: Context length
        device: Computation device

    Returns:
        Dictionary with predictions and confidence
    """
    model.eval()

    # Prepare features
    features = prepare_features(data)

    # Get last lookback timesteps
    x = torch.FloatTensor(features[-lookback:]).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get prediction
        prediction = model(x)

    return {
        'prediction': prediction.cpu().numpy(),
        'timestamp': data.index[-1],
        'context_length': lookback
    }


def compare_context_lengths(
    model_short: torch.nn.Module,  # Standard transformer with short context
    model_long: torch.nn.Module,   # Linformer with long context
    test_data: pd.DataFrame,
    short_lookback: int = 512,
    long_lookback: int = 2048
) -> pd.DataFrame:
    """
    Compare forecasting performance with different context lengths.

    This demonstrates Linformer's ability to use longer context
    for potentially better predictions.
    """
    results = []

    for i in range(long_lookback, len(test_data) - 1):
        # Short context prediction (standard transformer)
        x_short = prepare_sequence(test_data, i - short_lookback, i)
        pred_short = model_short(x_short)

        # Long context prediction (Linformer)
        x_long = prepare_sequence(test_data, i - long_lookback, i)
        pred_long = model_long(x_long)

        # Actual next value
        actual = test_data.iloc[i + 1]['log_return']

        results.append({
            'timestamp': test_data.index[i],
            'actual': actual,
            'pred_short_context': pred_short.item(),
            'pred_long_context': pred_long.item(),
            'error_short': abs(pred_short.item() - actual),
            'error_long': abs(pred_long.item() - actual)
        })

    df = pd.DataFrame(results)

    print("Context Length Comparison:")
    print(f"Short Context ({short_lookback}) MAE: {df['error_short'].mean():.6f}")
    print(f"Long Context ({long_lookback}) MAE: {df['error_long'].mean():.6f}")
    print(f"Improvement: {(1 - df['error_long'].mean() / df['error_short'].mean()) * 100:.2f}%")

    return df
```

### 05: Portfolio Backtesting

```python
# python/05_backtest.py

import pandas as pd
import numpy as np
from typing import Dict, List


def backtest_linformer_strategy(
    model: torch.nn.Module,
    test_data: Dict,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    position_sizing: str = 'equal',
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Backtest Linformer-based trading strategy.

    Args:
        model: Trained Linformer model
        test_data: Test data dictionary
        initial_capital: Starting capital
        transaction_cost: Transaction cost per trade
        position_sizing: 'equal' or 'proportional'
        device: Computation device

    Returns:
        DataFrame with backtest results
    """
    model.eval()

    X = torch.FloatTensor(test_data['X']).to(device)
    actual_returns = test_data['y']

    capital = initial_capital
    position = 0
    results = []

    for i in range(len(X)):
        with torch.no_grad():
            # Get model prediction
            prediction = model(X[i:i+1]).cpu().numpy().flatten()[0]

        # Generate signal
        if prediction > 0.001:  # Bullish threshold
            target_position = 1
        elif prediction < -0.001:  # Bearish threshold
            target_position = -1
        else:
            target_position = 0

        # Calculate trade cost
        position_change = abs(target_position - position)
        trade_cost = position_change * transaction_cost * capital

        # Update position
        position = target_position

        # Calculate return
        actual_return = actual_returns[i].item() if hasattr(actual_returns[i], 'item') else actual_returns[i]
        portfolio_return = position * actual_return

        # Update capital
        capital = capital * (1 + portfolio_return) - trade_cost

        results.append({
            'step': i,
            'capital': capital,
            'position': position,
            'prediction': prediction,
            'actual_return': actual_return,
            'portfolio_return': portfolio_return,
            'trade_cost': trade_cost
        })

    df = pd.DataFrame(results)

    # Calculate metrics
    total_return = (df['capital'].iloc[-1] / initial_capital - 1) * 100
    sharpe_ratio = calculate_sharpe_ratio(df['portfolio_return'])
    max_drawdown = calculate_max_drawdown(df['capital'])

    print("\nBacktest Results:")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Final Capital: ${df['capital'].iloc[-1]:,.2f}")

    return df


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown percentage."""
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return abs(drawdown.min()) * 100
```

## Rust Implementation

See [rust/](rust/) for complete Rust implementation.

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
│   ├── model/              # Linformer architecture
│   │   ├── mod.rs
│   │   ├── attention.rs    # Linear attention implementation
│   │   ├── encoder.rs      # Encoder layers
│   │   └── linformer.rs    # Complete model
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
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Train model with long sequences
cargo run --example train -- --seq-len 2048 --epochs 100

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── linformer.py            # Main model implementation
├── data_loader.py          # Bybit data loading
├── features.py             # Feature engineering
├── train.py                # Training script
├── backtest.py             # Backtesting utilities
├── requirements.txt        # Dependencies
└── examples/
    ├── 01_data_preparation.ipynb
    ├── 02_linformer_architecture.ipynb
    ├── 03_training.ipynb
    ├── 04_long_sequence_forecasting.ipynb
    └── 05_backtesting.ipynb
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch data
python data_loader.py --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Train model with long sequences
python train.py --seq-len 2048 --k 128 --epochs 100

# Run backtest
python backtest.py --model checkpoints/best_linformer.pt
```

## Best Practices

### When to Use Linformer

**Good use cases:**
- Long sequence analysis (1000+ tokens)
- Fixed-length time series forecasting
- Non-autoregressive tasks (encoding, classification)
- Memory-constrained environments
- High-frequency data with long lookback windows

**Not ideal for:**
- Autoregressive generation (use Performer or Longformer)
- Variable-length sequences (requires padding)
- Tasks requiring full attention patterns
- Very short sequences (standard attention is fine)

### Choosing the Projection Dimension k

```python
# Rule of thumb for choosing k:
# k should capture the effective rank of attention

# For financial time series:
# - Short sequences (n < 512): k = 64
# - Medium sequences (512 <= n < 2048): k = 128
# - Long sequences (n >= 2048): k = 256

# Empirical validation:
def find_optimal_k(model, data, k_values=[64, 128, 256, 512]):
    """Find optimal k by validation performance."""
    results = {}
    for k in k_values:
        model.attention.k = k
        val_loss = evaluate(model, data)
        results[k] = val_loss
    return min(results, key=results.get)
```

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `seq_len` | 2048-8192 | Linformer's sweet spot |
| `k` | 128-256 | Projection dimension |
| `d_model` | 256-512 | Match with n_heads |
| `n_heads` | 8-16 | Should divide d_model |
| `n_layers` | 4-6 | Deeper for long sequences |
| `dropout` | 0.1-0.2 | Higher for small datasets |

### Common Pitfalls

1. **Variable-length sequences**: Pad to fixed length or use chunking
2. **Attention interpretability**: Projected attention is harder to interpret
3. **Position encoding**: Critical for long sequences; consider learnable
4. **Memory-computation tradeoff**: k too small hurts quality; too large reduces benefits

## Resources

### Papers

- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768) — Original paper
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) — Comprehensive survey of efficient attention
- [Long Range Arena](https://arxiv.org/abs/2011.04006) — Benchmark for long-range tasks

### Implementations

- [lucidrains/linformer](https://github.com/lucidrains/linformer) — PyTorch implementation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) — Model hub
- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) — Time series library

### Related Chapters

- [Chapter 50: Memory-Augmented Transformers](../50_memory_augmented_transformers) — External memory mechanisms
- [Chapter 52: Performer Efficient Attention](../52_performer_efficient_attention) — Random feature attention
- [Chapter 53: BigBird Sparse Attention](../53_bigbird_sparse_attention) — Sparse attention patterns
- [Chapter 54: Reformer LSH Attention](../54_reformer_lsh_attention) — Locality-sensitive hashing

---

## Difficulty Level

**Intermediate to Advanced**

Prerequisites:
- Transformer architecture and self-attention
- Matrix factorization and low-rank approximation
- Time series analysis basics
- PyTorch/Rust ML libraries
