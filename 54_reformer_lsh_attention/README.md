# Chapter 54: Reformer - Locality-Sensitive Hashing (LSH) Attention

This chapter explores the **Reformer** architecture, which introduces **Locality-Sensitive Hashing (LSH) Attention** to achieve O(L log L) complexity instead of the standard O(L^2) attention mechanism. This makes Reformer particularly well-suited for processing long financial time series.

<p align="center">
<img src="https://i.imgur.com/reformer_arch.png" width="70%">
</p>

## Contents

1. [Introduction to Reformer](#introduction-to-reformer)
    * [Why Efficient Attention Matters](#why-efficient-attention-matters)
    * [Key Innovations](#key-innovations)
    * [Comparison with Other Efficient Transformers](#comparison-with-other-efficient-transformers)
2. [LSH Attention Mechanism](#lsh-attention-mechanism)
    * [Locality-Sensitive Hashing Explained](#locality-sensitive-hashing-explained)
    * [Hash Buckets and Chunking](#hash-buckets-and-chunking)
    * [Multi-Round Hashing](#multi-round-hashing)
3. [Reversible Layers](#reversible-layers)
    * [Memory Efficiency](#memory-efficiency)
    * [Implementation Details](#implementation-details)
4. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: LSH Attention Implementation](#02-lsh-attention-implementation)
    * [03: Model Training](#03-model-training)
    * [04: Long Sequence Prediction](#04-long-sequence-prediction)
    * [05: Trading Strategy Backtesting](#05-trading-strategy-backtesting)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to Reformer

The **Reformer** is a modified Transformer architecture introduced by Kitaev, Kaiser, and Levskaya (ICLR 2020) that addresses the memory and computational bottlenecks of standard Transformers when processing long sequences.

### Why Efficient Attention Matters

Standard self-attention computes attention scores between all pairs of tokens:

```
Standard Attention: O(L^2 * d)
- L = sequence length
- d = model dimension

For L = 65536 (financial tick data):
- Memory: ~17 billion attention weights
- Computation: Impractical for real-time trading
```

In financial applications, we often need to process:
- **High-frequency tick data**: Thousands of ticks per minute
- **Long historical context**: Days or weeks of hourly data
- **Multiple assets simultaneously**: Portfolio-level predictions

The Reformer makes this feasible by reducing complexity to **O(L log L)**.

### Key Innovations

1. **LSH Attention**
   - Uses locality-sensitive hashing to approximate attention
   - Only attends to similar keys, reducing computation
   - Complexity: O(L log L) instead of O(L^2)

2. **Reversible Residual Layers**
   - Allows recomputation during backpropagation
   - Reduces memory from O(N * L * d) to O(L * d)
   - N = number of layers

3. **Chunked Feed-Forward Layers**
   - Processes feed-forward layers in chunks
   - Further reduces peak memory usage

### Comparison with Other Efficient Transformers

| Model | Attention Complexity | Method | Trading Use Case |
|-------|---------------------|--------|------------------|
| Standard | O(L^2) | Full attention | Short sequences only |
| Linformer | O(L) | Linear projection | General forecasting |
| Performer | O(L) | Random features | Fast inference |
| **Reformer** | **O(L log L)** | **LSH hashing** | **Long tick sequences** |
| Longformer | O(L) | Local + global | Document-like data |
| BigBird | O(L) | Sparse patterns | Mixed patterns |

**Why Reformer for Trading?**
- Better at capturing exact nearest neighbors (important for pattern matching)
- Multi-round hashing provides accuracy vs. speed trade-off
- Works well with both crypto and stock data

## LSH Attention Mechanism

### Locality-Sensitive Hashing Explained

**LSH** is a technique that hashes similar items into the same "bucket" with high probability. For attention, this means queries and keys that would have high attention scores are likely to share the same hash.

```
STANDARD ATTENTION:
Query q attends to ALL keys k1, k2, k3, ..., kL
Compute: softmax(Q @ K^T / sqrt(d)) @ V
Cost: O(L^2)

LSH ATTENTION:
1. Hash queries and keys into buckets
2. Query q only attends to keys in the SAME bucket
3. Similar vectors -> Same bucket (with high probability)
Cost: O(L * bucket_size) ~ O(L log L)
```

The key insight: **Attention weights are often sparse**. Most attention is concentrated on a few keys, so we can skip the rest.

### Hash Buckets and Chunking

The LSH attention process:

```
┌─────────────────────────────────────────────────────────────────┐
│                     LSH ATTENTION FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Sequence [x1, x2, x3, ..., xL]                         │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────────────┐                   │
│  │         1. HASH PROJECTION               │                   │
│  │    h(x) = sign(x @ R) where R ~ N(0,1)  │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │         2. BUCKET ASSIGNMENT             │                   │
│  │    Bucket 0: [x1, x4, x7]               │                   │
│  │    Bucket 1: [x2, x5, x9]               │                   │
│  │    Bucket 2: [x3, x6, x8]               │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │         3. SORT BY BUCKET                │                   │
│  │    Sorted: [x1,x4,x7 | x2,x5,x9 | ...]  │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │         4. CHUNKED ATTENTION             │                   │
│  │    Attend within chunks + lookback       │                   │
│  │    [x1,x4,x7] attend to each other      │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  OUTPUT: Attention output (unsorted back)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Round Hashing

A single hash round might miss some similar pairs. Reformer uses **multiple hashing rounds** to improve accuracy:

```python
def multi_round_lsh_attention(x, n_rounds=4, n_buckets=64):
    """
    Multiple rounds of LSH increase the probability
    of similar vectors ending up in the same bucket.

    Probability of collision (similar vectors):
    - 1 round: ~70%
    - 4 rounds: ~99%
    """
    outputs = []
    for round in range(n_rounds):
        # Different random projection for each round
        hash_vectors = hash_with_random_rotation(x, round)
        buckets = assign_to_buckets(hash_vectors, n_buckets)
        attn_output = attend_within_buckets(x, buckets)
        outputs.append(attn_output)

    # Average across rounds
    return torch.mean(torch.stack(outputs), dim=0)
```

## Reversible Layers

### Memory Efficiency

Standard Transformer stores activations for each layer during forward pass:

```
Standard: Memory ~ N * L * d
N = 12 layers, L = 65536, d = 512
Memory = 12 * 65536 * 512 * 4 bytes = 1.5 GB (just activations!)

Reversible: Memory ~ L * d
Memory = 65536 * 512 * 4 bytes = 128 MB
```

### Implementation Details

Reversible layers split the input into two streams and apply functions alternately:

```python
class ReversibleBlock(nn.Module):
    """
    Y1 = X1 + Attention(X2)
    Y2 = X2 + FeedForward(Y1)

    Reverse (no stored activations needed):
    X2 = Y2 - FeedForward(Y1)
    X1 = Y1 - Attention(X2)
    """

    def __init__(self, attention, feed_forward):
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward

    def forward(self, x1, x2):
        y1 = x1 + self.attention(x2)
        y2 = x2 + self.feed_forward(y1)
        return y1, y2

    def reverse(self, y1, y2):
        x2 = y2 - self.feed_forward(y1)
        x1 = y1 - self.attention(x2)
        return x1, x2
```

## Practical Examples

### 01: Data Preparation

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

def prepare_long_sequence_data(
    symbols: List[str],
    lookback: int = 4096,  # Long sequence for Reformer
    horizon: int = 24,
    interval: str = '1h'
) -> Dict:
    """
    Prepare long sequence data for Reformer.

    Reformer excels at processing long sequences,
    so we can use 4096+ timesteps (weeks of hourly data).

    Args:
        symbols: Trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        lookback: Historical timesteps (can be much longer than standard)
        horizon: Prediction horizon
        interval: Data interval

    Returns:
        Dictionary with X, y arrays for training
    """
    all_data = []

    for symbol in symbols:
        # Load data from Bybit
        df = load_bybit_data(symbol, interval=interval)

        # Calculate features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(20).std()
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(100).mean()) / \
                              df['volume'].rolling(100).std()
        df['price_zscore'] = (df['close'] - df['close'].rolling(100).mean()) / \
                             df['close'].rolling(100).std()
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['trend'] = df['close'].rolling(50).mean() / df['close'].rolling(200).mean() - 1

        all_data.append(df)

    # Stack all symbols
    stacked = np.stack([df.values for df in all_data], axis=1)

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(stacked) - horizon):
        X.append(stacked[i-lookback:i])
        # Target: next period returns
        y.append(stacked[i+horizon-1:i+horizon, :, 0])  # log_return

    return {
        'X': np.array(X),  # [n_samples, lookback, n_symbols, features]
        'y': np.array(y).squeeze(),  # [n_samples, n_symbols]
        'symbols': symbols
    }


def load_bybit_data(symbol: str, interval: str = '1h') -> pd.DataFrame:
    """
    Load historical data from Bybit API.

    This is a simplified version - see the api module for full implementation.
    """
    import requests

    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": {"1h": "60", "4h": "240", "1d": "D"}[interval],
        "limit": 1000
    }

    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df
```

### 02: LSH Attention Implementation

```python
# python/02_lsh_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class LSHAttention(nn.Module):
    """
    Locality-Sensitive Hashing Attention

    Achieves O(L log L) complexity by:
    1. Hashing queries/keys into buckets
    2. Attending only within buckets
    3. Using multiple hash rounds for accuracy
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_buckets: int = 64,
        n_rounds: int = 4,
        chunk_size: int = 64,
        dropout: float = 0.1,
        causal: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_buckets = n_buckets
        self.n_rounds = n_rounds
        self.chunk_size = chunk_size
        self.causal = causal

        self.scale = math.sqrt(self.head_dim)

        # Shared QK projection (Reformer uses same vectors for Q and K)
        self.qk_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Random rotation matrices for hashing (one per round)
        self.register_buffer(
            'random_rotations',
            torch.randn(n_rounds, d_model // n_heads, n_buckets // 2)
        )

    def hash_vectors(
        self,
        vectors: torch.Tensor,
        round_idx: int
    ) -> torch.Tensor:
        """
        Hash vectors using random rotation.

        h(x) = argmax([x @ R; -x @ R])

        This ensures vectors pointing in similar directions
        get similar hash values.
        """
        batch, n_heads, seq_len, head_dim = vectors.shape

        # Get rotation matrix for this round
        rotation = self.random_rotations[round_idx]  # [head_dim, n_buckets//2]

        # Project vectors
        rotated = torch.einsum('bhld,dk->bhlk', vectors, rotation)
        # Concatenate with negation for n_buckets total
        rotated = torch.cat([rotated, -rotated], dim=-1)  # [B, H, L, n_buckets]

        # Hash is the argmax bucket
        hash_values = rotated.argmax(dim=-1)  # [B, H, L]

        return hash_values

    def sort_by_bucket(
        self,
        qk: torch.Tensor,
        v: torch.Tensor,
        buckets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sort sequences by bucket assignment.
        """
        batch, n_heads, seq_len, _ = qk.shape

        # Create sort indices: sort by (bucket, position) for stability
        sort_keys = buckets * seq_len + torch.arange(seq_len, device=qk.device)
        sort_indices = sort_keys.argsort(dim=-1)

        # Sort QK and V
        qk_sorted = qk.gather(2, sort_indices.unsqueeze(-1).expand_as(qk))
        v_sorted = v.gather(2, sort_indices.unsqueeze(-1).expand_as(v))
        buckets_sorted = buckets.gather(-1, sort_indices)

        # Compute unsort indices for later
        unsort_indices = sort_indices.argsort(dim=-1)

        return qk_sorted, v_sorted, buckets_sorted, unsort_indices

    def chunked_attention(
        self,
        qk: torch.Tensor,
        v: torch.Tensor,
        buckets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention within chunks.

        Each query attends to keys in:
        1. Same chunk
        2. Previous chunk (lookback for better coverage)
        """
        batch, n_heads, seq_len, head_dim = qk.shape
        n_chunks = seq_len // self.chunk_size

        # Reshape into chunks
        qk = qk.view(batch, n_heads, n_chunks, self.chunk_size, head_dim)
        v = v.view(batch, n_heads, n_chunks, self.chunk_size, head_dim)
        buckets = buckets.view(batch, n_heads, n_chunks, self.chunk_size)

        # Create attention mask: same bucket + lookback
        # For simplicity, attend within chunk and to previous chunk
        q = qk  # [B, H, C, chunk, D]
        k = torch.cat([
            F.pad(qk[:, :, :-1], (0, 0, 0, 0, 1, 0)),  # Previous chunk
            qk  # Current chunk
        ], dim=3)  # [B, H, C, 2*chunk, D]
        v_extended = torch.cat([
            F.pad(v[:, :, :-1], (0, 0, 0, 0, 1, 0)),
            v
        ], dim=3)

        # Compute attention scores
        scores = torch.einsum('bhcqd,bhckd->bhcqk', q, k) / self.scale

        # Create bucket mask: only attend to same bucket
        q_buckets = buckets  # [B, H, C, chunk]
        k_buckets = torch.cat([
            F.pad(buckets[:, :, :-1], (0, 0, 1, 0)),
            buckets
        ], dim=3)

        # Mask different buckets
        bucket_mask = (q_buckets.unsqueeze(-1) != k_buckets.unsqueeze(-2))

        # Causal mask (optional)
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(self.chunk_size, 2 * self.chunk_size, device=scores.device),
                diagonal=self.chunk_size + 1
            ).bool()
            bucket_mask = bucket_mask | causal_mask

        scores = scores.masked_fill(bucket_mask, float('-inf'))

        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Output
        out = torch.einsum('bhcqk,bhckd->bhcqd', attn, v_extended)
        out = out.view(batch, n_heads, seq_len, head_dim)

        return out

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with LSH attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_attention: Whether to return attention patterns

        Returns:
            output: [batch, seq_len, d_model]
            attention: Optional attention info for visualization
        """
        batch, seq_len, d_model = x.shape

        # Pad sequence to be divisible by chunk_size
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        padded_len = x.size(1)

        # Project to QK (shared) and V
        qk = self.qk_proj(x).view(batch, padded_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, padded_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Normalize QK for better hashing
        qk = F.normalize(qk, dim=-1)

        # Multi-round LSH attention
        outputs = []
        for round_idx in range(self.n_rounds):
            # Hash vectors
            buckets = self.hash_vectors(qk, round_idx)

            # Sort by bucket
            qk_sorted, v_sorted, buckets_sorted, unsort_indices = \
                self.sort_by_bucket(qk, v, buckets)

            # Chunked attention within buckets
            out_sorted = self.chunked_attention(qk_sorted, v_sorted, buckets_sorted)

            # Unsort
            out = out_sorted.gather(2, unsort_indices.unsqueeze(-1).expand_as(out_sorted))
            outputs.append(out)

        # Average across rounds
        output = torch.mean(torch.stack(outputs), dim=0)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch, padded_len, d_model)
        output = self.out_proj(output)

        # Remove padding
        if pad_len > 0:
            output = output[:, :seq_len]

        return output, None


class ReversibleBlock(nn.Module):
    """
    Reversible residual block for memory efficiency.

    Forward:
        Y1 = X1 + F(X2)
        Y2 = X2 + G(Y1)

    Reverse (recompute during backprop):
        X2 = Y2 - G(Y1)
        X1 = Y1 - F(X2)
    """

    def __init__(self, f: nn.Module, g: nn.Module):
        super().__init__()
        self.f = f  # Usually attention
        self.g = g  # Usually feed-forward

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y1 = x1 + self.f(x2)[0]  # f returns (output, attention)
        y2 = x2 + self.g(y1)
        return y1, y2


class ChunkedFeedForward(nn.Module):
    """
    Feed-forward network processed in chunks for memory efficiency.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        chunk_size: int = 1024
    ):
        super().__init__()
        self.chunk_size = chunk_size

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        if seq_len <= self.chunk_size:
            return self.ff(x)

        # Process in chunks
        outputs = []
        for i in range(0, seq_len, self.chunk_size):
            chunk = x[:, i:i+self.chunk_size]
            outputs.append(self.ff(chunk))

        return torch.cat(outputs, dim=1)
```

### 03: Model Training

```python
# python/03_train_model.py

import torch
import torch.nn as nn
from reformer import ReformerModel, ReformerConfig
from data import prepare_long_sequence_data
from torch.utils.data import DataLoader, TensorDataset

def train_reformer(
    symbols: list,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.0001,
    lookback: int = 4096
):
    """
    Train Reformer model on long financial sequences.
    """

    # Prepare data
    print("Loading data...")
    data = prepare_long_sequence_data(symbols, lookback=lookback)

    X = torch.FloatTensor(data['X'])
    y = torch.FloatTensor(data['y'])

    # Train/val split
    split = int(0.8 * len(X))
    train_dataset = TensorDataset(X[:split], y[:split])
    val_dataset = TensorDataset(X[split:], y[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model configuration
    config = ReformerConfig(
        input_features=X.shape[-1],
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        n_buckets=64,
        n_rounds=4,
        chunk_size=64,
        dropout=0.1,
        max_seq_len=lookback,
        num_tickers=len(symbols)
    )

    model = ReformerModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            output = model(batch_x)
            predictions = output['predictions']

            loss = criterion(predictions, batch_y)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output['predictions'], batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_reformer.pt')

    return model


if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
    model = train_reformer(symbols, epochs=50, lookback=4096)
```

### 04: Long Sequence Prediction

```python
# python/04_prediction.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from reformer import ReformerModel, ReformerConfig

def predict_with_long_context(
    model: ReformerModel,
    data: np.ndarray,
    symbols: list
) -> dict:
    """
    Make predictions using long historical context.

    Reformer's O(L log L) complexity allows us to use
    much longer sequences than standard Transformers.
    """
    model.eval()

    with torch.no_grad():
        x = torch.FloatTensor(data).unsqueeze(0)  # Add batch dim
        output = model(x, return_attention=True)

    predictions = output['predictions'].numpy().flatten()

    # Get confidence from prediction variance across sequence
    # (if using ensemble or dropout)

    result = {
        'predictions': {sym: pred for sym, pred in zip(symbols, predictions)},
        'direction': {sym: 'LONG' if pred > 0 else 'SHORT' for sym, pred in zip(symbols, predictions)},
        'confidence': {sym: abs(pred) for sym, pred in zip(symbols, predictions)}
    }

    return result


def analyze_pattern_matching(
    model: ReformerModel,
    sequence: torch.Tensor,
    pattern_length: int = 168  # 1 week of hourly data
) -> dict:
    """
    Analyze how Reformer matches patterns across long sequences.

    LSH attention naturally groups similar patterns together,
    making it effective at finding historical analogues.
    """
    model.eval()

    # Get hash buckets for the sequence
    with torch.no_grad():
        # Access internal LSH attention
        qk = model.encoder_layers[0].attention.qk_proj(sequence)
        qk = qk.view(1, -1, model.config.n_heads, model.config.head_dim).transpose(1, 2)
        qk = torch.nn.functional.normalize(qk, dim=-1)

        buckets = model.encoder_layers[0].attention.hash_vectors(qk, round_idx=0)

    # Find similar time periods (same bucket assignments)
    buckets = buckets.squeeze()  # [n_heads, seq_len]

    # For each recent timestep, find historical matches
    recent_start = len(sequence) - pattern_length
    matches = {}

    for t in range(recent_start, len(sequence)):
        current_bucket = buckets[:, t]

        # Find historical timesteps with same bucket
        historical_matches = (buckets[:, :recent_start] == current_bucket.unsqueeze(-1)).all(dim=0)
        match_indices = historical_matches.nonzero().squeeze().tolist()

        if isinstance(match_indices, int):
            match_indices = [match_indices]

        matches[t - recent_start] = match_indices

    return {
        'pattern_matches': matches,
        'match_counts': {k: len(v) for k, v in matches.items()},
        'avg_matches': np.mean([len(v) for v in matches.values()])
    }
```

### 05: Trading Strategy Backtesting

```python
# python/05_backtest.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    max_position_size: float = 0.2   # 20% of capital per position
    rebalance_frequency: int = 24    # Hours between rebalancing
    risk_free_rate: float = 0.02     # Annual risk-free rate

def backtest_reformer_strategy(
    model,
    test_data: Dict,
    config: BacktestConfig
) -> pd.DataFrame:
    """
    Backtest Reformer-based trading strategy.

    Strategy:
    1. Use long historical context (Reformer advantage)
    2. Predict direction for each asset
    3. Size positions by prediction confidence
    4. Rebalance at specified frequency
    """

    X = test_data['X']
    y_true = test_data['y']
    symbols = test_data['symbols']
    n_assets = len(symbols)

    # Initialize tracking
    capital = config.initial_capital
    positions = np.zeros(n_assets)
    portfolio_values = [capital]
    trades = []

    # Iterate through test data
    for i in range(0, len(X), config.rebalance_frequency):
        # Get model predictions
        with torch.no_grad():
            x = torch.FloatTensor(X[i:i+1])
            output = model(x)
            predictions = output['predictions'].numpy().flatten()

        # Calculate target positions based on predictions
        # Positive prediction -> long, negative -> short
        target_weights = np.tanh(predictions * 2)  # Scale and bound
        target_weights = target_weights / (np.abs(target_weights).sum() + 1e-8)

        # Apply position size limits
        target_weights = np.clip(
            target_weights,
            -config.max_position_size,
            config.max_position_size
        )

        # Calculate position changes
        current_weights = positions / (capital + 1e-8)
        weight_changes = target_weights - current_weights

        # Calculate transaction costs
        turnover = np.abs(weight_changes).sum()
        costs = turnover * config.transaction_cost * capital

        # Update positions
        positions = target_weights * capital

        # Calculate returns for this period
        period_returns = y_true[i:min(i+config.rebalance_frequency, len(y_true))]

        if len(period_returns) > 0:
            # Compound returns for the period
            for ret in period_returns:
                portfolio_return = np.sum(target_weights * ret)
                capital = capital * (1 + portfolio_return)
            capital -= costs

        portfolio_values.append(capital)

        trades.append({
            'step': i,
            'positions': target_weights.copy(),
            'predictions': predictions.copy(),
            'costs': costs,
            'capital': capital
        })

    # Calculate performance metrics
    returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])

    results = pd.DataFrame({
        'step': range(len(portfolio_values)),
        'portfolio_value': portfolio_values
    })

    # Add metrics
    results.attrs['sharpe_ratio'] = calculate_sharpe(returns, config.risk_free_rate)
    results.attrs['sortino_ratio'] = calculate_sortino(returns, config.risk_free_rate)
    results.attrs['max_drawdown'] = calculate_max_drawdown(portfolio_values)
    results.attrs['total_return'] = (capital - config.initial_capital) / config.initial_capital
    results.attrs['trades'] = trades

    return results


def calculate_sharpe(returns: np.ndarray, risk_free_rate: float) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / (365 * 24)  # Hourly risk-free
    if len(excess_returns) == 0 or excess_returns.std() == 0:
        return 0.0
    return np.sqrt(365 * 24) * excess_returns.mean() / excess_returns.std()


def calculate_sortino(returns: np.ndarray, risk_free_rate: float) -> float:
    """Calculate annualized Sortino ratio."""
    excess_returns = returns - risk_free_rate / (365 * 24)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf')

    downside_std = downside_returns.std()
    if downside_std == 0:
        return float('inf')

    return np.sqrt(365 * 24) * excess_returns.mean() / downside_std


def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """Calculate maximum drawdown."""
    peak = portfolio_values[0]
    max_dd = 0

    for value in portfolio_values:
        peak = max(peak, value)
        drawdown = (peak - value) / peak
        max_dd = max(max_dd, drawdown)

    return max_dd
```

## Rust Implementation

See [rust_reformer](rust_reformer/) for the complete Rust implementation.

```
rust_reformer/
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
│   ├── model/              # Reformer architecture
│   │   ├── mod.rs
│   │   ├── lsh_attention.rs    # LSH attention implementation
│   │   ├── reversible.rs       # Reversible layers
│   │   ├── embedding.rs        # Token embedding
│   │   └── reformer.rs         # Complete model
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
cd rust_reformer

# Fetch data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Train model
cargo run --example train -- --epochs 100 --batch-size 16 --seq-len 4096

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for the Python implementation.

```
python/
├── reformer.py             # Main model implementation
├── lsh_attention.py        # LSH attention module
├── data.py                 # Data loading from Bybit
├── features.py             # Feature engineering
├── train.py                # Training script
├── backtest.py             # Backtesting utilities
├── requirements.txt        # Dependencies
└── examples/
    ├── 01_data_preparation.ipynb
    ├── 02_lsh_attention.ipynb
    ├── 03_training.ipynb
    ├── 04_prediction.ipynb
    └── 05_backtesting.ipynb
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch data
python data.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1h

# Train model
python train.py --epochs 100 --batch-size 16 --seq-len 4096

# Run backtest
python backtest.py --model checkpoints/best_reformer.pt
```

## Best Practices

### When to Use Reformer

**Good use cases:**
- Long historical sequences (1000+ timesteps)
- High-frequency data analysis
- Pattern matching across long time horizons
- Memory-constrained environments

**Not ideal for:**
- Very short sequences (<256 tokens)
- When exact attention is required
- Simple forecasting tasks

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `n_buckets` | 64-128 | More buckets = more precision |
| `n_rounds` | 4-8 | More rounds = better accuracy |
| `chunk_size` | 64-128 | Match with bucket size |
| `d_model` | 256-512 | Standard Transformer guidance |
| `n_layers` | 4-8 | Use reversible layers for deep models |

### Common Pitfalls

1. **Too few hash rounds**: Use at least 4 rounds for reliable attention
2. **Bucket size mismatch**: Keep n_buckets ~= seq_len / chunk_size
3. **Ignoring causal mask**: Essential for autoregressive generation
4. **Small sequences**: Standard attention may be faster for L < 512

### Memory Considerations

```
Standard Transformer (L=4096, d=512, N=6):
- Activations: 6 * 4096 * 512 * 4 = 48 MB
- Attention: 4096^2 * 6 * 4 = 384 MB
- Total: ~432 MB per sample

Reformer (L=4096, d=512, N=6):
- Activations: 4096 * 512 * 4 = 8 MB (reversible)
- Attention: 4096 * 64 * 6 * 4 * 4 = 25 MB (LSH)
- Total: ~33 MB per sample

13x memory reduction!
```

## Resources

### Papers

- [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) - Original Reformer paper
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) - Related sparse attention
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - Alternative efficient attention
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) - Random feature attention

### Implementations

- [Reformer PyTorch (Lucidrains)](https://github.com/lucidrains/reformer-pytorch) - Popular implementation
- [Trax Reformer (Google)](https://github.com/google/trax/tree/master/trax/models/reformer) - Original implementation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/reformer) - Production-ready

### Related Chapters

- [Chapter 51: Linformer Long Sequences](../51_linformer_long_sequences) - Linear attention
- [Chapter 52: Performer Efficient Attention](../52_performer_efficient_attention) - FAVOR+ mechanism
- [Chapter 53: BigBird Sparse Attention](../53_bigbird_sparse_attention) - Sparse patterns
- [Chapter 55: FNet Fourier Transform](../55_fnet_fourier_transform) - Fourier mixing

---

## Difficulty Level

**Advanced**

Prerequisites:
- Transformer architecture and attention mechanisms
- Hashing algorithms and locality-sensitive hashing
- Time series forecasting basics
- PyTorch/Rust ML libraries
