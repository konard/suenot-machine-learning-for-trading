# Chapter 53: BigBird — Sparse Attention for Long Sequences in Trading

This chapter explores **BigBird**, a sparse attention mechanism that combines random, window, and global attention patterns to process long sequences with linear complexity. BigBird enables transformers to handle significantly longer context windows, making it particularly valuable for financial time series that require capturing long-range dependencies.

<p align="center">
<img src="https://i.imgur.com/JQW8k9M.png" width="70%">
</p>

## Contents

1. [Introduction to BigBird](#introduction-to-bigbird)
    * [The Attention Bottleneck](#the-attention-bottleneck)
    * [BigBird's Solution](#bigbirds-solution)
    * [Key Advantages](#key-advantages)
2. [BigBird Architecture](#bigbird-architecture)
    * [Random Attention](#random-attention)
    * [Window (Local) Attention](#window-local-attention)
    * [Global Attention](#global-attention)
    * [Combined Sparse Pattern](#combined-sparse-pattern)
3. [Financial Applications](#financial-applications)
    * [Long-Range Market Dependencies](#long-range-market-dependencies)
    * [Tick-Level Data Processing](#tick-level-data-processing)
    * [Multi-Timeframe Analysis](#multi-timeframe-analysis)
4. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: BigBird Architecture](#02-bigbird-architecture)
    * [03: Model Training](#03-model-training)
    * [04: Long Sequence Prediction](#04-long-sequence-prediction)
    * [05: Backtesting Strategy](#05-backtesting-strategy)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Comparison with Other Methods](#comparison-with-other-methods)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction to BigBird

### The Attention Bottleneck

Standard transformers compute attention scores between all pairs of tokens, resulting in **O(n²)** complexity:

```
Sequence length: 512   → Attention matrix: 262,144 elements
Sequence length: 4096  → Attention matrix: 16,777,216 elements
Sequence length: 8192  → Attention matrix: 67,108,864 elements
```

For financial applications requiring long historical context (e.g., one year of daily data = 252 points, one month of hourly data = 720 points, one day of minute data = 1440 points), this quadratic scaling becomes prohibitive.

### BigBird's Solution

BigBird introduces a **sparse attention pattern** that achieves **O(n)** complexity while maintaining:
- **Universal approximation**: Can approximate any sequence-to-sequence function
- **Turing completeness**: Can simulate any Turing machine

The key insight: not all token pairs need to attend to each other. A carefully designed sparse pattern captures both local and global dependencies.

```
Standard Transformer:          BigBird:
┌─────────────────┐           ┌─────────────────┐
│█████████████████│           │█ ░ █ ░ ░ █ ░ █ │  ← Global tokens
│█████████████████│           │░ █ █ █ ░ ░ ░ █ │
│█████████████████│           │█ █ █ █ █ ░ ░ ░ │  ← Window attention
│█████████████████│           │░ █ █ █ █ █ ░ ░ │
│█████████████████│           │░ ░ █ █ █ █ █ ░ │
│█████████████████│           │█ ░ ░ █ █ █ █ █ │
│█████████████████│           │░ ░ ░ ░ █ █ █ █ │
│█████████████████│           │█ █ ░ ░ ░ █ █ █ │  ← Random attention
└─────────────────┘           └─────────────────┘
  O(n²) dense                   O(n) sparse
```

### Key Advantages

1. **8x Longer Sequences**: Process sequences up to 8x longer on the same hardware
2. **Linear Complexity**: Memory and computation scale linearly with sequence length
3. **Theoretical Guarantees**: Proven universal approximation and Turing completeness
4. **Flexibility**: Can add global tokens for task-specific important positions

## BigBird Architecture

BigBird's sparse attention combines three complementary patterns:

### Random Attention

Each query attends to `r` randomly selected keys, enabling information flow across distant positions:

```python
def random_attention_pattern(seq_len: int, num_random: int) -> torch.Tensor:
    """
    Generate random attention pattern.

    Args:
        seq_len: Sequence length
        num_random: Number of random connections per query (r)

    Returns:
        Attention mask [seq_len, seq_len]
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):
        # Sample random indices (excluding self)
        candidates = list(range(seq_len))
        candidates.remove(i)
        random_indices = random.sample(candidates, min(num_random, len(candidates)))
        mask[i, random_indices] = True

    return mask
```

**Intuition**: Random connections create "shortcuts" in the attention graph, ensuring any two tokens are connected through a small number of hops (graph theory property).

### Window (Local) Attention

Each query attends to its local neighborhood of `w` tokens:

```python
def window_attention_pattern(seq_len: int, window_size: int) -> torch.Tensor:
    """
    Generate sliding window attention pattern.

    Args:
        seq_len: Sequence length
        window_size: Size of attention window (w)

    Returns:
        Attention mask [seq_len, seq_len]
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    half_window = window_size // 2

    for i in range(seq_len):
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        mask[i, start:end] = True

    return mask
```

**Intuition**: Financial time series have strong local dependencies (today's price depends heavily on yesterday's price). Window attention captures these patterns efficiently.

### Global Attention

Designated "global" tokens attend to all positions and are attended to by all positions:

```python
def global_attention_pattern(
    seq_len: int,
    global_indices: List[int]
) -> torch.Tensor:
    """
    Generate global attention pattern.

    Args:
        seq_len: Sequence length
        global_indices: Indices of global tokens (g)

    Returns:
        Attention mask [seq_len, seq_len]
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for idx in global_indices:
        mask[idx, :] = True  # Global token attends to all
        mask[:, idx] = True  # All tokens attend to global

    return mask
```

**Intuition**: Global tokens (like [CLS] in BERT) aggregate information from the entire sequence. In trading, these can represent key dates (earnings, FOMC meetings), market open/close, or learned important positions.

### Combined Sparse Pattern

BigBird combines all three patterns:

```python
class BigBirdAttentionPattern:
    """
    BigBird sparse attention pattern combining:
    - Random attention (r random connections per query)
    - Window attention (w local neighbors)
    - Global attention (g global tokens)
    """

    def __init__(
        self,
        seq_len: int,
        num_random: int = 3,
        window_size: int = 3,
        num_global: int = 2,
        global_tokens: str = 'first'  # 'first', 'last', 'both', 'random'
    ):
        self.seq_len = seq_len
        self.num_random = num_random
        self.window_size = window_size
        self.num_global = num_global

        # Determine global token positions
        if global_tokens == 'first':
            self.global_indices = list(range(num_global))
        elif global_tokens == 'last':
            self.global_indices = list(range(seq_len - num_global, seq_len))
        elif global_tokens == 'both':
            half = num_global // 2
            self.global_indices = list(range(half)) + list(range(seq_len - half, seq_len))
        else:  # random
            self.global_indices = random.sample(range(seq_len), num_global)

    def get_attention_mask(self) -> torch.Tensor:
        """Generate combined BigBird attention mask."""
        # Start with random attention
        mask = random_attention_pattern(self.seq_len, self.num_random)

        # Add window attention
        mask |= window_attention_pattern(self.seq_len, self.window_size)

        # Add global attention
        mask |= global_attention_pattern(self.seq_len, self.global_indices)

        # Ensure diagonal (self-attention)
        mask.fill_diagonal_(True)

        return mask
```

## Financial Applications

### Long-Range Market Dependencies

Financial markets exhibit dependencies across multiple time scales:

```
Short-term (minutes-hours):
- Intraday momentum
- Order flow imbalance
- Market microstructure effects

Medium-term (days-weeks):
- Trend following
- Mean reversion
- Earnings effects

Long-term (months-years):
- Business cycles
- Structural regime changes
- Seasonal patterns
```

BigBird's sparse attention captures all these with linear complexity:

```python
# Example: Processing one year of daily data
seq_len = 252  # Trading days in a year

# Standard transformer: 252 × 252 = 63,504 attention scores
# BigBird with window=5, random=3, global=2:
# Per token: 5 (window) + 3 (random) + 2 (global) ≈ 10 connections
# Total: 252 × 10 = 2,520 attention scores (25x reduction!)

pattern = BigBirdAttentionPattern(
    seq_len=252,
    window_size=5,      # Weekly local context
    num_random=3,       # Random long-range connections
    num_global=2        # First (year start) and last (most recent)
)
```

### Tick-Level Data Processing

For high-frequency applications, BigBird enables processing of long tick sequences:

```python
# Process 1 hour of tick data (approx. 10,000 ticks for liquid assets)
seq_len = 10000

# Standard transformer: 10000² = 100,000,000 attention scores (infeasible!)
# BigBird: 10000 × 15 = 150,000 attention scores

config = BigBirdConfig(
    seq_len=10000,
    window_size=11,     # Local microstructure (±5 ticks)
    num_random=3,       # Cross-session connections
    num_global=3,       # Key timestamps (open, significant events)
    d_model=128
)
```

### Multi-Timeframe Analysis

Use BigBird's global tokens to mark important timeframes:

```python
def create_multi_timeframe_globals(
    timestamps: pd.DatetimeIndex,
    mark_opens: bool = True,
    mark_closes: bool = True,
    mark_events: Optional[List[datetime]] = None
) -> List[int]:
    """
    Create global token indices for multi-timeframe analysis.

    Args:
        timestamps: Sequence timestamps
        mark_opens: Mark market open times as global
        mark_closes: Mark market close times as global
        mark_events: Custom event timestamps to mark as global

    Returns:
        List of global token indices
    """
    global_indices = []

    if mark_opens:
        # Find market open timestamps (9:30 ET for US markets)
        opens = timestamps[timestamps.hour == 9 & timestamps.minute == 30]
        global_indices.extend(timestamps.get_indexer(opens))

    if mark_closes:
        # Find market close timestamps (16:00 ET for US markets)
        closes = timestamps[timestamps.hour == 16 & timestamps.minute == 0]
        global_indices.extend(timestamps.get_indexer(closes))

    if mark_events:
        for event in mark_events:
            idx = timestamps.get_loc(event, method='nearest')
            global_indices.append(idx)

    return sorted(set(global_indices))
```

## Practical Examples

### 01: Data Preparation

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import ccxt
from datetime import datetime, timedelta

def fetch_bybit_data(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h',
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Bybit.

    Args:
        symbol: Trading pair
        timeframe: Candle timeframe
        limit: Number of candles to fetch

    Returns:
        DataFrame with OHLCV data
    """
    exchange = ccxt.bybit()

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for BigBird model.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional features
    """
    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility (rolling std of returns)
    df['volatility_20'] = df['log_return'].rolling(20).std()
    df['volatility_50'] = df['log_return'].rolling(50).std()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Volume features
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']

    # Price range
    df['range'] = (df['high'] - df['low']) / df['close']

    # VWAP (simplified)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['close']

    return df.dropna()

def create_sequences(
    df: pd.DataFrame,
    seq_len: int = 256,
    pred_len: int = 1,
    features: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for model training.

    Args:
        df: DataFrame with features
        seq_len: Input sequence length
        pred_len: Prediction horizon
        features: List of feature columns

    Returns:
        X: Input sequences [n_samples, seq_len, n_features]
        y: Targets [n_samples, pred_len]
    """
    if features is None:
        features = ['log_return', 'volatility_20', 'rsi', 'volume_ratio', 'range']

    data = df[features].values
    target = df['log_return'].values

    X, y = [], []
    for i in range(seq_len, len(data) - pred_len):
        X.append(data[i-seq_len:i])
        y.append(target[i:i+pred_len])

    return np.array(X), np.array(y)
```

### 02: BigBird Architecture

See [python/model.py](python/model.py) for complete implementation.

```python
# python/model.py (key components)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class BigBirdConfig:
    """BigBird model configuration"""
    seq_len: int = 512
    input_dim: int = 6
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # BigBird specific
    window_size: int = 7
    num_random: int = 3
    num_global: int = 2
    global_tokens: str = 'first'  # 'first', 'last', 'both'

    # Output
    output_dim: int = 1
    output_type: str = 'regression'  # 'regression', 'classification', 'quantile'

class BigBirdSparseAttention(nn.Module):
    """
    BigBird sparse attention mechanism combining:
    - Random attention
    - Sliding window attention
    - Global attention
    """

    def __init__(self, config: BigBirdConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Pre-compute attention pattern
        self.register_buffer('attention_mask',
                           self._create_attention_mask())

    def _create_attention_mask(self) -> torch.Tensor:
        """Create BigBird sparse attention mask."""
        seq_len = self.config.seq_len
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

        # Window attention
        half_w = self.config.window_size // 2
        for i in range(seq_len):
            start = max(0, i - half_w)
            end = min(seq_len, i + half_w + 1)
            mask[i, start:end] = True

        # Random attention (fixed pattern for reproducibility)
        torch.manual_seed(42)
        for i in range(seq_len):
            num_random = min(self.config.num_random, seq_len - self.config.window_size)
            if num_random > 0:
                candidates = torch.where(~mask[i])[0]
                if len(candidates) > 0:
                    indices = candidates[torch.randperm(len(candidates))[:num_random]]
                    mask[i, indices] = True

        # Global attention
        if self.config.global_tokens == 'first':
            global_indices = list(range(self.config.num_global))
        elif self.config.global_tokens == 'last':
            global_indices = list(range(seq_len - self.config.num_global, seq_len))
        else:  # 'both'
            half = self.config.num_global // 2
            global_indices = list(range(half)) + list(range(seq_len - half, seq_len))

        for idx in global_indices:
            mask[idx, :] = True
            mask[:, idx] = True

        return mask

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with sparse attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_attention: Whether to return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            attention: [batch, n_heads, seq_len, seq_len] (optional)
        """
        batch, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply sparse mask
        mask = self.attention_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.out_proj(out)

        return out, attn if return_attention else None


class BigBirdEncoderLayer(nn.Module):
    """BigBird transformer encoder layer."""

    def __init__(self, config: BigBirdConfig):
        super().__init__()
        self.attention = BigBirdSparseAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, return_attention)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        x = self.norm2(x + self.ff(x))

        return x, attn_weights


class BigBirdForTrading(nn.Module):
    """
    BigBird model for financial time series prediction.

    Example:
        config = BigBirdConfig(seq_len=512, input_dim=6, d_model=128)
        model = BigBirdForTrading(config)

        x = torch.randn(32, 512, 6)  # [batch, seq_len, features]
        output = model(x)
        print(output['predictions'].shape)  # [32, 1]
    """

    def __init__(self, config: BigBirdConfig):
        super().__init__()
        self.config = config

        # Input embedding
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.positional_encoding = self._create_positional_encoding(
            config.seq_len, config.d_model
        )
        self.dropout = nn.Dropout(config.dropout)

        # Encoder layers
        self.layers = nn.ModuleList([
            BigBirdEncoderLayer(config) for _ in range(config.n_layers)
        ])

        # Output head
        self.output_head = self._build_output_head(config)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def _build_output_head(self, config: BigBirdConfig) -> nn.Module:
        if config.output_type == 'classification':
            return nn.Linear(config.d_model, 3)  # down, neutral, up
        elif config.output_type == 'quantile':
            return nn.Linear(config.d_model, 3)  # q10, q50, q90
        else:  # regression
            return nn.Linear(config.d_model, config.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> dict:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with predictions and optional attention weights
        """
        batch, seq_len, _ = x.shape

        # Input projection and positional encoding
        x = self.input_proj(x)
        x = x + self.positional_encoding[:, :seq_len]
        x = self.dropout(x)

        # Encoder layers
        all_attention = {}
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, return_attention)
            if attn is not None:
                all_attention[f'layer_{i}'] = attn

        # Pool: use last position or mean
        pooled = x[:, -1, :]  # [batch, d_model]

        # Output projection
        predictions = self.output_head(pooled)

        if self.config.output_type == 'classification':
            predictions = F.softmax(predictions, dim=-1)

        return {
            'predictions': predictions,
            'attention_weights': all_attention if return_attention else None,
            'hidden_states': x
        }
```

### 03: Model Training

```python
# python/03_train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import BigBirdConfig, BigBirdForTrading
from data import fetch_bybit_data, prepare_features, create_sequences
import numpy as np

def train_bigbird_model(
    symbols: list = ['BTCUSDT', 'ETHUSDT'],
    seq_len: int = 256,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """
    Train BigBird model on cryptocurrency data.
    """
    # Prepare data
    print("Fetching and preparing data...")
    all_X, all_y = [], []

    for symbol in symbols:
        df = fetch_bybit_data(symbol, timeframe='1h', limit=5000)
        df = prepare_features(df)
        X, y = create_sequences(df, seq_len=seq_len)
        all_X.append(X)
        all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Train/val split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    config = BigBirdConfig(
        seq_len=seq_len,
        input_dim=X.shape[-1],
        d_model=128,
        n_heads=8,
        n_layers=4,
        window_size=7,
        num_random=3,
        num_global=2
    )

    model = BigBirdForTrading(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output['predictions'], batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                val_loss += criterion(output['predictions'], batch_y).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    return model

if __name__ == "__main__":
    model = train_bigbird_model()
```

### 04: Long Sequence Prediction

```python
# python/04_long_sequence_prediction.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns

def predict_and_visualize(
    model,
    X: torch.Tensor,
    timestamps: list = None
):
    """
    Make predictions and visualize attention patterns.
    """
    model.eval()
    with torch.no_grad():
        output = model(X, return_attention=True)

    predictions = output['predictions']
    attention = output['attention_weights']

    # Visualize attention from last layer
    if attention:
        last_layer_attn = attention['layer_3']  # [batch, heads, seq, seq]

        # Average over heads and batch
        avg_attn = last_layer_attn[0].mean(dim=0).cpu().numpy()

        plt.figure(figsize=(12, 10))
        sns.heatmap(avg_attn, cmap='Blues', vmax=0.1)
        plt.title('BigBird Sparse Attention Pattern')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.savefig('attention_pattern.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Analyze which positions get most attention
    attention_received = avg_attn.sum(axis=0)
    top_attended = np.argsort(attention_received)[-10:]

    print("Top 10 most attended positions:")
    for pos in reversed(top_attended):
        print(f"  Position {pos}: {attention_received[pos]:.4f}")

    return predictions, attention

def analyze_global_token_impact(model, X: torch.Tensor):
    """
    Analyze the impact of global tokens on predictions.
    """
    model.eval()

    # Get baseline prediction
    with torch.no_grad():
        baseline_output = model(X)
        baseline_pred = baseline_output['predictions']

    # Ablate global tokens (set to zero) and compare
    X_ablated = X.clone()
    global_indices = list(range(model.config.num_global))
    X_ablated[:, global_indices, :] = 0

    with torch.no_grad():
        ablated_output = model(X_ablated)
        ablated_pred = ablated_output['predictions']

    impact = (baseline_pred - ablated_pred).abs().mean()
    print(f"Global token impact on predictions: {impact:.6f}")

    return baseline_pred, ablated_pred, impact
```

### 05: Backtesting Strategy

```python
# python/05_backtest.py

import numpy as np
import pandas as pd
from typing import Dict, List

def backtest_bigbird_strategy(
    model,
    test_data: pd.DataFrame,
    seq_len: int = 256,
    features: List[str] = None,
    initial_capital: float = 100000,
    position_size: float = 0.1,
    transaction_cost: float = 0.001
) -> Dict:
    """
    Backtest BigBird prediction strategy.

    Args:
        model: Trained BigBird model
        test_data: DataFrame with OHLCV and features
        seq_len: Input sequence length
        features: Feature columns
        initial_capital: Starting capital
        position_size: Fraction of capital per trade
        transaction_cost: Transaction cost (0.1% = 0.001)

    Returns:
        Dictionary with backtest results
    """
    if features is None:
        features = ['log_return', 'volatility_20', 'rsi', 'volume_ratio', 'range']

    model.eval()
    device = next(model.parameters()).device

    capital = initial_capital
    position = 0  # -1 (short), 0 (flat), 1 (long)

    results = {
        'timestamp': [],
        'capital': [],
        'position': [],
        'prediction': [],
        'actual_return': [],
        'trade_return': []
    }

    data = test_data[features].values
    returns = test_data['log_return'].values

    for i in range(seq_len, len(data) - 1):
        # Get sequence
        x = torch.FloatTensor(data[i-seq_len:i]).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(x)
            pred = output['predictions'].item()

        # Trading logic
        actual_return = returns[i]

        # Determine signal
        if pred > 0.001:  # Predict up
            new_position = 1
        elif pred < -0.001:  # Predict down
            new_position = -1
        else:
            new_position = 0

        # Calculate trade return
        trade_return = position * actual_return

        # Apply transaction costs if position changed
        if new_position != position:
            cost = abs(new_position - position) * transaction_cost
            capital *= (1 - cost)

        # Update capital
        capital *= (1 + trade_return * position_size)
        position = new_position

        # Record results
        results['timestamp'].append(test_data.index[i])
        results['capital'].append(capital)
        results['position'].append(position)
        results['prediction'].append(pred)
        results['actual_return'].append(actual_return)
        results['trade_return'].append(trade_return)

    # Calculate metrics
    results_df = pd.DataFrame(results)
    results_df.set_index('timestamp', inplace=True)

    total_return = (capital - initial_capital) / initial_capital
    daily_returns = results_df['capital'].pct_change().dropna()

    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    sortino_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns[daily_returns < 0].std() if len(daily_returns[daily_returns < 0]) > 0 else 0

    cummax = results_df['capital'].cummax()
    drawdown = (results_df['capital'] - cummax) / cummax
    max_drawdown = drawdown.min()

    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'final_capital': capital,
        'num_trades': (results_df['position'].diff() != 0).sum()
    }

    return {
        'results': results_df,
        'metrics': metrics
    }

def compare_with_baseline(backtest_results: Dict, test_data: pd.DataFrame):
    """
    Compare strategy performance with buy-and-hold baseline.
    """
    strategy_return = backtest_results['metrics']['total_return']

    # Buy and hold return
    buy_hold_return = (test_data['close'].iloc[-1] / test_data['close'].iloc[0]) - 1

    print("\n" + "="*50)
    print("Strategy Comparison")
    print("="*50)
    print(f"BigBird Strategy Return: {strategy_return*100:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return*100:.2f}%")
    print(f"Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['metrics']['max_drawdown']*100:.2f}%")
    print(f"Number of Trades: {backtest_results['metrics']['num_trades']}")
    print("="*50)
```

## Rust Implementation

See [rust/](rust/) for complete Rust implementation using the `burn` deep learning framework.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── config.rs           # Configuration structures
│   ├── attention.rs        # BigBird sparse attention
│   ├── model.rs            # Full BigBird model
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Bybit API client
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset implementation
│   └── strategy/
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting engine
└── examples/
    ├── fetch_data.rs       # Download market data
    ├── train.rs            # Train model
    └── backtest.rs         # Run backtest
```

### Quick Start (Rust)

```bash
cd rust

# Fetch data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT --timeframe 1h

# Train model
cargo run --example train -- --epochs 100 --seq-len 256

# Run backtest
cargo run --example backtest -- --model checkpoints/best.safetensors
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── __init__.py
├── config.py               # Configuration classes
├── model.py                # BigBird model implementation
├── data.py                 # Data loading and preprocessing
├── train.py                # Training script
├── backtest.py             # Backtesting utilities
├── requirements.txt        # Dependencies
└── examples/
    ├── 01_data_preparation.py
    ├── 02_model_architecture.py
    ├── 03_training.py
    ├── 04_prediction.py
    └── 05_backtesting.py
```

### Quick Start (Python)

```bash
cd python

# Install dependencies
pip install -r requirements.txt

# Run examples
python examples/01_data_preparation.py
python examples/03_training.py --epochs 100
python examples/05_backtesting.py --model checkpoints/best.pt
```

## Comparison with Other Methods

| Method | Complexity | Max Sequence | Global Context | Local Context |
|--------|------------|--------------|----------------|---------------|
| Standard Transformer | O(n²) | ~512 | Full | Full |
| Linformer | O(n) | ~4096 | Approximated | Approximated |
| Performer | O(n) | ~8192 | Approximated | Limited |
| Longformer | O(n) | ~4096 | Global tokens | Window |
| **BigBird** | O(n) | ~8192 | Global tokens | Window + Random |
| Reformer | O(n log n) | ~64k | LSH-based | LSH-based |

### When to Use BigBird

**Ideal for:**
- Long historical sequences (>500 time steps)
- When both local and global patterns matter
- Multi-day or multi-week prediction horizons
- Tick-level data processing

**Consider alternatives when:**
- Short sequences (<256) - use standard transformer
- Purely local patterns - use convolutional models
- Real-time inference with strict latency - use simpler models

## Best Practices

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `seq_len` | 256-1024 | Longer for low-frequency data |
| `window_size` | 5-11 | Odd number, ~1-2% of seq_len |
| `num_random` | 2-5 | More for longer sequences |
| `num_global` | 2-4 | First and/or last positions |
| `d_model` | 128-256 | Scale with data complexity |
| `n_heads` | 4-8 | Must divide d_model |

### Common Pitfalls

1. **Mask Caching**: Pre-compute attention masks for efficiency
2. **Global Token Placement**: Place global tokens at meaningful positions (market open, key events)
3. **Sequence Length Mismatch**: Ensure training and inference use same seq_len
4. **Memory Management**: For very long sequences, use gradient checkpointing

### Training Tips

```python
# 1. Use learning rate warmup
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    total_steps=total_steps,
    pct_start=0.1  # 10% warmup
)

# 2. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 3. Mixed precision training for longer sequences
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(x)
    loss = criterion(output['predictions'], y)
```

## Resources

### Papers

- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) - Original BigBird paper (NeurIPS 2020)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - Related sliding window approach
- [ETC: Encoding Long and Structured Inputs](https://arxiv.org/abs/2004.08483) - Global-local attention

### Implementations

- [Google Research BigBird](https://github.com/google-research/bigbird) - Official implementation
- [Hugging Face BigBird](https://huggingface.co/docs/transformers/model_doc/big_bird) - PyTorch implementation
- [PyTorch Sparse Attention](https://pytorch.org/docs/stable/sparse.html) - Sparse tensor operations

### Related Chapters

- [Chapter 51: Linformer Long Sequences](../51_linformer_long_sequences) - Linear complexity alternative
- [Chapter 52: Performer Efficient Attention](../52_performer_efficient_attention) - Kernel-based attention
- [Chapter 54: Reformer LSH Attention](../54_reformer_lsh_attention) - Locality-sensitive hashing
- [Chapter 57: Longformer Financial](../57_longformer_financial) - Sliding window attention

---

## Difficulty Level

**Intermediate-Advanced**

Prerequisites:
- Transformer architecture fundamentals
- Attention mechanisms (self-attention, multi-head attention)
- PyTorch/Rust ML basics
- Time series forecasting concepts
