# Chapter 48: Positional Encoding for Time Series

This chapter explores **Positional Encoding** techniques specifically designed for time series and financial data. Unlike standard NLP transformers, time series require specialized encodings that capture temporal patterns, periodicity, and financial market dynamics.

<p align="center">
<img src="https://i.imgur.com/p2VxZcR.png" width="70%">
</p>

## Contents

1. [Introduction to Positional Encoding](#introduction-to-positional-encoding)
    * [Why Position Matters](#why-position-matters)
    * [Time Series Challenges](#time-series-challenges)
    * [Types of Positional Encoding](#types-of-positional-encoding)
2. [Sinusoidal Positional Encoding](#sinusoidal-positional-encoding)
    * [Mathematical Foundation](#mathematical-foundation)
    * [Implementation](#sinusoidal-implementation)
    * [Time Series Adaptations](#time-series-adaptations)
3. [Learned Positional Encoding](#learned-positional-encoding)
    * [Trainable Embeddings](#trainable-embeddings)
    * [Advantages for Financial Data](#advantages-for-financial-data)
4. [Relative Positional Encoding](#relative-positional-encoding)
    * [Shaw's Relative Attention](#shaws-relative-attention)
    * [XLNet Style Encoding](#xlnet-style-encoding)
5. [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
    * [Mathematical Formulation](#rope-mathematical-formulation)
    * [RoPE for Time Series](#rope-for-time-series)
6. [Temporal Encodings for Finance](#temporal-encodings-for-finance)
    * [Calendar Features](#calendar-features)
    * [Market Session Encoding](#market-session-encoding)
    * [Multi-Scale Temporal Encoding](#multi-scale-temporal-encoding)
7. [Practical Examples](#practical-examples)
    * [01: Comparing Encoding Methods](#01-comparing-encoding-methods)
    * [02: Time Series Transformer](#02-time-series-transformer)
    * [03: Crypto Price Prediction](#03-crypto-price-prediction)
    * [04: Stock Market Forecasting](#04-stock-market-forecasting)
    * [05: Backtesting Strategies](#05-backtesting-strategies)
8. [Rust Implementation](#rust-implementation)
9. [Python Implementation](#python-implementation)
10. [Best Practices](#best-practices)
11. [Resources](#resources)

## Introduction to Positional Encoding

Transformers process sequences without inherent order awareness. Unlike RNNs that process tokens sequentially, self-attention treats all positions equally. **Positional encoding** injects position information into the model.

### Why Position Matters

For time series, position carries critical information:

```
Without position:  [100, 105, 103, 108, 102] = Price values (unordered set)
With position:     t=1: 100 → t=2: 105 → t=3: 103 → t=4: 108 → t=5: 102

The sequence tells a story:
- Price went UP from 100 to 105 (+5%)
- Then DOWN to 103 (-2%)
- Then UP to 108 (+5%)
- Then DOWN to 102 (-6%)
```

**Position determines meaning**:
- `[100, 105]` = bullish trend (price rising)
- `[105, 100]` = bearish trend (price falling)

### Time Series Challenges

Financial time series have unique characteristics:

| Challenge | Description | Solution |
|-----------|-------------|----------|
| Variable lengths | Different prediction horizons | Relative encodings |
| Multiple time scales | Minutes, hours, days, weeks | Multi-scale encoding |
| Periodicity | Daily/weekly/monthly patterns | Sinusoidal encoding |
| Non-stationarity | Market regime changes | Learned + contextual encoding |
| Missing data | Market holidays, gaps | Masked position encoding |

### Types of Positional Encoding

```
┌────────────────────────────────────────────────────────────────┐
│                 POSITIONAL ENCODING TYPES                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. SINUSOIDAL (Fixed)                                         │
│     ├── No learnable parameters                                │
│     ├── Extrapolates to unseen lengths                         │
│     └── PE(pos, 2i) = sin(pos / 10000^(2i/d))                 │
│                                                                 │
│  2. LEARNED (Trainable)                                        │
│     ├── Embedding table for each position                      │
│     ├── Adapts to data patterns                                │
│     └── Limited to training sequence length                    │
│                                                                 │
│  3. RELATIVE (Shaw, T5)                                        │
│     ├── Encodes distance between tokens                        │
│     ├── Good for varying sequence lengths                      │
│     └── att(i,j) depends on (i-j)                             │
│                                                                 │
│  4. ROTARY (RoPE)                                              │
│     ├── Rotates query/key vectors                              │
│     ├── Relative position via rotation                         │
│     └── Used in LLaMA, GPT-NeoX                               │
│                                                                 │
│  5. TEMPORAL (Time Series Specific)                            │
│     ├── Calendar features (day, month, year)                   │
│     ├── Trading session indicators                             │
│     └── Multi-frequency components                             │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Sinusoidal Positional Encoding

The original Transformer uses sinusoidal functions to encode position:

### Mathematical Foundation

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos = position in sequence (0, 1, 2, ...)
- i = dimension index (0, 1, ..., d_model/2 - 1)
- d_model = embedding dimension
```

**Why sine/cosine?**
1. **Bounded values**: Always between [-1, 1]
2. **Uniqueness**: Each position has unique encoding
3. **Relative position**: `PE(pos+k)` can be represented as linear function of `PE(pos)`
4. **Extrapolation**: Works for sequences longer than training

### Sinusoidal Implementation

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from 'Attention Is All You Need'

    For time series, we extend this with:
    - Optional scaling for different time scales
    - Temperature parameter for frequency control
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        temperature: float = 10000.0
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Create position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Frequency terms
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(temperature) / d_model)
        )

        # Interleave sin and cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
        Returns:
            Position-encoded tensor [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### Time Series Adaptations

For financial time series, we can adapt sinusoidal encoding:

```python
class TimeSeriesSinusoidalEncoding(nn.Module):
    """
    Sinusoidal encoding adapted for time series with multiple frequencies

    Captures:
    - Intraday patterns (hourly cycles)
    - Daily patterns (market open/close)
    - Weekly patterns (Monday effect)
    - Monthly patterns (month-end rebalancing)
    """

    def __init__(
        self,
        d_model: int,
        frequencies: list = [24, 24*7, 24*30, 24*365],  # Hourly data periods
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.frequencies = frequencies
        self.dropout = nn.Dropout(p=dropout)

        # Allocate dimensions to each frequency
        dims_per_freq = d_model // (len(frequencies) * 2)
        self.dims_per_freq = dims_per_freq

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            timestamps: Optional absolute timestamps [batch, seq_len]
        Returns:
            Position-encoded tensor [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        device = x.device

        if timestamps is None:
            # Use sequential positions
            positions = torch.arange(seq_len, device=device).float()
        else:
            positions = timestamps.float()

        pe = torch.zeros(seq_len, d_model, device=device)

        dim_idx = 0
        for freq in self.frequencies:
            # Normalize position to frequency period
            pos_normalized = positions / freq

            for i in range(self.dims_per_freq):
                # Multiple harmonics per frequency
                harmonic = 2 ** i
                pe[:, dim_idx] = torch.sin(2 * math.pi * harmonic * pos_normalized)
                pe[:, dim_idx + 1] = torch.cos(2 * math.pi * harmonic * pos_normalized)
                dim_idx += 2

        # Fill remaining dimensions with standard sinusoidal
        if dim_idx < d_model:
            position = torch.arange(seq_len, device=device).unsqueeze(1)
            remaining_dims = d_model - dim_idx
            div_term = torch.exp(
                torch.arange(0, remaining_dims, 2, device=device).float() *
                (-math.log(10000.0) / remaining_dims)
            )
            pe[:, dim_idx::2] = torch.sin(position * div_term)
            pe[:, dim_idx+1::2] = torch.cos(position * div_term)

        x = x + pe.unsqueeze(0)
        return self.dropout(x)
```

## Learned Positional Encoding

Instead of fixed functions, learn position embeddings from data:

### Trainable Embeddings

```python
class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding using embedding table

    Advantages:
    - Adapts to dataset-specific patterns
    - Can learn asymmetric dependencies
    - Works well with fixed-length sequences

    Disadvantages:
    - Cannot extrapolate beyond training length
    - More parameters to optimize
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
        init_std: float = 0.02
    ):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Initialize with small values
        nn.init.normal_(self.embedding.weight, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
        Returns:
            Position-encoded tensor [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.embedding(positions)  # [seq_len, d_model]

        x = x + pos_emb.unsqueeze(0)
        return self.dropout(x)
```

### Advantages for Financial Data

Learned encodings can capture:
- **Recency bias**: Recent prices matter more
- **Asymmetric lookback**: Different weights for different lags
- **Non-linear decay**: Custom attention patterns over time

```python
class FinancialLearnedEncoding(nn.Module):
    """
    Learned encoding with financial market priors

    Incorporates:
    - Recency weighting (recent data more important)
    - Multi-scale time embeddings
    - Weekday/hour embeddings for market patterns
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        n_weekdays: int = 7,
        n_hours: int = 24,
        dropout: float = 0.1
    ):
        super().__init__()

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_len, d_model // 2)

        # Temporal embeddings
        self.weekday_embedding = nn.Embedding(n_weekdays, d_model // 4)
        self.hour_embedding = nn.Embedding(n_hours, d_model // 4)

        # Recency decay (learnable)
        self.decay = nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        weekdays: torch.Tensor = None,
        hours: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input [batch, seq_len, d_model]
            weekdays: Weekday indices [batch, seq_len] (0=Monday, 6=Sunday)
            hours: Hour indices [batch, seq_len] (0-23)
        """
        batch, seq_len, d_model = x.shape
        device = x.device

        # Position embedding
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.pos_embedding(positions)  # [seq_len, d_model//2]

        # Apply recency decay
        decay_weights = torch.exp(-self.decay * (seq_len - 1 - positions.float()) / seq_len)
        pos_emb = pos_emb * decay_weights.unsqueeze(-1)

        # Temporal embeddings (if provided)
        if weekdays is not None and hours is not None:
            weekday_emb = self.weekday_embedding(weekdays)  # [batch, seq_len, d_model//4]
            hour_emb = self.hour_embedding(hours)           # [batch, seq_len, d_model//4]

            # Combine all embeddings
            combined = torch.cat([
                pos_emb.unsqueeze(0).expand(batch, -1, -1),
                weekday_emb,
                hour_emb
            ], dim=-1)
        else:
            # Pad with zeros if temporal info not available
            combined = torch.cat([
                pos_emb.unsqueeze(0).expand(batch, -1, -1),
                torch.zeros(batch, seq_len, d_model // 2, device=device)
            ], dim=-1)

        x = x + combined
        return self.dropout(x)
```

## Relative Positional Encoding

Encodes the *distance* between positions rather than absolute positions:

### Shaw's Relative Attention

```python
class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding from Shaw et al.

    Instead of adding position to input, modify attention scores:
    attention(Q, K) = softmax((Q @ K^T + Q @ R^T) / sqrt(d))

    Where R is relative position embedding
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_relative_position: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_relative_position = max_relative_position

        # Relative position embeddings: [-max_pos, ..., 0, ..., max_pos]
        self.relative_embedding = nn.Embedding(
            2 * max_relative_position + 1,
            self.head_dim
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q: Query tensor [batch, n_heads, seq_len, head_dim]
            k: Key tensor [batch, n_heads, seq_len, head_dim]
            v: Value tensor [batch, n_heads, seq_len, head_dim]
        Returns:
            Attention output [batch, n_heads, seq_len, head_dim]
        """
        batch, n_heads, seq_len, head_dim = q.shape
        device = q.device

        # Standard QK attention scores
        qk_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq, seq]

        # Compute relative position indices
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq, seq]
        relative_positions = relative_positions.clamp(
            -self.max_relative_position,
            self.max_relative_position
        )
        relative_positions = relative_positions + self.max_relative_position  # Shift to positive

        # Get relative embeddings
        rel_emb = self.relative_embedding(relative_positions)  # [seq, seq, head_dim]

        # Compute relative attention scores: Q @ R^T
        q_expanded = q.unsqueeze(3)  # [batch, heads, seq, 1, head_dim]
        rel_emb_expanded = rel_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq, head_dim]

        relative_scores = torch.matmul(
            q_expanded,
            rel_emb_expanded.transpose(-2, -1)
        ).squeeze(-2)  # [batch, heads, seq, seq]

        # Combine scores
        scores = (qk_scores + relative_scores) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        return output
```

### XLNet Style Encoding

XLNet uses a more sophisticated relative encoding:

```python
class XLNetRelativeEncoding(nn.Module):
    """
    XLNet-style relative positional encoding

    Uses learnable parameters for:
    - Content-based attention (c)
    - Position-based attention (p)
    - Segment-based attention (optional)
    """

    def __init__(self, d_model: int, n_heads: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Relative position embedding
        self.r_embedding = nn.Embedding(max_len, d_model)

        # Content and position biases
        self.u = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.v = nn.Parameter(torch.zeros(n_heads, self.head_dim))

        # Initialize
        nn.init.normal_(self.u, std=0.02)
        nn.init.normal_(self.v, std=0.02)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Compute relative attention bias

        Args:
            q: Query [batch, n_heads, seq_len, head_dim]
            k: Key [batch, n_heads, seq_len, head_dim]
        Returns:
            Attention bias to add to scores
        """
        batch, n_heads, seq_len, head_dim = q.shape
        device = q.device

        # Get relative positions
        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq, seq]
        rel_pos = rel_pos.abs().clamp(max=seq_len-1)

        # Get relative embeddings
        r = self.r_embedding(rel_pos)  # [seq, seq, d_model]
        r = r.view(seq_len, seq_len, n_heads, head_dim)
        r = r.permute(2, 0, 1, 3)  # [n_heads, seq, seq, head_dim]

        # Content attention: (q + u) @ k^T
        q_u = q + self.u.unsqueeze(0).unsqueeze(2)
        content_attn = torch.matmul(q_u, k.transpose(-2, -1))

        # Position attention: (q + v) @ r^T
        q_v = q + self.v.unsqueeze(0).unsqueeze(2)
        position_attn = torch.einsum('bhid,hjid->bhij', q_v, r)

        return content_attn + position_attn
```

## Rotary Position Embedding (RoPE)

RoPE encodes position by rotating query and key vectors:

### RoPE Mathematical Formulation

```
For query q and key k at positions m and n:
RoPE(q, m) = R_m @ q
RoPE(k, n) = R_n @ k

Where R is rotation matrix:
R_m = [cos(mθ₁)  -sin(mθ₁)    0        0     ...
       sin(mθ₁)   cos(mθ₁)    0        0     ...
         0          0      cos(mθ₂) -sin(mθ₂) ...
         0          0      sin(mθ₂)  cos(mθ₂) ...
        ...       ...       ...      ...     ...]

Result: (R_m @ q)^T @ (R_n @ k) = q^T @ R_{n-m} @ k

The attention score depends on relative position (n-m)!
```

### RoPE for Time Series

```python
class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for time series

    Key insight: Rotate query/key vectors by position-dependent angle
    Result: Attention scores naturally encode relative position

    Benefits for time series:
    - Handles long sequences efficiently
    - Natural decay for distant positions
    - Works with variable-length sequences
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_len: int = 8192,
        base: float = 10000.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_len = max_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer('inv_freq', inv_freq)

        # Precompute sin/cos cache
        self._set_cos_sin_cache(max_len)

    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin values"""
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)

        # Stack sin and cos for rotation
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor = None
    ) -> tuple:
        """
        Apply rotary embeddings to queries and keys

        Args:
            q: Query tensor [batch, n_heads, seq_len, head_dim]
            k: Key tensor [batch, n_heads, seq_len, head_dim]
            positions: Optional position indices [batch, seq_len]

        Returns:
            Tuple of rotated (query, key) tensors
        """
        batch, n_heads, seq_len, head_dim = q.shape

        if positions is None:
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cached[positions].unsqueeze(1)
            sin = self.sin_cached[positions].unsqueeze(1)

        # Apply rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot


class RoPETimeSeriesAttention(nn.Module):
    """
    Self-attention with RoPE for time series data
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_len: int = 8192
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionalEncoding(d_model, n_heads, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input [batch, seq_len, d_model]
            mask: Optional attention mask
        Returns:
            Output [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.out_proj(out)

        return out
```

## Temporal Encodings for Finance

Specialized encodings that capture financial market patterns:

### Calendar Features

```python
class CalendarEncoding(nn.Module):
    """
    Encode calendar features important for financial markets

    Features:
    - Day of week (Monday effect)
    - Month (January effect, month-end rebalancing)
    - Quarter (earnings seasons)
    - Year (for regime awareness)
    - Trading session (market hours)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        # Dimension allocation
        self.d_dayofweek = d_model // 8
        self.d_month = d_model // 8
        self.d_quarter = d_model // 16
        self.d_hour = d_model // 8
        self.d_session = d_model // 16

        # Embeddings
        self.dayofweek_emb = nn.Embedding(7, self.d_dayofweek)      # Mon-Sun
        self.month_emb = nn.Embedding(12, self.d_month)             # Jan-Dec
        self.quarter_emb = nn.Embedding(4, self.d_quarter)          # Q1-Q4
        self.hour_emb = nn.Embedding(24, self.d_hour)               # 0-23
        self.session_emb = nn.Embedding(4, self.d_session)          # Pre/Regular/After/Closed

        # Project to model dimension
        total_dim = (self.d_dayofweek + self.d_month + self.d_quarter +
                     self.d_hour + self.d_session)
        self.proj = nn.Linear(total_dim, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        dayofweek: torch.Tensor,
        month: torch.Tensor,
        quarter: torch.Tensor,
        hour: torch.Tensor,
        session: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            All inputs: [batch, seq_len]
        Returns:
            Calendar encoding [batch, seq_len, d_model]
        """
        dow = self.dayofweek_emb(dayofweek)
        mon = self.month_emb(month)
        qtr = self.quarter_emb(quarter)
        hr = self.hour_emb(hour)
        ses = self.session_emb(session)

        combined = torch.cat([dow, mon, qtr, hr, ses], dim=-1)
        out = self.proj(combined)

        return self.dropout(out)
```

### Market Session Encoding

```python
class MarketSessionEncoding(nn.Module):
    """
    Encode market session information

    For crypto (24/7): Time-of-day patterns
    For stocks: Pre-market, regular, after-hours, closed
    """

    def __init__(
        self,
        d_model: int,
        market_type: str = 'crypto',  # 'crypto' or 'stock'
        dropout: float = 0.1
    ):
        super().__init__()
        self.market_type = market_type

        if market_type == 'crypto':
            # 24-hour cycle with Asian/European/American sessions
            self.session_emb = nn.Embedding(3, d_model // 3)  # Asia/Europe/US
            self.hour_emb = nn.Embedding(24, d_model // 3)
            self.remaining = d_model - 2 * (d_model // 3)
            self.proj = nn.Linear(2 * (d_model // 3), d_model)
        else:
            # Stock market sessions
            self.session_emb = nn.Embedding(4, d_model // 2)  # Pre/Regular/After/Closed
            self.time_in_session = nn.Embedding(100, d_model // 2)  # Minutes into session
            self.proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _get_crypto_session(self, hour: torch.Tensor) -> torch.Tensor:
        """Map hour to crypto trading session"""
        # Asia: 0-8 UTC, Europe: 8-16 UTC, US: 16-24 UTC
        session = torch.zeros_like(hour)
        session[(hour >= 0) & (hour < 8)] = 0   # Asia
        session[(hour >= 8) & (hour < 16)] = 1  # Europe
        session[(hour >= 16) & (hour < 24)] = 2 # US
        return session

    def forward(
        self,
        hour: torch.Tensor,
        session: torch.Tensor = None,
        time_in_session: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            hour: Hour of day [batch, seq_len]
            session: Market session (for stocks) [batch, seq_len]
            time_in_session: Minutes into session [batch, seq_len]
        """
        if self.market_type == 'crypto':
            crypto_session = self._get_crypto_session(hour)
            ses_emb = self.session_emb(crypto_session)
            hr_emb = self.hour_emb(hour)
            combined = torch.cat([ses_emb, hr_emb], dim=-1)
        else:
            ses_emb = self.session_emb(session)
            time_emb = self.time_in_session(time_in_session.clamp(0, 99))
            combined = torch.cat([ses_emb, time_emb], dim=-1)

        out = self.proj(combined)
        return self.dropout(out)
```

### Multi-Scale Temporal Encoding

```python
class MultiScaleTemporalEncoding(nn.Module):
    """
    Encode time at multiple scales for comprehensive temporal representation

    Scales:
    - Micro: Within trading session (minutes)
    - Intraday: Hours within day
    - Daily: Day patterns
    - Weekly: Week patterns
    - Monthly: Month patterns
    """

    def __init__(
        self,
        d_model: int,
        time_scales: list = ['minute', 'hour', 'day', 'week', 'month'],
        dropout: float = 0.1
    ):
        super().__init__()
        self.time_scales = time_scales

        # Dimension per scale
        d_per_scale = d_model // len(time_scales)
        self.d_per_scale = d_per_scale

        # Embeddings for each scale
        self.scale_embeddings = nn.ModuleDict()
        scale_sizes = {
            'minute': 60,
            'hour': 24,
            'day': 31,
            'week': 7,
            'month': 12
        }

        for scale in time_scales:
            self.scale_embeddings[scale] = nn.Embedding(
                scale_sizes[scale],
                d_per_scale
            )

        # Final projection
        self.proj = nn.Linear(d_per_scale * len(time_scales), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, timestamps: dict) -> torch.Tensor:
        """
        Args:
            timestamps: Dict with keys for each scale
                       e.g., {'minute': [batch, seq], 'hour': [batch, seq], ...}
        Returns:
            Multi-scale temporal encoding [batch, seq_len, d_model]
        """
        embeddings = []

        for scale in self.time_scales:
            if scale in timestamps:
                emb = self.scale_embeddings[scale](timestamps[scale])
                embeddings.append(emb)
            else:
                # Zero embedding if scale not provided
                batch, seq_len = next(iter(timestamps.values())).shape
                embeddings.append(
                    torch.zeros(batch, seq_len, self.d_per_scale,
                               device=next(iter(timestamps.values())).device)
                )

        combined = torch.cat(embeddings, dim=-1)
        out = self.proj(combined)

        return self.dropout(out)
```

## Practical Examples

### 01: Comparing Encoding Methods

See [python/01_compare_encodings.py](python/01_compare_encodings.py) for a comparison of all encoding methods.

```python
# Example: Compare encoding methods on crypto data

import torch
from positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    MultiScaleTemporalEncoding
)

# Create sample data
batch_size = 32
seq_len = 168  # 7 days of hourly data
d_model = 64

x = torch.randn(batch_size, seq_len, d_model)

# Initialize encoders
sinusoidal = SinusoidalPositionalEncoding(d_model)
learned = LearnedPositionalEncoding(d_model, max_len=512)
rope = RotaryPositionalEncoding(d_model, n_heads=4)
multiscale = MultiScaleTemporalEncoding(d_model)

# Apply encodings
x_sin = sinusoidal(x)
x_learned = learned(x)

# For RoPE, need to work with Q/K matrices
q = torch.randn(batch_size, 4, seq_len, d_model // 4)
k = torch.randn(batch_size, 4, seq_len, d_model // 4)
q_rope, k_rope = rope(q, k)

print(f"Sinusoidal output shape: {x_sin.shape}")
print(f"Learned output shape: {x_learned.shape}")
print(f"RoPE Q shape: {q_rope.shape}")
```

### 02: Time Series Transformer

See [python/model.py](python/model.py) for complete transformer with positional encoding.

### 03: Crypto Price Prediction

```python
# Example: BTCUSDT prediction with RoPE

from data import BybitDataLoader
from model import TimeSeriesTransformer

# Load Bybit data
loader = BybitDataLoader()
data = loader.load_klines('BTCUSDT', interval='1h', limit=2000)

# Create model with RoPE encoding
config = {
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 2,
    'encoding_type': 'rope',
    'dropout': 0.1
}

model = TimeSeriesTransformer(config)

# Train and predict
# ... (see examples/)
```

### 04: Stock Market Forecasting

```python
# Example: S&P 500 with calendar encoding

from model import StockTransformer

# Create model with calendar features
config = {
    'd_model': 128,
    'n_heads': 8,
    'use_calendar_encoding': True,
    'use_market_session': True
}

model = StockTransformer(config)

# The model will use both position and calendar information
# for improved predictions around market events
```

### 05: Backtesting Strategies

See [python/strategy.py](python/strategy.py) for backtesting framework.

## Rust Implementation

See [rust_positional_encoding](rust_positional_encoding/) for complete Rust implementation.

```
rust_positional_encoding/
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
│   │   └── features.rs     # Feature engineering
│   ├── encoding/           # Positional encoding implementations
│   │   ├── mod.rs
│   │   ├── sinusoidal.rs   # Sinusoidal encoding
│   │   ├── learned.rs      # Learned encoding
│   │   ├── relative.rs     # Relative encoding
│   │   ├── rope.rs         # Rotary position embedding
│   │   └── temporal.rs     # Calendar/temporal encoding
│   ├── model/              # Transformer model
│   │   ├── mod.rs
│   │   ├── attention.rs    # Self-attention with encoding
│   │   └── transformer.rs  # Complete model
│   └── strategy/           # Trading strategy
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting engine
└── examples/
    ├── compare_encodings.rs
    ├── fetch_data.rs
    ├── train.rs
    └── backtest.rs
```

### Quick Start (Rust)

```bash
# Navigate to Rust project
cd rust_positional_encoding

# Fetch data from Bybit
cargo run --example fetch_data -- --symbol BTCUSDT --interval 1h

# Compare encoding methods
cargo run --example compare_encodings

# Train model
cargo run --example train -- --epochs 100 --encoding rope

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── positional_encoding.py  # All encoding implementations
├── model.py                # Transformer model
├── data.py                 # Bybit data loading
├── strategy.py             # Trading strategy
├── train.py                # Training script
├── requirements.txt        # Dependencies
└── examples/
    ├── 01_compare_encodings.py
    ├── 02_crypto_prediction.py
    ├── 03_stock_prediction.py
    └── 04_backtesting.py
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Compare encoding methods
python examples/01_compare_encodings.py

# Train crypto prediction model
python train.py --symbol BTCUSDT --encoding rope

# Run backtest
python examples/04_backtesting.py
```

## Best Practices

### Choosing an Encoding Method

| Use Case | Recommended Encoding | Reason |
|----------|---------------------|--------|
| Fixed-length sequences | Sinusoidal or Learned | Simple, effective |
| Variable-length sequences | RoPE or Relative | Handles different lengths |
| Long sequences (>512) | RoPE | Better extrapolation |
| Calendar-dependent patterns | Calendar + Sinusoidal | Captures market effects |
| Crypto 24/7 markets | RoPE + Session | Continuous time awareness |
| Stock markets | Calendar + Market Session | Trading hour patterns |

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `d_model` | 64-256 | Larger for complex patterns |
| `dropout` | 0.1-0.2 | Higher for small datasets |
| `max_len` | 2x training length | Allow extrapolation |
| `temperature` (sinusoidal) | 10000 | Standard value |
| `base` (RoPE) | 10000 | Can tune for longer sequences |

### Common Pitfalls

1. **Fixed-length learned embeddings**: Cannot extrapolate
2. **Ignoring calendar features**: Missing market patterns
3. **Over-engineering**: Simple sinusoidal often sufficient
4. **Not scaling positions**: Normalize for long sequences

## Resources

### Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer with sinusoidal encoding
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE paper
- [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) — Shaw's relative encoding
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) — Segment-level recurrence
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) — Time series transformers

### Implementations

- [PyTorch Transformers](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) — Official PyTorch
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) — Pre-trained models
- [x-transformers](https://github.com/lucidrains/x-transformers) — Various position encodings

### Related Chapters

- [Chapter 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — Multi-horizon forecasting
- [Chapter 43: Stockformer Multivariate](../43_stockformer_multivariate) — Cross-asset attention
- [Chapter 47: Cross-Attention Multi-Asset](../47_cross_attention_multi_asset) — Cross-asset modeling
- [Chapter 51: Linformer Long Sequences](../51_linformer_long_sequences) — Efficient attention

---

## Difficulty Level

**Intermediate to Advanced**

Prerequisites:
- Transformer architecture fundamentals
- Self-attention mechanism
- Time series basics
- PyTorch/Rust ML libraries
