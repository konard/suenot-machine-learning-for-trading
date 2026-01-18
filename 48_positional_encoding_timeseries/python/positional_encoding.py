"""
Positional Encoding Implementations for Time Series

This module provides various positional encoding methods specifically
designed for time series and financial data processing.

Available Encodings:
- SinusoidalPositionalEncoding: Standard fixed sinusoidal encoding
- TimeSeriesSinusoidalEncoding: Multi-frequency sinusoidal for time series
- LearnedPositionalEncoding: Trainable position embeddings
- FinancialLearnedEncoding: Learned encoding with calendar features
- RelativePositionalEncoding: Shaw-style relative position encoding
- RotaryPositionalEncoding: RoPE for efficient long sequences
- RoPETimeSeriesAttention: Self-attention with RoPE
- CalendarEncoding: Calendar-aware features (day, month, etc.)
- MarketSessionEncoding: Trading session information
- MultiScaleTemporalEncoding: Multi-scale temporal features
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from 'Attention Is All You Need'.

    Uses sine and cosine functions of different frequencies to create
    unique position representations that can extrapolate to unseen lengths.

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        temperature: Base for frequency calculation (default: 10000)

    Example:
        >>> encoder = SinusoidalPositionalEncoding(d_model=64)
        >>> x = torch.randn(2, 100, 64)  # [batch, seq_len, d_model]
        >>> output = encoder(x)  # [batch, seq_len, d_model]
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
        Add positional encoding to input tensor.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Position-encoded tensor [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TimeSeriesSinusoidalEncoding(nn.Module):
    """
    Sinusoidal encoding adapted for time series with multiple frequencies.

    Captures various temporal patterns:
    - Intraday patterns (hourly cycles)
    - Daily patterns (market open/close)
    - Weekly patterns (Monday effect)
    - Monthly patterns (month-end rebalancing)

    Args:
        d_model: Embedding dimension
        frequencies: List of period lengths (in timesteps)
        dropout: Dropout probability

    Example:
        >>> # For hourly data, capture 24h, 1 week, 1 month cycles
        >>> encoder = TimeSeriesSinusoidalEncoding(
        ...     d_model=64,
        ...     frequencies=[24, 168, 720]
        ... )
    """

    def __init__(
        self,
        d_model: int,
        frequencies: list = [24, 24*7, 24*30, 24*365],
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.frequencies = frequencies
        self.dropout = nn.Dropout(p=dropout)

        # Allocate dimensions to each frequency
        dims_per_freq = d_model // (len(frequencies) * 2)
        self.dims_per_freq = dims_per_freq

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add multi-frequency positional encoding.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            timestamps: Optional absolute timestamps [batch, seq_len]

        Returns:
            Position-encoded tensor [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        device = x.device

        if timestamps is None:
            positions = torch.arange(seq_len, device=device).float()
        else:
            positions = timestamps.float()
            if positions.dim() > 1:
                positions = positions[0]  # Use first batch's timestamps

        pe = torch.zeros(seq_len, d_model, device=device)

        dim_idx = 0
        for freq in self.frequencies:
            pos_normalized = positions / freq

            for i in range(self.dims_per_freq):
                harmonic = 2 ** i
                if dim_idx + 1 < d_model:
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
            if remaining_dims > 1:
                pe[:, dim_idx::2] = torch.sin(position * div_term)
                pe[:, dim_idx+1::2] = torch.cos(position * div_term)

        x = x + pe.unsqueeze(0)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding using an embedding table.

    Position embeddings are learned during training, allowing the model
    to discover dataset-specific position patterns.

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        init_std: Standard deviation for initialization

    Note:
        Cannot extrapolate beyond max_len during inference.

    Example:
        >>> encoder = LearnedPositionalEncoding(d_model=64, max_len=512)
        >>> x = torch.randn(2, 100, 64)
        >>> output = encoder(x)
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
        self.max_len = max_len

        # Initialize with small values
        nn.init.normal_(self.embedding.weight, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Position-encoded tensor [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.embedding(positions)  # [seq_len, d_model]

        x = x + pos_emb.unsqueeze(0)
        return self.dropout(x)


class FinancialLearnedEncoding(nn.Module):
    """
    Learned encoding with financial market priors.

    Combines position embeddings with temporal features relevant
    to financial markets:
    - Recency weighting (recent data more important)
    - Weekday embeddings
    - Hour embeddings

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
        n_weekdays: Number of weekdays (default: 7)
        n_hours: Number of hours (default: 24)
        dropout: Dropout probability

    Example:
        >>> encoder = FinancialLearnedEncoding(d_model=64)
        >>> x = torch.randn(2, 100, 64)
        >>> weekdays = torch.randint(0, 7, (2, 100))
        >>> hours = torch.randint(0, 24, (2, 100))
        >>> output = encoder(x, weekdays, hours)
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
        self.max_len = max_len

    def forward(
        self,
        x: torch.Tensor,
        weekdays: Optional[torch.Tensor] = None,
        hours: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add financial-aware positional encoding.

        Args:
            x: Input [batch, seq_len, d_model]
            weekdays: Weekday indices [batch, seq_len] (0=Monday, 6=Sunday)
            hours: Hour indices [batch, seq_len] (0-23)

        Returns:
            Position-encoded tensor [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        device = x.device

        # Position embedding
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.pos_embedding(positions)  # [seq_len, d_model//2]

        # Apply recency decay
        decay_weights = torch.exp(
            -self.decay * (seq_len - 1 - positions.float()) / seq_len
        )
        pos_emb = pos_emb * decay_weights.unsqueeze(-1)

        # Temporal embeddings (if provided)
        if weekdays is not None and hours is not None:
            weekday_emb = self.weekday_embedding(weekdays)
            hour_emb = self.hour_embedding(hours)

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


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding from Shaw et al.

    Instead of adding position to input, modifies attention scores
    based on relative positions between tokens.

    Args:
        d_model: Embedding dimension
        n_heads: Number of attention heads
        max_relative_position: Maximum relative distance to encode
        dropout: Dropout probability

    Example:
        >>> encoder = RelativePositionalEncoding(d_model=64, n_heads=4)
        >>> q = torch.randn(2, 4, 100, 16)  # [batch, heads, seq, head_dim]
        >>> k = torch.randn(2, 4, 100, 16)
        >>> v = torch.randn(2, 4, 100, 16)
        >>> output = encoder(q, k, v)
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
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply relative positional attention.

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
        qk_scores = torch.matmul(q, k.transpose(-2, -1))

        # Compute relative position indices
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions.clamp(
            -self.max_relative_position,
            self.max_relative_position
        )
        relative_positions = relative_positions + self.max_relative_position

        # Get relative embeddings
        rel_emb = self.relative_embedding(relative_positions)

        # Compute relative attention scores: Q @ R^T
        q_for_rel = q.unsqueeze(3)
        rel_emb_expanded = rel_emb.unsqueeze(0).unsqueeze(0)

        relative_scores = torch.matmul(
            q_for_rel,
            rel_emb_expanded.transpose(-2, -1)
        ).squeeze(-2)

        # Combine scores
        scores = (qk_scores + relative_scores) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        return output


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for time series.

    Encodes position by rotating query and key vectors. The attention
    score between positions naturally depends on their relative distance.

    Benefits for time series:
    - Handles long sequences efficiently
    - Natural decay for distant positions
    - Works with variable-length sequences

    Args:
        d_model: Embedding dimension
        n_heads: Number of attention heads
        max_len: Maximum sequence length for caching
        base: Base for frequency calculation

    Example:
        >>> rope = RotaryPositionalEncoding(d_model=64, n_heads=4)
        >>> q = torch.randn(2, 4, 100, 16)
        >>> k = torch.randn(2, 4, 100, 16)
        >>> q_rot, k_rot = rope(q, k)
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
        """Precompute cos and sin values."""
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)

        # Stack sin and cos for rotation
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.

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
    Self-attention with RoPE for time series data.

    Combines multi-head self-attention with rotary position embeddings
    for efficient and effective time series modeling.

    Args:
        d_model: Embedding dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        max_len: Maximum sequence length

    Example:
        >>> attn = RoPETimeSeriesAttention(d_model=64, n_heads=4)
        >>> x = torch.randn(2, 100, 64)
        >>> output = attn(x)
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
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply self-attention with RoPE.

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
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

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


class CalendarEncoding(nn.Module):
    """
    Encode calendar features important for financial markets.

    Captures temporal patterns:
    - Day of week (Monday effect)
    - Month (January effect, month-end rebalancing)
    - Quarter (earnings seasons)
    - Hour (market hours)
    - Trading session (pre/regular/after/closed)

    Args:
        d_model: Output dimension
        dropout: Dropout probability

    Example:
        >>> encoder = CalendarEncoding(d_model=64)
        >>> dayofweek = torch.randint(0, 7, (2, 100))
        >>> month = torch.randint(0, 12, (2, 100))
        >>> quarter = torch.randint(0, 4, (2, 100))
        >>> hour = torch.randint(0, 24, (2, 100))
        >>> session = torch.randint(0, 4, (2, 100))
        >>> output = encoder(dayofweek, month, quarter, hour, session)
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
        self.dayofweek_emb = nn.Embedding(7, self.d_dayofweek)
        self.month_emb = nn.Embedding(12, self.d_month)
        self.quarter_emb = nn.Embedding(4, self.d_quarter)
        self.hour_emb = nn.Embedding(24, self.d_hour)
        self.session_emb = nn.Embedding(4, self.d_session)

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
        Generate calendar encoding.

        Args:
            dayofweek: Day of week [batch, seq_len] (0=Monday)
            month: Month [batch, seq_len] (0=January)
            quarter: Quarter [batch, seq_len] (0=Q1)
            hour: Hour [batch, seq_len] (0-23)
            session: Session [batch, seq_len] (0=pre, 1=regular, 2=after, 3=closed)

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


class MarketSessionEncoding(nn.Module):
    """
    Encode market session information.

    For crypto (24/7): Time-of-day patterns with Asian/European/American sessions
    For stocks: Pre-market, regular, after-hours, closed

    Args:
        d_model: Output dimension
        market_type: 'crypto' or 'stock'
        dropout: Dropout probability

    Example:
        >>> encoder = MarketSessionEncoding(d_model=64, market_type='crypto')
        >>> hour = torch.randint(0, 24, (2, 100))
        >>> output = encoder(hour)
    """

    def __init__(
        self,
        d_model: int,
        market_type: str = 'crypto',
        dropout: float = 0.1
    ):
        super().__init__()
        self.market_type = market_type

        if market_type == 'crypto':
            # 24-hour cycle with Asian/European/American sessions
            self.session_emb = nn.Embedding(3, d_model // 3)
            self.hour_emb = nn.Embedding(24, d_model // 3)
            self.proj = nn.Linear(2 * (d_model // 3), d_model)
        else:
            # Stock market sessions
            self.session_emb = nn.Embedding(4, d_model // 2)
            self.time_in_session = nn.Embedding(100, d_model // 2)
            self.proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _get_crypto_session(self, hour: torch.Tensor) -> torch.Tensor:
        """Map hour to crypto trading session."""
        session = torch.zeros_like(hour)
        session[(hour >= 0) & (hour < 8)] = 0   # Asia
        session[(hour >= 8) & (hour < 16)] = 1  # Europe
        session[(hour >= 16) & (hour < 24)] = 2 # US
        return session

    def forward(
        self,
        hour: torch.Tensor,
        session: Optional[torch.Tensor] = None,
        time_in_session: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate market session encoding.

        Args:
            hour: Hour of day [batch, seq_len]
            session: Market session (for stocks) [batch, seq_len]
            time_in_session: Minutes into session [batch, seq_len]

        Returns:
            Market session encoding [batch, seq_len, d_model]
        """
        if self.market_type == 'crypto':
            crypto_session = self._get_crypto_session(hour)
            ses_emb = self.session_emb(crypto_session)
            hr_emb = self.hour_emb(hour)
            combined = torch.cat([ses_emb, hr_emb], dim=-1)
        else:
            if session is None or time_in_session is None:
                raise ValueError("Stock market requires session and time_in_session")
            ses_emb = self.session_emb(session)
            time_emb = self.time_in_session(time_in_session.clamp(0, 99))
            combined = torch.cat([ses_emb, time_emb], dim=-1)

        out = self.proj(combined)
        return self.dropout(out)


class MultiScaleTemporalEncoding(nn.Module):
    """
    Encode time at multiple scales for comprehensive temporal representation.

    Scales:
    - Micro: Within trading session (minutes)
    - Intraday: Hours within day
    - Daily: Day patterns
    - Weekly: Week patterns
    - Monthly: Month patterns

    Args:
        d_model: Output dimension
        time_scales: List of time scales to encode
        dropout: Dropout probability

    Example:
        >>> encoder = MultiScaleTemporalEncoding(d_model=64)
        >>> timestamps = {
        ...     'minute': torch.randint(0, 60, (2, 100)),
        ...     'hour': torch.randint(0, 24, (2, 100)),
        ...     'day': torch.randint(0, 31, (2, 100)),
        ...     'week': torch.randint(0, 7, (2, 100)),
        ...     'month': torch.randint(0, 12, (2, 100))
        ... }
        >>> output = encoder(timestamps)
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
            if scale in scale_sizes:
                self.scale_embeddings[scale] = nn.Embedding(
                    scale_sizes[scale],
                    d_per_scale
                )

        # Final projection
        self.proj = nn.Linear(d_per_scale * len(time_scales), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, timestamps: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate multi-scale temporal encoding.

        Args:
            timestamps: Dict with keys for each scale
                       e.g., {'minute': [batch, seq], 'hour': [batch, seq], ...}

        Returns:
            Multi-scale temporal encoding [batch, seq_len, d_model]
        """
        embeddings = []

        # Get batch and seq_len from first available timestamp
        sample_tensor = next(iter(timestamps.values()))
        batch, seq_len = sample_tensor.shape
        device = sample_tensor.device

        for scale in self.time_scales:
            if scale in timestamps and scale in self.scale_embeddings:
                emb = self.scale_embeddings[scale](timestamps[scale])
                embeddings.append(emb)
            else:
                embeddings.append(
                    torch.zeros(batch, seq_len, self.d_per_scale, device=device)
                )

        combined = torch.cat(embeddings, dim=-1)
        out = self.proj(combined)

        return self.dropout(out)


if __name__ == "__main__":
    # Test all encodings
    print("Testing Positional Encoding Implementations...")
    print("=" * 60)

    batch_size = 2
    seq_len = 100
    d_model = 64
    n_heads = 4

    x = torch.randn(batch_size, seq_len, d_model)

    # Test 1: Sinusoidal
    print("\n1. SinusoidalPositionalEncoding")
    encoder = SinusoidalPositionalEncoding(d_model)
    output = encoder(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")

    # Test 2: Time Series Sinusoidal
    print("\n2. TimeSeriesSinusoidalEncoding")
    encoder = TimeSeriesSinusoidalEncoding(d_model)
    output = encoder(x)
    print(f"   Output shape: {output.shape}")

    # Test 3: Learned
    print("\n3. LearnedPositionalEncoding")
    encoder = LearnedPositionalEncoding(d_model)
    output = encoder(x)
    print(f"   Output shape: {output.shape}")

    # Test 4: Financial Learned
    print("\n4. FinancialLearnedEncoding")
    encoder = FinancialLearnedEncoding(d_model)
    weekdays = torch.randint(0, 7, (batch_size, seq_len))
    hours = torch.randint(0, 24, (batch_size, seq_len))
    output = encoder(x, weekdays, hours)
    print(f"   Output shape: {output.shape}")

    # Test 5: Relative
    print("\n5. RelativePositionalEncoding")
    encoder = RelativePositionalEncoding(d_model, n_heads)
    head_dim = d_model // n_heads
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim)
    output = encoder(q, k, v)
    print(f"   Output shape: {output.shape}")

    # Test 6: RoPE
    print("\n6. RotaryPositionalEncoding")
    rope = RotaryPositionalEncoding(d_model, n_heads)
    q_rot, k_rot = rope(q, k)
    print(f"   Q rotated shape: {q_rot.shape}")
    print(f"   K rotated shape: {k_rot.shape}")

    # Test 7: RoPE Attention
    print("\n7. RoPETimeSeriesAttention")
    attn = RoPETimeSeriesAttention(d_model, n_heads)
    output = attn(x)
    print(f"   Output shape: {output.shape}")

    # Test 8: Calendar
    print("\n8. CalendarEncoding")
    encoder = CalendarEncoding(d_model)
    dayofweek = torch.randint(0, 7, (batch_size, seq_len))
    month = torch.randint(0, 12, (batch_size, seq_len))
    quarter = torch.randint(0, 4, (batch_size, seq_len))
    hour = torch.randint(0, 24, (batch_size, seq_len))
    session = torch.randint(0, 4, (batch_size, seq_len))
    output = encoder(dayofweek, month, quarter, hour, session)
    print(f"   Output shape: {output.shape}")

    # Test 9: Market Session
    print("\n9. MarketSessionEncoding")
    encoder = MarketSessionEncoding(d_model, market_type='crypto')
    output = encoder(hour)
    print(f"   Output shape: {output.shape}")

    # Test 10: Multi-Scale Temporal
    print("\n10. MultiScaleTemporalEncoding")
    encoder = MultiScaleTemporalEncoding(d_model)
    timestamps = {
        'minute': torch.randint(0, 60, (batch_size, seq_len)),
        'hour': torch.randint(0, 24, (batch_size, seq_len)),
        'day': torch.randint(0, 31, (batch_size, seq_len)),
        'week': torch.randint(0, 7, (batch_size, seq_len)),
        'month': torch.randint(0, 12, (batch_size, seq_len))
    }
    output = encoder(timestamps)
    print(f"   Output shape: {output.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
