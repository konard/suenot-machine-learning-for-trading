# Chapter 124: Attention Rollout Trading

## Introduction

Attention Rollout is a powerful interpretability technique for transformer models that tracks how attention flows through multiple layers. In trading applications, this method helps explain why a model makes specific predictions, enabling traders to understand which historical patterns, time periods, or features most influence buy/sell decisions.

This chapter covers the theoretical foundations of attention rollout, its mathematical formulation, and practical implementations in both Python and Rust for financial market analysis.

## Table of Contents

1. [Theoretical Foundations](#theoretical-foundations)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Attention Rollout Algorithm](#attention-rollout-algorithm)
4. [Trading Applications](#trading-applications)
5. [Python Implementation](#python-implementation)
6. [Rust Implementation](#rust-implementation)
7. [Practical Examples](#practical-examples)
8. [Backtesting Framework](#backtesting-framework)
9. [Advanced Topics](#advanced-topics)
10. [References](#references)

---

## Theoretical Foundations

### What is Attention Rollout?

Attention Rollout, introduced by Abnar & Zuidema (2020), is a method for quantifying information flow in transformer architectures. Unlike raw attention weights that only show layer-specific patterns, attention rollout recursively combines attention matrices across all layers to reveal the cumulative influence of input tokens on the final output.

### Why Attention Rollout for Trading?

Traditional black-box models pose significant risks in financial applications:

1. **Regulatory Compliance**: Financial institutions must explain model decisions
2. **Risk Management**: Understanding why a model predicts a crash is crucial
3. **Strategy Validation**: Confirming models use sensible market indicators
4. **Debugging**: Identifying when models rely on spurious correlations

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER TRADING MODEL                     │
├─────────────────────────────────────────────────────────────────┤
│  Input: [Price_t-5, Price_t-4, Price_t-3, Price_t-2, Price_t-1] │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Layer 1 Attention: Which past prices matter?           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Layer 2 Attention: Refined pattern recognition         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Layer N Attention: Final decision weighting            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  Output: BUY/SELL Signal + Attention Rollout Explanation        │
└─────────────────────────────────────────────────────────────────┘
```

### Comparison with Other Interpretability Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Attention Rollout** | Captures multi-layer flow | Assumes linear combination | Sequence models |
| **Attention Flow** | Graph-theoretic foundation | Computationally expensive | Deep analysis |
| **Gradient-based** | Model-agnostic | Can be noisy | Any differentiable model |
| **SHAP** | Theoretically grounded | Slow for transformers | Feature importance |
| **LIME** | Local explanations | Approximation errors | Instance-level |

---

## Mathematical Formulation

### Single-Head Attention

For a single attention head, the attention weights are computed as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q: Query matrix (n × d_k)
- K: Key matrix (n × d_k)
- V: Value matrix (n × d_v)
- d_k: Dimension of keys

The attention weight matrix A is:

```
A = softmax(QK^T / √d_k)
```

### Multi-Head Attention Aggregation

For multi-head attention with h heads, we aggregate attention weights:

```
A_combined = (1/h) Σ_{i=1}^{h} A_i
```

Or using attention head importance weighting:

```
A_combined = Σ_{i=1}^{h} w_i · A_i,  where Σw_i = 1
```

### Attention Rollout Formula

The key insight of attention rollout is incorporating residual connections. At each layer, the effective attention becomes:

```
Ã_l = 0.5 · I + 0.5 · A_l
```

Where I is the identity matrix (representing the residual connection).

The rollout matrix R after L layers is computed recursively:

```
R_1 = Ã_1
R_l = Ã_l · R_{l-1}  for l = 2, ..., L
```

The final rollout matrix R_L shows the cumulative attention from each input position to the output.

### Normalized Rollout

To ensure proper probability distribution:

```
R̂_L = R_L / Σ_j R_L[i,j]
```

Each row sums to 1, representing attention distribution.

---

## Attention Rollout Algorithm

### Algorithm Pseudocode

```
Algorithm: Attention Rollout
Input: Attention matrices A_1, A_2, ..., A_L from L layers
Output: Rollout matrix R showing input-to-output attention flow

1. Initialize: R ← I (identity matrix)
2. For l = 1 to L:
   a. If multi-head: A_l ← mean(A_l, axis=heads)
   b. Add residual: Ã_l ← 0.5 · I + 0.5 · A_l
   c. Accumulate: R ← Ã_l · R
3. Normalize rows: R ← R / row_sum(R)
4. Return R
```

### Computational Complexity

- Time: O(L · n²) for L layers and sequence length n
- Space: O(n²) for storing attention matrices

For trading with typical sequence lengths (50-200 time steps), this is highly efficient.

---

## Trading Applications

### 1. Feature Attribution for Price Prediction

Understanding which historical prices influence predictions:

```
Input Sequence: [Day-10, Day-9, Day-8, ..., Day-1, Day-0]
                   ↓       ↓      ↓           ↓      ↓
Rollout Weights: [0.05,  0.08,  0.15,  ...,  0.25,  0.20]

Interpretation: Days -1 and -2 have highest influence on prediction
```

### 2. Multi-Asset Attention Analysis

For portfolio models processing multiple assets:

```
┌─────────────────────────────────────────────────┐
│  Assets: [AAPL, GOOGL, MSFT, AMZN, TSLA]       │
│                                                 │
│  Attention Rollout for AAPL prediction:        │
│  AAPL  ████████████████████  0.35              │
│  GOOGL ██████████            0.20              │
│  MSFT  ████████              0.18              │
│  AMZN  ██████                0.15              │
│  TSLA  ██████                0.12              │
│                                                 │
│  → AAPL prediction heavily influenced by       │
│    itself and tech sector peers                │
└─────────────────────────────────────────────────┘
```

### 3. Temporal Pattern Discovery

Identifying important time windows:

```
Market Regime Detection via Attention Rollout:

Bull Market: Attention concentrated on recent momentum
  [0.05, 0.08, 0.12, 0.20, 0.25, 0.30] → Recent bias

Bear Market: Attention spread across history
  [0.15, 0.18, 0.17, 0.16, 0.18, 0.16] → Uniform attention

Volatility Spike: Attention on specific events
  [0.05, 0.40, 0.05, 0.05, 0.40, 0.05] → Event-focused
```

---

## Python Implementation

### Requirements

```python
# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
torch>=1.9.0
transformers>=4.10.0
yfinance>=0.1.70
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
requests>=2.26.0
```

### Core Attention Rollout Module

```python
"""
attention_rollout.py - Core implementation of Attention Rollout for trading
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn


class AttentionRollout:
    """
    Compute attention rollout for transformer models.

    Attention rollout tracks how attention flows through transformer layers,
    providing interpretable explanations for model predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layer_name: str = "attn",
        head_fusion: str = "mean",
        discard_ratio: float = 0.0
    ):
        """
        Initialize AttentionRollout.

        Args:
            model: PyTorch transformer model
            attention_layer_name: Name pattern for attention layers
            head_fusion: Method to combine heads ('mean', 'max', 'min')
            discard_ratio: Fraction of lowest attention weights to discard
        """
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions: List[torch.Tensor] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks to capture attention weights."""
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name:
                module.register_forward_hook(self._attention_hook)

    def _attention_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: Tuple
    ) -> None:
        """Hook function to capture attention weights."""
        # Handle different output formats
        if isinstance(output, tuple):
            attention = output[1] if len(output) > 1 else output[0]
        else:
            attention = output
        self.attentions.append(attention.detach())

    def _fuse_heads(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Fuse multiple attention heads into single attention matrix.

        Args:
            attention: Tensor of shape (batch, heads, seq_len, seq_len)

        Returns:
            Fused attention of shape (batch, seq_len, seq_len)
        """
        if self.head_fusion == "mean":
            return attention.mean(dim=1)
        elif self.head_fusion == "max":
            return attention.max(dim=1)[0]
        elif self.head_fusion == "min":
            return attention.min(dim=1)[0]
        else:
            raise ValueError(f"Unknown head fusion method: {self.head_fusion}")

    def _discard_low_attention(
        self,
        attention: torch.Tensor
    ) -> torch.Tensor:
        """Discard lowest attention weights based on discard_ratio."""
        if self.discard_ratio <= 0:
            return attention

        flat = attention.flatten()
        threshold = torch.quantile(flat, self.discard_ratio)
        attention = torch.where(
            attention > threshold,
            attention,
            torch.zeros_like(attention)
        )
        # Re-normalize
        attention = attention / attention.sum(dim=-1, keepdim=True)
        return attention

    def compute_rollout(
        self,
        input_tensor: torch.Tensor,
        start_layer: int = 0
    ) -> np.ndarray:
        """
        Compute attention rollout for given input.

        Args:
            input_tensor: Input tensor for the model
            start_layer: Layer to start rollout computation

        Returns:
            Rollout matrix of shape (seq_len, seq_len)
        """
        self.attentions = []

        # Forward pass to collect attention weights
        with torch.no_grad():
            _ = self.model(input_tensor)

        if not self.attentions:
            raise RuntimeError("No attention weights captured. Check layer name.")

        # Process attention matrices
        batch_size = self.attentions[0].shape[0]
        seq_len = self.attentions[0].shape[-1]

        # Initialize rollout with identity matrix
        rollout = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1)
        rollout = rollout.to(self.attentions[0].device)

        for i, attention in enumerate(self.attentions[start_layer:]):
            # Fuse attention heads
            attention = self._fuse_heads(attention)

            # Discard low attention weights
            attention = self._discard_low_attention(attention)

            # Add residual connection (identity matrix)
            # This accounts for skip connections in transformers
            identity = torch.eye(seq_len).unsqueeze(0).to(attention.device)
            attention = 0.5 * attention + 0.5 * identity

            # Accumulate rollout
            rollout = torch.bmm(attention, rollout)

        # Normalize rows
        rollout = rollout / rollout.sum(dim=-1, keepdim=True)

        return rollout.cpu().numpy()

    def get_input_attribution(
        self,
        input_tensor: torch.Tensor,
        output_position: int = -1
    ) -> np.ndarray:
        """
        Get attribution scores for input positions.

        Args:
            input_tensor: Input tensor
            output_position: Position to get attribution for (-1 for last)

        Returns:
            Attribution scores for each input position
        """
        rollout = self.compute_rollout(input_tensor)
        # Get attribution for specified output position
        attribution = rollout[0, output_position, :]
        return attribution


class TradingAttentionRollout(AttentionRollout):
    """
    Specialized attention rollout for trading applications.

    Extends base AttentionRollout with trading-specific features:
    - Feature importance analysis
    - Temporal pattern detection
    - Multi-asset correlation analysis
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize TradingAttentionRollout.

        Args:
            model: Trading transformer model
            feature_names: Names of input features for interpretation
            **kwargs: Arguments passed to AttentionRollout
        """
        super().__init__(model, **kwargs)
        self.feature_names = feature_names

    def analyze_temporal_importance(
        self,
        input_tensor: torch.Tensor,
        timestamps: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Analyze which time periods are most important for prediction.

        Args:
            input_tensor: Input tensor with shape (batch, seq_len, features)
            timestamps: Optional timestamp labels

        Returns:
            Dictionary mapping timestamps to importance scores
        """
        attribution = self.get_input_attribution(input_tensor)

        if timestamps is None:
            timestamps = [f"t-{i}" for i in range(len(attribution)-1, -1, -1)]

        return dict(zip(timestamps, attribution))

    def detect_attention_regime(
        self,
        input_tensor: torch.Tensor,
        threshold_recent: float = 0.6
    ) -> str:
        """
        Detect market regime based on attention pattern.

        Args:
            input_tensor: Input tensor
            threshold_recent: Threshold for recent attention concentration

        Returns:
            Detected regime: 'momentum', 'mean_reversion', or 'mixed'
        """
        attribution = self.get_input_attribution(input_tensor)
        seq_len = len(attribution)

        # Calculate attention on recent vs historical periods
        recent_window = seq_len // 4
        recent_attention = attribution[-recent_window:].sum()

        if recent_attention > threshold_recent:
            return "momentum"  # Focus on recent prices
        elif recent_attention < 1 - threshold_recent:
            return "mean_reversion"  # Focus on historical
        else:
            return "mixed"

    def compute_feature_importance(
        self,
        input_tensor: torch.Tensor,
        n_features: int
    ) -> Dict[str, float]:
        """
        Compute importance of each feature type across time.

        Args:
            input_tensor: Input tensor (batch, seq_len, features)
            n_features: Number of features per time step

        Returns:
            Feature importance scores
        """
        rollout = self.compute_rollout(input_tensor)
        seq_len = rollout.shape[-1]
        time_steps = seq_len // n_features

        # Reshape and aggregate by feature
        importance = {}
        for f in range(n_features):
            feature_positions = list(range(f, seq_len, n_features))
            feature_attention = rollout[0, -1, feature_positions].sum()

            feature_name = (
                self.feature_names[f]
                if self.feature_names and f < len(self.feature_names)
                else f"feature_{f}"
            )
            importance[feature_name] = float(feature_attention)

        return importance
```

### Trading Transformer Model

```python
"""
model.py - Transformer model for trading with attention extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with attention weight storage."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        self.attention_weights = attention.detach()
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.out_linear(context)
        return output, self.attention_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with accessible attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual
        attn_out, attn_weights = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x, attn_weights


class TradingTransformer(nn.Module):
    """
    Transformer model for trading prediction with attention extraction.

    This model processes sequential market data and provides
    buy/sell predictions along with interpretable attention weights.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        n_classes: int = 3  # buy, hold, sell
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

        self.attention_maps: list = []

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass with attention extraction.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            logits: Classification logits
            attention_maps: List of attention weight tensors
        """
        self.attention_maps = []

        # Input projection and positional encoding
        x = self.input_projection(x)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        # Process through transformer layers
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            self.attention_maps.append(attn_weights)

        # Use last position for classification
        x = x[:, -1, :]
        logits = self.classifier(x)

        return logits, self.attention_maps

    def get_attention_maps(self) -> list:
        """Return stored attention maps from last forward pass."""
        return self.attention_maps
```

### Data Loader Module

```python
"""
data_loader.py - Data loading utilities for stock and crypto markets
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict
import requests
from datetime import datetime, timedelta


def load_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Load stock data using yfinance.

    Args:
        symbol: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval (1d, 1h, etc.)

    Returns:
        DataFrame with OHLCV data
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    df = df.reset_index()
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    return df


def load_bybit_data(
    symbol: str = "BTCUSDT",
    interval: str = "D",
    limit: int = 200
) -> pd.DataFrame:
    """
    Load cryptocurrency data from Bybit exchange.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Candle interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        limit: Number of candles to fetch (max 200)

    Returns:
        DataFrame with OHLCV data
    """
    url = "https://api.bybit.com/v5/market/kline"

    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["retCode"] != 0:
        raise ValueError(f"Bybit API error: {data['retMsg']}")

    candles = data["result"]["list"]

    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def prepare_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    lookback: int = 20,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features for transformer model.

    Args:
        df: DataFrame with OHLCV data
        feature_columns: Columns to use as features
        lookback: Number of time steps to look back
        normalize: Whether to normalize features

    Returns:
        X: Feature sequences (n_samples, lookback, n_features)
        y: Target labels (n_samples,)
    """
    if feature_columns is None:
        feature_columns = ["open", "high", "low", "close", "volume"]

    # Add technical indicators
    df = add_technical_indicators(df)

    # Forward returns for labels
    df["returns"] = df["close"].pct_change().shift(-1)

    # Create labels: 0=sell, 1=hold, 2=buy
    df["label"] = 1  # hold
    df.loc[df["returns"] > 0.01, "label"] = 2  # buy
    df.loc[df["returns"] < -0.01, "label"] = 0  # sell

    # Normalize features
    if normalize:
        for col in feature_columns:
            if col in df.columns:
                df[col] = (df[col] - df[col].rolling(lookback).mean()) / (
                    df[col].rolling(lookback).std() + 1e-8
                )

    df = df.dropna()

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(df[feature_columns].iloc[i-lookback:i].values)
        y.append(df["label"].iloc[i])

    return np.array(X), np.array(y)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to DataFrame."""

    # Moving averages
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std

    # Volatility
    df["volatility"] = df["close"].pct_change().rolling(20).std()

    return df


class TradingDataset:
    """Dataset class for trading data."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_samples = len(X)

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.random.permutation(self.n_samples)
        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]
```

### Backtesting Module

```python
"""
backtest.py - Backtesting framework with attention analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch


@dataclass
class BacktestResult:
    """Results from backtesting a trading strategy."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    returns: np.ndarray
    equity_curve: np.ndarray
    attention_analysis: Optional[Dict] = None


class AttentionBacktester:
    """
    Backtesting framework that incorporates attention rollout analysis.

    This backtester not only evaluates trading performance but also
    analyzes attention patterns to understand model behavior.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        attention_rollout,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001
    ):
        """
        Initialize backtester.

        Args:
            model: Trained trading model
            attention_rollout: AttentionRollout instance
            initial_capital: Starting capital
            transaction_cost: Cost per transaction (fraction)
        """
        self.model = model
        self.attention_rollout = attention_rollout
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def run_backtest(
        self,
        X: np.ndarray,
        prices: np.ndarray,
        threshold: float = 0.6
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            X: Feature sequences (n_samples, seq_len, n_features)
            prices: Close prices aligned with X
            threshold: Confidence threshold for trading

        Returns:
            BacktestResult with performance metrics
        """
        self.model.eval()

        n_samples = len(X)
        positions = np.zeros(n_samples)
        returns = np.zeros(n_samples)
        attention_patterns = []

        capital = self.initial_capital
        equity_curve = [capital]
        current_position = 0
        n_trades = 0
        wins = 0

        for i in range(n_samples - 1):
            # Get model prediction
            x_tensor = torch.FloatTensor(X[i:i+1])

            with torch.no_grad():
                logits, _ = self.model(x_tensor)
                probs = torch.softmax(logits, dim=-1).numpy()[0]

            # Get attention rollout
            attribution = self.attention_rollout.get_input_attribution(x_tensor)
            attention_patterns.append(attribution)

            # Determine position
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]

            if confidence > threshold:
                if pred_class == 2:  # Buy signal
                    new_position = 1
                elif pred_class == 0:  # Sell signal
                    new_position = -1
                else:  # Hold
                    new_position = current_position
            else:
                new_position = 0  # No position if low confidence

            # Calculate returns
            price_return = (prices[i+1] - prices[i]) / prices[i]

            # Apply transaction costs on position changes
            if new_position != current_position:
                n_trades += 1
                trade_cost = self.transaction_cost * abs(new_position - current_position)
            else:
                trade_cost = 0

            position_return = current_position * price_return - trade_cost
            returns[i] = position_return

            if position_return > 0:
                wins += 1

            capital *= (1 + position_return)
            equity_curve.append(capital)

            positions[i] = current_position
            current_position = new_position

        # Calculate metrics
        equity_curve = np.array(equity_curve)

        result = BacktestResult(
            total_return=(capital - self.initial_capital) / self.initial_capital,
            sharpe_ratio=self._calculate_sharpe(returns),
            sortino_ratio=self._calculate_sortino(returns),
            max_drawdown=self._calculate_max_drawdown(equity_curve),
            win_rate=wins / max(n_trades, 1),
            n_trades=n_trades,
            returns=returns,
            equity_curve=equity_curve,
            attention_analysis=self._analyze_attention_patterns(
                attention_patterns, returns
            )
        )

        return result

    def _calculate_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_sortino(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate annualized Sortino ratio."""
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / downside.std()

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())

    def _analyze_attention_patterns(
        self,
        attention_patterns: List[np.ndarray],
        returns: np.ndarray
    ) -> Dict:
        """
        Analyze attention patterns relative to trading performance.

        Returns insights about which attention patterns correlate
        with profitable vs unprofitable trades.
        """
        attention_matrix = np.array(attention_patterns)

        # Separate winning and losing trades
        winning_mask = returns > 0
        losing_mask = returns < 0

        # Average attention patterns
        avg_winning_attention = attention_matrix[winning_mask[:-1]].mean(axis=0)
        avg_losing_attention = attention_matrix[losing_mask[:-1]].mean(axis=0)

        # Attention concentration (entropy)
        def entropy(p):
            p = p + 1e-10
            return -np.sum(p * np.log(p))

        winning_concentration = np.mean([
            entropy(att) for att in attention_matrix[winning_mask[:-1]]
        ])
        losing_concentration = np.mean([
            entropy(att) for att in attention_matrix[losing_mask[:-1]]
        ])

        return {
            "avg_winning_attention": avg_winning_attention.tolist(),
            "avg_losing_attention": avg_losing_attention.tolist(),
            "winning_attention_entropy": winning_concentration,
            "losing_attention_entropy": losing_concentration,
            "recent_bias_winning": float(avg_winning_attention[-5:].sum()),
            "recent_bias_losing": float(avg_losing_attention[-5:].sum())
        }


def print_backtest_report(result: BacktestResult) -> None:
    """Print formatted backtest report."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Return:     {result.total_return*100:>10.2f}%")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:    {result.sortino_ratio:>10.2f}")
    print(f"Max Drawdown:     {result.max_drawdown*100:>10.2f}%")
    print(f"Win Rate:         {result.win_rate*100:>10.2f}%")
    print(f"Number of Trades: {result.n_trades:>10d}")
    print("="*60)

    if result.attention_analysis:
        print("\nATTENTION ANALYSIS")
        print("-"*60)
        print(f"Winning trades attention entropy: "
              f"{result.attention_analysis['winning_attention_entropy']:.4f}")
        print(f"Losing trades attention entropy:  "
              f"{result.attention_analysis['losing_attention_entropy']:.4f}")
        print(f"Recent bias (winning): "
              f"{result.attention_analysis['recent_bias_winning']:.4f}")
        print(f"Recent bias (losing):  "
              f"{result.attention_analysis['recent_bias_losing']:.4f}")
    print()
```

### Visualization Module

```python
"""
visualization.py - Visualization tools for attention rollout
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


def plot_attention_rollout(
    attention: np.ndarray,
    timestamps: Optional[List[str]] = None,
    title: str = "Attention Rollout",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot attention rollout heatmap.

    Args:
        attention: Attention matrix (seq_len, seq_len)
        timestamps: Optional timestamp labels
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    if timestamps is None:
        timestamps = [f"t-{i}" for i in range(attention.shape[0]-1, -1, -1)]

    sns.heatmap(
        attention,
        xticklabels=timestamps,
        yticklabels=timestamps,
        cmap="YlOrRd",
        annot=False,
        fmt=".2f"
    )

    plt.title(title)
    plt.xlabel("Input Position")
    plt.ylabel("Output Position")
    plt.tight_layout()
    plt.show()


def plot_input_attribution(
    attribution: np.ndarray,
    timestamps: Optional[List[str]] = None,
    title: str = "Input Attribution Scores",
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """
    Plot bar chart of input attribution scores.

    Args:
        attribution: Attribution scores for each input position
        timestamps: Optional timestamp labels
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    if timestamps is None:
        timestamps = [f"t-{i}" for i in range(len(attribution)-1, -1, -1)]

    colors = plt.cm.YlOrRd(attribution / attribution.max())

    plt.bar(range(len(attribution)), attribution, color=colors)
    plt.xticks(range(len(attribution)), timestamps, rotation=45, ha='right')
    plt.xlabel("Time Step")
    plt.ylabel("Attribution Score")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_attention_comparison(
    winning_attention: np.ndarray,
    losing_attention: np.ndarray,
    timestamps: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Compare attention patterns between winning and losing trades.

    Args:
        winning_attention: Average attention for winning trades
        losing_attention: Average attention for losing trades
        timestamps: Optional timestamp labels
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    if timestamps is None:
        n = len(winning_attention)
        timestamps = [f"t-{i}" for i in range(n-1, -1, -1)]

    x = np.arange(len(timestamps))
    width = 0.35

    plt.bar(x - width/2, winning_attention, width, label='Winning Trades',
            color='green', alpha=0.7)
    plt.bar(x + width/2, losing_attention, width, label='Losing Trades',
            color='red', alpha=0.7)

    plt.xlabel('Time Step')
    plt.ylabel('Average Attention')
    plt.title('Attention Pattern Comparison: Winning vs Losing Trades')
    plt.xticks(x, timestamps, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_equity_curve(
    equity_curve: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Plot equity curve with optional benchmark comparison.

    Args:
        equity_curve: Strategy equity values
        benchmark: Optional benchmark equity values
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    plt.plot(equity_curve, label='Strategy', linewidth=2)

    if benchmark is not None:
        plt.plot(benchmark, label='Benchmark', linewidth=2, linestyle='--')

    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

---

## Rust Implementation

### Cargo.toml

```toml
[package]
name = "attention_rollout_trading"
version = "0.1.0"
edition = "2021"
description = "Attention Rollout for interpretable trading models"
license = "MIT"

[dependencies]
ndarray = "0.15"
ndarray-linalg = { version = "0.16", features = ["openblas-system"] }
ndarray-stats = "0.5"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["json", "blocking"] }
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
thiserror = "1.0"

[[example]]
name = "trading_example"
path = "examples/trading_example.rs"
```

### Core Library (src/lib.rs)

```rust
//! Attention Rollout Trading Library
//!
//! This library provides attention rollout computation for interpretable
//! trading models implemented in Rust for high performance.

use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during attention rollout computation
#[derive(Error, Debug)]
pub enum AttentionRolloutError {
    #[error("Empty attention list provided")]
    EmptyAttentionList,

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid layer index: {0}")]
    InvalidLayerIndex(usize),

    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Method for fusing multiple attention heads
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeadFusion {
    Mean,
    Max,
    Min,
}

/// Configuration for attention rollout computation
#[derive(Debug, Clone)]
pub struct AttentionRolloutConfig {
    pub head_fusion: HeadFusion,
    pub discard_ratio: f64,
    pub add_residual: bool,
    pub residual_weight: f64,
}

impl Default for AttentionRolloutConfig {
    fn default() -> Self {
        Self {
            head_fusion: HeadFusion::Mean,
            discard_ratio: 0.0,
            add_residual: true,
            residual_weight: 0.5,
        }
    }
}

/// Attention Rollout computation engine
pub struct AttentionRollout {
    config: AttentionRolloutConfig,
}

impl AttentionRollout {
    /// Create new AttentionRollout with default configuration
    pub fn new() -> Self {
        Self {
            config: AttentionRolloutConfig::default(),
        }
    }

    /// Create AttentionRollout with custom configuration
    pub fn with_config(config: AttentionRolloutConfig) -> Self {
        Self { config }
    }

    /// Fuse multiple attention heads into single matrix
    fn fuse_heads(&self, attention: &Array3<f64>) -> Array2<f64> {
        match self.config.head_fusion {
            HeadFusion::Mean => attention.mean_axis(Axis(0)).unwrap(),
            HeadFusion::Max => {
                let shape = (attention.shape()[1], attention.shape()[2]);
                let mut result = Array2::zeros(shape);
                for i in 0..shape.0 {
                    for j in 0..shape.1 {
                        let mut max_val = f64::NEG_INFINITY;
                        for h in 0..attention.shape()[0] {
                            max_val = max_val.max(attention[[h, i, j]]);
                        }
                        result[[i, j]] = max_val;
                    }
                }
                result
            }
            HeadFusion::Min => {
                let shape = (attention.shape()[1], attention.shape()[2]);
                let mut result = Array2::zeros(shape);
                for i in 0..shape.0 {
                    for j in 0..shape.1 {
                        let mut min_val = f64::INFINITY;
                        for h in 0..attention.shape()[0] {
                            min_val = min_val.min(attention[[h, i, j]]);
                        }
                        result[[i, j]] = min_val;
                    }
                }
                result
            }
        }
    }

    /// Discard lowest attention weights
    fn discard_low_attention(&self, attention: &mut Array2<f64>) {
        if self.config.discard_ratio <= 0.0 {
            return;
        }

        let mut values: Vec<f64> = attention.iter().cloned().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_idx = (values.len() as f64 * self.config.discard_ratio) as usize;
        let threshold = values[threshold_idx.min(values.len() - 1)];

        attention.mapv_inplace(|x| if x > threshold { x } else { 0.0 });

        // Renormalize rows
        for mut row in attention.rows_mut() {
            let sum: f64 = row.sum();
            if sum > 0.0 {
                row.mapv_inplace(|x| x / sum);
            }
        }
    }

    /// Compute attention rollout from list of attention matrices
    ///
    /// # Arguments
    /// * `attentions` - List of attention matrices (n_heads, seq_len, seq_len)
    /// * `start_layer` - Layer to start rollout computation
    ///
    /// # Returns
    /// Rollout matrix of shape (seq_len, seq_len)
    pub fn compute_rollout(
        &self,
        attentions: &[Array3<f64>],
        start_layer: usize,
    ) -> Result<Array2<f64>, AttentionRolloutError> {
        if attentions.is_empty() {
            return Err(AttentionRolloutError::EmptyAttentionList);
        }

        if start_layer >= attentions.len() {
            return Err(AttentionRolloutError::InvalidLayerIndex(start_layer));
        }

        let seq_len = attentions[0].shape()[1];

        // Initialize rollout with identity matrix
        let mut rollout = Array2::eye(seq_len);

        for attention in attentions.iter().skip(start_layer) {
            // Fuse attention heads
            let mut fused = self.fuse_heads(attention);

            // Discard low attention weights
            self.discard_low_attention(&mut fused);

            // Add residual connection
            if self.config.add_residual {
                let identity = Array2::eye(seq_len);
                let w = self.config.residual_weight;
                fused = &fused * (1.0 - w) + &identity * w;
            }

            // Accumulate rollout: R = A @ R
            rollout = fused.dot(&rollout);
        }

        // Normalize rows
        for mut row in rollout.rows_mut() {
            let sum: f64 = row.sum();
            if sum > 0.0 {
                row.mapv_inplace(|x| x / sum);
            }
        }

        Ok(rollout)
    }

    /// Get attribution scores for input positions
    ///
    /// # Arguments
    /// * `attentions` - List of attention matrices
    /// * `output_position` - Position to get attribution for (use seq_len-1 for last)
    pub fn get_input_attribution(
        &self,
        attentions: &[Array3<f64>],
        output_position: usize,
    ) -> Result<Array1<f64>, AttentionRolloutError> {
        let rollout = self.compute_rollout(attentions, 0)?;
        Ok(rollout.row(output_position).to_owned())
    }
}

impl Default for AttentionRollout {
    fn default() -> Self {
        Self::new()
    }
}

/// Trading-specific attention analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingAttentionAnalysis {
    pub temporal_importance: Vec<f64>,
    pub regime: AttentionRegime,
    pub concentration_score: f64,
    pub recent_bias: f64,
}

/// Detected attention regime
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AttentionRegime {
    Momentum,
    MeanReversion,
    Mixed,
}

/// Trading-specific attention rollout analyzer
pub struct TradingAttentionAnalyzer {
    rollout: AttentionRollout,
    momentum_threshold: f64,
}

impl TradingAttentionAnalyzer {
    pub fn new() -> Self {
        Self {
            rollout: AttentionRollout::new(),
            momentum_threshold: 0.6,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.momentum_threshold = threshold;
        self
    }

    /// Analyze attention patterns for trading insights
    pub fn analyze(
        &self,
        attentions: &[Array3<f64>],
    ) -> Result<TradingAttentionAnalysis, AttentionRolloutError> {
        let seq_len = attentions[0].shape()[1];
        let attribution = self.rollout.get_input_attribution(attentions, seq_len - 1)?;

        // Calculate recent bias
        let recent_window = seq_len / 4;
        let recent_bias: f64 = attribution
            .slice(ndarray::s![seq_len - recent_window..])
            .sum();

        // Determine regime
        let regime = if recent_bias > self.momentum_threshold {
            AttentionRegime::Momentum
        } else if recent_bias < 1.0 - self.momentum_threshold {
            AttentionRegime::MeanReversion
        } else {
            AttentionRegime::Mixed
        };

        // Calculate concentration (entropy)
        let concentration_score = self.calculate_entropy(&attribution);

        Ok(TradingAttentionAnalysis {
            temporal_importance: attribution.to_vec(),
            regime,
            concentration_score,
            recent_bias,
        })
    }

    fn calculate_entropy(&self, probs: &Array1<f64>) -> f64 {
        -probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }
}

impl Default for TradingAttentionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Bybit market data fetcher
pub mod data {
    use super::*;
    use reqwest::blocking::Client;

    #[derive(Debug, Deserialize)]
    struct BybitResponse {
        #[serde(rename = "retCode")]
        ret_code: i32,
        #[serde(rename = "retMsg")]
        ret_msg: String,
        result: BybitResult,
    }

    #[derive(Debug, Deserialize)]
    struct BybitResult {
        list: Vec<Vec<String>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Candle {
        pub timestamp: i64,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
    }

    /// Fetch candle data from Bybit
    pub fn fetch_bybit_candles(
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>, anyhow::Error> {
        let client = Client::new();
        let url = format!(
            "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            symbol, interval, limit
        );

        let response: BybitResponse = client.get(&url).send()?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let candles: Vec<Candle> = response
            .result
            .list
            .into_iter()
            .map(|row| Candle {
                timestamp: row[0].parse().unwrap_or(0),
                open: row[1].parse().unwrap_or(0.0),
                high: row[2].parse().unwrap_or(0.0),
                low: row[3].parse().unwrap_or(0.0),
                close: row[4].parse().unwrap_or(0.0),
                volume: row[5].parse().unwrap_or(0.0),
            })
            .collect();

        Ok(candles)
    }
}

/// Backtesting utilities
pub mod backtest {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BacktestResult {
        pub total_return: f64,
        pub sharpe_ratio: f64,
        pub sortino_ratio: f64,
        pub max_drawdown: f64,
        pub win_rate: f64,
        pub n_trades: usize,
    }

    /// Calculate Sharpe ratio
    pub fn calculate_sharpe(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let std = variance.sqrt();

        if std == 0.0 {
            return 0.0;
        }

        let daily_rf = risk_free_rate / 252.0;
        (252.0_f64).sqrt() * (mean - daily_rf) / std
    }

    /// Calculate Sortino ratio
    pub fn calculate_sortino(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

        let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if downside.is_empty() {
            return if mean > 0.0 { f64::INFINITY } else { 0.0 };
        }

        let downside_var: f64 = downside.iter().map(|r| r.powi(2)).sum::<f64>()
            / downside.len() as f64;
        let downside_std = downside_var.sqrt();

        if downside_std == 0.0 {
            return 0.0;
        }

        let daily_rf = risk_free_rate / 252.0;
        (252.0_f64).sqrt() * (mean - daily_rf) / downside_std
    }

    /// Calculate maximum drawdown
    pub fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let mut max_dd = 0.0;
        let mut peak = equity_curve[0];

        for &value in equity_curve.iter() {
            if value > peak {
                peak = value;
            }
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_rollout_basic() {
        let rollout = AttentionRollout::new();

        // Create simple attention matrices
        let attention1 = Array3::from_shape_fn((2, 4, 4), |(_, i, j)| {
            if i == j { 0.5 } else { 0.5 / 3.0 }
        });
        let attention2 = attention1.clone();

        let attentions = vec![attention1, attention2];
        let result = rollout.compute_rollout(&attentions, 0).unwrap();

        assert_eq!(result.shape(), &[4, 4]);

        // Check row normalization
        for row in result.rows() {
            let sum: f64 = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_head_fusion() {
        let rollout = AttentionRollout::with_config(AttentionRolloutConfig {
            head_fusion: HeadFusion::Max,
            ..Default::default()
        });

        let attention = Array3::from_shape_fn((2, 3, 3), |(h, i, j)| {
            if h == 0 { 0.3 } else { 0.7 }
        });

        let fused = rollout.fuse_heads(&attention);
        assert!((fused[[0, 0]] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_trading_regime_detection() {
        let analyzer = TradingAttentionAnalyzer::new();

        // Create attention that focuses on recent data (momentum)
        let mut attention = Array3::zeros((1, 10, 10));
        for i in 0..10 {
            attention[[0, i, 9]] = 0.8; // Focus on last position
            for j in 0..9 {
                attention[[0, i, j]] = 0.2 / 9.0;
            }
        }

        let result = analyzer.analyze(&[attention]).unwrap();
        assert_eq!(result.regime, AttentionRegime::Momentum);
    }
}
```

### Example Trading Application

```rust
// examples/trading_example.rs

use attention_rollout_trading::{
    AttentionRollout, AttentionRolloutConfig, HeadFusion,
    TradingAttentionAnalyzer, AttentionRegime,
    data::fetch_bybit_candles,
    backtest::{calculate_sharpe, calculate_sortino, calculate_max_drawdown},
};
use ndarray::{Array3, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Attention Rollout Trading Example ===\n");

    // Fetch market data from Bybit
    println!("Fetching BTC/USDT data from Bybit...");
    let candles = fetch_bybit_candles("BTCUSDT", "D", 100)?;
    println!("Fetched {} candles\n", candles.len());

    // Create simulated attention matrices (in practice, from transformer model)
    let seq_len = 20;
    let n_heads = 4;
    let n_layers = 3;

    let attentions: Vec<Array3<f64>> = (0..n_layers)
        .map(|layer| {
            Array3::from_shape_fn((n_heads, seq_len, seq_len), |(h, i, j)| {
                // Simulate attention pattern with recency bias
                let recency_weight = (j as f64 + 1.0) / seq_len as f64;
                let layer_factor = 1.0 + layer as f64 * 0.1;
                let base = recency_weight * layer_factor;

                // Normalize
                base / (seq_len as f64 * 0.5)
            })
        })
        .collect();

    // Compute attention rollout
    println!("Computing attention rollout...");
    let config = AttentionRolloutConfig {
        head_fusion: HeadFusion::Mean,
        discard_ratio: 0.1,
        add_residual: true,
        residual_weight: 0.5,
    };

    let rollout = AttentionRollout::with_config(config);
    let rollout_matrix = rollout.compute_rollout(&attentions, 0)?;

    println!("Rollout matrix shape: {:?}", rollout_matrix.shape());

    // Get input attribution for last output position
    let attribution = rollout.get_input_attribution(&attentions, seq_len - 1)?;

    println!("\nInput Attribution Scores (most recent = rightmost):");
    println!("{:=<60}", "");
    for (i, &score) in attribution.iter().enumerate() {
        let bar_len = (score * 50.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("t-{:>2}: {} {:.4}", seq_len - 1 - i, bar, score);
    }

    // Analyze trading regime
    println!("\n{:=<60}", "");
    println!("Trading Regime Analysis");
    println!("{:=<60}", "");

    let analyzer = TradingAttentionAnalyzer::new();
    let analysis = analyzer.analyze(&attentions)?;

    let regime_str = match analysis.regime {
        AttentionRegime::Momentum => "MOMENTUM (Recent price focus)",
        AttentionRegime::MeanReversion => "MEAN REVERSION (Historical focus)",
        AttentionRegime::Mixed => "MIXED (Balanced attention)",
    };

    println!("Detected Regime: {}", regime_str);
    println!("Recent Bias Score: {:.4}", analysis.recent_bias);
    println!("Attention Concentration: {:.4}", analysis.concentration_score);

    // Calculate backtest metrics (using simulated returns)
    println!("\n{:=<60}", "");
    println!("Backtest Metrics (Simulated)");
    println!("{:=<60}", "");

    // Simulate some returns
    let returns: Vec<f64> = candles
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect();

    let sharpe = calculate_sharpe(&returns, 0.02);
    let sortino = calculate_sortino(&returns, 0.02);

    let equity: Vec<f64> = returns
        .iter()
        .scan(100000.0, |capital, &r| {
            *capital *= 1.0 + r;
            Some(*capital)
        })
        .collect();

    let max_dd = calculate_max_drawdown(&equity);

    println!("Sharpe Ratio:  {:.4}", sharpe);
    println!("Sortino Ratio: {:.4}", sortino);
    println!("Max Drawdown:  {:.2}%", max_dd * 100.0);

    println!("\n=== Example Complete ===");

    Ok(())
}
```

---

## Practical Examples

### Example 1: Stock Market Prediction

```python
"""
Example: Using attention rollout for stock prediction interpretation
"""

import torch
import numpy as np
from attention_rollout import TradingAttentionRollout
from model import TradingTransformer
from data_loader import load_stock_data, prepare_features
from visualization import plot_input_attribution

# Load data
df = load_stock_data("AAPL", "2023-01-01", "2024-01-01")
X, y = prepare_features(df, lookback=20)

# Initialize model
model = TradingTransformer(
    input_dim=X.shape[2],
    d_model=128,
    n_heads=8,
    n_layers=4
)

# Initialize attention rollout
rollout = TradingAttentionRollout(
    model,
    attention_layer_name="attn",
    head_fusion="mean"
)

# Make prediction and get attribution
sample_idx = 100
x_sample = torch.FloatTensor(X[sample_idx:sample_idx+1])

with torch.no_grad():
    logits, _ = model(x_sample)
    prediction = torch.argmax(logits, dim=-1).item()

attribution = rollout.get_input_attribution(x_sample)

# Interpret results
labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
print(f"Prediction: {labels[prediction]}")
print(f"\nMost influential time periods:")
top_5_idx = np.argsort(attribution)[-5:][::-1]
for idx in top_5_idx:
    print(f"  t-{19-idx}: {attribution[idx]:.4f}")

# Visualize
timestamps = [f"t-{i}" for i in range(19, -1, -1)]
plot_input_attribution(attribution, timestamps)
```

### Example 2: Crypto Trading with Bybit Data

```python
"""
Example: Cryptocurrency trading with attention analysis using Bybit
"""

from data_loader import load_bybit_data, prepare_features
from attention_rollout import TradingAttentionRollout
from model import TradingTransformer
from backtest import AttentionBacktester, print_backtest_report

# Load Bybit data
df = load_bybit_data(symbol="BTCUSDT", interval="60", limit=200)
print(f"Loaded {len(df)} hourly candles")

# Prepare features
X, y = prepare_features(df, lookback=24)  # 24 hours lookback
prices = df["close"].values[-len(X)-1:-1]

# Initialize model (in practice, load trained weights)
model = TradingTransformer(
    input_dim=X.shape[2],
    d_model=64,
    n_heads=4,
    n_layers=3
)

# Initialize attention rollout
rollout = TradingAttentionRollout(model)

# Run backtest with attention analysis
backtester = AttentionBacktester(
    model=model,
    attention_rollout=rollout,
    initial_capital=10000.0,
    transaction_cost=0.001
)

result = backtester.run_backtest(X, prices, threshold=0.6)
print_backtest_report(result)

# Analyze attention patterns
print("\nAttention Pattern Insights:")
if result.attention_analysis['recent_bias_winning'] > result.attention_analysis['recent_bias_losing']:
    print("  → Winning trades show stronger recent price focus")
else:
    print("  → Losing trades show stronger recent price focus")
```

---

## Backtesting Framework

### Complete Backtesting Pipeline

```python
"""
Complete backtesting pipeline with attention rollout analysis
"""

import numpy as np
import torch
from typing import Dict, Tuple
from dataclasses import dataclass

from model import TradingTransformer
from attention_rollout import TradingAttentionRollout
from data_loader import load_stock_data, load_bybit_data, prepare_features
from backtest import AttentionBacktester, BacktestResult, print_backtest_report


def run_complete_backtest(
    data_source: str = "stock",
    symbol: str = "AAPL",
    lookback: int = 20,
    model_params: Dict = None
) -> Tuple[BacktestResult, TradingTransformer]:
    """
    Run complete backtest pipeline.

    Args:
        data_source: "stock" or "crypto"
        symbol: Asset symbol
        lookback: Number of time steps to look back
        model_params: Model hyperparameters

    Returns:
        BacktestResult and trained model
    """
    # Load data
    if data_source == "stock":
        df = load_stock_data(symbol, "2022-01-01", "2024-01-01")
    else:
        df = load_bybit_data(symbol, interval="D", limit=200)

    # Prepare features
    X, y = prepare_features(df, lookback=lookback)
    prices = df["close"].values[-len(X)-1:-1]

    # Split data
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    prices_test = prices[train_size:]

    # Default model parameters
    if model_params is None:
        model_params = {
            "input_dim": X.shape[2],
            "d_model": 128,
            "n_heads": 8,
            "n_layers": 4,
            "dropout": 0.1
        }

    # Initialize and train model
    model = TradingTransformer(**model_params)
    model = train_model(model, X_train, y_train, epochs=50)

    # Initialize attention rollout
    rollout = TradingAttentionRollout(model)

    # Run backtest
    backtester = AttentionBacktester(
        model=model,
        attention_rollout=rollout,
        initial_capital=100000.0,
        transaction_cost=0.001
    )

    result = backtester.run_backtest(X_test, prices_test)

    return result, model


def train_model(
    model: TradingTransformer,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32
) -> TradingTransformer:
    """Train the trading model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        indices = torch.randperm(len(X))

        for i in range(0, len(X), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_tensor[batch_idx]
            y_batch = y_tensor[batch_idx]

            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")

    model.eval()
    return model


if __name__ == "__main__":
    # Run stock backtest
    print("Running Stock Market Backtest (AAPL)")
    print("="*60)
    stock_result, _ = run_complete_backtest(
        data_source="stock",
        symbol="AAPL"
    )
    print_backtest_report(stock_result)

    # Run crypto backtest
    print("\nRunning Crypto Backtest (BTCUSDT)")
    print("="*60)
    crypto_result, _ = run_complete_backtest(
        data_source="crypto",
        symbol="BTCUSDT"
    )
    print_backtest_report(crypto_result)
```

---

## Advanced Topics

### 1. Gradient-Weighted Attention Rollout

Combining attention rollout with gradient information for more accurate attribution:

```python
class GradientWeightedAttentionRollout(AttentionRollout):
    """
    Attention rollout weighted by gradient magnitudes.

    This method combines attention flow with gradient-based saliency
    for more accurate feature attribution.
    """

    def compute_gradient_weighted_rollout(
        self,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> np.ndarray:
        """Compute gradient-weighted attention rollout."""
        self.model.zero_grad()
        input_tensor.requires_grad_(True)

        self.attentions = []
        logits, _ = self.model(input_tensor)

        # Backward pass for target class
        logits[0, target_class].backward()

        # Get gradients
        gradients = input_tensor.grad.abs().mean(dim=-1)

        # Compute standard rollout
        rollout = self.compute_rollout(input_tensor.detach())

        # Weight by gradients
        grad_weights = gradients.numpy()[0]
        grad_weights = grad_weights / grad_weights.sum()

        weighted_rollout = rollout * grad_weights.reshape(-1, 1)
        weighted_rollout = weighted_rollout / weighted_rollout.sum(axis=1, keepdims=True)

        return weighted_rollout
```

### 2. Multi-Asset Attention Analysis

Analyzing attention patterns across multiple correlated assets:

```python
def analyze_multi_asset_attention(
    model: TradingTransformer,
    assets_data: Dict[str, np.ndarray],
    target_asset: str
) -> Dict[str, float]:
    """
    Analyze how attention flows between multiple assets.

    Args:
        model: Multi-asset transformer model
        assets_data: Dictionary of asset name to feature sequences
        target_asset: Asset to analyze predictions for

    Returns:
        Dictionary mapping asset names to influence scores
    """
    # Concatenate all asset features
    asset_names = list(assets_data.keys())
    combined_features = np.concatenate(
        [assets_data[name] for name in asset_names],
        axis=1  # Concatenate along sequence dimension
    )

    x_tensor = torch.FloatTensor(combined_features).unsqueeze(0)

    rollout = TradingAttentionRollout(model)
    attribution = rollout.get_input_attribution(x_tensor)

    # Map attribution back to assets
    seq_len_per_asset = assets_data[asset_names[0]].shape[0]
    asset_influence = {}

    for i, name in enumerate(asset_names):
        start_idx = i * seq_len_per_asset
        end_idx = (i + 1) * seq_len_per_asset
        asset_influence[name] = float(attribution[start_idx:end_idx].sum())

    return asset_influence
```

### 3. Time-Varying Attention Patterns

Tracking how attention patterns evolve over time:

```python
def track_attention_evolution(
    model: TradingTransformer,
    X: np.ndarray,
    window_size: int = 50
) -> List[Dict]:
    """
    Track how attention patterns evolve over time.

    Returns list of attention analyses for rolling windows.
    """
    rollout = TradingAttentionRollout(model)
    evolution = []

    for i in range(window_size, len(X)):
        x_window = torch.FloatTensor(X[i:i+1])

        analysis = {
            "index": i,
            "regime": rollout.detect_attention_regime(x_window),
            "temporal_importance": rollout.analyze_temporal_importance(x_window),
            "recent_bias": float(
                rollout.get_input_attribution(x_window)[-5:].sum()
            )
        }
        evolution.append(analysis)

    return evolution
```

---

## References

1. **Abnar, S., & Zuidema, W. (2020)**. "Quantifying Attention Flow in Transformers." *ACL 2020*. [https://arxiv.org/abs/2005.00928](https://arxiv.org/abs/2005.00928)

2. **Vaswani, A., et al. (2017)**. "Attention Is All You Need." *NeurIPS 2017*. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

3. **Chefer, H., Gur, S., & Wolf, L. (2021)**. "Transformer Interpretability Beyond Attention Visualization." *CVPR 2021*. [https://arxiv.org/abs/2012.09838](https://arxiv.org/abs/2012.09838)

4. **Ding, Q., et al. (2020)**. "Hierarchical Multi-Scale Gaussian Transformer for Stock Movement Prediction." *IJCAI 2020*. [https://www.ijcai.org/proceedings/2020/640](https://www.ijcai.org/proceedings/2020/640)

5. **Zhang, L., et al. (2022)**. "Transformer-based Stock Trend Prediction with Attention Analysis." *Expert Systems with Applications*. [https://doi.org/10.1016/j.eswa.2022.117239](https://doi.org/10.1016/j.eswa.2022.117239)

---

## Running the Examples

### Python Setup

```bash
cd 124_attention_rollout_trading/python
pip install -r requirements.txt

# Run basic example
python -c "
from attention_rollout import AttentionRollout
from model import TradingTransformer
import torch

model = TradingTransformer(input_dim=5, d_model=64, n_heads=4, n_layers=2)
rollout = AttentionRollout(model, attention_layer_name='attn')

x = torch.randn(1, 20, 5)
attribution = rollout.get_input_attribution(x)
print('Attribution scores:', attribution)
"
```

### Rust Setup

```bash
cd 124_attention_rollout_trading/rust
cargo build --release
cargo run --example trading_example
```

### Jupyter Notebook

```bash
cd 124_attention_rollout_trading/python/notebooks
jupyter notebook 01_attention_rollout_trading.ipynb
```

---

## Conclusion

Attention Rollout provides a powerful lens into transformer-based trading models, enabling:

1. **Interpretable predictions** - Understand why the model predicts buy/sell
2. **Risk management** - Validate model focus areas before trading
3. **Strategy development** - Discover temporal patterns the model learns
4. **Debugging** - Identify when models rely on spurious correlations

By combining attention rollout with robust backtesting, traders can build more trustworthy and profitable algorithmic trading systems.
