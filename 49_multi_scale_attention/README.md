# Chapter 49: Multi-Scale Attention for Financial Time Series

This chapter explores **Multi-Scale Attention** mechanisms for financial time series prediction. Unlike single-scale approaches that process data at one temporal resolution, multi-scale attention captures patterns across different time horizons simultaneously — from short-term tick-by-tick fluctuations to long-term trend dynamics.

## Contents

1. [Introduction to Multi-Scale Attention](#introduction-to-multi-scale-attention)
    * [Why Multiple Time Scales?](#why-multiple-time-scales)
    * [Key Advantages](#key-advantages)
    * [Comparison with Single-Scale Models](#comparison-with-single-scale-models)
2. [Multi-Scale Attention Architecture](#multi-scale-attention-architecture)
    * [Scale-Specific Encoders](#scale-specific-encoders)
    * [Multi-Resolution Attention](#multi-resolution-attention)
    * [Cross-Scale Fusion](#cross-scale-fusion)
    * [Hierarchical Aggregation](#hierarchical-aggregation)
3. [Time Scale Decomposition](#time-scale-decomposition)
    * [Temporal Downsampling](#temporal-downsampling)
    * [Wavelet-Based Decomposition](#wavelet-based-decomposition)
    * [Variational Mode Decomposition (VMD)](#variational-mode-decomposition-vmd)
4. [Practical Examples](#practical-examples)
    * [01: Data Preparation with Multi-Scale Features](#01-data-preparation-with-multi-scale-features)
    * [02: Multi-Scale Attention Architecture](#02-multi-scale-attention-architecture)
    * [03: Model Training](#03-model-training)
    * [04: Multi-Horizon Prediction](#04-multi-horizon-prediction)
    * [05: Strategy Backtesting](#05-strategy-backtesting)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to Multi-Scale Attention

Financial markets exhibit patterns across multiple time scales. A minute trader sees noise where a day trader sees trends, and a day trader sees noise where a swing trader sees cycles. **Multi-Scale Attention** explicitly models these different temporal resolutions, allowing a single model to understand both short-term momentum and long-term trends.

### Why Multiple Time Scales?

Traditional models process time series at a single resolution:

```
Price data (1-minute) → Model → Prediction

Problems:
- Short window: Captures noise, misses trends
- Long window: Captures trends, loses fine details
- Fixed scale: Can't adapt to changing market conditions
```

Multi-Scale Attention processes at multiple resolutions simultaneously:

```
Price data (1-minute)  ]
Price data (5-minute)  ]  → Multi-Scale    → Prediction
Price data (1-hour)    ]    Attention        (understanding all patterns!)
Price data (1-day)     ]

Benefits:
- Short scales: Capture entry/exit timing
- Medium scales: Capture intraday patterns
- Long scales: Capture trend direction
- Fusion: Combine all insights intelligently
```

### Key Advantages

1. **Hierarchical Pattern Discovery**
   - Minute-level: Order flow, microstructure effects
   - Hourly-level: Session patterns, volume profiles
   - Daily-level: Trend following, mean reversion cycles
   - Weekly-level: Macro regime changes

2. **Adaptive Focus**
   - Learn which scales matter for each prediction
   - Attention weights reveal scale importance
   - Dynamic reweighting based on market conditions

3. **Robust Predictions**
   - Short-term noise doesn't dominate
   - Long-term trends provide context
   - Cross-scale validation reduces false signals

4. **Multi-Horizon Forecasting**
   - Single model for multiple prediction horizons
   - Consistent signals across timeframes
   - Unified risk management

### Comparison with Single-Scale Models

| Feature | LSTM | Transformer | TFT | Multi-Scale Attention |
|---------|------|-------------|-----|----------------------|
| Multi-resolution | No | No | Limited | Full |
| Scale-aware attention | No | No | No | Yes |
| Cross-scale fusion | No | No | No | Yes |
| Adaptive scale weights | No | No | Partial | Yes |
| Long sequence handling | Poor | O(L^2) | O(L) | O(L) per scale |
| Interpretability | Low | Medium | High | Very High |

## Multi-Scale Attention Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-SCALE ATTENTION NETWORK                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Raw Time Series                                                              │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    SCALE DECOMPOSITION LAYER                             │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │ │
│  │  │ Scale 1  │  │ Scale 2  │  │ Scale 3  │  │ Scale 4  │                │ │
│  │  │ (1-min)  │  │ (5-min)  │  │ (1-hour) │  │ (1-day)  │                │ │
│  │  │ L=1440   │  │ L=288    │  │ L=24     │  │ L=30     │                │ │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                │ │
│  └───────│─────────────│─────────────│─────────────│───────────────────────┘ │
│          │             │             │             │                          │
│          ▼             ▼             ▼             ▼                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │ Encoder  │  │ Encoder  │  │ Encoder  │  │ Encoder  │                      │
│  │ (short)  │  │ (medium) │  │  (long)  │  │ (trend)  │                      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                      │
│       │             │             │             │                             │
│       └──────────┬──┴──────────┬──┴──────────┬──┘                            │
│                  │             │             │                                │
│                  ▼             ▼             ▼                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    CROSS-SCALE ATTENTION FUSION                         │ │
│  │                                                                          │ │
│  │    Scale 1 ←──attention──→ Scale 2 ←──attention──→ Scale 3             │ │
│  │         └──────────────── attention ─────────────────→ Scale 4          │ │
│  │                                                                          │ │
│  │    Learns: "Short-term momentum aligns with long-term trend?"           │ │
│  │            "Scale 2 confirms Scale 1 signal?"                            │ │
│  └──────────────────────────────────┬──────────────────────────────────────┘ │
│                                     │                                         │
│                                     ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    HIERARCHICAL AGGREGATION                              │ │
│  │                                                                          │ │
│  │    Weighted combination based on learned scale importance                │ │
│  │    α₁·Scale1 + α₂·Scale2 + α₃·Scale3 + α₄·Scale4                        │ │
│  │                                                                          │ │
│  └──────────────────────────────────┬──────────────────────────────────────┘ │
│                                     │                                         │
│                                     ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         PREDICTION HEAD                                  │ │
│  │    • Short-term prediction (next 5 min)                                  │ │
│  │    • Medium-term prediction (next 1 hour)                                │ │
│  │    • Long-term prediction (next 1 day)                                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Scale-Specific Encoders

Each temporal scale has its own encoder optimized for that resolution:

```python
class ScaleEncoder(nn.Module):
    """
    Encoder for a specific time scale.

    Different scales need different architectures:
    - Short scales (1-min): CNN for local patterns, dropout for noise
    - Long scales (1-day): Deeper attention for global patterns
    """
    def __init__(
        self,
        scale_name: str,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.scale_name = scale_name

        # Positional encoding for temporal order
        self.pos_encoding = LearnablePositionalEncoding(d_model)

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Scale-specific output
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            encoded: [batch, seq_len, d_model]
        """
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return self.output_norm(x)
```

**Scale-specific configurations:**

| Scale | Sequence Length | n_layers | n_heads | Notes |
|-------|----------------|----------|---------|-------|
| 1-min | 1440 (1 day) | 2 | 4 | Focus on local patterns |
| 5-min | 288 (1 day) | 3 | 8 | Balance local/global |
| 1-hour | 168 (1 week) | 4 | 8 | Capture daily patterns |
| 1-day | 252 (1 year) | 4 | 8 | Long-term trends |

### Multi-Resolution Attention

The key mechanism: attention that operates across scales:

```python
class MultiResolutionAttention(nn.Module):
    """
    Attention mechanism that queries across different time scales.

    Query from one scale can attend to keys/values from other scales,
    enabling the model to discover cross-scale dependencies.
    """
    def __init__(self, d_model: int, n_heads: int, n_scales: int):
        super().__init__()
        self.n_scales = n_scales
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Separate projections for each scale
        self.query_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_scales)
        ])
        self.key_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_scales)
        ])
        self.value_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_scales)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model * n_scales, d_model)

    def forward(
        self,
        scale_features: List[torch.Tensor],
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            scale_features: List of [batch, seq_len_i, d_model] for each scale
        Returns:
            fused: [batch, d_model] - fused representation
            attention_weights: [batch, n_scales, n_scales] if requested
        """
        batch_size = scale_features[0].shape[0]

        # Compute queries, keys, values for each scale
        # Use the last timestep as query (most recent information)
        queries = [
            self.query_projs[i](feat[:, -1, :])  # [batch, d_model]
            for i, feat in enumerate(scale_features)
        ]

        # Use all timesteps as keys/values
        keys = [
            self.key_projs[i](feat)  # [batch, seq_len, d_model]
            for i, feat in enumerate(scale_features)
        ]
        values = [
            self.value_projs[i](feat)  # [batch, seq_len, d_model]
            for i, feat in enumerate(scale_features)
        ]

        # Cross-scale attention: each scale attends to all scales
        attended_features = []
        attention_weights = []

        for i in range(self.n_scales):
            q = queries[i].unsqueeze(1)  # [batch, 1, d_model]

            # Attend to all scales
            scale_attended = []
            scale_attention = []

            for j in range(self.n_scales):
                k = keys[j]  # [batch, seq_len_j, d_model]
                v = values[j]

                # Compute attention scores
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attn = torch.softmax(scores, dim=-1)  # [batch, 1, seq_len_j]

                # Attend to values
                context = torch.matmul(attn, v)  # [batch, 1, d_model]
                scale_attended.append(context.squeeze(1))

                # Store attention for interpretability
                scale_attention.append(attn.mean(dim=-1))

            # Combine attention from all scales
            attended_features.append(torch.stack(scale_attended, dim=1).mean(dim=1))
            attention_weights.append(torch.stack(scale_attention, dim=-1))

        # Concatenate and project
        fused = torch.cat(attended_features, dim=-1)  # [batch, d_model * n_scales]
        fused = self.output_proj(fused)  # [batch, d_model]

        if return_attention:
            attention_matrix = torch.stack(attention_weights, dim=1)
            return fused, attention_matrix

        return fused, None
```

### Cross-Scale Fusion

Different scales contain complementary information. The fusion layer learns to combine them:

```python
class CrossScaleFusion(nn.Module):
    """
    Fuses information from multiple time scales with learnable weights.

    Uses attention to determine which scales are most relevant
    for the current prediction task.
    """
    def __init__(self, d_model: int, n_scales: int):
        super().__init__()

        # Learnable scale importance weights
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        # Gate for each scale
        self.scale_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()
            )
            for _ in range(n_scales)
        ])

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * n_scales, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(
        self,
        scale_features: List[torch.Tensor],
        return_weights: bool = False
    ) -> torch.Tensor:
        """
        Args:
            scale_features: List of [batch, d_model] from each scale
        Returns:
            fused: [batch, d_model]
        """
        # Compute dynamic gates for each scale
        gates = []
        for i, feat in enumerate(scale_features):
            gate = self.scale_gates[i](feat)  # [batch, 1]
            gates.append(gate)

        # Combine with learnable weights
        weights = torch.softmax(self.scale_weights, dim=0)
        weighted_features = []

        for i, feat in enumerate(scale_features):
            # Scale importance = learned weight * dynamic gate
            importance = weights[i] * gates[i]
            weighted_features.append(importance * feat)

        # Concatenate and fuse
        concat = torch.cat(weighted_features, dim=-1)
        fused = self.fusion(concat)

        if return_weights:
            return fused, weights, torch.cat(gates, dim=-1)

        return fused
```

### Hierarchical Aggregation

For very long sequences, hierarchical aggregation reduces computational cost:

```python
class HierarchicalAggregation(nn.Module):
    """
    Hierarchical aggregation for efficient multi-scale processing.

    Instead of attending to all tokens, aggregate in a pyramid:
    Level 0: Full resolution (all tokens)
    Level 1: 2x downsampled (average pooling)
    Level 2: 4x downsampled
    ...
    """
    def __init__(
        self,
        d_model: int,
        n_levels: int = 4,
        pool_size: int = 2
    ):
        super().__init__()
        self.n_levels = n_levels
        self.pool_size = pool_size

        # Attention at each hierarchical level
        self.level_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            for _ in range(n_levels)
        ])

        # Level fusion
        self.level_fusion = nn.Sequential(
            nn.Linear(d_model * n_levels, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, d_model]
        """
        level_outputs = []
        current = x

        for level in range(self.n_levels):
            # Self-attention at current level
            attn_out, _ = self.level_attention[level](current, current, current)

            # Take last token as level representation
            level_outputs.append(attn_out[:, -1, :])

            # Downsample for next level
            if level < self.n_levels - 1:
                batch, seq_len, d = current.shape
                if seq_len >= self.pool_size:
                    # Average pooling
                    current = current.view(batch, seq_len // self.pool_size, self.pool_size, d)
                    current = current.mean(dim=2)

        # Fuse all levels
        fused = torch.cat(level_outputs, dim=-1)
        return self.level_fusion(fused)
```

## Time Scale Decomposition

Before feeding data to multi-scale encoders, we need to decompose the time series into different scales.

### Temporal Downsampling

The simplest approach: resample data at different intervals:

```python
def temporal_downsample(
    df: pd.DataFrame,
    target_intervals: List[str] = ['1min', '5min', '1H', '1D']
) -> Dict[str, pd.DataFrame]:
    """
    Downsample OHLCV data to multiple time scales.

    Args:
        df: DataFrame with 1-minute OHLCV data
        target_intervals: List of target resolutions

    Returns:
        Dictionary mapping interval to DataFrame
    """
    result = {}

    for interval in target_intervals:
        if interval == '1min':
            result[interval] = df.copy()
        else:
            # Resample OHLCV data properly
            resampled = df.resample(interval).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            result[interval] = resampled

    return result
```

### Wavelet-Based Decomposition

Wavelets naturally decompose signals into different frequency components:

```python
import pywt
import numpy as np

def wavelet_decomposition(
    prices: np.ndarray,
    wavelet: str = 'db4',
    levels: int = 4
) -> Dict[str, np.ndarray]:
    """
    Decompose price series using wavelet transform.

    Returns approximation (trend) and detail (noise) coefficients
    at each level.

    Args:
        prices: 1D numpy array of prices
        wavelet: Wavelet type (db4, haar, sym5, etc.)
        levels: Number of decomposition levels

    Returns:
        Dictionary with 'trend' and 'detail_i' for each level
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(prices, wavelet, level=levels)

    result = {
        'trend': coeffs[0]  # Approximation (lowest frequency)
    }

    # Detail coefficients (high to low frequency)
    for i, detail in enumerate(coeffs[1:], 1):
        result[f'detail_{i}'] = detail

    return result


def wavelet_reconstruction(
    coeffs: Dict[str, np.ndarray],
    wavelet: str = 'db4',
    keep_levels: List[int] = None
) -> np.ndarray:
    """
    Reconstruct signal from selected wavelet coefficients.

    Useful for filtering: keep only trend + some details.
    """
    all_coeffs = [coeffs['trend']]
    n_details = len([k for k in coeffs if k.startswith('detail')])

    for i in range(1, n_details + 1):
        if keep_levels is None or i in keep_levels:
            all_coeffs.append(coeffs[f'detail_{i}'])
        else:
            # Zero out unwanted levels
            all_coeffs.append(np.zeros_like(coeffs[f'detail_{i}']))

    return pywt.waverec(all_coeffs, wavelet)
```

### Variational Mode Decomposition (VMD)

VMD provides adaptive decomposition into frequency bands:

```python
from vmdpy import VMD

def vmd_decomposition(
    prices: np.ndarray,
    n_modes: int = 4,
    alpha: float = 2000,
    tau: float = 0,
    DC: int = 0,
    init: int = 1,
    tol: float = 1e-7
) -> Dict[str, np.ndarray]:
    """
    Decompose price series using Variational Mode Decomposition.

    VMD adaptively finds the optimal center frequencies,
    making it better suited for financial data than wavelets.

    Args:
        prices: 1D numpy array of prices
        n_modes: Number of modes to extract
        alpha: Bandwidth constraint parameter
        tau: Noise tolerance (0 for no noise)
        DC: Include DC component (0 or 1)
        init: Initialization (1 = uniform, 2 = random)
        tol: Convergence tolerance

    Returns:
        Dictionary with mode_i for each decomposed mode
    """
    # Run VMD
    modes, freqs, _ = VMD(
        prices, alpha, tau, n_modes, DC, init, tol
    )

    result = {}
    for i, mode in enumerate(modes):
        # Lower index = lower frequency (trend)
        # Higher index = higher frequency (noise)
        result[f'mode_{i}'] = mode

    # Also return center frequencies
    result['frequencies'] = freqs

    return result
```

## Practical Examples

### 01: Data Preparation with Multi-Scale Features

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import requests
from datetime import datetime, timedelta

def fetch_bybit_klines(
    symbol: str = 'BTCUSDT',
    interval: str = '1',  # 1 minute
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Bybit.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval in minutes ('1', '5', '60', '240', 'D')
        limit: Number of candles to fetch

    Returns:
        DataFrame with OHLCV data
    """
    url = 'https://api.bybit.com/v5/market/kline'

    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data['retCode'] != 0:
        raise ValueError(f"API error: {data['retMsg']}")

    # Parse response
    klines = data['result']['list']
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    df = df.set_index('timestamp').sort_index()

    return df


def create_multi_scale_features(
    df: pd.DataFrame,
    scales: List[str] = ['1min', '5min', '15min', '1H', '4H']
) -> Dict[str, pd.DataFrame]:
    """
    Create feature DataFrames for multiple time scales.

    Args:
        df: 1-minute OHLCV DataFrame
        scales: List of target time scales

    Returns:
        Dictionary mapping scale to feature DataFrame
    """
    result = {}

    for scale in scales:
        # Resample if needed
        if scale == '1min':
            resampled = df.copy()
        else:
            resampled = df.resample(scale).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        # Calculate features
        features = pd.DataFrame(index=resampled.index)

        # Returns
        features['log_return'] = np.log(resampled['close'] / resampled['close'].shift(1))
        features['return_1'] = resampled['close'].pct_change(1)
        features['return_5'] = resampled['close'].pct_change(5)
        features['return_10'] = resampled['close'].pct_change(10)

        # Volatility
        features['volatility_5'] = features['log_return'].rolling(5).std()
        features['volatility_20'] = features['log_return'].rolling(20).std()

        # Volume features
        features['volume_ratio'] = resampled['volume'] / resampled['volume'].rolling(20).mean()
        features['volume_change'] = resampled['volume'].pct_change()

        # Price features
        features['high_low_ratio'] = (resampled['high'] - resampled['low']) / resampled['close']
        features['close_position'] = (resampled['close'] - resampled['low']) / (resampled['high'] - resampled['low'] + 1e-8)

        # Moving averages
        for period in [5, 10, 20, 50]:
            ma = resampled['close'].rolling(period).mean()
            features[f'ma_{period}_ratio'] = resampled['close'] / ma - 1

        # RSI
        delta = resampled['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        features['rsi'] = 100 - (100 / (1 + rs))

        # Normalize RSI to [-1, 1]
        features['rsi_normalized'] = (features['rsi'] - 50) / 50

        # Drop NaN and store
        result[scale] = features.dropna()

    return result


def create_sequences(
    multi_scale_features: Dict[str, pd.DataFrame],
    lookback_periods: Dict[str, int],
    prediction_horizon: int = 1,
    target_scale: str = '1H'
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Create sequences for multi-scale training.

    Args:
        multi_scale_features: Dictionary of feature DataFrames by scale
        lookback_periods: Number of periods to look back for each scale
        prediction_horizon: How many periods ahead to predict
        target_scale: Scale for the prediction target

    Returns:
        X: Dictionary of feature arrays by scale
        y: Target array
    """
    # Align all scales to target scale timestamps
    target_df = multi_scale_features[target_scale]

    X = {scale: [] for scale in multi_scale_features.keys()}
    y = []

    for i in range(max(lookback_periods.values()), len(target_df) - prediction_horizon):
        target_time = target_df.index[i]

        # Get features for each scale
        valid = True
        for scale, features in multi_scale_features.items():
            lookback = lookback_periods.get(scale, 24)

            # Find the index closest to target_time
            if scale == target_scale:
                idx = i
            else:
                idx = features.index.get_indexer([target_time], method='ffill')[0]

            if idx < lookback:
                valid = False
                break

            # Extract sequence
            seq = features.iloc[idx - lookback:idx].values
            if len(seq) == lookback:
                X[scale].append(seq)
            else:
                valid = False
                break

        if valid:
            # Target: next period return
            target_return = target_df['log_return'].iloc[i + prediction_horizon]
            y.append(target_return)

    # Convert to numpy arrays
    X = {scale: np.array(seqs) for scale, seqs in X.items()}
    y = np.array(y)

    return X, y
```

### 02: Multi-Scale Attention Architecture

```python
# python/02_model.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class MultiScaleAttention(nn.Module):
    """
    Multi-Scale Attention Network for financial time series prediction.

    Processes data at multiple temporal resolutions and learns
    to combine insights from different time scales.
    """
    def __init__(
        self,
        scale_configs: Dict[str, Dict],
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        """
        Args:
            scale_configs: Configuration for each scale
                {
                    'scale_name': {
                        'input_dim': int,
                        'seq_len': int
                    }
                }
            d_model: Model dimension
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder layers per scale
            dropout: Dropout rate
            output_dim: Output dimension
        """
        super().__init__()
        self.scale_names = list(scale_configs.keys())
        self.n_scales = len(self.scale_names)
        self.d_model = d_model

        # Scale-specific encoders
        self.scale_encoders = nn.ModuleDict()
        for name, config in scale_configs.items():
            self.scale_encoders[name] = ScaleEncoder(
                scale_name=name,
                input_dim=config['input_dim'],
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_encoder_layers,
                dropout=dropout
            )

        # Multi-resolution attention
        self.multi_res_attention = MultiResolutionAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_scales=self.n_scales
        )

        # Cross-scale fusion
        self.cross_scale_fusion = CrossScaleFusion(
            d_model=d_model,
            n_scales=self.n_scales
        )

        # Prediction heads for different horizons
        self.short_term_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.medium_term_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.long_term_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        # Direction classification head
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        scale_inputs: Dict[str, torch.Tensor],
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            scale_inputs: Dictionary mapping scale name to input tensor
                Each tensor has shape [batch, seq_len, input_dim]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with predictions and optionally attention weights
        """
        # Encode each scale
        scale_encodings = []
        for name in self.scale_names:
            x = scale_inputs[name]
            encoded = self.scale_encoders[name](x)
            scale_encodings.append(encoded)

        # Multi-resolution attention
        fused, attention = self.multi_res_attention(
            scale_encodings,
            return_attention=return_attention
        )

        # Cross-scale fusion
        # Extract last timestep from each scale
        scale_summaries = [enc[:, -1, :] for enc in scale_encodings]
        final_repr = self.cross_scale_fusion(scale_summaries)

        # Combine attention output with scale fusion
        combined = fused + final_repr

        # Predictions
        result = {
            'short_term': self.short_term_head(combined),
            'medium_term': self.medium_term_head(combined),
            'long_term': self.long_term_head(combined),
            'direction': self.direction_head(combined)
        }

        if return_attention:
            result['attention'] = attention

        return result


class ScaleEncoder(nn.Module):
    """Encoder for a specific time scale."""

    def __init__(
        self,
        scale_name: str,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.scale_name = scale_name

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Encode
        x = self.encoder(x)

        return self.output_norm(x)


class MultiResolutionAttention(nn.Module):
    """Cross-scale attention mechanism."""

    def __init__(self, d_model: int, n_heads: int, n_scales: int):
        super().__init__()
        self.n_scales = n_scales
        self.d_model = d_model

        # Cross-attention for each scale
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(n_scales)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model * n_scales, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        scale_features: List[torch.Tensor],
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = scale_features[0].shape[0]

        # Concatenate all scales for keys/values
        all_features = torch.cat(scale_features, dim=1)  # [batch, total_len, d_model]

        attended = []
        attention_weights = []

        for i, features in enumerate(scale_features):
            # Query: last position of this scale
            query = features[:, -1:, :]  # [batch, 1, d_model]

            # Attend to all scales
            attn_out, attn_weight = self.cross_attention[i](
                query, all_features, all_features,
                need_weights=return_attention
            )

            attended.append(attn_out.squeeze(1))
            if return_attention:
                attention_weights.append(attn_weight)

        # Combine
        combined = torch.cat(attended, dim=-1)
        output = self.output_proj(combined)
        output = self.output_norm(output)

        if return_attention:
            return output, torch.stack(attention_weights, dim=1)
        return output, None


class CrossScaleFusion(nn.Module):
    """Learnable fusion of multi-scale representations."""

    def __init__(self, d_model: int, n_scales: int):
        super().__init__()
        self.n_scales = n_scales

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(d_model * n_scales, n_scales),
            nn.Softmax(dim=-1)
        )

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        # Stack features
        stacked = torch.stack(scale_features, dim=1)  # [batch, n_scales, d_model]

        # Compute dynamic gates
        concat = torch.cat(scale_features, dim=-1)  # [batch, n_scales * d_model]
        gates = self.gate(concat)  # [batch, n_scales]

        # Combine with learned weights
        combined_weights = torch.softmax(self.scale_weights, dim=0) * gates
        combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted sum
        weighted = (stacked * combined_weights.unsqueeze(-1)).sum(dim=1)

        # Fusion
        return self.fusion(weighted)
```

### 03: Model Training

```python
# python/03_train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

def train_multi_scale_model(
    model: nn.Module,
    train_data: Tuple[Dict[str, np.ndarray], np.ndarray],
    val_data: Tuple[Dict[str, np.ndarray], np.ndarray],
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Train the multi-scale attention model.

    Args:
        model: MultiScaleAttention model
        train_data: (X_dict, y) for training
        val_data: (X_dict, y) for validation
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        device: Device to train on

    Returns:
        Training history dictionary
    """
    model = model.to(device)

    # Prepare data loaders
    train_X, train_y = train_data
    val_X, val_y = val_data

    # Convert to tensors
    train_tensors = {
        k: torch.FloatTensor(v) for k, v in train_X.items()
    }
    train_y_tensor = torch.FloatTensor(train_y).unsqueeze(-1)

    val_tensors = {
        k: torch.FloatTensor(v) for k, v in val_X.items()
    }
    val_y_tensor = torch.FloatTensor(val_y).unsqueeze(-1)

    # Create a simple batch iterator
    n_train = len(train_y)
    n_val = len(val_y)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # Loss functions
    regression_loss = nn.MSELoss()
    direction_loss = nn.BCELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_direction_acc': []
    }

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        # Shuffle indices
        indices = torch.randperm(n_train)

        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i + batch_size]

            # Get batch
            batch_X = {
                k: v[batch_idx].to(device)
                for k, v in train_tensors.items()
            }
            batch_y = train_y_tensor[batch_idx].to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(batch_X)

            # Compute loss
            # Regression loss for all horizons
            loss = regression_loss(outputs['short_term'], batch_y)
            loss += regression_loss(outputs['medium_term'], batch_y)
            loss += regression_loss(outputs['long_term'], batch_y)

            # Direction loss
            direction_target = (batch_y > 0).float()
            loss += direction_loss(outputs['direction'], direction_target)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        correct_directions = 0

        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                batch_X = {
                    k: v[i:i + batch_size].to(device)
                    for k, v in val_tensors.items()
                }
                batch_y = val_y_tensor[i:i + batch_size].to(device)

                outputs = model(batch_X)

                # Compute loss
                loss = regression_loss(outputs['short_term'], batch_y)
                loss += regression_loss(outputs['medium_term'], batch_y)
                loss += regression_loss(outputs['long_term'], batch_y)

                direction_target = (batch_y > 0).float()
                loss += direction_loss(outputs['direction'], direction_target)

                val_losses.append(loss.item())

                # Direction accuracy
                pred_direction = (outputs['direction'] > 0.5).float()
                correct_directions += (pred_direction == direction_target).sum().item()

        # Record history
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        direction_acc = correct_directions / n_val

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_direction_acc'].append(direction_acc)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {avg_val_loss:.6f}")
            print(f"  Direction Acc: {direction_acc:.4f}")

    return history
```

### 04: Multi-Horizon Prediction

```python
# python/04_prediction.py

import torch
import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt

def multi_horizon_predict(
    model: torch.nn.Module,
    data: Dict[str, np.ndarray],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Make predictions for multiple horizons.

    Args:
        model: Trained MultiScaleAttention model
        data: Dictionary of input arrays by scale
        device: Device to use

    Returns:
        Dictionary with predictions for each horizon
    """
    model.eval()
    model = model.to(device)

    # Convert to tensors
    inputs = {
        k: torch.FloatTensor(v).to(device)
        for k, v in data.items()
    }

    with torch.no_grad():
        outputs = model(inputs, return_attention=True)

    return {
        'short_term': outputs['short_term'].cpu().numpy(),
        'medium_term': outputs['medium_term'].cpu().numpy(),
        'long_term': outputs['long_term'].cpu().numpy(),
        'direction': outputs['direction'].cpu().numpy(),
        'attention': outputs['attention'].cpu().numpy()
    }


def visualize_scale_importance(
    model: torch.nn.Module,
    data: Dict[str, np.ndarray],
    scale_names: List[str]
) -> None:
    """
    Visualize the importance of different time scales.
    """
    model.eval()

    # Get fusion weights
    weights = torch.softmax(
        model.cross_scale_fusion.scale_weights, dim=0
    ).detach().cpu().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(scale_names, weights, color='steelblue')
    ax.set_ylabel('Scale Importance')
    ax.set_xlabel('Time Scale')
    ax.set_title('Learned Scale Importance Weights')

    # Add value labels
    for bar, weight in zip(bars, weights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{weight:.3f}',
            ha='center', va='bottom'
        )

    plt.tight_layout()
    plt.savefig('scale_importance.png', dpi=150)
    plt.close()


def visualize_attention_across_scales(
    attention: np.ndarray,
    scale_names: List[str]
) -> None:
    """
    Visualize cross-scale attention patterns.

    Args:
        attention: Attention weights [batch, n_scales, total_seq_len]
        scale_names: Names of scales
    """
    # Average over batch
    avg_attention = attention.mean(axis=0)  # [n_scales, total_seq_len]

    fig, axes = plt.subplots(len(scale_names), 1, figsize=(12, 3 * len(scale_names)))

    for i, (ax, name) in enumerate(zip(axes, scale_names)):
        ax.plot(avg_attention[i], color='steelblue', alpha=0.7)
        ax.fill_between(
            range(len(avg_attention[i])),
            avg_attention[i],
            alpha=0.3
        )
        ax.set_title(f'Attention from {name}')
        ax.set_xlabel('Position (concatenated scales)')
        ax.set_ylabel('Attention Weight')

    plt.tight_layout()
    plt.savefig('cross_scale_attention.png', dpi=150)
    plt.close()
```

### 05: Strategy Backtesting

```python
# python/05_backtest.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    equity_curve: pd.Series


def backtest_multi_scale_strategy(
    predictions: Dict[str, np.ndarray],
    prices: pd.DataFrame,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    confidence_threshold: float = 0.6,
    position_sizing: str = 'confidence'
) -> BacktestResult:
    """
    Backtest a multi-scale attention strategy.

    Args:
        predictions: Model predictions
        prices: DataFrame with OHLCV data
        initial_capital: Starting capital
        transaction_cost: Cost per transaction (fraction)
        confidence_threshold: Minimum direction confidence to trade
        position_sizing: 'fixed', 'confidence', or 'kelly'

    Returns:
        BacktestResult with performance metrics
    """
    n_samples = len(predictions['direction'])

    capital = initial_capital
    position = 0.0  # -1 to 1
    equity = [capital]
    returns = []
    trades = []

    for i in range(n_samples - 1):
        # Get signals
        direction_prob = predictions['direction'][i, 0]
        short_pred = predictions['short_term'][i, 0]

        # Determine signal strength
        confidence = abs(direction_prob - 0.5) * 2  # 0 to 1

        # Determine position
        if confidence > confidence_threshold:
            if direction_prob > 0.5:
                target_position = confidence if position_sizing == 'confidence' else 1.0
            else:
                target_position = -confidence if position_sizing == 'confidence' else -1.0
        else:
            target_position = 0.0

        # Calculate position change
        position_change = target_position - position

        # Transaction costs
        cost = abs(position_change) * transaction_cost * capital

        # Execute trade
        if abs(position_change) > 0.01:
            capital -= cost
            trades.append({
                'index': i,
                'type': 'long' if target_position > 0 else 'short',
                'size': abs(target_position),
                'cost': cost
            })

        position = target_position

        # Calculate return
        if i + 1 < len(prices):
            price_return = (prices['close'].iloc[i + 1] - prices['close'].iloc[i]) / prices['close'].iloc[i]
            pnl = position * price_return * capital
            capital += pnl
            returns.append(pnl / equity[-1])

        equity.append(capital)

    # Calculate metrics
    equity_series = pd.Series(equity)
    returns_series = pd.Series(returns)

    total_return = (capital - initial_capital) / initial_capital

    # Sharpe ratio (annualized, assuming hourly data)
    periods_per_year = 24 * 365
    sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(periods_per_year) if returns_series.std() > 0 else 0

    # Sortino ratio
    downside_returns = returns_series[returns_series < 0]
    sortino = (returns_series.mean() / downside_returns.std()) * np.sqrt(periods_per_year) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0

    # Max drawdown
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    winning_returns = returns_series[returns_series > 0]
    win_rate = len(winning_returns) / len(returns_series) if len(returns_series) > 0 else 0

    # Profit factor
    gross_profit = winning_returns.sum() if len(winning_returns) > 0 else 0
    gross_loss = abs(returns_series[returns_series < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_trades=len(trades),
        equity_curve=equity_series
    )


def compare_scale_strategies(
    predictions: Dict[str, np.ndarray],
    prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare strategies using different horizon predictions.
    """
    results = {}

    for horizon in ['short_term', 'medium_term', 'long_term']:
        # Create predictions using only this horizon
        single_horizon_preds = {
            'direction': (predictions[horizon] > 0).astype(float),
            'short_term': predictions[horizon],
            'medium_term': predictions[horizon],
            'long_term': predictions[horizon]
        }

        result = backtest_multi_scale_strategy(
            single_horizon_preds,
            prices,
            confidence_threshold=0.0
        )

        results[horizon] = {
            'Total Return': f"{result.total_return * 100:.2f}%",
            'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{result.sortino_ratio:.2f}",
            'Max Drawdown': f"{result.max_drawdown * 100:.2f}%",
            'Win Rate': f"{result.win_rate * 100:.1f}%",
            'Trades': result.n_trades
        }

    # Add combined strategy
    result = backtest_multi_scale_strategy(predictions, prices)
    results['multi_scale'] = {
        'Total Return': f"{result.total_return * 100:.2f}%",
        'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
        'Sortino Ratio': f"{result.sortino_ratio:.2f}",
        'Max Drawdown': f"{result.max_drawdown * 100:.2f}%",
        'Win Rate': f"{result.win_rate * 100:.1f}%",
        'Trades': result.n_trades
    }

    return pd.DataFrame(results).T
```

## Rust Implementation

See [rust_multi_scale](rust/) for the complete Rust implementation.

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
│   │   └── scales.rs       # Multi-scale decomposition
│   ├── model/              # Multi-scale attention
│   │   ├── mod.rs
│   │   ├── encoder.rs      # Scale-specific encoders
│   │   ├── attention.rs    # Multi-resolution attention
│   │   ├── fusion.rs       # Cross-scale fusion
│   │   └── network.rs      # Complete model
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
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT --interval 1

# Train model
cargo run --example train -- --epochs 100 --batch-size 32

# Run backtest
cargo run --example backtest -- --model checkpoints/best_model.ot
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── __init__.py
├── model.py               # Multi-scale attention model
├── data.py                # Data loading and preprocessing
├── features.py            # Feature engineering
├── train.py               # Training script
├── backtest.py            # Backtesting utilities
├── requirements.txt       # Dependencies
└── examples/
    ├── example_usage.py   # Complete example
    └── visualization.py   # Visualization utilities
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete example
python examples/example_usage.py

# Or step by step:
# 1. Fetch and prepare data
python data.py --symbol BTCUSDT --days 30

# 2. Train model
python train.py --config configs/default.yaml

# 3. Run backtest
python backtest.py --model checkpoints/best_model.pt
```

## Best Practices

### When to Use Multi-Scale Attention

**Good use cases:**
- Multi-horizon forecasting (predict multiple timeframes)
- Volatile markets with patterns at different scales
- Portfolio rebalancing across timeframes
- Risk management with multi-scale volatility

**Not ideal for:**
- High-frequency trading (too much overhead)
- Single, very short horizon predictions
- Limited data (need enough for all scales)

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `n_scales` | 3-5 | More scales = more complexity |
| `d_model` | 64-256 | Scale with data size |
| `n_heads` | 4-8 | Must divide d_model |
| `n_encoder_layers` | 2-4 | Deeper for longer sequences |
| `lookback` | 24-168 per scale | Depends on data frequency |
| `dropout` | 0.1-0.3 | Higher for small datasets |

### Common Pitfalls

1. **Scale alignment**: Ensure timestamps align across scales
2. **Data leakage**: Don't use future data when downsampling
3. **Imbalanced scales**: Some scales may dominate; use balanced losses
4. **Overfitting**: Multi-scale models have many parameters; use regularization

### Performance Optimization

1. **Memory**: Use gradient checkpointing for long sequences
2. **Speed**: Process scales in parallel when possible
3. **Inference**: Cache intermediate representations
4. **Batch size**: Balance memory vs. throughput

## Resources

### Papers

- [Multi-Scale Temporal Memory for Financial Time Series](https://arxiv.org/abs/2201.08586) — Foundation paper
- [VMD-MSANet: Multi-Scale Attention with VMD](https://www.sciencedirect.com/science/article/abs/pii/S0925231225015267) — VMD integration
- [MSTAN: Multi-Scale Temporal Attention Network](https://www.researchgate.net/publication/398598476_MSTAN_A_multi-scale_temporal_attention_network_for_stock_prediction) — Stock prediction variant
- [Multi-Scale Temporal Neural Network](https://www.ijcai.org/proceedings/2025/0364.pdf) — IJCAI 2025
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer

### Related Chapters

- [Chapter 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — Multi-horizon forecasting
- [Chapter 43: Stockformer Multivariate](../43_stockformer_multivariate) — Cross-asset attention
- [Chapter 46: Temporal Attention Networks](../46_temporal_attention_networks) — Temporal attention
- [Chapter 48: Positional Encoding Timeseries](../48_positional_encoding_timeseries) — Positional encodings

### Implementations

- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) — Time series library
- [Informer](https://github.com/zhouhaoyi/Informer2020) — Efficient transformers
- [vmdpy](https://github.com/vrcarva/vmdpy) — VMD implementation

---

## Difficulty Level

**Advanced**

Prerequisites:
- Transformer architecture and attention mechanisms
- Time series analysis and feature engineering
- Multi-horizon forecasting concepts
- PyTorch/Rust ML frameworks
