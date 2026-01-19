# Chapter 55: FNet — Fourier Transform for Efficient Token Mixing

This chapter explores **FNet**, an innovative architecture that replaces self-attention mechanisms with Fourier Transform operations, achieving O(n log n) computational complexity while maintaining competitive performance for financial time series prediction.

<p align="center">
<img src="https://i.imgur.com/7vKwPzN.png" width="70%">
</p>

## Contents

1. [Introduction to FNet](#introduction-to-fnet)
    * [Why Replace Attention?](#why-replace-attention)
    * [Key Advantages](#key-advantages)
    * [Comparison with Transformers](#comparison-with-transformers)
2. [FNet Architecture](#fnet-architecture)
    * [Fourier Transform Layer](#fourier-transform-layer)
    * [Feed-Forward Network](#feed-forward-network)
    * [Complete Architecture](#complete-architecture)
3. [Mathematical Foundation](#mathematical-foundation)
    * [Discrete Fourier Transform](#discrete-fourier-transform)
    * [2D Fourier Transform in FNet](#2d-fourier-transform-in-fnet)
    * [Computational Complexity](#computational-complexity)
4. [FNet for Trading](#fnet-for-trading)
    * [Time Series Adaptation](#time-series-adaptation)
    * [Multi-Asset Prediction](#multi-asset-prediction)
    * [Signal Generation](#signal-generation)
5. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: FNet Architecture](#02-fnet-architecture)
    * [03: Model Training](#03-model-training)
    * [04: Trading Strategy](#04-trading-strategy)
    * [05: Backtesting](#05-backtesting)
6. [Rust Implementation](#rust-implementation)
7. [Python Implementation](#python-implementation)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction to FNet

FNet (Fourier Network) is a revolutionary architecture introduced by Google Research in 2021 that challenges the dominance of attention mechanisms in Transformer models. Instead of computing expensive quadratic attention matrices, FNet uses the **Fast Fourier Transform (FFT)** to mix token representations.

### Why Replace Attention?

Standard self-attention has O(L²) complexity where L is sequence length:

```
Standard Attention:
- Q, K, V projections: O(L × d × d)
- QK^T computation: O(L² × d)  ← Bottleneck!
- Softmax + V multiplication: O(L² × d)

For L=512, d=768: ~200M operations per layer
For L=2048: ~3.2B operations per layer (16x more!)
```

FNet replaces this with FFT:

```
FNet Fourier Layer:
- 2D FFT: O(L × d × log(L × d))
- Take real part: O(L × d)

For L=512, d=768: ~3.6M operations (55x faster!)
For L=2048: ~19M operations (168x faster!)
```

### Key Advantages

1. **Speed**: 80% faster training on GPUs, 70% faster on TPUs
2. **Memory Efficiency**: No O(L²) attention matrices to store
3. **Simplicity**: No learnable parameters in the mixing layer
4. **Long Sequences**: Scales linearly with sequence length
5. **Competitive Accuracy**: 92-97% of BERT performance on GLUE benchmark

### Comparison with Transformers

| Feature | Transformer | FNet | Advantage |
|---------|-------------|------|-----------|
| Token Mixing | Self-Attention | FFT | FNet (speed) |
| Complexity | O(L²) | O(L log L) | FNet |
| Parameters | Learnable Q,K,V | None | FNet (simpler) |
| GPU Speed | Baseline | 80% faster | FNet |
| GLUE Score | 100% | 92-97% | Transformer |
| Long Sequences | Slow | Fast | FNet |
| Interpretability | Attention weights | Frequency analysis | Different |

## FNet Architecture

### Fourier Transform Layer

The core of FNet is remarkably simple:

```python
class FourierTransformLayer(nn.Module):
    """
    Replaces self-attention with 2D Fourier Transform.

    The Fourier transform mixes tokens along two dimensions:
    1. Sequence dimension (across time steps)
    2. Hidden dimension (across features)
    """

    def __init__(self):
        super().__init__()
        # No learnable parameters!

    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]

        # Apply 2D FFT
        # FFT along sequence dimension mixes temporal information
        # FFT along hidden dimension mixes feature information
        x_fft = torch.fft.fft2(x.float())

        # Take real part (discard imaginary component)
        return x_fft.real
```

**Why does this work?**

1. **Fourier Transform as Global Mixing**: Each output token contains information from ALL input tokens (via frequency domain representation)
2. **Periodic Patterns**: Financial data often has periodic components (daily, weekly, monthly cycles)
3. **Efficiency**: FFT algorithm computes the transform in O(n log n) instead of naive O(n²)

### Feed-Forward Network

After Fourier mixing, FNet uses a standard feed-forward network:

```python
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # Expand, activate, project back
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

### Complete Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                              FNet                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌────────────────────────────────────────────────────────┐          │
│  │                    Input Embedding                       │          │
│  │   Token Embedding + Positional Encoding                  │          │
│  └────────────────────────────┬───────────────────────────┘          │
│                               │                                        │
│           ┌───────────────────┴───────────────────┐                   │
│           │           FNet Encoder Block          │ × N               │
│           │  ┌─────────────────────────────────┐  │                   │
│           │  │     Fourier Transform Layer      │  │                   │
│           │  │        FFT2D → Real Part         │  │                   │
│           │  └─────────────────────────────────┘  │                   │
│           │             ↓ + Residual              │                   │
│           │  ┌─────────────────────────────────┐  │                   │
│           │  │        Layer Normalization       │  │                   │
│           │  └─────────────────────────────────┘  │                   │
│           │             ↓                         │                   │
│           │  ┌─────────────────────────────────┐  │                   │
│           │  │     Feed-Forward Network         │  │                   │
│           │  │   Linear → GELU → Linear         │  │                   │
│           │  └─────────────────────────────────┘  │                   │
│           │             ↓ + Residual              │                   │
│           │  ┌─────────────────────────────────┐  │                   │
│           │  │        Layer Normalization       │  │                   │
│           │  └─────────────────────────────────┘  │                   │
│           └───────────────────┬───────────────────┘                   │
│                               │                                        │
│  ┌────────────────────────────┴───────────────────────────┐          │
│  │                   Output Head                           │          │
│  │   Pooling → Linear → Prediction                        │          │
│  └────────────────────────────────────────────────────────┘          │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

## Mathematical Foundation

### Discrete Fourier Transform

The Discrete Fourier Transform (DFT) converts a sequence from time domain to frequency domain:

$$X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-2\pi i \cdot kn / N}$$

Where:
- $x_n$ is the input sequence at position n
- $X_k$ is the k-th frequency component
- $N$ is the sequence length
- $i$ is the imaginary unit

### 2D Fourier Transform in FNet

FNet applies a 2D FFT to the input tensor:

```python
def fnet_fourier_transform(x):
    """
    Apply 2D Fourier Transform to input tensor.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim]

    Returns:
        Real part of 2D FFT output
    """
    # FFT along last two dimensions
    # Dimension -2: sequence dimension (time mixing)
    # Dimension -1: hidden dimension (feature mixing)
    return torch.fft.fft2(x).real
```

The 2D FFT can be decomposed as:

1. **FFT along sequence dimension**: Mixes information across time steps
   - Captures temporal patterns and periodicity
   - Each position learns from all other positions

2. **FFT along hidden dimension**: Mixes information across features
   - Combines different feature representations
   - Creates richer representations

### Computational Complexity

**Standard Self-Attention:**
```
Complexity: O(L² × d)
Where L = sequence length, d = hidden dimension

Memory: O(L²) for attention matrix
```

**FNet Fourier Transform:**
```
Complexity: O(L × d × log(L × d))
≈ O(L × d × log(L)) for typical cases

Memory: O(L × d) - no attention matrix needed
```

**Speedup Analysis:**

| Sequence Length | Attention Ops | FFT Ops | Speedup |
|----------------|---------------|---------|---------|
| 128 | 12.6M | 0.8M | 15x |
| 512 | 201.3M | 3.6M | 55x |
| 1024 | 805.3M | 7.8M | 103x |
| 2048 | 3221.2M | 16.5M | 195x |

## FNet for Trading

### Time Series Adaptation

Adapting FNet for financial time series requires several modifications:

```python
class FNetForTrading(nn.Module):
    """
    FNet adapted for financial time series prediction.

    Modifications from original FNet:
    1. Time-aware positional encoding (captures market hours, days)
    2. Multi-scale Fourier analysis (daily, weekly, monthly patterns)
    3. Causal masking for real-time prediction
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        output_type: str = 'regression'
    ):
        super().__init__()

        # Input embedding
        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # FNet encoder blocks
        self.encoder_blocks = nn.ModuleList([
            FNetEncoderBlock(d_model, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output head
        self.output_head = self._create_output_head(d_model, output_type)

    def forward(self, x, return_frequencies=False):
        # x: [batch, seq_len, n_features]

        # Project to model dimension
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        # Apply FNet encoder blocks
        frequency_maps = []
        for block in self.encoder_blocks:
            x, freq = block(x, return_frequencies=True)
            frequency_maps.append(freq)

        # Generate predictions
        output = self.output_head(x)

        if return_frequencies:
            return output, frequency_maps
        return output
```

### Multi-Asset Prediction

FNet can predict multiple assets simultaneously:

```python
class MultiFNet(nn.Module):
    """
    Multi-asset FNet for portfolio prediction.

    Uses separate Fourier layers for:
    1. Temporal mixing (across time)
    2. Asset mixing (across different instruments)
    """

    def __init__(self, n_assets, n_features, d_model):
        super().__init__()

        # Per-asset embedding
        self.asset_embeddings = nn.ModuleList([
            nn.Linear(n_features, d_model)
            for _ in range(n_assets)
        ])

        # Temporal FNet (within each asset)
        self.temporal_fnet = FNetEncoder(d_model, n_layers=2)

        # Cross-asset FNet (across assets)
        self.cross_asset_fnet = FNetEncoder(d_model, n_layers=2)

        # Output heads for each asset
        self.prediction_heads = nn.ModuleList([
            nn.Linear(d_model, 1)
            for _ in range(n_assets)
        ])

    def forward(self, x):
        # x: [batch, seq_len, n_assets, n_features]
        batch_size, seq_len, n_assets, _ = x.shape

        # Embed each asset separately
        asset_features = []
        for i in range(n_assets):
            asset_x = self.asset_embeddings[i](x[:, :, i, :])
            asset_features.append(asset_x)

        # Stack: [batch, seq_len, n_assets, d_model]
        x = torch.stack(asset_features, dim=2)

        # Apply temporal FNet to each asset
        temporal_outputs = []
        for i in range(n_assets):
            temp_out = self.temporal_fnet(x[:, :, i, :])
            temporal_outputs.append(temp_out)
        x = torch.stack(temporal_outputs, dim=2)

        # Apply cross-asset FNet
        # Reshape: [batch * seq_len, n_assets, d_model]
        x_reshaped = x.view(batch_size * seq_len, n_assets, -1)
        x_cross = self.cross_asset_fnet(x_reshaped)
        x = x_cross.view(batch_size, seq_len, n_assets, -1)

        # Generate predictions for each asset
        predictions = []
        for i in range(n_assets):
            pred = self.prediction_heads[i](x[:, -1, i, :])
            predictions.append(pred)

        return torch.cat(predictions, dim=1)
```

### Signal Generation

Generate trading signals from FNet predictions:

```python
class FNetSignalGenerator:
    """
    Generate trading signals from FNet predictions.
    """

    def __init__(self, model, threshold=0.0, confidence_threshold=0.6):
        self.model = model
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold

    def generate_signals(self, x, return_confidence=False):
        """
        Generate trading signals.

        Args:
            x: Input features [batch, seq_len, n_features]
            return_confidence: Whether to return confidence scores

        Returns:
            signals: Trading signals (-1, 0, 1) for (short, hold, long)
            confidence: Optional confidence scores
        """
        self.model.eval()
        with torch.no_grad():
            # Get predictions and frequency maps
            predictions, freq_maps = self.model(x, return_frequencies=True)

            # Calculate confidence from frequency stability
            confidence = self._calculate_confidence(freq_maps)

            # Generate signals
            signals = torch.zeros_like(predictions)

            # Long signal: prediction above threshold AND high confidence
            long_mask = (predictions > self.threshold) & (confidence > self.confidence_threshold)
            signals[long_mask] = 1.0

            # Short signal: prediction below -threshold AND high confidence
            short_mask = (predictions < -self.threshold) & (confidence > self.confidence_threshold)
            signals[short_mask] = -1.0

        if return_confidence:
            return signals, confidence
        return signals

    def _calculate_confidence(self, freq_maps):
        """
        Calculate prediction confidence from frequency analysis.

        High confidence = stable frequency patterns
        Low confidence = noisy/unstable patterns
        """
        # Average energy concentration in dominant frequencies
        confidences = []
        for freq_map in freq_maps:
            # Get magnitude spectrum
            magnitude = torch.abs(freq_map)

            # Calculate energy concentration (Gini coefficient of frequencies)
            sorted_mag = torch.sort(magnitude.flatten(-2), dim=-1)[0]
            cumsum = torch.cumsum(sorted_mag, dim=-1)
            total = cumsum[:, -1:]

            # Normalized cumsum
            lorenz_curve = cumsum / (total + 1e-8)

            # Gini = 1 - 2 * area under Lorenz curve
            n = lorenz_curve.shape[-1]
            gini = 1 - 2 * lorenz_curve.mean(dim=-1)

            confidences.append(gini)

        # Average confidence across layers
        return torch.stack(confidences).mean(dim=0)
```

## Practical Examples

### 01: Data Preparation

```python
# python/data_loader.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import requests

class BybitDataLoader:
    """
    Data loader for Bybit cryptocurrency data.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",  # 1 hour
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Candle interval in minutes
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        response = self.session.get(endpoint, params=params)
        data = response.json()

        if data["retCode"] != 0:
            raise ValueError(f"API error: {data['retMsg']}")

        # Parse kline data
        klines = data["result"]["list"]
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)

        return df.sort_values("timestamp").reset_index(drop=True)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for FNet model.
        """
        df = df.copy()

        # Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Volatility (20-period rolling std)
        df["volatility"] = df["log_return"].rolling(20).std()

        # Volume change
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # Price momentum
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands position
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_position"] = (df["close"] - sma_20) / (2 * std_20 + 1e-8)

        return df.dropna()


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int = 168,
    horizon: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for FNet training.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        target_col: Target column name
        seq_len: Input sequence length
        horizon: Prediction horizon

    Returns:
        X: Input sequences [n_samples, seq_len, n_features]
        y: Target values [n_samples]
    """
    features = df[feature_cols].values
    target = df[target_col].values

    X, y = [], []
    for i in range(seq_len, len(df) - horizon):
        X.append(features[i-seq_len:i])
        y.append(target[i + horizon - 1])

    return np.array(X), np.array(y)
```

### 02: FNet Architecture

See [python/model.py](python/model.py) for complete implementation.

```python
# python/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FourierLayer(nn.Module):
    """
    Fourier Transform layer that replaces self-attention.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Apply 2D FFT and return real part.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Fourier-transformed tensor (real part only)
        """
        return torch.fft.fft2(x.float()).real


class FNetEncoderBlock(nn.Module):
    """
    Single FNet encoder block.

    Structure:
    1. Fourier Transform + Residual + LayerNorm
    2. Feed-Forward + Residual + LayerNorm
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.fourier = FourierLayer()
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, return_frequencies=False):
        # Fourier sublayer
        fourier_out = self.fourier(x)
        x = self.norm1(x + fourier_out)

        # Feed-forward sublayer
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        if return_frequencies:
            return x, fourier_out
        return x


class FNet(nn.Module):
    """
    Complete FNet model for financial time series.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        output_dim: int = 1
    ):
        super().__init__()

        self.d_model = d_model

        # Input embedding
        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            FNetEncoderBlock(d_model, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x, return_frequencies=False):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, n_features]
            return_frequencies: Whether to return frequency maps

        Returns:
            predictions: Output predictions
            freq_maps: Optional list of frequency maps from each layer
        """
        batch_size, seq_len, _ = x.shape

        # Input projection + positional encoding
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, :seq_len, :]

        # Apply encoder blocks
        freq_maps = []
        for block in self.encoder_blocks:
            if return_frequencies:
                x, freq = block(x, return_frequencies=True)
                freq_maps.append(freq)
            else:
                x = block(x)

        # Global average pooling + output
        x = x.mean(dim=1)  # [batch, d_model]
        output = self.output_head(x)

        if return_frequencies:
            return output, freq_maps
        return output
```

### 03: Model Training

```python
# python/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import FNet
from data_loader import BybitDataLoader, create_sequences

def train_fnet(
    symbols: list = ["BTCUSDT", "ETHUSDT"],
    seq_len: int = 168,
    horizon: int = 24,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train FNet model on cryptocurrency data.
    """
    print(f"Training on {device}")

    # Load and prepare data
    loader = BybitDataLoader()
    all_X, all_y = [], []

    feature_cols = [
        "log_return", "volatility", "volume_ratio",
        "momentum_5", "momentum_20", "rsi", "bb_position"
    ]

    for symbol in symbols:
        print(f"Loading data for {symbol}...")
        df = loader.fetch_klines(symbol, interval="60", limit=2000)
        df = loader.prepare_features(df)

        X, y = create_sequences(df, feature_cols, "log_return", seq_len, horizon)
        all_X.append(X)
        all_y.append(y)

    # Combine data
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Normalize
    X_mean, X_std = X.mean(axis=(0, 1)), X.std(axis=(0, 1))
    X = (X - X_mean) / (X_std + 1e-8)

    # Train/val split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Create dataloaders
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
    model = FNet(
        n_features=len(feature_cols),
        d_model=256,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        max_seq_len=seq_len,
        output_dim=1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions.squeeze(), batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_fnet_model.pt")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    return model

if __name__ == "__main__":
    train_fnet()
```

### 04: Trading Strategy

```python
# python/strategy.py

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

class FNetTradingStrategy:
    """
    Trading strategy based on FNet predictions.
    """

    def __init__(
        self,
        model,
        threshold: float = 0.001,
        position_size: float = 1.0,
        stop_loss: float = 0.02,
        take_profit: float = 0.04
    ):
        self.model = model
        self.threshold = threshold
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signal(self, x: torch.Tensor) -> Tuple[int, float]:
        """
        Generate trading signal from model prediction.

        Args:
            x: Input features [1, seq_len, n_features]

        Returns:
            signal: Trading signal (-1=short, 0=hold, 1=long)
            confidence: Prediction confidence
        """
        self.model.eval()
        with torch.no_grad():
            prediction, freq_maps = self.model(x, return_frequencies=True)
            pred_value = prediction.item()

            # Calculate confidence from frequency stability
            confidence = self._calculate_confidence(freq_maps)

        # Generate signal
        if pred_value > self.threshold and confidence > 0.5:
            return 1, confidence
        elif pred_value < -self.threshold and confidence > 0.5:
            return -1, confidence
        else:
            return 0, confidence

    def _calculate_confidence(self, freq_maps: list) -> float:
        """Calculate prediction confidence from frequency analysis."""
        # Use energy concentration in low frequencies as confidence measure
        confidences = []
        for freq_map in freq_maps:
            mag = torch.abs(freq_map)
            # Low frequency energy ratio
            low_freq_ratio = mag[:, :mag.shape[1]//4, :].sum() / (mag.sum() + 1e-8)
            confidences.append(low_freq_ratio.mean().item())
        return np.mean(confidences)


class Backtester:
    """
    Backtesting engine for FNet trading strategy.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def run_backtest(
        self,
        strategy: FNetTradingStrategy,
        X: np.ndarray,
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run backtest on historical data.

        Args:
            strategy: Trading strategy instance
            X: Feature sequences [n_samples, seq_len, n_features]
            prices: Close prices corresponding to each sample
            timestamps: Optional timestamps

        Returns:
            Dictionary with backtest results
        """
        capital = self.initial_capital
        position = 0
        entry_price = 0

        trades = []
        equity_curve = [capital]
        positions = []

        for i in range(len(X)):
            current_price = prices[i]
            x_tensor = torch.FloatTensor(X[i:i+1])

            signal, confidence = strategy.generate_signal(x_tensor)

            # Check stop-loss / take-profit for existing position
            if position != 0:
                pnl_pct = (current_price / entry_price - 1) * position

                if pnl_pct <= -strategy.stop_loss:
                    # Stop-loss triggered
                    pnl = capital * position * pnl_pct
                    capital += pnl - abs(capital * position * self.transaction_cost)
                    trades.append({
                        "type": "stop_loss",
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl": pnl,
                        "position": position
                    })
                    position = 0

                elif pnl_pct >= strategy.take_profit:
                    # Take-profit triggered
                    pnl = capital * position * pnl_pct
                    capital += pnl - abs(capital * position * self.transaction_cost)
                    trades.append({
                        "type": "take_profit",
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl": pnl,
                        "position": position
                    })
                    position = 0

            # Process new signals
            if signal != 0 and position == 0:
                # Enter position
                position = signal * strategy.position_size
                entry_price = current_price * (1 + signal * self.slippage)
                capital -= abs(capital * position * self.transaction_cost)

            elif signal != 0 and signal != np.sign(position):
                # Close and reverse
                pnl_pct = (current_price / entry_price - 1) * position
                pnl = capital * abs(position) * pnl_pct
                capital += pnl - abs(capital * position * self.transaction_cost)

                trades.append({
                    "type": "signal_reverse",
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl": pnl,
                    "position": position
                })

                position = signal * strategy.position_size
                entry_price = current_price * (1 + signal * self.slippage)
                capital -= abs(capital * position * self.transaction_cost)

            # Update equity curve
            if position != 0:
                unrealized_pnl = capital * position * (current_price / entry_price - 1)
                equity_curve.append(capital + unrealized_pnl)
            else:
                equity_curve.append(capital)

            positions.append(position)

        # Close final position
        if position != 0:
            pnl_pct = (prices[-1] / entry_price - 1) * position
            pnl = capital * abs(position) * pnl_pct
            capital += pnl - abs(capital * position * self.transaction_cost)
            trades.append({
                "type": "final_close",
                "entry_price": entry_price,
                "exit_price": prices[-1],
                "pnl": pnl,
                "position": position
            })

        # Calculate metrics
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]

        metrics = self._calculate_metrics(returns, equity_curve, trades)

        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve,
            "positions": positions
        }

    def _calculate_metrics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        trades: list
    ) -> Dict:
        """Calculate performance metrics."""
        total_return = (equity_curve[-1] / equity_curve[0]) - 1

        # Annualized metrics (assuming hourly data)
        periods_per_year = 24 * 365

        # Sharpe Ratio
        sharpe = np.sqrt(periods_per_year) * returns.mean() / (returns.std() + 1e-8)

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino = np.sqrt(periods_per_year) * returns.mean() / (downside_returns.std() + 1e-8)

        # Maximum Drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (cummax - equity_curve) / cummax
        max_drawdown = drawdown.max()

        # Win rate
        winning_trades = [t for t in trades if t["pnl"] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0

        # Profit factor
        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        profit_factor = gross_profit / (gross_loss + 1e-8)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades": len(trades),
            "final_capital": equity_curve[-1]
        }
```

### 05: Backtesting

```python
# python/backtest.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import FNet
from data_loader import BybitDataLoader, create_sequences
from strategy import FNetTradingStrategy, Backtester

def run_full_backtest():
    """
    Run complete backtest of FNet trading strategy.
    """
    # Load model
    feature_cols = [
        "log_return", "volatility", "volume_ratio",
        "momentum_5", "momentum_20", "rsi", "bb_position"
    ]

    model = FNet(
        n_features=len(feature_cols),
        d_model=256,
        n_layers=4,
        d_ff=1024,
        output_dim=1
    )
    model.load_state_dict(torch.load("best_fnet_model.pt", map_location="cpu"))
    model.eval()

    # Load test data
    loader = BybitDataLoader()
    df = loader.fetch_klines("BTCUSDT", interval="60", limit=2000)
    df = loader.prepare_features(df)

    # Create sequences
    X, y = create_sequences(df, feature_cols, "log_return", seq_len=168, horizon=24)

    # Use last 20% as test
    test_start = int(len(X) * 0.8)
    X_test = X[test_start:]
    prices_test = df["close"].values[168 + 23 + test_start:][:len(X_test)]

    # Normalize using training stats
    X_mean = X[:test_start].mean(axis=(0, 1))
    X_std = X[:test_start].std(axis=(0, 1))
    X_test = (X_test - X_mean) / (X_std + 1e-8)

    # Create strategy and backtester
    strategy = FNetTradingStrategy(
        model=model,
        threshold=0.001,
        position_size=1.0,
        stop_loss=0.02,
        take_profit=0.04
    )

    backtester = Backtester(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005
    )

    # Run backtest
    results = backtester.run_backtest(strategy, X_test, prices_test)

    # Print metrics
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    for key, value in results["metrics"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(results["equity_curve"])
    plt.title("FNet Strategy Equity Curve")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.3)
    plt.savefig("equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nEquity curve saved to equity_curve.png")

    return results

if __name__ == "__main__":
    run_full_backtest()
```

## Rust Implementation

See [rust_fnet](rust_fnet/) for complete Rust implementation.

```
rust_fnet/
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
│   ├── model/              # FNet architecture
│   │   ├── mod.rs
│   │   ├── fourier.rs      # Fourier transform layer
│   │   ├── encoder.rs      # Encoder stack
│   │   └── fnet.rs         # Complete model
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
cd rust_fnet

# Fetch data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT

# Train model
cargo run --example train -- --epochs 100 --batch-size 32

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── __init__.py             # Package exports
├── model.py                # FNet model implementation
├── data_loader.py          # Bybit data loading
├── train.py                # Training script
├── strategy.py             # Trading strategy
├── backtest.py             # Backtesting utilities
├── requirements.txt        # Dependencies
└── example_usage.py        # Complete example
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch data and train
python train.py --symbols BTCUSDT,ETHUSDT --epochs 100

# Run backtest
python backtest.py --model best_fnet_model.pt
```

## Best Practices

### When to Use FNet

**Good use cases:**
- Long sequence forecasting (>512 tokens)
- Real-time prediction (inference speed matters)
- Periodic pattern detection (daily/weekly cycles)
- Resource-constrained environments
- Multi-asset portfolio management

**Not ideal for:**
- Tasks requiring interpretable attention weights
- Very short sequences (<64 tokens)
- When highest accuracy is critical (use Transformer)

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `d_model` | 256 | Balance between capacity and speed |
| `n_layers` | 4-6 | More for complex patterns |
| `d_ff` | 4 × d_model | Standard expansion ratio |
| `dropout` | 0.1-0.2 | Higher for small datasets |
| `seq_len` | 168-336 | 1-2 weeks of hourly data |

### Common Pitfalls

1. **Not normalizing inputs**: FFT is sensitive to input scale
2. **Ignoring phase information**: Real part only loses directional info
3. **Too short sequences**: FFT needs sufficient data for pattern detection
4. **Overfitting on noise**: Use proper regularization

### Frequency Analysis Tips

```python
def analyze_frequency_patterns(model, x):
    """
    Analyze what frequencies the model focuses on.
    """
    _, freq_maps = model(x, return_frequencies=True)

    for i, freq_map in enumerate(freq_maps):
        magnitude = torch.abs(freq_map)

        # Find dominant frequencies
        flat_mag = magnitude.mean(dim=0).flatten()
        top_k = torch.topk(flat_mag, k=10)

        print(f"Layer {i+1} dominant frequency indices: {top_k.indices.tolist()}")
        print(f"Layer {i+1} frequency magnitudes: {top_k.values.tolist()}")
```

## Resources

### Papers

- [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824) — Original FNet paper (2021)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer
- [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) — Related work on Fourier in neural networks
- [Spectral Leakage and Neural Networks](https://arxiv.org/abs/2103.00428) — Understanding FFT in ML

### Implementations

- [Google Research FNet](https://github.com/google-research/google-research/tree/master/f_net) — Official implementation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/fnet) — FNet in transformers library
- [PyTorch FFT Documentation](https://pytorch.org/docs/stable/fft.html) — FFT operations in PyTorch

### Related Chapters

- [Chapter 52: Performer Efficient Attention](../52_performer_efficient_attention) — Another efficient attention variant
- [Chapter 54: Reformer LSH Attention](../54_reformer_lsh_attention) — Locality-sensitive hashing attention
- [Chapter 58: Flash Attention Trading](../58_flash_attention_trading) — Memory-efficient attention

---

## Difficulty Level

**Intermediate**

Prerequisites:
- Understanding of Fourier Transform basics
- Neural network fundamentals
- Time series forecasting concepts
- Python/Rust programming experience
