# Chapter 47: Cross-Attention for Multi-Asset Trading

This chapter explores **Cross-Attention mechanisms** for modeling relationships between multiple financial assets simultaneously. Unlike traditional single-asset forecasting, cross-attention enables the model to capture inter-asset dependencies, correlations, and lead-lag relationships that are crucial for portfolio management and multi-asset trading strategies.

<p align="center">
<img src="https://i.imgur.com/YqN3rZm.png" width="70%">
</p>

## Contents

1. [Introduction to Cross-Attention](#introduction-to-cross-attention)
    * [Why Cross-Attention for Multi-Asset Trading?](#why-cross-attention-for-multi-asset-trading)
    * [Key Advantages](#key-advantages)
    * [Comparison with Other Approaches](#comparison-with-other-approaches)
2. [Cross-Attention Architecture](#cross-attention-architecture)
    * [Query-Key-Value Mechanism](#query-key-value-mechanism)
    * [Multi-Head Cross-Attention](#multi-head-cross-attention)
    * [Temporal Cross-Attention](#temporal-cross-attention)
    * [Hierarchical Cross-Attention](#hierarchical-cross-attention)
3. [Mathematical Foundation](#mathematical-foundation)
    * [Attention Score Computation](#attention-score-computation)
    * [Cross-Attention vs Self-Attention](#cross-attention-vs-self-attention)
    * [Scaled Dot-Product Attention](#scaled-dot-product-attention)
4. [Data Representation](#data-representation)
    * [Multi-Asset Feature Engineering](#multi-asset-feature-engineering)
    * [Data from Stock Markets](#data-from-stock-markets)
    * [Data from Cryptocurrency Markets (Bybit)](#data-from-cryptocurrency-markets-bybit)
5. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: Cross-Attention Model](#02-cross-attention-model)
    * [03: Model Training](#03-model-training)
    * [04: Multi-Asset Prediction](#04-multi-asset-prediction)
    * [05: Portfolio Backtesting](#05-portfolio-backtesting)
6. [Rust Implementation](#rust-implementation)
7. [Python Implementation](#python-implementation)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction to Cross-Attention

Cross-attention is an attention mechanism where queries come from one sequence (or asset) while keys and values come from another. In multi-asset trading, this allows each asset to "attend to" other assets, learning which assets provide predictive information for others.

### Why Cross-Attention for Multi-Asset Trading?

Traditional approaches treat each asset independently:

```
Asset A → Model_A → Prediction_A
Asset B → Model_B → Prediction_B
Asset C → Model_C → Prediction_C
```

Cross-attention models all assets jointly:

```
┌─────────────────────────────────────────────────┐
│           Cross-Attention Network                │
│                                                  │
│   Asset A ←→ Asset B ←→ Asset C                 │
│      ↑           ↑           ↑                   │
│      └───────────┴───────────┘                   │
│         Bidirectional attention                  │
│                                                  │
│                    ↓                             │
│   [Prediction_A, Prediction_B, Prediction_C]    │
└─────────────────────────────────────────────────┘
```

**Key insight**: Financial markets are interconnected. When Bitcoin moves, Ethereum often follows. When oil prices rise, airline stocks typically fall. Cross-attention explicitly models these dependencies.

### Key Advantages

1. **Inter-Asset Dependency Learning**
   - Captures correlations between different asset classes
   - Models lead-lag relationships (e.g., BTC leading altcoins)
   - Learns time-varying relationships

2. **Attention-Based Interpretability**
   - Attention weights reveal which assets influence predictions
   - Visualize cross-asset information flow
   - Identify market leaders and followers

3. **Portfolio-Level Optimization**
   - Optimize Sharpe ratio directly instead of individual predictions
   - Learn optimal asset allocation weights
   - Account for diversification benefits

4. **Adaptive Regime Detection**
   - Attention patterns change during different market regimes
   - Detect correlation breakdowns during crises
   - Adapt to structural market changes

### Comparison with Other Approaches

| Feature | Single-Asset LSTM | Multi-Asset RNN | Self-Attention | Cross-Attention |
|---------|-------------------|-----------------|----------------|-----------------|
| Inter-asset modeling | ✗ | Limited | Implicit | ✓ Explicit |
| Bidirectional influence | ✗ | ✗ | ✓ | ✓ |
| Asymmetric relationships | ✗ | ✗ | ✗ | ✓ |
| Lead-lag detection | ✗ | ✗ | Limited | ✓ |
| Interpretable | ✗ | ✗ | ✓ | ✓ |
| Portfolio optimization | ✗ | ✗ | ✗ | ✓ |

## Cross-Attention Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    CROSS-ATTENTION MULTI-ASSET MODEL                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │ BTC     │  │ ETH     │  │ SOL     │  │ AAPL    │                     │
│  │ (Query) │  │ (Query) │  │ (Query) │  │ (Query) │                     │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                     │
│       │            │            │            │                            │
│       ▼            ▼            ▼            ▼                            │
│  ┌──────────────────────────────────────────────────┐                    │
│  │             Token Embedding Layer                 │                    │
│  │    (1D-CNN or Linear projection per asset)       │                    │
│  └───────────────────────┬──────────────────────────┘                    │
│                          │                                                │
│                          ▼                                                │
│  ┌──────────────────────────────────────────────────┐                    │
│  │          Temporal Self-Attention                  │                    │
│  │    (Model temporal patterns within each asset)    │                    │
│  └───────────────────────┬──────────────────────────┘                    │
│                          │                                                │
│                          ▼                                                │
│  ┌──────────────────────────────────────────────────┐                    │
│  │         Cross-Asset Cross-Attention               │                    │
│  │                                                   │                    │
│  │   Q(BTC) attends to K,V(ETH), K,V(SOL), K,V(AAPL)│                    │
│  │   Q(ETH) attends to K,V(BTC), K,V(SOL), K,V(AAPL)│                    │
│  │   ...                                            │                    │
│  │                                                   │                    │
│  │   Learns: "BTC leads ETH with weight 0.7"        │                    │
│  │           "ETH leads SOL with weight 0.5"        │                    │
│  └───────────────────────┬──────────────────────────┘                    │
│                          │                                                │
│                          ▼                                                │
│  ┌──────────────────────────────────────────────────┐                    │
│  │         Encoder Stack (N layers)                  │                    │
│  │    Temporal Attention + Cross-Asset Attention     │                    │
│  └───────────────────────┬──────────────────────────┘                    │
│                          │                                                │
│                          ▼                                                │
│  ┌──────────────────────────────────────────────────┐                    │
│  │              Prediction Heads                     │                    │
│  │   • Returns prediction (regression)               │                    │
│  │   • Direction prediction (classification)         │                    │
│  │   • Portfolio weights (softmax/tanh)              │                    │
│  └──────────────────────────────────────────────────┘                    │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Query-Key-Value Mechanism

In cross-attention, one asset generates queries while other assets provide keys and values:

```python
class CrossAssetAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_assets):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        # Separate projections for each role
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query_asset, key_value_assets):
        """
        Args:
            query_asset: [batch, seq_len, d_model] - Asset to predict
            key_value_assets: [batch, n_other_assets, seq_len, d_model]

        Returns:
            context: [batch, seq_len, d_model] - Attended representation
            attention: [batch, n_heads, seq_len, n_other_assets]
        """
        batch, seq_len, d_model = query_asset.shape
        n_other = key_value_assets.shape[1]

        # Project queries from target asset
        Q = self.query_proj(query_asset)

        # Project keys and values from other assets
        K = self.key_proj(key_value_assets.view(-1, seq_len, d_model))
        V = self.value_proj(key_value_assets.view(-1, seq_len, d_model))

        # Reshape for multi-head attention
        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, n_other, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch, n_other, seq_len, self.n_heads, self.head_dim)

        # Cross-attention: each query position attends to all positions of all other assets
        # Simplified: attend to last timestep of other assets
        K_last = K[:, :, -1, :, :].transpose(1, 2)  # [batch, n_heads, n_other, head_dim]
        V_last = V[:, :, -1, :, :].transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K_last.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)

        # Weighted values
        context = torch.matmul(attention, V_last)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        return self.output_proj(context), attention
```

### Multi-Head Cross-Attention

Multiple attention heads capture different types of cross-asset relationships:

```python
class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention with different heads specializing in:
    - Correlation-based relationships
    - Lead-lag relationships
    - Volatility spillover
    - Sector/industry groupings
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x_query, x_key_value, mask=None):
        """
        Args:
            x_query: [batch, n_query_assets, seq_len, d_model]
            x_key_value: [batch, n_kv_assets, seq_len, d_model]

        Returns:
            output: [batch, n_query_assets, seq_len, d_model]
            attention: [batch, n_heads, n_query_assets, n_kv_assets]
        """
        batch, n_q, seq_len, d_model = x_query.shape
        n_kv = x_key_value.shape[1]

        # Pool temporal dimension for cross-asset attention
        q = x_query.mean(dim=2)  # [batch, n_q, d_model]
        k = x_key_value.mean(dim=2)  # [batch, n_kv, d_model]
        v = x_key_value.mean(dim=2)

        # Project
        Q = self.W_q(q).view(batch, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(k).view(batch, n_kv, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(v).view(batch, n_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [batch, n_heads, n_q, n_kv]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Weighted sum: [batch, n_heads, n_q, head_dim]
        context = torch.matmul(attention, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch, n_q, d_model)
        output = self.W_o(context)

        # Broadcast back to sequence length
        output = output.unsqueeze(2).expand(-1, -1, seq_len, -1)
        output = self.layer_norm(x_query + output)

        return output, attention
```

### Temporal Cross-Attention

Captures lead-lag relationships across time:

```python
class TemporalCrossAttention(nn.Module):
    """
    Cross-attention that considers temporal shifts between assets.

    Example: BTC at time t-1 predicts ETH at time t
    """

    def __init__(self, d_model, n_heads, max_lag=5):
        super().__init__()
        self.max_lag = max_lag
        self.attention = MultiHeadCrossAttention(d_model, n_heads)

        # Learnable lag weights
        self.lag_weights = nn.Parameter(torch.ones(max_lag + 1) / (max_lag + 1))

    def forward(self, x_query, x_key_value):
        """
        Args:
            x_query: [batch, n_q, seq_len, d_model]
            x_key_value: [batch, n_kv, seq_len, d_model]

        Returns:
            output: Attended representation with temporal alignment
            attention: Cross-asset attention weights per lag
        """
        batch, n_q, seq_len, d_model = x_query.shape
        n_kv = x_key_value.shape[1]

        outputs = []
        attentions = []

        # Compute attention at different lags
        for lag in range(self.max_lag + 1):
            if lag == 0:
                kv_lagged = x_key_value
            else:
                # Shift key_value backward by lag steps
                kv_lagged = F.pad(x_key_value[:, :, :-lag], (0, 0, lag, 0))

            out, attn = self.attention(x_query, kv_lagged)
            outputs.append(out)
            attentions.append(attn)

        # Weighted combination across lags
        lag_weights = F.softmax(self.lag_weights, dim=0)
        output = sum(w * o for w, o in zip(lag_weights, outputs))

        return output, torch.stack(attentions, dim=1)
```

### Hierarchical Cross-Attention

Models relationships at multiple levels (assets, sectors, markets):

```python
class HierarchicalCrossAttention(nn.Module):
    """
    Three-level hierarchy:
    1. Asset level: Individual asset relationships
    2. Sector level: Sector/industry relationships
    3. Market level: Cross-market relationships (crypto vs stocks)
    """

    def __init__(self, d_model, n_heads, sector_mapping, market_mapping):
        super().__init__()
        self.sector_mapping = sector_mapping  # asset_id -> sector_id
        self.market_mapping = market_mapping  # asset_id -> market_id

        # Asset-level attention
        self.asset_attention = MultiHeadCrossAttention(d_model, n_heads)

        # Sector-level attention
        self.sector_attention = MultiHeadCrossAttention(d_model, n_heads // 2)

        # Market-level attention
        self.market_attention = MultiHeadCrossAttention(d_model, n_heads // 4)

        # Combine hierarchies
        self.combine = nn.Linear(d_model * 3, d_model)

    def forward(self, x):
        """
        Args:
            x: [batch, n_assets, seq_len, d_model]

        Returns:
            output: Hierarchically attended representation
        """
        # Asset-level cross-attention
        asset_out, _ = self.asset_attention(x, x)

        # Aggregate to sectors
        sector_repr = self._aggregate_to_sectors(x)
        sector_out, _ = self.sector_attention(sector_repr, sector_repr)
        sector_out = self._broadcast_from_sectors(sector_out, x.shape)

        # Aggregate to markets
        market_repr = self._aggregate_to_markets(x)
        market_out, _ = self.market_attention(market_repr, market_repr)
        market_out = self._broadcast_from_markets(market_out, x.shape)

        # Combine all levels
        combined = torch.cat([asset_out, sector_out, market_out], dim=-1)
        return self.combine(combined)
```

## Mathematical Foundation

### Attention Score Computation

The attention score between query asset $i$ and key asset $j$ is:

$$\text{Attention}(Q_i, K_j, V_j) = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right) V_j$$

Where:
- $Q_i \in \mathbb{R}^{T \times d_k}$ - Query representations for asset $i$
- $K_j \in \mathbb{R}^{T \times d_k}$ - Key representations for asset $j$
- $V_j \in \mathbb{R}^{T \times d_v}$ - Value representations for asset $j$
- $d_k$ - Dimension of keys (scaling factor)

### Cross-Attention vs Self-Attention

| Aspect | Self-Attention | Cross-Attention |
|--------|---------------|-----------------|
| Q, K, V source | Same sequence | Q from one, K/V from another |
| Use case | Temporal patterns | Inter-asset relationships |
| Symmetry | Symmetric | Can be asymmetric |
| Complexity | $O(T^2)$ | $O(T^2 \cdot N)$ for N assets |

### Scaled Dot-Product Attention

For multi-asset scenarios with $N$ assets:

$$\text{MultiAssetAttention}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

Where each head $i$ computes:

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

## Data Representation

### Multi-Asset Feature Engineering

```python
def create_multi_asset_features(df_dict: dict, lookback: int = 100) -> np.ndarray:
    """
    Create feature tensor for multiple assets.

    Args:
        df_dict: Dictionary mapping asset symbol to DataFrame with OHLCV
        lookback: Number of historical timesteps

    Returns:
        features: [n_samples, n_assets, lookback, n_features]
    """
    features = []

    for symbol, df in df_dict.items():
        asset_features = []

        # Price features
        asset_features.append(np.log(df['close'] / df['close'].shift(1)))  # Log returns
        asset_features.append((df['close'] - df['open']) / df['open'])     # Intraday return
        asset_features.append((df['high'] - df['low']) / df['close'])      # Range

        # Volume features
        asset_features.append(df['volume'] / df['volume'].rolling(20).mean())  # Relative volume

        # Technical indicators
        asset_features.append(compute_rsi(df['close'], 14))
        asset_features.append(compute_macd(df['close']))

        features.append(np.column_stack(asset_features))

    return np.stack(features, axis=1)  # [time, n_assets, n_features]
```

### Data from Stock Markets

```python
import yfinance as yf

def fetch_stock_data(symbols: list, start: str, end: str) -> dict:
    """
    Fetch stock data from Yahoo Finance.

    Args:
        symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    data = {}

    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval='1h')
        df.columns = df.columns.str.lower()
        data[symbol] = df

    return data

# Example usage
stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
stock_data = fetch_stock_data(stock_symbols, '2023-01-01', '2024-01-01')
```

### Data from Cryptocurrency Markets (Bybit)

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

class BybitDataLoader:
    """Load cryptocurrency data from Bybit exchange."""

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    def __init__(self):
        self.session = requests.Session()

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",  # 60 minutes = 1 hour
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            limit: Number of candles (max 1000)

        Returns:
            DataFrame with OHLCV data
        """
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        response = self.session.get(self.BASE_URL, params=params)
        data = response.json()

        if data['retCode'] != 0:
            raise Exception(f"API Error: {data['retMsg']}")

        klines = data['result']['list']

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        return df.sort_values('timestamp').reset_index(drop=True)

    def fetch_multi_asset(self, symbols: list, **kwargs) -> dict:
        """Fetch data for multiple assets."""
        return {symbol: self.fetch_klines(symbol, **kwargs) for symbol in symbols}

# Example usage
loader = BybitDataLoader()
crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT']
crypto_data = loader.fetch_multi_asset(crypto_symbols, interval='60', limit=1000)
```

## Practical Examples

### 01: Data Preparation

```python
# python/01_data_preparation.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler

def prepare_cross_attention_data(
    asset_data: Dict[str, pd.DataFrame],
    lookback: int = 168,  # 7 days hourly
    horizon: int = 24,    # 24 hours ahead
    features: List[str] = ['log_return', 'volume_ratio', 'volatility', 'rsi']
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for cross-attention multi-asset model.

    Returns:
        X: [n_samples, n_assets, lookback, n_features]
        y: [n_samples, n_assets] - Future returns
        symbols: List of asset symbols
    """
    symbols = list(asset_data.keys())
    n_assets = len(symbols)

    # Compute features for each asset
    processed = {}
    for symbol, df in asset_data.items():
        feat = pd.DataFrame(index=df.index)

        feat['log_return'] = np.log(df['close'] / df['close'].shift(1))
        feat['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        feat['volatility'] = feat['log_return'].rolling(20).std()
        feat['rsi'] = compute_rsi(df['close'], 14)

        processed[symbol] = feat

    # Align timestamps
    common_idx = processed[symbols[0]].index
    for symbol in symbols[1:]:
        common_idx = common_idx.intersection(processed[symbol].index)

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(common_idx) - horizon):
        x_sample = []
        y_sample = []

        for symbol in symbols:
            df = processed[symbol].loc[common_idx]
            x_sample.append(df.iloc[i-lookback:i][features].values)
            y_sample.append(df.iloc[i+horizon]['log_return'])

        X.append(np.stack(x_sample, axis=0))
        y.append(np.array(y_sample))

    return np.array(X), np.array(y), symbols

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

### 02: Cross-Attention Model

See [python/model.py](python/model.py) for complete implementation.

```python
# python/model.py (simplified)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttentionMultiAsset(nn.Module):
    """
    Cross-Attention model for multi-asset prediction.

    Features:
    - Temporal self-attention within each asset
    - Cross-asset attention between all pairs
    - Multi-head attention for diverse relationships
    - Flexible output: regression, classification, or portfolio weights
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        output_type: str = 'regression'
    ):
        super().__init__()
        self.n_assets = n_assets
        self.output_type = output_type

        # Embedding
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Encoder layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output head
        if output_type == 'regression':
            self.output_head = nn.Linear(d_model, 1)
        elif output_type == 'classification':
            self.output_head = nn.Linear(d_model, 3)  # Down, Neutral, Up
        elif output_type == 'portfolio':
            self.output_head = nn.Linear(d_model, 1)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: [batch, n_assets, seq_len, n_features]

        Returns:
            predictions: [batch, n_assets] or [batch, n_assets, n_classes]
            attention: Optional attention weights
        """
        batch, n_assets, seq_len, n_features = x.shape

        # Embed each asset
        x = self.input_proj(x)  # [batch, n_assets, seq_len, d_model]

        # Add positional encoding
        for a in range(n_assets):
            x[:, a] = self.pos_encoding(x[:, a])

        # Apply encoder layers
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, return_attention)
            if return_attention:
                attentions.append(attn)

        # Pool temporal dimension
        x = x[:, :, -1, :]  # Take last timestep: [batch, n_assets, d_model]

        # Output
        if self.output_type == 'portfolio':
            logits = self.output_head(x).squeeze(-1)  # [batch, n_assets]
            output = F.softmax(logits, dim=-1)  # Portfolio weights
        elif self.output_type == 'classification':
            output = self.output_head(x)  # [batch, n_assets, 3]
        else:
            output = self.output_head(x).squeeze(-1)  # [batch, n_assets]

        if return_attention:
            return output, attentions
        return output
```

### 03: Model Training

```python
# python/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_cross_attention_model(
    model: nn.Module,
    train_data: tuple,
    val_data: tuple,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = 'cuda'
):
    """
    Train cross-attention model.

    Args:
        model: CrossAttentionMultiAsset model
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
    """
    X_train, y_train = train_data
    X_val, y_val = val_data

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    if model.output_type == 'regression':
        criterion = nn.MSELoss()
    elif model.output_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:  # portfolio
        criterion = lambda pred, ret: -torch.mean(torch.sum(pred * ret, dim=-1))  # Negative returns

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    model = model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)

            if model.output_type == 'classification':
                # Reshape for cross-entropy
                predictions = predictions.view(-1, 3)
                batch_y = (batch_y > 0).long().view(-1)

            loss = criterion(predictions, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_x = torch.FloatTensor(X_val).to(device)
            val_y = torch.FloatTensor(y_val).to(device)
            val_pred = model(val_x)

            if model.output_type == 'classification':
                val_pred = val_pred.view(-1, 3)
                val_y = (val_y > 0).long().view(-1)

            val_loss = criterion(val_pred, val_y).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.6f}, "
                  f"Val Loss = {val_loss:.6f}")

    return model
```

### 04: Multi-Asset Prediction

```python
# python/predict.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def predict_and_visualize(
    model,
    X: np.ndarray,
    symbols: list,
    device: str = 'cuda'
):
    """
    Make predictions and visualize attention patterns.
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        x = torch.FloatTensor(X).to(device)
        predictions, attentions = model(x, return_attention=True)

    predictions = predictions.cpu().numpy()

    # Visualize cross-asset attention
    if attentions:
        # Get attention from last layer
        cross_attn = attentions[-1]['cross_asset']  # [batch, n_heads, n_assets, n_assets]
        avg_attn = cross_attn.mean(dim=[0, 1]).cpu().numpy()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            avg_attn,
            xticklabels=symbols,
            yticklabels=symbols,
            annot=True,
            fmt='.2f',
            cmap='Blues'
        )
        plt.title('Cross-Asset Attention Weights')
        plt.xlabel('Key (Source Asset)')
        plt.ylabel('Query (Target Asset)')
        plt.tight_layout()
        plt.savefig('cross_attention_heatmap.png', dpi=150)
        plt.close()

    return predictions

def analyze_lead_lag_relationships(
    model,
    X: np.ndarray,
    symbols: list
):
    """
    Analyze which assets lead/lag others based on attention patterns.
    """
    model.eval()

    with torch.no_grad():
        _, attentions = model(torch.FloatTensor(X), return_attention=True)

    # Extract cross-asset attention
    cross_attn = attentions[-1]['cross_asset'].mean(dim=[0, 1]).numpy()

    # Compute influence scores
    influence = {}
    for i, symbol in enumerate(symbols):
        # How much does this asset influence others?
        influence[symbol] = {
            'as_leader': cross_attn[:, i].mean(),  # Others attend to this
            'as_follower': cross_attn[i, :].mean()  # This attends to others
        }

    # Rank by leadership
    leaders = sorted(
        influence.items(),
        key=lambda x: x[1]['as_leader'],
        reverse=True
    )

    print("\nAsset Leadership Ranking:")
    print("-" * 40)
    for symbol, scores in leaders:
        print(f"{symbol}: Leader={scores['as_leader']:.3f}, "
              f"Follower={scores['as_follower']:.3f}")

    return influence
```

### 05: Portfolio Backtesting

```python
# python/backtest.py

import numpy as np
import pandas as pd
from typing import Dict, List

class CrossAttentionBacktest:
    """
    Backtest cross-attention portfolio strategy.
    """

    def __init__(
        self,
        model,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        rebalance_freq: int = 24  # Hours
    ):
        self.model = model
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_freq = rebalance_freq

    def run(
        self,
        X: np.ndarray,
        returns: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Run backtest on test data.

        Args:
            X: [n_samples, n_assets, lookback, n_features]
            returns: [n_samples, n_assets] - Actual future returns
            timestamps: DatetimeIndex for results

        Returns:
            DataFrame with portfolio metrics over time
        """
        import torch

        self.model.eval()
        n_samples, n_assets, _, _ = X.shape

        capital = self.initial_capital
        positions = np.zeros(n_assets)

        results = []

        for i in range(0, n_samples, self.rebalance_freq):
            # Get model predictions (portfolio weights)
            with torch.no_grad():
                x = torch.FloatTensor(X[i:i+1])
                weights = self.model(x).numpy().flatten()

            # Normalize weights
            if self.model.output_type == 'regression':
                # Convert return predictions to weights
                weights = np.clip(weights, -1, 1)
                weights = weights / (np.abs(weights).sum() + 1e-8)

            # Calculate transaction costs
            position_change = np.abs(weights - positions).sum()
            costs = position_change * self.transaction_cost * capital

            # Calculate period returns
            period_returns = returns[i:min(i+self.rebalance_freq, n_samples)]

            for j, ret in enumerate(period_returns):
                portfolio_return = np.sum(positions * ret)
                capital = capital * (1 + portfolio_return)

                if j == 0:
                    capital -= costs

                results.append({
                    'timestamp': timestamps[i+j] if i+j < len(timestamps) else None,
                    'capital': capital,
                    'return': portfolio_return,
                    'positions': positions.copy(),
                    'weights': weights.copy()
                })

            # Update positions
            positions = weights

        return pd.DataFrame(results)

    def compute_metrics(self, results: pd.DataFrame) -> Dict:
        """Compute performance metrics."""
        returns = results['return'].values

        # Sharpe Ratio (annualized for hourly data)
        sharpe = np.sqrt(365 * 24) * returns.mean() / (returns.std() + 1e-8)

        # Sortino Ratio
        downside = returns[returns < 0]
        sortino = np.sqrt(365 * 24) * returns.mean() / (downside.std() + 1e-8)

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Total Return
        total_return = (results['capital'].iloc[-1] / self.initial_capital - 1) * 100

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown * 100,
            'volatility': returns.std() * np.sqrt(365 * 24) * 100,
            'win_rate': (returns > 0).mean() * 100
        }
```

## Rust Implementation

See [rust/](rust/) for complete Rust implementation using the `candle` ML framework.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                 # Library exports
│   ├── model/                 # Model implementation
│   │   ├── mod.rs
│   │   ├── attention.rs       # Cross-attention layers
│   │   ├── embedding.rs       # Token embeddings
│   │   └── cross_attention.rs # Main model
│   ├── data/                  # Data handling
│   │   ├── mod.rs
│   │   ├── bybit.rs          # Bybit API client
│   │   ├── features.rs       # Feature engineering
│   │   └── dataset.rs        # Training dataset
│   └── strategy/             # Trading strategy
│       ├── mod.rs
│       ├── signals.rs        # Signal generation
│       └── backtest.rs       # Backtesting engine
└── examples/
    ├── fetch_data.rs         # Download data from Bybit
    ├── train.rs              # Train the model
    └── backtest.rs           # Run backtest
```

### Quick Start (Rust)

```bash
# Navigate to Rust project
cd rust

# Fetch data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT,AVAXUSDT

# Train model
cargo run --release --example train -- --epochs 50 --batch-size 32

# Run backtest
cargo run --release --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── __init__.py
├── model.py                  # Cross-attention model
├── data.py                   # Data loading (Bybit + Yahoo Finance)
├── features.py               # Feature engineering
├── train.py                  # Training script
├── backtest.py               # Backtesting utilities
├── requirements.txt          # Dependencies
└── examples/
    ├── 01_data_preparation.py
    ├── 02_model_training.py
    ├── 03_prediction.py
    └── 04_backtesting.py
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Run example
python examples/01_data_preparation.py
python examples/02_model_training.py
python examples/03_prediction.py
python examples/04_backtesting.py
```

## Best Practices

### When to Use Cross-Attention

**Good use cases:**
- Trading correlated asset classes (crypto, tech stocks, commodities)
- Portfolio optimization across multiple assets
- Detecting lead-lag relationships
- Multi-asset risk management

**Not ideal for:**
- Single asset prediction (use simpler models)
- Very short-term prediction (latency concerns)
- Uncorrelated assets (cross-attention won't help)

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `d_model` | 64-128 | Match computational budget |
| `n_heads` | 4-8 | More heads for more assets |
| `n_layers` | 2-4 | Deeper for complex relationships |
| `dropout` | 0.1-0.2 | Higher for small datasets |
| `lookback` | 168 (7 days hourly) | Match prediction horizon |

### Common Pitfalls

1. **Correlation collapse**: All attention goes to one dominant asset
   - Solution: Use dropout, attention regularization

2. **Overfitting cross-asset patterns**: Model memorizes spurious correlations
   - Solution: More data, simpler model, regularization

3. **Ignoring regime changes**: Cross-asset relationships change over time
   - Solution: Rolling training windows, regime detection

4. **Computational cost**: O(N² * T²) for N assets, T timesteps
   - Solution: Sparse attention, efficient implementations

## Resources

### Papers

- [Portfolio Transformer for Attention-Based Asset Allocation](https://arxiv.org/abs/2206.03246) — End-to-end portfolio optimization with attention
- [Attention-Based Ensemble Learning for Portfolio Optimisation](https://arxiv.org/abs/2404.08935) — MASAAT framework with multi-agent attention
- [Large-scale Time-Varying Portfolio Optimisation using Graph Attention Networks](https://arxiv.org/abs/2407.15532) — GAT-based portfolio management
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer paper

### Implementations

- [PyTorch Multi-Head Attention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Candle ML Framework (Rust)](https://github.com/huggingface/candle)

### Related Chapters

- [Chapter 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — Multi-horizon forecasting
- [Chapter 43: Stockformer Multivariate](../43_stockformer_multivariate) — Cross-ticker attention
- [Chapter 44: ProbSparse Attention](../44_probsparse_attention) — Efficient attention mechanisms
- [Chapter 46: Temporal Attention Networks](../46_temporal_attention_networks) — Temporal attention

---

## Difficulty Level

**Advanced**

Prerequisites:
- Transformer architecture and attention mechanisms
- Multi-asset portfolio theory
- Time series forecasting
- PyTorch or Rust ML libraries
