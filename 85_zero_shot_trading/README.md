# Chapter 85: Zero-Shot Trading

## Overview

Zero-shot trading represents a paradigm shift in algorithmic trading, enabling models to make predictions on entirely new assets, markets, or regimes **without any task-specific training examples**. Unlike few-shot learning that requires a small support set, zero-shot learning leverages transferred knowledge and semantic understanding to generalize to unseen scenarios immediately.

This approach is particularly powerful for cryptocurrency markets where new tokens constantly emerge, or for adapting to sudden market regime shifts where historical patterns become obsolete.

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Zero-Shot vs Few-Shot Learning](#zero-shot-vs-few-shot-learning)
4. [Architecture Design](#architecture-design)
5. [Implementation Strategy](#implementation-strategy)
6. [Bybit Integration](#bybit-integration)
7. [Trading Strategy](#trading-strategy)
8. [Risk Management](#risk-management)
9. [Performance Metrics](#performance-metrics)
10. [References](#references)

---

## Introduction

### The Zero-Shot Challenge in Trading

Traditional machine learning for trading follows a predictable pattern:
1. Collect historical data for target asset
2. Train model on that data
3. Make predictions for that same asset

But what happens when:
- A new cryptocurrency is listed with no historical data?
- Market regime shifts dramatically, invalidating historical patterns?
- You want to trade in a completely new market segment?

**Zero-shot trading** addresses these challenges by learning **transferable representations** that generalize across assets and market conditions without requiring target-specific training.

### Why Zero-Shot for Trading?

```
+-------------------------------------------------------------------------+
|                    The Zero-Shot Trading Problem                          |
+-------------------------------------------------------------------------+
|                                                                           |
|   Traditional Approach:              Zero-Shot Approach:                  |
|   --------------------              --------------------                  |
|                                                                           |
|   New Asset Listed:                  New Asset Listed:                    |
|   "Wait 6 months for data"           "Trade immediately!"                 |
|   "Then train a model"               "Use transferred knowledge"          |
|   "Then start trading"                                                    |
|                                                                           |
|   Regime Shift Detected:             Regime Shift Detected:               |
|   "Model is broken"                  "Adapt via semantic features"        |
|   "Retrain from scratch"             "Continue trading"                   |
|                                                                           |
|   Market Crash:                      Market Crash:                        |
|   "Historical patterns fail"         "Leverage cross-market invariants"   |
|   "Large losses incurred"            "Robust predictions continue"        |
|                                                                           |
+-------------------------------------------------------------------------+
```

### Key Advantages

| Aspect | Traditional ML | Few-Shot | Zero-Shot |
|--------|---------------|----------|-----------|
| Data requirements | 1000s of samples | 5-20 samples | 0 samples |
| New asset adaptation | Full retraining | Few examples needed | Immediate |
| Regime change handling | Poor | Moderate | Excellent |
| Computational cost | High for retraining | Low | Very low |
| Time to first trade | Days/weeks | Hours | Seconds |

---

## Theoretical Foundation

### The Zero-Shot Learning Framework

Zero-shot learning works by mapping both inputs (market data) and outputs (predictions) into a shared semantic embedding space where relationships can be transferred.

### Mathematical Formulation

**Embedding Functions**:

Let $f_\theta: \mathcal{X} \rightarrow \mathbb{R}^d$ be the market data encoder that maps market features to embeddings.

Let $g_\phi: \mathcal{A} \rightarrow \mathbb{R}^d$ be the attribute encoder that maps asset/regime attributes to the same embedding space.

**Compatibility Function**:

$$F(x, a) = f_\theta(x)^T g_\phi(a)$$

This measures compatibility between market data $x$ and attributes $a$.

**Zero-Shot Prediction**:

For a new target class $c$ with attributes $a_c$:

$$\hat{y} = \arg\max_{c \in \mathcal{C}_{new}} F(x, a_c) = \arg\max_{c} f_\theta(x)^T g_\phi(a_c)$$

### Attribute-Based Transfer

The key insight is that assets/regimes can be described by **semantic attributes**:

```
+-------------------------------------------------------------------------+
|                    Asset Attribute Description                            |
+-------------------------------------------------------------------------+
|                                                                           |
|   Bitcoin (BTC):                                                          |
|   - Asset type: Cryptocurrency                                            |
|   - Market cap: Large                                                     |
|   - Volatility: High                                                      |
|   - Correlation with: Tech stocks, Risk-on assets                        |
|   - Typical daily range: 3-5%                                            |
|   - Trading hours: 24/7                                                   |
|   - Liquidity: High                                                       |
|                                                                           |
|   New Altcoin (Unknown):                                                  |
|   - Asset type: Cryptocurrency  <-- Same!                                 |
|   - Market cap: Small                                                     |
|   - Volatility: Very High                                                 |
|   - Correlation with: BTC, Risk-on assets  <-- Similar!                  |
|   - Typical daily range: 10-20%                                          |
|   - Trading hours: 24/7  <-- Same!                                       |
|   - Liquidity: Low                                                        |
|                                                                           |
|   By matching attributes, model transfers BTC knowledge to new altcoin    |
|                                                                           |
+-------------------------------------------------------------------------+
```

### Gaussian Mixture Meta-Learning for Zero-Shot Forecasting

Based on recent research (Liu et al., 2025), a powerful approach uses:

1. **Learned Embeddings**: Neural network learns to embed time series into latent space
2. **GMM Clustering**: Gaussian Mixture Models softly cluster embeddings into latent regimes
3. **Dual Task Learning**:
   - **Intra-cluster tasks**: Learning patterns within similar assets/regimes
   - **Inter-cluster tasks**: Learning transferable patterns across different clusters
4. **Hard Task Mining**: Focusing on difficult cross-cluster transfers to strengthen generalization

```
+-------------------------------------------------------------------------+
|                    GMM-Based Zero-Shot Architecture                       |
+-------------------------------------------------------------------------+
|                                                                           |
|   Input: Market Time Series                                               |
|   [price, volume, volatility, ...]                                       |
|            |                                                              |
|            v                                                              |
|   +------------------+                                                    |
|   | Time Series      |                                                    |
|   | Encoder f_theta  |                                                    |
|   +------------------+                                                    |
|            |                                                              |
|            v                                                              |
|   Embedding z in R^d                                                      |
|            |                                                              |
|            v                                                              |
|   +------------------+                                                    |
|   | GMM Clustering   |-----> K latent clusters (regimes)                  |
|   +------------------+       c1, c2, ..., cK                              |
|            |                                                              |
|            v                                                              |
|   +------------------+     +------------------+                           |
|   | Intra-Cluster    |     | Inter-Cluster    |                          |
|   | Meta-Tasks       |     | Meta-Tasks       |                          |
|   | (same regime)    |     | (cross regime)   |                          |
|   +------------------+     +------------------+                           |
|            |                        |                                     |
|            v                        v                                     |
|   +------------------------------------------+                           |
|   |        Combined Meta-Learning            |                           |
|   |   Learns both local and global patterns  |                           |
|   +------------------------------------------+                           |
|            |                                                              |
|            v                                                              |
|   Zero-Shot Prediction for New Series                                     |
|                                                                           |
+-------------------------------------------------------------------------+
```

---

## Zero-Shot vs Few-Shot Learning

### Comparison Framework

```
+-------------------------------------------------------------------------+
|               Zero-Shot vs Few-Shot for Trading                           |
+-------------------------------------------------------------------------+
|                                                                           |
|   Few-Shot Learning (e.g., Prototypical Networks):                       |
|   ------------------------------------------------                        |
|   - Given: 5-20 examples of new asset/regime                             |
|   - Method: Compute prototype, classify by distance                       |
|   - Strength: Can adapt to truly novel patterns                          |
|   - Weakness: Needs at least some examples                               |
|                                                                           |
|   Zero-Shot Learning:                                                     |
|   -------------------                                                     |
|   - Given: Semantic description of new asset/regime                      |
|   - Method: Match via shared embedding space                             |
|   - Strength: No examples needed at all                                  |
|   - Weakness: Limited by attribute quality                               |
|                                                                           |
|   Hybrid Approach (Recommended):                                          |
|   ------------------------------                                          |
|   - Start: Zero-shot prediction for immediate trading                    |
|   - Evolve: Collect examples over time                                   |
|   - Improve: Transition to few-shot as data accumulates                  |
|   - Best of both worlds!                                                 |
|                                                                           |
+-------------------------------------------------------------------------+
```

### When to Use Each Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Brand new token listing | Zero-shot |
| Flash crash (sudden regime) | Zero-shot |
| New market (forex to crypto) | Zero-shot then few-shot |
| Asset with 1 week of data | Few-shot |
| Asset with 1+ month of data | Traditional or few-shot |
| Cross-asset strategy | Zero-shot for initialization |

---

## Architecture Design

### Zero-Shot Trading Network

```
+-------------------------------------------------------------------------+
|                    Zero-Shot Trading Architecture                         |
+-------------------------------------------------------------------------+
|                                                                           |
|   MARKET DATA ENCODER (f_theta)                                          |
|   =============================                                           |
|   Input: [price, volume, volatility, indicators]                         |
|   Shape: (batch, sequence_length, features)                              |
|                                                                           |
|   +-----------------------+                                               |
|   | Temporal Embedding    |                                               |
|   | - Conv1D layers       |                                               |
|   | - Positional encoding |                                               |
|   +-----------------------+                                               |
|            |                                                              |
|            v                                                              |
|   +-----------------------+                                               |
|   | Transformer Encoder   |                                               |
|   | - Self-attention      |                                               |
|   | - Feed-forward        |                                               |
|   +-----------------------+                                               |
|            |                                                              |
|            v                                                              |
|   +-----------------------+                                               |
|   | Projection Head       |                                               |
|   | - Linear layers       |                                               |
|   | - L2 normalization    |                                               |
|   +-----------------------+                                               |
|            |                                                              |
|            v                                                              |
|   Market Embedding: z_market in R^d                                       |
|                                                                           |
|   ATTRIBUTE ENCODER (g_phi)                                              |
|   =========================                                               |
|   Input: [asset_type, volatility_class, correlation, ...]                |
|                                                                           |
|   +-----------------------+                                               |
|   | Attribute Embedding   |                                               |
|   | - Categorical embed   |                                               |
|   | - Numerical scaling   |                                               |
|   +-----------------------+                                               |
|            |                                                              |
|            v                                                              |
|   +-----------------------+                                               |
|   | MLP Projection        |                                               |
|   | - Hidden layers       |                                               |
|   | - L2 normalization    |                                               |
|   +-----------------------+                                               |
|            |                                                              |
|            v                                                              |
|   Attribute Embedding: z_attr in R^d                                      |
|                                                                           |
|   COMPATIBILITY SCORING                                                   |
|   =====================                                                   |
|   score = z_market . z_attr (dot product)                                |
|   prediction = softmax(scores across possible classes)                   |
|                                                                           |
+-------------------------------------------------------------------------+
```

### Semantic Attribute Design for Trading

```
+-------------------------------------------------------------------------+
|                    Trading Attributes for Zero-Shot                       |
+-------------------------------------------------------------------------+
|                                                                           |
|   Asset-Level Attributes:                                                 |
|   -----------------------                                                 |
|   - asset_type: [crypto, stock, forex, commodity]                        |
|   - market_cap_class: [large, medium, small, micro]                      |
|   - volatility_regime: [low, medium, high, extreme]                      |
|   - liquidity_class: [highly_liquid, liquid, illiquid]                   |
|   - sector: [defi, layer1, layer2, meme, gaming, ...]                    |
|   - correlation_btc: continuous [-1, 1]                                  |
|   - correlation_sp500: continuous [-1, 1]                                |
|   - beta: continuous risk measure                                         |
|                                                                           |
|   Regime-Level Attributes:                                                |
|   ------------------------                                                |
|   - trend: [strong_up, weak_up, sideways, weak_down, strong_down]        |
|   - volatility_state: [contracting, stable, expanding]                   |
|   - volume_profile: [accumulation, distribution, neutral]                |
|   - market_sentiment: [euphoria, optimism, neutral, fear, panic]         |
|   - funding_rate: continuous                                              |
|   - open_interest_trend: [rising, flat, falling]                         |
|                                                                           |
|   Temporal Attributes:                                                    |
|   --------------------                                                    |
|   - time_of_day: [asian_session, european_session, us_session]           |
|   - day_of_week: [monday, ..., friday, weekend]                          |
|   - market_event: [earnings, fomc, opex, normal]                         |
|                                                                           |
+-------------------------------------------------------------------------+
```

### GMM-Based Regime Discovery

```python
# Pseudocode for GMM regime clustering
def train_gmm_regime_model(embeddings, n_components=5):
    """
    Discover latent market regimes using GMM.

    Args:
        embeddings: Time series embeddings from encoder
        n_components: Number of latent regimes to discover

    Returns:
        gmm: Trained GMM model
        cluster_assignments: Soft cluster probabilities
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        init_params='kmeans',
        max_iter=100
    )

    gmm.fit(embeddings)

    # Soft assignment (probability of belonging to each regime)
    cluster_probs = gmm.predict_proba(embeddings)

    return gmm, cluster_probs
```

---

## Implementation Strategy

### Python Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class MarketEncoder(nn.Module):
    """
    Encodes market time series data into embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Temporal feature extraction
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Transformer for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Market data tensor of shape (batch, seq_len, features)

        Returns:
            Embedding tensor of shape (batch, embed_dim)
        """
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Convolutional feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # x: (batch, hidden, seq_len) -> (batch, seq_len, hidden)
        x = x.transpose(1, 2)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Project to embedding space
        x = self.projection(x)

        # L2 normalize
        x = F.normalize(x, p=2, dim=1)

        return x


class AttributeEncoder(nn.Module):
    """
    Encodes asset/regime attributes into embeddings.
    """

    def __init__(
        self,
        categorical_dims: Dict[str, int],  # {attr_name: num_categories}
        numerical_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.categorical_dims = categorical_dims
        self.numerical_dim = numerical_dim

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(num_cats, hidden_dim // len(categorical_dims))
            for name, num_cats in categorical_dims.items()
        })

        # Numerical feature processing
        self.num_mlp = nn.Sequential(
            nn.Linear(numerical_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        # Combined projection
        total_dim = (hidden_dim // len(categorical_dims)) * len(categorical_dims) + hidden_dim // 2
        self.projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(
        self,
        categorical_attrs: Dict[str, torch.Tensor],
        numerical_attrs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            categorical_attrs: Dict mapping attr names to category indices
            numerical_attrs: Tensor of shape (batch, numerical_dim)

        Returns:
            Embedding tensor of shape (batch, embed_dim)
        """
        # Embed categorical attributes
        cat_embeds = []
        for name in self.categorical_dims.keys():
            cat_embeds.append(self.cat_embeddings[name](categorical_attrs[name]))
        cat_embed = torch.cat(cat_embeds, dim=1)

        # Process numerical attributes
        num_embed = self.num_mlp(numerical_attrs)

        # Combine and project
        combined = torch.cat([cat_embed, num_embed], dim=1)
        x = self.projection(combined)

        # L2 normalize
        x = F.normalize(x, p=2, dim=1)

        return x


class ZeroShotTradingModel(nn.Module):
    """
    Complete zero-shot trading model with market and attribute encoders.
    """

    def __init__(
        self,
        market_input_dim: int,
        categorical_dims: Dict[str, int],
        numerical_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        temperature: float = 0.1
    ):
        super().__init__()

        self.market_encoder = MarketEncoder(
            input_dim=market_input_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim
        )

        self.attribute_encoder = AttributeEncoder(
            categorical_dims=categorical_dims,
            numerical_dim=numerical_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
        )

        self.temperature = temperature

    def encode_market(self, market_data: torch.Tensor) -> torch.Tensor:
        """Encode market data to embedding space."""
        return self.market_encoder(market_data)

    def encode_attributes(
        self,
        categorical_attrs: Dict[str, torch.Tensor],
        numerical_attrs: torch.Tensor
    ) -> torch.Tensor:
        """Encode attributes to embedding space."""
        return self.attribute_encoder(categorical_attrs, numerical_attrs)

    def compute_compatibility(
        self,
        market_embed: torch.Tensor,
        attr_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute compatibility scores between market and attribute embeddings.

        Args:
            market_embed: (batch_size, embed_dim)
            attr_embed: (num_classes, embed_dim) or (batch_size, num_classes, embed_dim)

        Returns:
            Compatibility scores (batch_size, num_classes)
        """
        if attr_embed.dim() == 2:
            # attr_embed: (num_classes, embed_dim)
            scores = torch.matmul(market_embed, attr_embed.T) / self.temperature
        else:
            # attr_embed: (batch_size, num_classes, embed_dim)
            scores = torch.bmm(
                attr_embed,
                market_embed.unsqueeze(-1)
            ).squeeze(-1) / self.temperature

        return scores

    def forward(
        self,
        market_data: torch.Tensor,
        categorical_attrs: Dict[str, torch.Tensor],
        numerical_attrs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for zero-shot prediction.

        Returns:
            (compatibility_scores, predicted_class_probabilities)
        """
        market_embed = self.encode_market(market_data)
        attr_embed = self.encode_attributes(categorical_attrs, numerical_attrs)

        scores = self.compute_compatibility(market_embed, attr_embed)
        probs = F.softmax(scores, dim=-1)

        return scores, probs
```

### Training Strategy

```python
class ZeroShotTrainer:
    """
    Trainer for zero-shot trading model using contrastive learning.
    """

    def __init__(
        self,
        model: ZeroShotTradingModel,
        learning_rate: float = 1e-4,
        margin: float = 0.2
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.margin = margin

    def contrastive_loss(
        self,
        market_embed: torch.Tensor,
        positive_attr_embed: torch.Tensor,
        negative_attr_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss to align market data with correct attributes.

        Args:
            market_embed: Market data embedding (batch, embed_dim)
            positive_attr_embed: Correct attribute embedding (batch, embed_dim)
            negative_attr_embeds: Incorrect attribute embeddings (batch, num_neg, embed_dim)
        """
        # Positive similarity
        pos_sim = F.cosine_similarity(market_embed, positive_attr_embed)

        # Negative similarities
        neg_sims = F.cosine_similarity(
            market_embed.unsqueeze(1),
            negative_attr_embeds,
            dim=2
        )

        # Triplet margin loss
        loss = F.relu(self.margin - pos_sim.unsqueeze(1) + neg_sims).mean()

        return loss

    def train_step(
        self,
        market_data: torch.Tensor,
        positive_attrs: Tuple[Dict[str, torch.Tensor], torch.Tensor],
        negative_attrs: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
    ) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Encode market data
        market_embed = self.model.encode_market(market_data)

        # Encode positive attributes
        pos_attr_embed = self.model.encode_attributes(*positive_attrs)

        # Encode negative attributes
        neg_attr_embeds = torch.stack([
            self.model.encode_attributes(*neg_attr)
            for neg_attr in negative_attrs
        ], dim=1)

        # Compute loss
        loss = self.contrastive_loss(market_embed, pos_attr_embed, neg_attr_embeds)

        # Backprop
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

---

## Bybit Integration

### Data Fetching for Zero-Shot Trading

```python
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

class BybitZeroShotClient:
    """
    Bybit client designed for zero-shot trading data collection.
    Fetches data and computes attribute features for multiple assets.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",  # 1 hour
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Fetch OHLCV kline data from Bybit.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Kline interval in minutes
            limit: Number of klines to fetch

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

        async with self.session.get(endpoint, params=params) as response:
            data = await response.json()

        if data["retCode"] != 0:
            raise ValueError(f"API error: {data['retMsg']}")

        klines = data["result"]["list"]

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")

        return df.sort_values("timestamp").reset_index(drop=True)

    async def fetch_ticker_info(self, symbol: str) -> Dict:
        """Fetch current ticker information for attribute computation."""
        endpoint = f"{self.BASE_URL}/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}

        async with self.session.get(endpoint, params=params) as response:
            data = await response.json()

        if data["retCode"] != 0:
            raise ValueError(f"API error: {data['retMsg']}")

        return data["result"]["list"][0]

    async def compute_asset_attributes(
        self,
        symbol: str,
        reference_symbols: List[str] = ["BTCUSDT", "ETHUSDT"]
    ) -> Dict:
        """
        Compute semantic attributes for an asset for zero-shot learning.

        Returns attributes like volatility class, correlation, etc.
        """
        # Fetch data for target and reference assets
        tasks = [self.fetch_klines(symbol, "60", 168)]  # 1 week of hourly data
        tasks.extend([self.fetch_klines(ref, "60", 168) for ref in reference_symbols])

        results = await asyncio.gather(*tasks)
        target_df = results[0]
        ref_dfs = results[1:]

        # Compute returns
        target_returns = target_df["close"].pct_change().dropna()
        ref_returns = [df["close"].pct_change().dropna() for df in ref_dfs]

        # Volatility class
        annualized_vol = target_returns.std() * np.sqrt(24 * 365)
        if annualized_vol < 0.3:
            volatility_class = "low"
        elif annualized_vol < 0.6:
            volatility_class = "medium"
        elif annualized_vol < 1.0:
            volatility_class = "high"
        else:
            volatility_class = "extreme"

        # Correlations with reference assets
        min_len = min(len(target_returns), min(len(r) for r in ref_returns))
        correlations = {}
        for ref_name, ref_ret in zip(reference_symbols, ref_returns):
            corr = np.corrcoef(
                target_returns.iloc[-min_len:],
                ref_ret.iloc[-min_len:]
            )[0, 1]
            correlations[f"corr_{ref_name}"] = corr

        # Market cap class (approximated from volume)
        avg_volume = target_df["turnover"].mean()
        if avg_volume > 1e9:
            market_cap_class = "large"
        elif avg_volume > 1e8:
            market_cap_class = "medium"
        elif avg_volume > 1e7:
            market_cap_class = "small"
        else:
            market_cap_class = "micro"

        # Trend detection
        sma_20 = target_df["close"].rolling(20).mean().iloc[-1]
        sma_50 = target_df["close"].rolling(50).mean().iloc[-1]
        current_price = target_df["close"].iloc[-1]

        if current_price > sma_20 > sma_50:
            trend = "strong_up"
        elif current_price > sma_20:
            trend = "weak_up"
        elif current_price < sma_20 < sma_50:
            trend = "strong_down"
        elif current_price < sma_20:
            trend = "weak_down"
        else:
            trend = "sideways"

        return {
            "asset_type": "crypto",
            "volatility_class": volatility_class,
            "market_cap_class": market_cap_class,
            "trend": trend,
            "annualized_vol": annualized_vol,
            **correlations
        }


async def fetch_multi_asset_data(
    symbols: List[str],
    interval: str = "60",
    limit: int = 200
) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
    """
    Fetch data and attributes for multiple assets.

    Returns:
        Dict mapping symbol to (DataFrame, attributes)
    """
    async with BybitZeroShotClient() as client:
        results = {}

        for symbol in symbols:
            df = await client.fetch_klines(symbol, interval, limit)
            attrs = await client.compute_asset_attributes(symbol)
            results[symbol] = (df, attrs)

    return results
```

---

## Trading Strategy

### Zero-Shot Regime-Based Trading

```python
class ZeroShotTradingStrategy:
    """
    Trading strategy using zero-shot regime prediction.
    """

    def __init__(
        self,
        model: ZeroShotTradingModel,
        regime_attributes: Dict[str, Tuple[Dict, np.ndarray]],
        confidence_threshold: float = 0.6
    ):
        """
        Args:
            model: Trained zero-shot model
            regime_attributes: Dict mapping regime names to (categorical, numerical) attributes
            confidence_threshold: Minimum confidence for trading
        """
        self.model = model
        self.regime_attributes = regime_attributes
        self.confidence_threshold = confidence_threshold

        # Precompute regime embeddings
        self._precompute_regime_embeddings()

    def _precompute_regime_embeddings(self):
        """Precompute attribute embeddings for all known regimes."""
        self.model.eval()
        self.regime_embeddings = {}

        with torch.no_grad():
            for regime_name, (cat_attrs, num_attrs) in self.regime_attributes.items():
                cat_tensors = {k: torch.tensor([v]) for k, v in cat_attrs.items()}
                num_tensor = torch.tensor([num_attrs], dtype=torch.float32)

                embed = self.model.encode_attributes(cat_tensors, num_tensor)
                self.regime_embeddings[regime_name] = embed.squeeze(0)

    def predict_regime(
        self,
        market_data: torch.Tensor
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict market regime using zero-shot inference.

        Args:
            market_data: Market features (1, seq_len, features)

        Returns:
            (predicted_regime, confidence, all_regime_probabilities)
        """
        self.model.eval()

        with torch.no_grad():
            market_embed = self.model.encode_market(market_data)

            # Compute similarities to all regime embeddings
            similarities = {}
            for regime_name, regime_embed in self.regime_embeddings.items():
                sim = F.cosine_similarity(
                    market_embed,
                    regime_embed.unsqueeze(0)
                ).item()
                similarities[regime_name] = sim

        # Convert to probabilities via softmax
        sim_values = list(similarities.values())
        exp_sims = np.exp(np.array(sim_values) / self.model.temperature)
        probs = exp_sims / exp_sims.sum()

        regime_probs = dict(zip(similarities.keys(), probs))

        # Get prediction
        predicted_regime = max(regime_probs, key=regime_probs.get)
        confidence = regime_probs[predicted_regime]

        return predicted_regime, confidence, regime_probs

    def generate_signal(
        self,
        market_data: torch.Tensor,
        current_position: float = 0.0
    ) -> Dict:
        """
        Generate trading signal based on zero-shot regime prediction.

        Args:
            market_data: Market features tensor
            current_position: Current position (-1 to 1, where 1 is full long)

        Returns:
            Trading signal with action, size, and reasoning
        """
        regime, confidence, regime_probs = self.predict_regime(market_data)

        # Regime to action mapping
        regime_actions = {
            "strong_uptrend": {"action": "long", "base_size": 1.0},
            "weak_uptrend": {"action": "long", "base_size": 0.5},
            "sideways": {"action": "neutral", "base_size": 0.0},
            "weak_downtrend": {"action": "short", "base_size": 0.5},
            "strong_downtrend": {"action": "short", "base_size": 1.0},
        }

        signal = regime_actions.get(regime, {"action": "neutral", "base_size": 0.0})

        # Adjust size based on confidence
        if confidence < self.confidence_threshold:
            signal["base_size"] *= 0.5  # Reduce size if uncertain

        # Compute target position
        if signal["action"] == "long":
            target_position = signal["base_size"] * confidence
        elif signal["action"] == "short":
            target_position = -signal["base_size"] * confidence
        else:
            target_position = 0.0

        # Determine trade
        position_change = target_position - current_position

        return {
            "regime": regime,
            "confidence": confidence,
            "regime_probabilities": regime_probs,
            "action": signal["action"],
            "target_position": target_position,
            "position_change": position_change,
            "reasoning": f"Zero-shot detected {regime} regime with {confidence:.1%} confidence"
        }
```

### Hybrid Zero-Shot + Few-Shot Strategy

```
+-------------------------------------------------------------------------+
|                    Hybrid Trading Strategy                                |
+-------------------------------------------------------------------------+
|                                                                           |
|   Phase 1: Pure Zero-Shot (Day 0)                                        |
|   ================================                                        |
|   - New asset listed on Bybit                                            |
|   - Compute asset attributes from:                                        |
|     * Initial volatility estimate                                         |
|     * Asset category (e.g., DeFi token)                                  |
|     * Similar assets correlation                                          |
|   - Zero-shot regime prediction                                           |
|   - Conservative position sizing (0.5x base)                             |
|                                                                           |
|   Phase 2: Zero-Shot + Accumulating (Days 1-7)                           |
|   =============================================                           |
|   - Continue zero-shot predictions                                        |
|   - Collect and label market data                                         |
|   - Build support set for few-shot                                        |
|   - Gradually increase position sizing                                    |
|                                                                           |
|   Phase 3: Hybrid Mode (Days 7-30)                                       |
|   ================================                                        |
|   - Few-shot learning with small support set                             |
|   - Combine zero-shot + few-shot predictions:                            |
|     final_pred = alpha * zero_shot + (1-alpha) * few_shot                |
|   - Alpha decreases as more data accumulates                             |
|   - Full position sizing available                                        |
|                                                                           |
|   Phase 4: Few-Shot Dominant (Day 30+)                                   |
|   =====================================                                   |
|   - Few-shot predictions primary                                          |
|   - Zero-shot as fallback for regime shifts                              |
|   - Update support set with recent examples                              |
|   - Full trading capacity                                                 |
|                                                                           |
+-------------------------------------------------------------------------+
```

---

## Risk Management

### Zero-Shot Specific Risks

```
+-------------------------------------------------------------------------+
|                    Risk Considerations for Zero-Shot                      |
+-------------------------------------------------------------------------+
|                                                                           |
|   1. Attribute Mismatch Risk                                             |
|   ===========================                                             |
|   Risk: Asset attributes computed incorrectly                            |
|   Mitigation:                                                             |
|   - Use multiple attribute estimation methods                            |
|   - Require attribute confidence threshold                               |
|   - Compare with similar assets                                           |
|                                                                           |
|   2. Distribution Shift Risk                                             |
|   ==========================                                              |
|   Risk: New asset fundamentally different from training                  |
|   Mitigation:                                                             |
|   - Monitor embedding distances to training distribution                 |
|   - Flag outliers for manual review                                       |
|   - Use uncertainty quantification                                        |
|                                                                           |
|   3. Low Confidence Predictions                                          |
|   =============================                                           |
|   Risk: Model uncertain but still trading                                |
|   Mitigation:                                                             |
|   - Strict confidence thresholds (e.g., >60%)                           |
|   - Position size proportional to confidence                             |
|   - No trading below minimum confidence                                  |
|                                                                           |
|   4. Regime Transition Risk                                              |
|   =========================                                               |
|   Risk: Regime changes faster than detection                             |
|   Mitigation:                                                             |
|   - Continuous regime monitoring                                          |
|   - Stop-loss always active                                               |
|   - Maximum holding period limits                                         |
|                                                                           |
+-------------------------------------------------------------------------+
```

### Position Sizing Algorithm

```python
def compute_zero_shot_position_size(
    base_position: float,
    prediction_confidence: float,
    attribute_confidence: float,
    distribution_distance: float,
    max_position: float = 1.0
) -> float:
    """
    Compute position size with multiple confidence factors.

    Args:
        base_position: Base position from strategy
        prediction_confidence: Regime prediction confidence
        attribute_confidence: Confidence in attribute estimation
        distribution_distance: Distance from training distribution (lower = better)
        max_position: Maximum allowed position

    Returns:
        Adjusted position size
    """
    # Confidence multiplier (0.5 to 1.0)
    confidence_mult = 0.5 + 0.5 * prediction_confidence

    # Attribute quality multiplier (0.5 to 1.0)
    attribute_mult = 0.5 + 0.5 * attribute_confidence

    # Distribution penalty (penalize out-of-distribution)
    # distribution_distance normalized to [0, 1], lower is better
    distribution_mult = max(0.3, 1.0 - distribution_distance)

    # Combined position
    position = base_position * confidence_mult * attribute_mult * distribution_mult

    # Clip to max
    return min(abs(position), max_position) * np.sign(position)
```

---

## Performance Metrics

### Evaluation Framework

| Metric | Description | Target |
|--------|-------------|--------|
| Zero-Shot Accuracy | Regime classification on unseen assets | >60% |
| Adaptation Speed | Time to reach 70% accuracy on new asset | <24 hours |
| Transfer Ratio | Performance on new asset / trained asset | >0.7 |
| Sharpe Ratio | Risk-adjusted returns | >1.5 |
| Max Drawdown | Largest peak-to-trough decline | <15% |
| Sortino Ratio | Downside risk-adjusted returns | >2.0 |
| Win Rate | Percentage of profitable trades | >55% |
| Profit Factor | Gross profit / Gross loss | >1.3 |

### Backtesting Results Format

```python
@dataclass
class ZeroShotBacktestResults:
    """Results from zero-shot trading backtest."""

    # Overall performance
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float

    # Zero-shot specific metrics
    regime_accuracy: float
    average_confidence: float
    out_of_distribution_rate: float

    # Trade statistics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_duration: timedelta

    # Per-regime performance
    regime_returns: Dict[str, float]
    regime_accuracies: Dict[str, float]

    # Adaptation metrics (for new assets)
    time_to_profit: Optional[timedelta]
    adaptation_curve: List[float]  # Accuracy over time
```

---

## References

### Academic Papers

1. **Adapting to the Unknown: Robust Meta-Learning for Zero-Shot Financial Time Series Forecasting**
   - Liu, Ma, Zhang (2025)
   - URL: https://arxiv.org/abs/2504.09664
   - Key contribution: GMM-based meta-learning for zero-shot forecasting

2. **Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly**
   - Xian et al. (2018)
   - Key contribution: Comprehensive ZSL benchmark and evaluation

3. **Learning to Learn with Compound HD Models**
   - Key contribution: Attribute-based zero-shot learning framework

### Related Chapters

- [Chapter 81: MAML for Trading](../81_maml_for_trading/)
- [Chapter 82: Reptile Algorithm Trading](../82_reptile_algorithm_trading/)
- [Chapter 83: Prototypical Networks Finance](../83_prototypical_networks_finance/)
- [Chapter 84: Matching Networks Finance](../84_matching_networks_finance/)
- [Chapter 86: Few-Shot Market Prediction](../86_few_shot_market_prediction/)

### Libraries and Tools

- **PyTorch**: Deep learning framework
- **Bybit API**: Cryptocurrency market data
- **scikit-learn**: GMM and clustering
- **pandas/numpy**: Data processing

---

## Directory Structure

```
85_zero_shot_trading/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Simplified explanation (English)
├── readme.simple.ru.md          # Simplified explanation (Russian)
├── Cargo.toml                   # Rust project configuration
├── src/                         # Rust source code
│   ├── lib.rs                   # Library root
│   ├── model/                   # Model implementations
│   ├── data/                    # Data handling & Bybit client
│   ├── training/                # Training logic
│   ├── strategy/                # Trading strategy
│   └── backtest/                # Backtesting engine
├── python/                      # Python implementation
│   └── zero_shot_trading.py     # Main Python module
└── examples/                    # Example scripts
    ├── basic_zero_shot.rs       # Basic Rust example
    ├── multi_asset.rs           # Multi-asset example
    └── trading_strategy.rs      # Full strategy example
```

---

## Quick Start

### Python

```python
import asyncio
from zero_shot_trading import (
    ZeroShotTradingModel,
    ZeroShotTradingStrategy,
    BybitZeroShotClient,
    prepare_features
)

async def main():
    # Initialize model
    model = ZeroShotTradingModel(
        market_input_dim=15,
        categorical_dims={
            "asset_type": 4,
            "volatility_class": 4,
            "market_cap_class": 4,
            "trend": 5
        },
        numerical_dim=3,
        embed_dim=64
    )

    # Load pretrained weights
    model.load_state_dict(torch.load("zero_shot_trading.pth"))

    # Define regime attributes
    regime_attributes = {
        "strong_uptrend": (
            {"trend": 0, "volatility_class": 2},
            np.array([0.7, 0.8, 0.6])  # corr_btc, corr_eth, momentum
        ),
        # ... other regimes
    }

    # Create strategy
    strategy = ZeroShotTradingStrategy(
        model=model,
        regime_attributes=regime_attributes,
        confidence_threshold=0.6
    )

    # Fetch data for new asset
    async with BybitZeroShotClient() as client:
        df = await client.fetch_klines("NEWTOKEN", "60", 100)
        attrs = await client.compute_asset_attributes("NEWTOKEN")

    # Prepare features and predict
    features = prepare_features(df)
    signal = strategy.generate_signal(features)

    print(f"Regime: {signal['regime']}")
    print(f"Confidence: {signal['confidence']:.1%}")
    print(f"Action: {signal['action']}")
    print(f"Reasoning: {signal['reasoning']}")

asyncio.run(main())
```

### Rust

```rust
use zero_shot_trading::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize model
    let model = ZeroShotModel::load("model.bin")?;

    // Fetch data from Bybit
    let client = BybitClient::new();
    let klines = client.fetch_klines("BTCUSDT", "1h", 100).await?;

    // Compute attributes
    let attrs = compute_asset_attributes(&klines)?;

    // Generate features and predict
    let features = prepare_features(&klines);
    let prediction = model.predict_regime(&features, &attrs)?;

    println!("Predicted regime: {:?}", prediction.regime);
    println!("Confidence: {:.1}%", prediction.confidence * 100.0);

    Ok(())
}
```

---

*This chapter is part of the Machine Learning for Trading series. For questions or contributions, please open an issue on GitHub.*
