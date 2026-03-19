# Chapter 238: Disentangled VAE Trading

This chapter explores **Disentangled Variational Autoencoders** (beta-VAE, FactorVAE, DIP-VAE, beta-TCVAE) for learning interpretable, disentangled latent representations of financial market data. Unlike standard VAEs that produce entangled latent spaces where each dimension encodes a mixture of generative factors, disentangled VAEs encourage each latent dimension to capture a single, independent generative factor such as trend direction, volatility regime, momentum strength, or cross-asset correlation structure. These isolated factors can then be directly used as interpretable trading signals.

<p align="center">
<img src="https://i.imgur.com/placeholder_disentangled_vae.png" width="70%">
</p>

## Contents

1. [Introduction to Disentangled VAEs](#introduction-to-disentangled-vaes)
    * [The Entanglement Problem](#the-entanglement-problem)
    * [Key Advantages](#key-advantages)
    * [Comparison of Disentanglement Methods](#comparison-of-disentanglement-methods)
2. [Disentanglement Methods](#disentanglement-methods)
    * [beta-VAE](#beta-vae)
    * [FactorVAE](#factorvae)
    * [DIP-VAE](#dip-vae)
    * [beta-TCVAE](#beta-tcvae)
3. [Mathematical Foundation](#mathematical-foundation)
    * [ELBO Decomposition](#elbo-decomposition)
    * [Total Correlation](#total-correlation)
    * [beta-VAE Objective](#beta-vae-objective)
    * [FactorVAE Objective](#factorvae-objective)
    * [Disentanglement Metrics](#disentanglement-metrics)
4. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: beta-VAE Architecture](#02-beta-vae-architecture)
    * [03: Model Training with Disentanglement](#03-model-training-with-disentanglement)
    * [04: Latent Factor Analysis](#04-latent-factor-analysis)
    * [05: Trading Strategy Using Disentangled Factors](#05-trading-strategy-using-disentangled-factors)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to Disentangled VAEs

Disentangled Variational Autoencoders are a family of generative models that extend the standard VAE framework by encouraging the learned latent representation to be **disentangled** -- meaning that each latent dimension responds to changes in a single underlying generative factor while remaining invariant to changes in other factors. In financial markets, these generative factors correspond to the independent forces driving price dynamics: macroeconomic trends, sector rotations, volatility regimes, liquidity conditions, and momentum cycles.

### The Entanglement Problem

Standard VAEs learn latent spaces where individual dimensions encode arbitrary mixtures of generative factors, making the representation difficult to interpret or control:

```
Standard VAE Latent Space (Entangled):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  z_1 = 0.3*trend + 0.5*volatility + 0.2*momentum       │
│  z_2 = 0.6*trend - 0.1*volatility + 0.3*correlation    │
│  z_3 = 0.1*trend + 0.4*volatility - 0.5*momentum       │
│  z_4 = 0.2*correlation + 0.3*momentum + 0.5*liquidity  │
│                                                         │
│  Problem: Changing z_1 simultaneously affects           │
│  trend, volatility, AND momentum predictions.           │
│  No single dimension is interpretable.                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

Disentangled VAE Latent Space:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  z_1 ≈ trend        (bull/bear market direction)        │
│  z_2 ≈ volatility   (calm/turbulent regime)             │
│  z_3 ≈ momentum     (mean-reverting/trending)           │
│  z_4 ≈ correlation  (diversified/correlated regime)     │
│                                                         │
│  Benefit: Each dimension captures ONE factor.           │
│  Traversing z_2 only changes volatility regime.         │
│  Factors can be used directly as trading signals.       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

The entanglement problem is particularly severe in financial data because:
- **Regime transitions** affect multiple market characteristics simultaneously
- **Factor interactions** are complex but often have clear economic interpretations
- **Risk management** requires understanding individual factor contributions
- **Regulatory requirements** increasingly demand model interpretability

### Key Advantages

1. **Interpretability**
   - Each latent dimension maps to a recognizable market factor
   - Factor traversals reveal how individual forces shape market dynamics
   - Portfolio attribution becomes straightforward

2. **Controllable Generation**
   - Generate synthetic scenarios by manipulating individual factors
   - Stress test portfolios by isolating specific risk dimensions
   - Create targeted training data for downstream models

3. **Factor Isolation**
   - Construct trading signals from individual market drivers
   - Avoid signal contamination from correlated factors
   - Build factor-neutral strategies with precision

4. **Regime Detection**
   - Monitor latent dimensions for regime shifts
   - Detect when volatility or correlation regimes change
   - Adapt strategies based on identified factor states

### Comparison of Disentanglement Methods

| Method | Mechanism | Strengths | Weaknesses | Financial Use Case |
|--------|-----------|-----------|------------|-------------------|
| **beta-VAE** | KL penalty weight beta > 1 | Simple, stable training | Reconstruction vs disentanglement tradeoff | Quick prototyping, regime detection |
| **FactorVAE** | Discriminator on TC | Better reconstruction at same disentanglement | Requires auxiliary discriminator | High-fidelity factor modeling |
| **DIP-VAE** | Covariance matching | No auxiliary network needed | Weaker disentanglement guarantees | Decorrelated factor extraction |
| **beta-TCVAE** | TC decomposition | Targets TC directly, no discriminator | Higher variance gradients | Precise factor separation |
| **Standard VAE** | beta = 1 | Best reconstruction | Poor disentanglement | Baseline, anomaly detection |

## Disentanglement Methods

### beta-VAE

The beta-VAE (Higgins et al., 2017) is the simplest disentanglement approach, modifying the standard VAE by introducing a hyperparameter beta > 1 that increases the weight of the KL divergence term:

```
beta-VAE Training Process:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Input: Market features x = [OHLCV, indicators, ...]    │
│                     │                                    │
│                     ▼                                    │
│            ┌────────────────┐                            │
│            │    Encoder     │                            │
│            │  q(z|x)        │                            │
│            └───────┬────────┘                            │
│                    │                                     │
│              ┌─────┴─────┐                               │
│              │ mu, sigma  │                              │
│              └─────┬──────┘                              │
│                    │ z ~ N(mu, sigma^2)                   │
│                    ▼                                     │
│            ┌────────────────┐                            │
│            │    Decoder     │                            │
│            │  p(x|z)        │                            │
│            └───────┬────────┘                            │
│                    │                                     │
│                    ▼                                     │
│  Loss = -E[log p(x|z)] + β * KL(q(z|x) || p(z))       │
│           ↑ reconstruction    ↑ disentanglement          │
│                                                          │
│  β = 1: Standard VAE (no disentanglement pressure)      │
│  β > 1: Forces latent dimensions to be independent      │
│  β >> 1: Strong disentanglement but poor reconstruction  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

The increased KL penalty forces the encoder to use the latent capacity efficiently, allocating each dimension to capture the most informative independent factor.

### FactorVAE

FactorVAE (Kim & Mnih, 2018) directly penalizes the **Total Correlation** (TC) of the latent distribution using a discriminator network, avoiding the reconstruction-disentanglement tradeoff of beta-VAE:

```
FactorVAE Architecture:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Input x ──► Encoder q(z|x) ──► z ──► Decoder p(x|z)   │
│                                  │                       │
│                                  ▼                       │
│                        ┌──────────────────┐              │
│                        │  Discriminator D  │             │
│                        │                  │              │
│                        │  Distinguishes:  │              │
│                        │  q(z) vs q~(z)   │              │
│                        │                  │              │
│                        │  q(z): joint      │             │
│                        │  q~(z): product   │             │
│                        │   of marginals   │              │
│                        └──────────────────┘              │
│                                                          │
│  q~(z) is created by independently shuffling each       │
│  latent dimension across the batch:                      │
│                                                          │
│  Batch z:   [z1_a, z2_a, z3_a]                          │
│             [z1_b, z2_b, z3_b]                           │
│             [z1_c, z2_c, z3_c]                           │
│                                                          │
│  Shuffled:  [z1_c, z2_a, z3_b]  ← product of marginals │
│             [z1_a, z2_c, z3_a]                           │
│             [z1_b, z2_b, z3_c]                           │
│                                                          │
│  Loss_VAE = Reconstruction + KL + γ * TC(z)             │
│  Loss_D   = Binary cross-entropy(real vs shuffled)      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### DIP-VAE

DIP-VAE (Disentangled Inferred Prior, Kumar et al., 2018) encourages disentanglement by matching the covariance structure of the aggregated posterior to a diagonal matrix:

```
DIP-VAE Constraint:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Standard VAE matches: q(z|x) ≈ p(z) per data point    │
│                                                          │
│  DIP-VAE additionally matches the aggregate posterior:   │
│                                                          │
│  Cov_x[E_q(z|x)[z]] ≈ I  (identity matrix)             │
│                                                          │
│  DIP-VAE-I:  Penalizes off-diagonal elements of         │
│              Cov(mu_z) to decorrelate means              │
│                                                          │
│  DIP-VAE-II: Penalizes off-diagonal elements of         │
│              Cov_q(z) (full covariance)                  │
│                                                          │
│  Covariance Matrix Target:                               │
│                                                          │
│  ┌─────────────────┐     ┌─────────────────┐            │
│  │ 1.0  0.3  0.5   │     │ 1.0  0.0  0.0   │           │
│  │ 0.3  1.0  0.2   │ ──► │ 0.0  1.0  0.0   │           │
│  │ 0.5  0.2  1.0   │     │ 0.0  0.0  1.0   │           │
│  └─────────────────┘     └─────────────────┘            │
│    Entangled              Disentangled                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### beta-TCVAE

beta-TCVAE (Chen et al., 2018) decomposes the KL divergence into three interpretable terms and applies a stronger penalty specifically to the Total Correlation term:

```
KL Decomposition in beta-TCVAE:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  KL(q(z|x) || p(z)) = Index-Code MI                     │
│                       + Total Correlation (TC)           │
│                       + Dimension-wise KL                │
│                                                          │
│  1. Index-Code MI:   I(x; z)                             │
│     Mutual info between data and latent code             │
│     → Preserves useful information about input           │
│                                                          │
│  2. Total Correlation: KL(q(z) || Π_j q(z_j))          │
│     Divergence from factorial (independent) posterior    │
│     → THIS is what we want to minimize!                  │
│                                                          │
│  3. Dimension-wise KL: Σ_j KL(q(z_j) || p(z_j))       │
│     Per-dimension prior matching                         │
│     → Regularizes individual latent dimensions           │
│                                                          │
│  beta-TCVAE Loss:                                        │
│  L = Reconstruction + α * MI + β * TC + γ * DW-KL      │
│                                                          │
│  Typical: α = 1, β > 1, γ = 1                          │
│  Only the TC term gets extra weight                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Mathematical Foundation

### ELBO Decomposition

The standard VAE maximizes the Evidence Lower Bound (ELBO):

```
log p(x) ≥ ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

Where:
  p(x|z) : Decoder (likelihood)
  q(z|x) : Encoder (approximate posterior)
  p(z)   : Prior, typically N(0, I)
```

The KL term can be decomposed (Hoffman & Johnson, 2016; Chen et al., 2018):

```
E_p(x)[KL(q(z|x) || p(z))]

  = E_p(x)[KL(q(z|x) || q(z))]     ... Index-Code MI
  + KL(q(z) || Π_j q(z_j))          ... Total Correlation
  + Σ_j KL(q(z_j) || p(z_j))        ... Dimension-wise KL

Where:
  q(z) = E_p(x)[q(z|x)]             ... Aggregated posterior
  q(z_j) = ∫ q(z) dz_{\j}           ... j-th marginal
```

### Total Correlation

Total Correlation (TC) measures the degree of statistical dependence among the latent dimensions:

```
TC(z) = KL(q(z) || Π_j q(z_j))

     = E_q(z)[log q(z) - Σ_j log q(z_j)]

Properties:
  TC ≥ 0                  (always non-negative)
  TC = 0  ⟺  z_j ⊥ z_k   (zero iff all dimensions independent)

For Gaussian q(z) with covariance Σ:
  TC = (1/2) * [Σ_j log σ_jj - log det(Σ)]

  If Σ is diagonal:  TC = 0  (independent dimensions)
  If Σ has off-diag:  TC > 0  (correlated dimensions)
```

In the financial context, minimizing TC ensures that each latent factor captures a distinct market phenomenon:

```
Financial Interpretation of Low TC:
┌───────────────────────────────────────────────────────┐
│                                                       │
│  High TC (entangled):                                 │
│  z_trend and z_vol are correlated                     │
│  → Cannot trade trend without volatility exposure     │
│  → Factor attribution is ambiguous                    │
│                                                       │
│  Low TC (disentangled):                               │
│  z_trend ⊥ z_vol                                     │
│  → Pure trend signal, independent of vol regime       │
│  → Clean factor decomposition for risk management     │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### beta-VAE Objective

The beta-VAE modifies the ELBO with a single hyperparameter:

```
L_beta-VAE = E_q(z|x)[log p(x|z)] - β * KL(q(z|x) || p(z))

For β > 1:
  - Stronger pressure for q(z|x) to match N(0, I)
  - Forces encoder to use latent capacity efficiently
  - Each dimension must independently justify its deviation from prior

Information bottleneck perspective:
  - β controls the tradeoff between compression and reconstruction
  - Higher β → more compressed, more disentangled representation
  - Optimal β depends on data complexity and latent dimensionality

Practical range for financial data:
  β ∈ [1, 10]:   Mild disentanglement
  β ∈ [10, 50]:  Strong disentanglement
  β ∈ [50, 200]: Very strong (may lose reconstruction quality)
```

### FactorVAE Objective

FactorVAE adds a density-ratio trick to estimate TC using a discriminator:

```
L_FactorVAE = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z)) - γ * TC(z)

Where TC(z) is estimated via a discriminator D:

  TC(z) ≈ E_q(z)[log D(z) / (1 - D(z))]

The discriminator is trained to distinguish:
  - Real samples:    z ~ q(z)           (joint posterior)
  - Shuffled samples: z~ ~ Π_j q(z_j)  (product of marginals)

Loss_D = -E_q(z)[log D(z)] - E_Π q(z_j)[log(1 - D(z~))]

Advantage over beta-VAE:
  - Only penalizes TC, not MI or dimension-wise KL
  - Better reconstruction quality at same disentanglement level
  - γ typically in range [10, 100] for financial data
```

### Disentanglement Metrics

Three key metrics for evaluating disentanglement quality:

```
1. DCI (Disentanglement, Completeness, Informativeness):
   ──────────────────────────────────────────────────────
   Train a predictor from each z_j to each ground truth factor v_k.

   Disentanglement: Each z_j should predict only one v_k
   D_j = 1 - H(p_j) / log(K)
   where p_jk = |R_jk| / Σ_k |R_jk| and R is the importance matrix

   Completeness: Each v_k should be predicted by only one z_j
   C_k = 1 - H(p_k) / log(J)

2. MIG (Mutual Information Gap):
   ──────────────────────────────
   For each ground truth factor v_k:
     Compute MI(z_j; v_k) for all j
     Sort MI values in descending order
     MIG_k = [MI_top1 - MI_top2] / H(v_k)

   MIG = (1/K) * Σ_k MIG_k

   Higher MIG → each factor is primarily captured by one z_j

3. SAP (Separated Attribute Predictability):
   ──────────────────────────────────────────
   For each factor v_k:
     Train linear classifiers from each z_j to v_k
     SAP_k = accuracy_top1 - accuracy_top2

   Higher SAP → clear separation between most and second-most
                predictive latent dimension for each factor
```

## Practical Examples

### 01: Data Preparation

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader


def prepare_disentangled_vae_data(
    symbols: List[str],
    window_size: int = 60,
    features: List[str] = None,
    source: str = 'bybit'
) -> Dict:
    """
    Prepare multi-asset financial data for Disentangled VAE training.

    The data is structured so that the VAE can discover independent
    generative factors across multiple assets and feature types.

    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'] for Bybit
                 or ['AAPL', 'MSFT', 'GOOGL'] for stocks)
        window_size: Number of time steps per sample
        features: Feature columns to compute
        source: Data source ('bybit' or 'yfinance')

    Returns:
        Dictionary with X (features), metadata, and normalization params
    """
    if features is None:
        features = [
            'log_return', 'volume_ratio', 'volatility_20',
            'rsi_14', 'macd_signal', 'bb_position',
            'spread', 'skewness_20'
        ]

    all_frames = []

    for symbol in symbols:
        if source == 'bybit':
            df = load_bybit_klines(symbol, interval='1h')
        else:
            df = load_stock_data(symbol, interval='1d')

        # Core features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility_20'] = df['log_return'].rolling(20).std()
        df['rsi_14'] = compute_rsi(df['close'], period=14)

        # MACD signal
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        df['macd_signal'] = macd - macd.ewm(span=9).mean()

        # Bollinger Band position
        ma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - ma_20) / (2 * std_20 + 1e-8)

        # Bid-ask spread proxy
        df['spread'] = (df['high'] - df['low']) / df['close']

        # Return skewness
        df['skewness_20'] = df['log_return'].rolling(20).skew()

        all_frames.append(df[features].dropna())

    # Concatenate features across assets
    combined = pd.concat(all_frames, axis=1, keys=symbols)
    combined = combined.dropna()

    # Normalize each feature column to zero mean, unit variance
    means = combined.mean()
    stds = combined.std() + 1e-8
    normalized = (combined - means) / stds

    # Create sliding windows
    X = []
    for i in range(window_size, len(normalized)):
        window = normalized.iloc[i - window_size:i].values
        X.append(window)

    X = np.array(X, dtype=np.float32)

    # Flatten window into feature vector for VAE input
    # Shape: (num_samples, window_size * num_assets * num_features)
    X_flat = X.reshape(X.shape[0], -1)

    return {
        'X': X_flat,
        'X_windowed': X,
        'symbols': symbols,
        'features': features,
        'window_size': window_size,
        'means': means.values,
        'stds': stds.values,
        'input_dim': X_flat.shape[1]
    }


class DisentangledVAEDataset(Dataset):
    """PyTorch Dataset for Disentangled VAE training."""

    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def load_bybit_klines(symbol: str, interval: str = '1h') -> pd.DataFrame:
    """Load historical kline data from Bybit API."""
    import requests

    url = "https://api.bybit.com/v5/market/kline"
    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': '60' if interval == '1h' else interval,
        'limit': 1000
    }

    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def load_stock_data(symbol: str, interval: str = '1d') -> pd.DataFrame:
    """Load stock data using yfinance."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(period='2y', interval=interval)
    df.columns = [c.lower() for c in df.columns]
    return df.reset_index()


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))
```

### 02: beta-VAE Architecture

See [python/model.py](python/model.py) for complete implementation.

```python
# python/model.py - Core Disentangled VAE implementations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class Encoder(nn.Module):
    """
    Shared encoder architecture for all disentangled VAE variants.

    Maps financial feature vectors to latent distribution parameters.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list = None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim

        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.network(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    Shared decoder architecture for all disentangled VAE variants.

    Reconstructs financial feature vectors from latent codes.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list = None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 512]

        layers = []
        in_dim = latent_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class BetaVAE(nn.Module):
    """
    beta-VAE for financial time series disentanglement.

    Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a
    Constrained Variational Framework" (ICLR 2017)

    The beta parameter controls the disentanglement-reconstruction tradeoff.
    Higher beta encourages more independent latent factors at the cost of
    reconstruction fidelity.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        hidden_dims: list = None,
        beta: float = 4.0
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = Encoder(input_dim, latent_dim, hidden_dims)
        self.decoder = Decoder(
            latent_dim, input_dim,
            hidden_dims=list(reversed(hidden_dims)) if hidden_dims else None
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        return {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        beta-VAE loss: L = Recon + beta * KL

        Returns individual loss components for monitoring.
        """
        x_recon = outputs['x_recon']
        mu = outputs['mu']
        logvar = outputs['logvar']

        # Reconstruction loss (MSE for continuous financial data)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta': torch.tensor(self.beta)
        }


class FactorVAE(nn.Module):
    """
    FactorVAE for financial factor disentanglement.

    Kim & Mnih, "Disentangling by Factorising" (ICML 2018)

    Uses a discriminator to directly penalize Total Correlation,
    achieving better disentanglement without sacrificing reconstruction.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        hidden_dims: list = None,
        gamma: float = 35.0
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.gamma = gamma

        self.encoder = Encoder(input_dim, latent_dim, hidden_dims)
        self.decoder = Decoder(
            latent_dim, input_dim,
            hidden_dims=list(reversed(hidden_dims)) if hidden_dims else None
        )

        # Discriminator for TC estimation
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2)  # Binary: joint vs product of marginals
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        return {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

    def permute_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Create samples from product of marginals by permuting
        each latent dimension independently across the batch.
        """
        z_perm = z.clone()
        batch_size = z.shape[0]

        for j in range(self.latent_dim):
            perm_idx = torch.randperm(batch_size, device=z.device)
            z_perm[:, j] = z[perm_idx, j]

        return z_perm

    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """FactorVAE loss: Recon + KL + gamma * TC."""
        x_recon = outputs['x_recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        z = outputs['z']

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # Standard KL divergence (beta = 1)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        # TC estimation via discriminator
        d_z = self.discriminator(z)
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()

        total_loss = recon_loss + kl_loss + self.gamma * tc_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'tc_loss': tc_loss
        }

    def discriminator_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Train discriminator to distinguish q(z) from product of marginals."""
        z = z.detach()
        z_perm = self.permute_latent(z)

        d_real = self.discriminator(z)
        d_perm = self.discriminator(z_perm)

        # Real samples labeled as 0, permuted as 1
        real_labels = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        perm_labels = torch.ones(z.shape[0], dtype=torch.long, device=z.device)

        loss = 0.5 * (
            F.cross_entropy(d_real, real_labels) +
            F.cross_entropy(d_perm, perm_labels)
        )

        return loss
```

### 03: Model Training with Disentanglement

```python
# python/03_train_model.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import BetaVAE, FactorVAE
from data import DisentangledVAEDataset, prepare_disentangled_vae_data
import numpy as np


def train_beta_vae(
    model: BetaVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 200,
    lr: float = 1e-3,
    beta_schedule: str = 'warmup'
) -> dict:
    """
    Train beta-VAE with optional beta warmup schedule.

    Beta warmup gradually increases beta from 0 to target value,
    allowing the model to first learn good reconstructions before
    enforcing disentanglement constraints.

    Args:
        model: BetaVAE instance
        train_loader: Training data
        val_loader: Validation data
        epochs: Number of training epochs
        lr: Learning rate
        beta_schedule: 'fixed', 'warmup', or 'cyclical'

    Returns:
        Training history dictionary
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    target_beta = model.beta
    history = {
        'train_loss': [], 'val_loss': [],
        'recon_loss': [], 'kl_loss': [], 'beta_values': []
    }

    for epoch in range(epochs):
        model.train()
        epoch_losses = {'loss': 0, 'recon_loss': 0, 'kl_loss': 0}

        # Beta scheduling
        if beta_schedule == 'warmup':
            warmup_epochs = min(50, epochs // 4)
            current_beta = target_beta * min(1.0, epoch / warmup_epochs)
        elif beta_schedule == 'cyclical':
            cycle_length = 30
            cycle_pos = (epoch % cycle_length) / cycle_length
            current_beta = target_beta * cycle_pos
        else:
            current_beta = target_beta

        model.beta = current_beta
        history['beta_values'].append(current_beta)

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(batch)
            losses = model.loss_function(batch, outputs)

            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

        # Average epoch losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Validation
        val_loss = evaluate(model, val_loader)

        history['train_loss'].append(epoch_losses['loss'])
        history['val_loss'].append(val_loss)
        history['recon_loss'].append(epoch_losses['recon_loss'])
        history['kl_loss'].append(epoch_losses['kl_loss'])

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {epoch_losses['loss']:.4f} | "
                f"Recon: {epoch_losses['recon_loss']:.4f} | "
                f"KL: {epoch_losses['kl_loss']:.4f} | "
                f"Beta: {current_beta:.2f} | "
                f"Val: {val_loss:.4f}"
            )

    model.beta = target_beta
    return history


def train_factor_vae(
    model: FactorVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 200,
    lr_vae: float = 1e-3,
    lr_disc: float = 1e-4
) -> dict:
    """
    Train FactorVAE with alternating VAE and discriminator updates.

    The discriminator learns to distinguish joint samples from
    products of marginals, providing a TC gradient to the VAE.
    """
    # Separate optimizers for VAE and discriminator
    vae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer_vae = optim.Adam(vae_params, lr=lr_vae, weight_decay=1e-5)
    optimizer_disc = optim.Adam(
        model.discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.9)
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'recon_loss': [], 'kl_loss': [],
        'tc_loss': [], 'disc_loss': []
    }

    for epoch in range(epochs):
        model.train()
        epoch_losses = {
            'loss': 0, 'recon_loss': 0, 'kl_loss': 0,
            'tc_loss': 0, 'disc_loss': 0
        }

        for batch in train_loader:
            # Step 1: Update VAE (encoder + decoder)
            optimizer_vae.zero_grad()
            outputs = model(batch)
            vae_losses = model.loss_function(batch, outputs)
            vae_losses['loss'].backward(retain_graph=True)
            optimizer_vae.step()

            # Step 2: Update discriminator
            optimizer_disc.zero_grad()
            z = outputs['z'].detach()
            disc_loss = model.discriminator_loss(z)
            disc_loss.backward()
            optimizer_disc.step()

            epoch_losses['loss'] += vae_losses['loss'].item()
            epoch_losses['recon_loss'] += vae_losses['recon_loss'].item()
            epoch_losses['kl_loss'] += vae_losses['kl_loss'].item()
            epoch_losses['tc_loss'] += vae_losses['tc_loss'].item()
            epoch_losses['disc_loss'] += disc_loss.item()

        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        val_loss = evaluate(model, val_loader)

        for key in epoch_losses:
            if key in history:
                history[key].append(epoch_losses[key])
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {epoch_losses['loss']:.4f} | "
                f"Recon: {epoch_losses['recon_loss']:.4f} | "
                f"TC: {epoch_losses['tc_loss']:.4f} | "
                f"Disc: {epoch_losses['disc_loss']:.4f}"
            )

    return history


def evaluate(model, loader):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            losses = model.loss_function(batch, outputs)
            total_loss += losses['loss'].item()

    return total_loss / len(loader)


# Example usage
if __name__ == '__main__':
    # Prepare data
    data = prepare_disentangled_vae_data(
        symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        window_size=60,
        source='bybit'
    )

    dataset = DisentangledVAEDataset(data['X'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)

    # Train beta-VAE
    beta_vae = BetaVAE(
        input_dim=data['input_dim'],
        latent_dim=10,
        hidden_dims=[512, 256, 128],
        beta=8.0
    )
    print(f"beta-VAE parameters: {sum(p.numel() for p in beta_vae.parameters()):,}")

    history = train_beta_vae(
        beta_vae, train_loader, val_loader,
        epochs=200, beta_schedule='warmup'
    )
```

### 04: Latent Factor Analysis

```python
# python/04_latent_analysis.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from model import BetaVAE


def extract_latent_factors(
    model: BetaVAE,
    data_loader,
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Extract latent factors from trained disentangled VAE.

    Returns:
        Dictionary with latent means, logvars, and samples
    """
    model.eval()
    all_mu, all_logvar, all_z = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch)
            all_mu.append(outputs['mu'].cpu().numpy())
            all_logvar.append(outputs['logvar'].cpu().numpy())
            all_z.append(outputs['z'].cpu().numpy())

    return {
        'mu': np.concatenate(all_mu, axis=0),
        'logvar': np.concatenate(all_logvar, axis=0),
        'z': np.concatenate(all_z, axis=0)
    }


def latent_traversal(
    model: BetaVAE,
    base_sample: torch.Tensor,
    dim: int,
    range_vals: np.ndarray = None,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Perform latent traversal along a single dimension.

    Fixes all latent dimensions at base_sample values except dim,
    which is varied across range_vals. This reveals what market
    characteristic that dimension encodes.

    Args:
        model: Trained disentangled VAE
        base_sample: Reference input sample
        dim: Latent dimension to traverse
        range_vals: Values to sweep (default: -3 to 3)

    Returns:
        Reconstructions for each traversal value
    """
    if range_vals is None:
        range_vals = np.linspace(-3, 3, 11)

    model.eval()
    base_sample = base_sample.to(device)

    with torch.no_grad():
        mu, logvar = model.encoder(base_sample.unsqueeze(0))

    reconstructions = []

    for val in range_vals:
        z = mu.clone()
        z[0, dim] = val

        with torch.no_grad():
            recon = model.decoder(z)
            reconstructions.append(recon.cpu().numpy().squeeze())

    return np.array(reconstructions)


def identify_factor_meanings(
    latent_factors: Dict[str, np.ndarray],
    market_indicators: Dict[str, np.ndarray],
    method: str = 'correlation'
) -> Dict[int, Dict]:
    """
    Identify what each latent dimension represents by correlating
    with known market indicators.

    Args:
        latent_factors: Output from extract_latent_factors
        market_indicators: Dict of indicator_name -> values
            e.g., {'realized_vol': [...], 'trend_strength': [...]}
        method: 'correlation' or 'mutual_information'

    Returns:
        Mapping from latent dim to identified factor and correlation
    """
    mu = latent_factors['mu']
    latent_dim = mu.shape[1]

    factor_map = {}

    for dim in range(latent_dim):
        z_dim = mu[:, dim]
        best_indicator = None
        best_score = 0

        for name, values in market_indicators.items():
            # Ensure aligned lengths
            min_len = min(len(z_dim), len(values))
            z_slice = z_dim[:min_len]
            v_slice = values[:min_len]

            if method == 'correlation':
                score = abs(np.corrcoef(z_slice, v_slice)[0, 1])
            else:
                score = mutual_information(z_slice, v_slice)

            if score > best_score:
                best_score = score
                best_indicator = name

        factor_map[dim] = {
            'indicator': best_indicator,
            'score': best_score,
            'active_range': (float(z_dim.mean() - 2 * z_dim.std()),
                           float(z_dim.mean() + 2 * z_dim.std()))
        }

    return factor_map


def compute_disentanglement_metrics(
    latent_factors: Dict[str, np.ndarray],
    ground_truth_factors: np.ndarray
) -> Dict[str, float]:
    """
    Compute DCI, MIG, and SAP disentanglement metrics.

    Args:
        latent_factors: Extracted latent representations
        ground_truth_factors: Known generative factors (K dimensions)

    Returns:
        Dictionary with metric scores
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    mu = latent_factors['mu']
    J = mu.shape[1]  # Number of latent dims
    K = ground_truth_factors.shape[1]  # Number of true factors

    # --- DCI Metric ---
    importance_matrix = np.zeros((J, K))

    for k in range(K):
        reg = GradientBoostingRegressor(n_estimators=50, max_depth=3)
        reg.fit(mu, ground_truth_factors[:, k])
        importance_matrix[:, k] = reg.feature_importances_

    # Disentanglement: each z_j predicts only one v_k
    dci_disentanglement = 0
    for j in range(J):
        row = importance_matrix[j, :]
        if row.sum() > 0:
            p = row / row.sum()
            entropy = -np.sum(p * np.log(p + 1e-10))
            dci_disentanglement += 1 - entropy / np.log(K)
    dci_disentanglement /= J

    # Completeness: each v_k is predicted by only one z_j
    dci_completeness = 0
    for k in range(K):
        col = importance_matrix[:, k]
        if col.sum() > 0:
            p = col / col.sum()
            entropy = -np.sum(p * np.log(p + 1e-10))
            dci_completeness += 1 - entropy / np.log(J)
    dci_completeness /= K

    # --- MIG Metric ---
    mig = 0
    for k in range(K):
        mi_values = []
        for j in range(J):
            mi = mutual_information(mu[:, j], ground_truth_factors[:, k])
            mi_values.append(mi)
        mi_sorted = sorted(mi_values, reverse=True)
        h_k = entropy_estimate(ground_truth_factors[:, k])
        mig += (mi_sorted[0] - mi_sorted[1]) / (h_k + 1e-10)
    mig /= K

    # --- SAP Metric ---
    sap = 0
    for k in range(K):
        accuracies = []
        for j in range(J):
            reg = LinearRegression()
            reg.fit(mu[:, j:j+1], ground_truth_factors[:, k])
            acc = reg.score(mu[:, j:j+1], ground_truth_factors[:, k])
            accuracies.append(max(0, acc))
        acc_sorted = sorted(accuracies, reverse=True)
        sap += acc_sorted[0] - acc_sorted[1]
    sap /= K

    return {
        'dci_disentanglement': dci_disentanglement,
        'dci_completeness': dci_completeness,
        'mig': mig,
        'sap': sap
    }


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Estimate mutual information between two continuous variables."""
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi = 0
    for i in range(bins):
        for j in range(bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
    return mi


def entropy_estimate(x: np.ndarray, bins: int = 20) -> float:
    """Estimate entropy of a continuous variable."""
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist / hist.sum()
    return -np.sum(hist * np.log(hist + 1e-10))


def visualize_traversals(
    model: BetaVAE,
    base_sample: torch.Tensor,
    factor_map: Dict[int, Dict],
    features: List[str],
    num_dims: int = 5
):
    """
    Visualize latent traversals for the top disentangled dimensions.

    Creates a grid showing how each latent dimension affects
    the reconstructed market features.
    """
    fig, axes = plt.subplots(num_dims, 1, figsize=(14, 3 * num_dims))

    sorted_dims = sorted(
        factor_map.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )[:num_dims]

    for idx, (dim, info) in enumerate(sorted_dims):
        traversals = latent_traversal(model, base_sample, dim)
        range_vals = np.linspace(-3, 3, traversals.shape[0])

        # Plot how key features change during traversal
        ax = axes[idx]
        num_features_to_show = min(4, traversals.shape[1])
        for f in range(num_features_to_show):
            ax.plot(range_vals, traversals[:, f], label=features[f % len(features)])

        ax.set_title(
            f"z_{dim}: {info['indicator']} "
            f"(correlation: {info['score']:.3f})"
        )
        ax.set_xlabel(f"z_{dim} value")
        ax.set_ylabel("Feature value")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('latent_traversals.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### 05: Trading Strategy Using Disentangled Factors

```python
# python/05_trading_strategy.py

import torch
import numpy as np
from typing import Dict, List, Optional
from model import BetaVAE
from data import prepare_disentangled_vae_data


class DisentangledFactorStrategy:
    """
    Trading strategy based on disentangled latent factors.

    Each latent dimension captures an independent market factor
    (trend, volatility, momentum, etc.). The strategy generates
    signals by monitoring these factors and combining them with
    factor-specific rules.
    """

    def __init__(
        self,
        model: BetaVAE,
        factor_map: Dict[int, Dict],
        trend_dim: int,
        vol_dim: int,
        momentum_dim: int,
        correlation_dim: Optional[int] = None
    ):
        self.model = model
        self.factor_map = factor_map
        self.trend_dim = trend_dim
        self.vol_dim = vol_dim
        self.momentum_dim = momentum_dim
        self.correlation_dim = correlation_dim

        self.model.eval()

    def extract_factors(self, x: torch.Tensor) -> Dict[str, float]:
        """Extract disentangled factors from market data."""
        with torch.no_grad():
            mu, _ = self.model.encoder(x.unsqueeze(0))
            z = mu.squeeze(0).numpy()

        factors = {
            'trend': z[self.trend_dim],
            'volatility': z[self.vol_dim],
            'momentum': z[self.momentum_dim],
        }

        if self.correlation_dim is not None:
            factors['correlation'] = z[self.correlation_dim]

        return factors

    def generate_signal(
        self,
        factors: Dict[str, float],
        trend_threshold: float = 0.5,
        vol_threshold: float = 1.0,
        momentum_threshold: float = 0.3
    ) -> Dict[str, float]:
        """
        Generate trading signal from disentangled factors.

        The key advantage: each factor contributes independently
        to the final signal, making attribution transparent.

        Returns:
            Dictionary with signal, position size, and factor contributions
        """
        # Trend component: bullish if trend factor is positive
        trend_signal = np.tanh(factors['trend'] / trend_threshold)

        # Volatility filter: reduce position in high volatility
        vol_factor = factors['volatility']
        vol_scaling = 1.0 / (1.0 + np.exp(vol_factor - vol_threshold))

        # Momentum confirmation: strengthen signal if aligned
        momentum_signal = np.tanh(factors['momentum'] / momentum_threshold)
        momentum_weight = 0.3

        # Combined signal
        raw_signal = (
            (1 - momentum_weight) * trend_signal +
            momentum_weight * momentum_signal
        )

        # Apply volatility scaling
        position_size = raw_signal * vol_scaling

        # Correlation regime adjustment
        if self.correlation_dim is not None:
            corr_factor = factors['correlation']
            # Reduce exposure in high-correlation regimes (systemic risk)
            if corr_factor > 1.5:
                position_size *= 0.5

        return {
            'signal': float(np.sign(position_size)),
            'position_size': float(np.clip(position_size, -1, 1)),
            'trend_contribution': float(trend_signal),
            'vol_scaling': float(vol_scaling),
            'momentum_contribution': float(momentum_signal),
            'factor_values': factors
        }


def backtest_disentangled_strategy(
    model: BetaVAE,
    factor_map: Dict[int, Dict],
    test_data: np.ndarray,
    test_returns: np.ndarray,
    trend_dim: int,
    vol_dim: int,
    momentum_dim: int,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001
) -> Dict:
    """
    Backtest the disentangled factor strategy.

    Args:
        model: Trained disentangled VAE
        factor_map: Latent dimension to factor mapping
        test_data: Test feature vectors
        test_returns: Actual returns for PnL calculation
        trend_dim: Latent dimension identified as trend
        vol_dim: Latent dimension identified as volatility
        momentum_dim: Latent dimension identified as momentum
        initial_capital: Starting capital
        transaction_cost: Fee per trade as fraction

    Returns:
        Backtest results with performance metrics
    """
    strategy = DisentangledFactorStrategy(
        model=model,
        factor_map=factor_map,
        trend_dim=trend_dim,
        vol_dim=vol_dim,
        momentum_dim=momentum_dim
    )

    capital = initial_capital
    position = 0.0
    capital_history = [capital]
    returns_history = []
    positions_history = []
    factor_history = []

    for i in range(len(test_data)):
        x = torch.tensor(test_data[i], dtype=torch.float32)

        # Extract factors and generate signal
        factors = strategy.extract_factors(x)
        signal = strategy.generate_signal(factors)

        target_position = signal['position_size']

        # Transaction costs
        position_change = abs(target_position - position)
        costs = position_change * transaction_cost * capital

        # PnL
        actual_return = test_returns[i]
        pnl = position * actual_return * capital - costs

        capital += pnl
        position = target_position

        capital_history.append(capital)
        returns_history.append(pnl / capital_history[-2])
        positions_history.append(position)
        factor_history.append(factors)

    returns = np.array(returns_history)

    # Performance metrics
    total_return = (capital - initial_capital) / initial_capital
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
    max_dd = calculate_max_drawdown(capital_history)
    win_rate = (returns > 0).sum() / len(returns)

    # Factor attribution
    factor_df = {
        'trend': [f['trend'] for f in factor_history],
        'volatility': [f['volatility'] for f in factor_history],
        'momentum': [f['momentum'] for f in factor_history]
    }

    results = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'capital_history': capital_history,
        'returns_history': returns_history,
        'positions_history': positions_history,
        'factor_history': factor_df
    }

    print(f"Total Return:  {total_return:.2%}")
    print(f"Sharpe Ratio:  {sharpe:.3f}")
    print(f"Max Drawdown:  {max_dd:.2%}")
    print(f"Win Rate:      {win_rate:.2%}")

    return results


def calculate_max_drawdown(capital_history: list) -> float:
    """Calculate maximum drawdown from capital history."""
    peak = capital_history[0]
    max_dd = 0

    for capital in capital_history:
        if capital > peak:
            peak = capital
        drawdown = (peak - capital) / peak
        max_dd = max(max_dd, drawdown)

    return max_dd


def plot_strategy_results(results: Dict, symbol: str = 'BTCUSDT'):
    """Visualize backtest results with factor attribution."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # 1. Equity curve
    ax = axes[0]
    ax.plot(results['capital_history'], 'b-', linewidth=1.5)
    ax.set_ylabel('Capital ($)')
    ax.set_title(f'Disentangled Factor Strategy - {symbol}')
    ax.grid(True, alpha=0.3)

    # 2. Positions
    ax = axes[1]
    ax.fill_between(
        range(len(results['positions_history'])),
        results['positions_history'],
        alpha=0.5, color='green', where=np.array(results['positions_history']) > 0
    )
    ax.fill_between(
        range(len(results['positions_history'])),
        results['positions_history'],
        alpha=0.5, color='red', where=np.array(results['positions_history']) < 0
    )
    ax.set_ylabel('Position')
    ax.grid(True, alpha=0.3)

    # 3. Trend and momentum factors
    ax = axes[2]
    ax.plot(results['factor_history']['trend'], label='Trend Factor', alpha=0.8)
    ax.plot(results['factor_history']['momentum'], label='Momentum Factor', alpha=0.8)
    ax.set_ylabel('Factor Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Volatility factor
    ax = axes[3]
    ax.plot(results['factor_history']['volatility'], color='orange', alpha=0.8)
    ax.set_ylabel('Volatility Factor')
    ax.set_xlabel('Time Step')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('strategy_results.png', dpi=150, bbox_inches='tight')
    plt.show()
```

## Rust Implementation

See [disentangled_vae/](disentangled_vae/) for complete Rust implementation.

```
disentangled_vae/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                  # Main library exports
│   ├── api/                    # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs           # HTTP client for Bybit REST API
│   │   └── types.rs            # Kline, OrderBook response types
│   ├── data/                   # Data processing pipeline
│   │   ├── mod.rs
│   │   ├── loader.rs           # Data loading (Bybit + CSV)
│   │   ├── features.rs         # Feature engineering
│   │   ├── normalization.rs    # Z-score, min-max normalization
│   │   └── dataset.rs          # Windowed dataset iterator
│   ├── model/                  # Disentangled VAE architectures
│   │   ├── mod.rs
│   │   ├── config.rs           # Model hyperparameters
│   │   ├── encoder.rs          # Shared encoder network
│   │   ├── decoder.rs          # Shared decoder network
│   │   ├── beta_vae.rs         # beta-VAE implementation
│   │   ├── factor_vae.rs       # FactorVAE with discriminator
│   │   ├── dip_vae.rs          # DIP-VAE (I and II variants)
│   │   ├── beta_tcvae.rs       # beta-TCVAE implementation
│   │   └── metrics.rs          # DCI, MIG, SAP metrics
│   ├── training/               # Training loop and scheduling
│   │   ├── mod.rs
│   │   ├── trainer.rs          # Generic trainer for all variants
│   │   ├── beta_schedule.rs    # Beta warmup / cyclical schedules
│   │   └── early_stopping.rs   # Patience-based early stopping
│   ├── analysis/               # Latent space analysis
│   │   ├── mod.rs
│   │   ├── traversal.rs        # Latent dimension traversals
│   │   ├── factor_id.rs        # Factor identification via correlation
│   │   └── visualization.rs    # Plot generation (SVG output)
│   └── strategy/               # Trading strategy
│       ├── mod.rs
│       ├── signals.rs          # Factor-based signal generation
│       ├── portfolio.rs        # Position sizing and risk management
│       └── backtest.rs         # Backtesting engine with metrics
└── examples/
    ├── fetch_bybit_data.rs     # Download historical klines
    ├── train_beta_vae.rs       # Train beta-VAE model
    ├── train_factor_vae.rs     # Train FactorVAE model
    ├── analyze_factors.rs      # Latent traversal and factor ID
    └── backtest.rs             # Run full backtest
```

### Quick Start (Rust)

```bash
# Navigate to Rust project
cd disentangled_vae

# Fetch data from Bybit
cargo run --example fetch_bybit_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Train beta-VAE
cargo run --example train_beta_vae -- \
    --latent-dim 10 \
    --beta 8.0 \
    --epochs 200 \
    --batch-size 128 \
    --beta-schedule warmup

# Train FactorVAE
cargo run --example train_factor_vae -- \
    --latent-dim 10 \
    --gamma 35.0 \
    --epochs 200

# Analyze latent factors
cargo run --example analyze_factors -- --model-path models/beta_vae.bin

# Run backtest
cargo run --example backtest -- \
    --model-path models/beta_vae.bin \
    --start 2024-01-01 \
    --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── model.py                    # BetaVAE, FactorVAE, DIP-VAE, beta-TCVAE
├── data.py                     # Data loading and preprocessing
├── train.py                    # Training loops for all variants
├── analysis.py                 # Latent traversals and factor identification
├── strategy.py                 # Disentangled factor trading strategy
├── metrics.py                  # Disentanglement metrics (DCI, MIG, SAP)
├── example_usage.py            # Complete end-to-end example
├── requirements.txt            # Dependencies
└── __init__.py                 # Package initialization
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete example
python example_usage.py

# Train beta-VAE
python train.py --model beta_vae --beta 8.0 --latent-dim 10 --epochs 200

# Train FactorVAE
python train.py --model factor_vae --gamma 35.0 --latent-dim 10 --epochs 200

# Analyze learned factors
python analysis.py --model-path checkpoints/beta_vae_best.pt

# Run backtest
python strategy.py --model-path checkpoints/beta_vae_best.pt --symbol BTCUSDT
```

## Best Practices

### When to Use Disentangled VAEs

**Ideal use cases:**
- Interpretable factor discovery in multi-asset portfolios
- Regime detection with clear factor attribution
- Stress testing by manipulating individual risk factors
- Generating synthetic scenarios for specific market conditions
- Building factor-neutral or factor-targeted strategies

**Consider alternatives for:**
- Pure prediction tasks where interpretability is not required (use standard VAE or supervised models)
- Very low-dimensional data where factors are already observable
- Real-time inference with extreme latency constraints (disentangled models are larger)
- Small datasets where the disentanglement pressure causes underfitting

### Hyperparameter Recommendations

| Parameter | beta-VAE | FactorVAE | DIP-VAE | beta-TCVAE | Notes |
|-----------|----------|-----------|---------|------------|-------|
| `latent_dim` | 8-15 | 8-15 | 8-15 | 8-15 | More dims than expected factors |
| `beta` / `gamma` | 4-50 | 10-100 | N/A | 2-20 | Start low, increase gradually |
| `lambda_diag` | N/A | N/A | 10-100 | N/A | DIP covariance penalty |
| `lambda_offdiag` | N/A | N/A | 5-50 | N/A | DIP off-diagonal penalty |
| `hidden_dims` | [512,256,128] | [512,256,128] | [512,256,128] | [512,256,128] | Encoder/decoder width |
| `learning_rate` | 1e-3 | 1e-3 (VAE), 1e-4 (disc) | 1e-3 | 1e-3 | Separate LR for discriminator |
| `batch_size` | 128-256 | 128-256 | 128-256 | 128-256 | FactorVAE needs larger batches |
| `beta_schedule` | warmup | N/A | N/A | warmup | Prevents posterior collapse |

### Common Pitfalls

1. **Posterior Collapse**
   - Problem: All latent dimensions collapse to the prior, producing no useful information
   - Solution: Use beta warmup schedule, start with beta=0 and increase linearly over 50 epochs
   - Monitor: Check KL divergence per dimension; collapsed dims have KL near zero

2. **Reconstruction vs Disentanglement Tradeoff**
   - Problem: High beta destroys reconstruction quality
   - Solution: Use FactorVAE or beta-TCVAE which target TC directly
   - Monitor: Track both reconstruction error and disentanglement metrics

3. **Choosing Latent Dimensionality**
   - Problem: Too few dims cannot capture all factors; too many lead to unused dims
   - Solution: Set latent_dim = 1.5x to 2x the expected number of independent factors
   - Monitor: Check KL per dimension; active dims have KL significantly above zero

4. **Unstable Discriminator (FactorVAE)**
   - Problem: Discriminator training oscillates, producing noisy TC estimates
   - Solution: Use lower learning rate for discriminator (10x smaller than VAE), apply spectral normalization
   - Monitor: Discriminator accuracy should stabilize near 60-70%, not 50% or 100%

5. **Spurious Disentanglement**
   - Problem: Metric scores are high but factors do not correspond to meaningful market phenomena
   - Solution: Always validate with latent traversals and economic interpretation
   - Monitor: Correlate latent dims with known indicators (VIX, trend measures, etc.)

6. **Non-Stationarity of Financial Data**
   - Problem: Factor meanings may shift over time as market regimes change
   - Solution: Retrain periodically, use rolling windows, monitor factor stability
   - Monitor: Track correlation between latent dims and market indicators over time

### Factor Interpretation Checklist

```
For each active latent dimension, verify:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  1. [ ] Single high correlation with one market          │
│         indicator (r > 0.5)                              │
│                                                          │
│  2. [ ] Low correlation with all other indicators        │
│         (r < 0.2 ideally)                                │
│                                                          │
│  3. [ ] Latent traversal produces economically           │
│         meaningful reconstruction changes                │
│                                                          │
│  4. [ ] Factor values are stable within regime           │
│         but shift across regime transitions              │
│                                                          │
│  5. [ ] Factor is uncorrelated with other active         │
│         latent dimensions (off-diagonal cov < 0.1)       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Resources

### Papers

- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) -- Higgins et al. (ICLR 2017). Introduced beta weighting of KL divergence for disentanglement.
- [Disentangling by Factorising](https://arxiv.org/abs/1802.05983) -- Kim & Mnih (ICML 2018). FactorVAE with discriminator-based Total Correlation penalty.
- [Variational Inference of Disentangled Latent Concepts from Unlabeled Observations](https://arxiv.org/abs/1711.00848) -- Kumar, Sattigeri & Balakrishnan (ICLR 2018). DIP-VAE with covariance matching.
- [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942) -- Chen et al. (NeurIPS 2018). beta-TCVAE with KL decomposition.
- [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359) -- Locatello et al. (ICML 2019). Critical analysis of disentanglement methods and metrics.
- [A Framework for the Quantitative Evaluation of Disentangled Representations](https://openreview.net/forum?id=By-7dz-AZ) -- Eastwood & Williams (ICLR 2018). DCI metrics for disentanglement evaluation.

### Implementations

- [disentanglement_lib](https://github.com/google-research/disentanglement_lib) -- Google Research library with all major methods and metrics
- [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE) -- Collection of VAE variants including beta-VAE, FactorVAE, DIP-VAE, beta-TCVAE
- [disentangled-representation-papers](https://github.com/sootlasten/disentangled-representation-papers) -- Curated list of disentanglement research papers

### Related Chapters

- [Chapter 231: VAE Factor Model](../231_vae_factor_model) -- Standard VAE for financial factor modeling
- [Chapter 234: Beta-VAE Trading](../234_beta_vae_trading) -- Deep dive into beta-VAE for trading applications
- [Chapter 236: Conditional VAE Trading](../236_conditional_vae_trading) -- Conditional generation for regime-specific modeling
- [Chapter 237: Hierarchical VAE Trading](../237_hierarchical_vae_trading) -- Multi-scale latent hierarchies for market structure

---

## Difficulty Level

**Advanced**

Prerequisites:
- Variational Autoencoder fundamentals (ELBO, reparameterization trick)
- Information theory (KL divergence, mutual information, entropy)
- Probabilistic graphical models (latent variable models, posterior inference)
- PyTorch/Rust ML programming
- Financial time series analysis (technical indicators, factor models)
