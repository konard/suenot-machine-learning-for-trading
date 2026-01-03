# Chapter 334: GLOW (Generative Flow) for Trading

## Overview

GLOW (Generative Flow with Invertible 1x1 Convolutions) is a powerful flow-based generative model introduced by OpenAI in 2018. Unlike GANs or VAEs, GLOW allows for exact likelihood computation and efficient sampling through its invertible architecture. For trading applications, GLOW provides unique advantages in understanding market distributions, generating realistic scenarios, and detecting anomalies.

This chapter explores how to apply GLOW to financial markets, using its exact likelihood computation for risk assessment, its generative capabilities for scenario analysis, and its latent space for market regime detection.

## Core Concepts

### What is GLOW?

GLOW is a normalizing flow model that transforms simple distributions (like Gaussians) into complex data distributions through a series of invertible transformations:

```
Simple Distribution (z ~ N(0,I))
         ↓
    [Invertible Transformations]
         ↓
Complex Data Distribution (x)

Key Properties:
├── Exact likelihood: log p(x) = log p(z) + log|det(∂z/∂x)|
├── Efficient sampling: x = f(z), where z ~ N(0,I)
├── Invertible: z = f⁻¹(x) for encoding
└── Learnable: All transformations are parameterized
```

### Why GLOW for Trading?

1. **Exact Likelihood**: Unlike VAEs (lower bound) or GANs (no likelihood), GLOW computes exact log-probabilities
2. **Anomaly Detection**: Low likelihood = unusual market conditions
3. **Scenario Generation**: Sample realistic market scenarios for stress testing
4. **Latent Representations**: Compress market states for regime analysis
5. **Invertibility**: Both encode (market → latent) and decode (latent → market)

### GLOW Architecture

```
GLOW Architecture:
├── Multi-Scale Structure
│   ├── Level 1: Full resolution
│   ├── Level 2: Half resolution (after split)
│   ├── Level 3: Quarter resolution (after split)
│   └── ... (hierarchical compression)
│
├── Flow Steps (repeated K times per level):
│   ├── ActNorm: Data-dependent normalization
│   ├── Invertible 1x1 Conv: Channel mixing
│   └── Affine Coupling: Split → Transform → Concat
│
└── Split Operation:
    ├── Half channels → Next level
    └── Half channels → Latent z_i
```

## Trading Strategy

**Strategy Overview:** Use GLOW to model the distribution of market states. Trading signals are generated based on:
1. Likelihood-based regime detection
2. Latent space analysis for market structure
3. Scenario generation for risk management

### Signal Generation

```
1. Feature Extraction:
   - Compute market features: returns, volatility, momentum
   - Normalize to training distribution scale

2. Likelihood Computation:
   - Encode: z = f⁻¹(x)
   - Compute: log p(x) = log p(z) + log|det(Jacobian)|

3. Signal Interpretation:
   - High likelihood → Familiar market state → Trade normally
   - Low likelihood → Unusual conditions → Reduce exposure
   - Latent direction → Market regime indicator

4. Scenario Analysis:
   - Generate samples from learned distribution
   - Compute VaR/CVaR from generated scenarios
```

### Entry Signals

- **Long Signal**: Latent encoding indicates bullish regime with high likelihood
- **Short Signal**: Latent encoding indicates bearish regime with high likelihood
- **No Trade**: Low likelihood indicates unusual/uncertain conditions

### Risk Management

- **Likelihood Filter**: Only trade when log p(x) > threshold
- **Position Sizing**: Scale by normalized likelihood
- **Scenario VaR**: Use generated samples for risk limits

## Technical Specification

### Mathematical Foundation

#### Normalizing Flows

The key insight is the change of variables formula:

```
For invertible transformation f: z → x = f(z)

p_x(x) = p_z(f⁻¹(x)) · |det(∂f⁻¹/∂x)|
       = p_z(z) · |det(∂z/∂x)|

Log-likelihood:
log p_x(x) = log p_z(z) + log|det(∂z/∂x)|
           = log p_z(z) + Σ log|det(∂h_i/∂h_{i-1})|

Where h_0 = x, h_N = z, and each h_i → h_{i+1} is invertible
```

#### GLOW Flow Steps

Each flow step consists of three invertible operations:

**1. Activation Normalization (ActNorm)**
```
Forward: y = (x - μ) / σ
Inverse: x = y · σ + μ

Log-det: Σ log|σ| (per channel)

Note: μ, σ initialized with first batch statistics
```

**2. Invertible 1x1 Convolution**
```
Forward: y = Wx, where W is d×d weight matrix
Inverse: x = W⁻¹y

Log-det: log|det(W)|

Optimization: LU decomposition for O(d) instead of O(d³)
W = PL(U + diag(s)), where L lower triangular, U upper triangular
log|det(W)| = Σ log|s_i|
```

**3. Affine Coupling Layer**
```
Split: x → [x_a, x_b] (split on channel dimension)

Forward:
  x_a unchanged
  (log_s, t) = NN(x_a)  # Neural network
  y_b = x_b ⊙ exp(log_s) + t
  y = [x_a, y_b]

Inverse:
  [y_a, y_b] = y
  (log_s, t) = NN(y_a)
  x_b = (y_b - t) ⊙ exp(-log_s)
  x = [y_a, x_b]

Log-det: Σ log_s
```

### Architecture Diagram

```
                    Market Data Stream
                           │
                           ▼
            ┌─────────────────────────────┐
            │    Feature Engineering      │
            │  ├── Returns (multi-scale)  │
            │  ├── Volatility measures    │
            │  ├── Volume patterns        │
            │  └── Technical indicators   │
            └──────────────┬──────────────┘
                           │
                           ▼ x (market state)
            ┌─────────────────────────────┐
            │         GLOW Model          │
            │                             │
            │  ┌───────────────────────┐  │
            │  │    Level 1 (K flows)  │  │
            │  │  ├── ActNorm          │  │
            │  │  ├── 1x1 Conv         │  │
            │  │  └── Affine Coupling  │  │
            │  └───────────┬───────────┘  │
            │              │ split        │
            │  ┌───────────┴───────────┐  │
            │  │     z_1    │  h_1     │  │
            │  │   (latent) │  (next)  │  │
            │  └────────────┴────┬─────┘  │
            │                    │        │
            │  ┌───────────────────────┐  │
            │  │    Level 2 (K flows)  │  │
            │  └───────────┬───────────┘  │
            │              │ split        │
            │  ┌───────────┴───────────┐  │
            │  │     z_2    │  h_2     │  │
            │  └────────────┴────┬─────┘  │
            │                    │        │
            │              ... (L levels) │
            │                    │        │
            │  ┌───────────────────────┐  │
            │  │    Level L (K flows)  │  │
            │  └───────────┬───────────┘  │
            │              ▼              │
            │           z_L (final)       │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼              ▼              ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │   Log-      │ │   Latent    │ │  Scenario   │
     │   Likelihood│ │   Analysis  │ │  Generation │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
            ┌─────────────────────────────┐
            │     Trading Decision        │
            │  ├── Regime Detection       │
            │  ├── Risk Assessment        │
            │  ├── Position Sizing        │
            │  └── Signal Generation      │
            └─────────────────────────────┘
```

### Feature Engineering for GLOW

```python
import numpy as np
import pandas as pd

def compute_glow_features(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """
    Create feature vector for GLOW model
    Features are designed to capture market state in multiple dimensions
    """
    features = {}

    # Returns at multiple scales
    returns = df['close'].pct_change()
    for period in [1, 5, 10, 20]:
        features[f'return_{period}'] = returns.rolling(period).sum().iloc[-1]

    # Volatility features
    features['volatility_5'] = returns.rolling(5).std().iloc[-1]
    features['volatility_20'] = returns.rolling(20).std().iloc[-1]
    features['vol_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)

    # Momentum features
    features['momentum_10'] = df['close'].iloc[-1] / df['close'].iloc[-10] - 1
    features['momentum_20'] = df['close'].iloc[-1] / df['close'].iloc[-20] - 1

    # Volume features
    volume_ma = df['volume'].rolling(20).mean()
    features['volume_ratio'] = df['volume'].iloc[-1] / (volume_ma.iloc[-1] + 1e-8)

    # Price position in range
    high_20 = df['high'].rolling(20).max().iloc[-1]
    low_20 = df['low'].rolling(20).min().iloc[-1]
    features['price_position'] = (df['close'].iloc[-1] - low_20) / (high_20 - low_20 + 1e-8)

    # OHLC patterns
    features['body_ratio'] = (df['close'].iloc[-1] - df['open'].iloc[-1]) / (df['high'].iloc[-1] - df['low'].iloc[-1] + 1e-8)
    features['upper_shadow'] = (df['high'].iloc[-1] - max(df['open'].iloc[-1], df['close'].iloc[-1])) / (df['high'].iloc[-1] - df['low'].iloc[-1] + 1e-8)
    features['lower_shadow'] = (min(df['open'].iloc[-1], df['close'].iloc[-1]) - df['low'].iloc[-1]) / (df['high'].iloc[-1] - df['low'].iloc[-1] + 1e-8)

    return np.array(list(features.values()))
```

### GLOW Model Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActNorm(nn.Module):
    """
    Activation Normalization - data-dependent initialization
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.log_scale = nn.Parameter(torch.zeros(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.initialized = False

    def initialize(self, x: torch.Tensor):
        """Initialize with first batch statistics"""
        with torch.no_grad():
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + 1e-6
            self.bias.data = -mean
            self.log_scale.data = -torch.log(std)
        self.initialized = True

    def forward(self, x: torch.Tensor, reverse: bool = False):
        if not self.initialized:
            self.initialize(x)

        if reverse:
            x = (x - self.bias) * torch.exp(-self.log_scale)
            log_det = -self.log_scale.sum() * x.shape[0]
        else:
            x = x * torch.exp(self.log_scale) + self.bias
            log_det = self.log_scale.sum() * x.shape[0]

        return x, log_det


class InvertibleConv1x1(nn.Module):
    """
    Invertible 1x1 convolution with LU decomposition for efficient log-det
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

        # Initialize with random orthogonal matrix
        W = torch.linalg.qr(torch.randn(num_features, num_features))[0]

        # LU decomposition
        P, L, U = torch.linalg.lu(W)

        # Store components
        self.register_buffer('P', P)
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
        self.log_s = nn.Parameter(torch.zeros(num_features))

        # Masks for triangular matrices
        self.register_buffer('L_mask', torch.tril(torch.ones(num_features, num_features), -1))
        self.register_buffer('U_mask', torch.triu(torch.ones(num_features, num_features), 1))

    def get_weight(self):
        """Reconstruct W from LU decomposition"""
        L = self.L * self.L_mask + torch.eye(self.num_features, device=self.L.device)
        U = self.U * self.U_mask + torch.diag(torch.exp(self.log_s))
        return self.P @ L @ U

    def forward(self, x: torch.Tensor, reverse: bool = False):
        W = self.get_weight()
        log_det = self.log_s.sum() * x.shape[0]

        if reverse:
            W_inv = torch.inverse(W)
            x = x @ W_inv
            log_det = -log_det
        else:
            x = x @ W

        return x, log_det


class AffineCoupling(nn.Module):
    """
    Affine coupling layer - the key component for expressiveness
    """
    def __init__(self, num_features: int, hidden_dim: int = 128):
        super().__init__()
        self.num_features = num_features
        self.split_dim = num_features // 2

        # Neural network for computing scale and translation
        self.net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (num_features - self.split_dim) * 2)
        )

        # Initialize last layer to zero for identity initialization
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x: torch.Tensor, reverse: bool = False):
        x_a = x[:, :self.split_dim]
        x_b = x[:, self.split_dim:]

        # Compute scale and translation from first half
        h = self.net(x_a)
        log_s, t = h.chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * 2  # Constrain scale

        if reverse:
            y_b = (x_b - t) * torch.exp(-log_s)
            log_det = -log_s.sum(dim=-1).sum()
        else:
            y_b = x_b * torch.exp(log_s) + t
            log_det = log_s.sum(dim=-1).sum()

        y = torch.cat([x_a, y_b], dim=-1)
        return y, log_det


class FlowStep(nn.Module):
    """
    Single flow step: ActNorm → 1x1 Conv → Affine Coupling
    """
    def __init__(self, num_features: int, hidden_dim: int = 128):
        super().__init__()
        self.actnorm = ActNorm(num_features)
        self.conv1x1 = InvertibleConv1x1(num_features)
        self.coupling = AffineCoupling(num_features, hidden_dim)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        total_log_det = 0

        if reverse:
            x, log_det = self.coupling(x, reverse=True)
            total_log_det += log_det
            x, log_det = self.conv1x1(x, reverse=True)
            total_log_det += log_det
            x, log_det = self.actnorm(x, reverse=True)
            total_log_det += log_det
        else:
            x, log_det = self.actnorm(x)
            total_log_det += log_det
            x, log_det = self.conv1x1(x)
            total_log_det += log_det
            x, log_det = self.coupling(x)
            total_log_det += log_det

        return x, total_log_det


class GLOW(nn.Module):
    """
    GLOW: Generative Flow with Invertible 1x1 Convolutions

    Adapted for 1D time series / financial features
    """
    def __init__(self, num_features: int, num_levels: int = 3,
                 num_steps: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.num_features = num_features
        self.num_levels = num_levels
        self.num_steps = num_steps

        self.levels = nn.ModuleList()
        current_features = num_features

        for level in range(num_levels):
            steps = nn.ModuleList([
                FlowStep(current_features, hidden_dim)
                for _ in range(num_steps)
            ])
            self.levels.append(steps)

            # After each level (except last), split features
            if level < num_levels - 1:
                current_features = current_features // 2

    def forward(self, x: torch.Tensor):
        """
        Forward pass: x → z (encode)
        Returns: z, log_det_jacobian
        """
        total_log_det = 0
        z_list = []

        h = x
        for level_idx, level_steps in enumerate(self.levels):
            # Apply flow steps
            for step in level_steps:
                h, log_det = step(h)
                total_log_det += log_det

            # Split (except last level)
            if level_idx < self.num_levels - 1:
                split_dim = h.shape[-1] // 2
                z_i, h = h[:, :split_dim], h[:, split_dim:]
                z_list.append(z_i)

        z_list.append(h)  # Final latent
        z = torch.cat(z_list, dim=-1)

        return z, total_log_det

    def inverse(self, z: torch.Tensor):
        """
        Inverse pass: z → x (decode/sample)
        """
        # Split z into components
        z_sizes = []
        size = self.num_features
        for level in range(self.num_levels - 1):
            z_sizes.append(size // 2)
            size = size - size // 2
        z_sizes.append(size)

        z_list = []
        start = 0
        for size in z_sizes:
            z_list.append(z[:, start:start+size])
            start += size

        # Reverse through levels
        h = z_list[-1]
        for level_idx in range(self.num_levels - 1, -1, -1):
            # Merge (except for last level in reverse)
            if level_idx < self.num_levels - 1:
                h = torch.cat([z_list[level_idx], h], dim=-1)

            # Reverse flow steps
            for step in reversed(self.levels[level_idx]):
                h, _ = step(h, reverse=True)

        return h

    def log_prob(self, x: torch.Tensor):
        """
        Compute log probability of x
        """
        z, log_det = self.forward(x)

        # Log probability under prior (standard Gaussian)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)

        # log p(x) = log p(z) + log|det(dz/dx)|
        log_px = log_pz + log_det / x.shape[0]  # Normalize by batch

        return log_px

    def sample(self, num_samples: int, device: str = 'cpu', temperature: float = 1.0):
        """
        Sample from the model
        """
        z = torch.randn(num_samples, self.num_features, device=device) * temperature
        x = self.inverse(z)
        return x


class GLOWTrainer:
    """
    Training loop for GLOW model
    """
    def __init__(self, model: GLOW, lr: float = 1e-4, weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        total_samples = 0

        for batch in data_loader:
            self.optimizer.zero_grad()

            # Negative log-likelihood
            log_prob = self.model.log_prob(batch)
            loss = -log_prob.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * batch.shape[0]
            total_samples += batch.shape[0]

        return total_loss / total_samples

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        total_log_prob = 0
        total_samples = 0

        for batch in data_loader:
            log_prob = self.model.log_prob(batch)
            total_log_prob += log_prob.sum().item()
            total_samples += batch.shape[0]

        avg_log_prob = total_log_prob / total_samples
        self.scheduler.step(avg_log_prob)

        return avg_log_prob
```

### GLOW Trading System

```python
import numpy as np
import pandas as pd

class GLOWTrader:
    """
    Trading system based on GLOW model
    """
    def __init__(self, model: GLOW,
                 likelihood_threshold: float = -10.0,
                 regime_clusters: int = 4):
        self.model = model
        self.likelihood_threshold = likelihood_threshold
        self.regime_clusters = regime_clusters

        # For regime clustering
        self.regime_centers = None

    def fit_regimes(self, train_data: torch.Tensor):
        """
        Cluster latent representations to identify market regimes
        """
        from sklearn.cluster import KMeans

        self.model.eval()
        with torch.no_grad():
            z, _ = self.model.forward(train_data)
            z_np = z.numpy()

        kmeans = KMeans(n_clusters=self.regime_clusters, random_state=42)
        kmeans.fit(z_np)

        self.regime_centers = kmeans.cluster_centers_
        self.kmeans = kmeans

    def get_regime(self, z: np.ndarray) -> int:
        """Get regime label for latent representation"""
        if self.kmeans is None:
            return -1
        return self.kmeans.predict(z.reshape(1, -1))[0]

    def generate_signal(self, market_state: torch.Tensor) -> dict:
        """
        Generate trading signal from current market state
        """
        self.model.eval()

        with torch.no_grad():
            # Compute log probability
            log_prob = self.model.log_prob(market_state.unsqueeze(0))

            # Get latent representation
            z, _ = self.model.forward(market_state.unsqueeze(0))

        log_prob = log_prob.item()
        z = z.squeeze().numpy()

        # Check if in distribution
        in_distribution = log_prob > self.likelihood_threshold

        # Get regime
        regime = self.get_regime(z)

        # Analyze latent for directional signal
        # First component often captures main variation (like returns direction)
        latent_signal = z[0]  # Use first latent component

        if not in_distribution:
            return {
                'signal': 0.0,
                'log_likelihood': log_prob,
                'in_distribution': False,
                'regime': regime,
                'latent': z,
                'confidence': 0.0
            }

        # Normalize signal strength by likelihood
        confidence = np.clip((log_prob - self.likelihood_threshold) / 10.0, 0, 1)
        signal = np.tanh(latent_signal) * confidence

        return {
            'signal': signal,
            'log_likelihood': log_prob,
            'in_distribution': True,
            'regime': regime,
            'latent': z,
            'confidence': confidence
        }

    def generate_scenarios(self, num_scenarios: int = 1000, temperature: float = 1.0):
        """
        Generate market scenarios for risk analysis
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(num_scenarios, temperature=temperature)
        return samples.numpy()

    def compute_var(self, scenarios: np.ndarray, confidence: float = 0.95) -> float:
        """
        Compute Value at Risk from generated scenarios
        """
        # Assuming first feature is returns
        returns = scenarios[:, 0]
        var = np.percentile(returns, (1 - confidence) * 100)
        return var

    def compute_cvar(self, scenarios: np.ndarray, confidence: float = 0.95) -> float:
        """
        Compute Conditional Value at Risk (Expected Shortfall)
        """
        returns = scenarios[:, 0]
        var = self.compute_var(scenarios, confidence)
        cvar = returns[returns <= var].mean()
        return cvar


class GLOWBacktest:
    """
    Backtesting framework for GLOW trading strategy
    """
    def __init__(self, trader: GLOWTrader, lookback: int = 20):
        self.trader = trader
        self.lookback = lookback

    def run(self, prices: pd.DataFrame, warmup: int = 100) -> pd.DataFrame:
        """
        Run backtest on price data
        """
        results = {
            'timestamp': [],
            'price': [],
            'signal': [],
            'log_likelihood': [],
            'in_distribution': [],
            'regime': [],
            'position': [],
            'pnl': [],
            'cumulative_pnl': []
        }

        position = 0.0
        cumulative_pnl = 0.0

        for i in range(warmup, len(prices)):
            window = prices.iloc[i-self.lookback:i]
            state = compute_glow_features(window)
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Get signal
            signal_info = self.trader.generate_signal(state_tensor)

            # Calculate PnL
            if i > warmup:
                daily_return = prices['close'].iloc[i] / prices['close'].iloc[i-1] - 1
                pnl = position * daily_return
                cumulative_pnl += pnl
            else:
                pnl = 0.0

            # Update position
            position = signal_info['signal']

            results['timestamp'].append(prices.index[i])
            results['price'].append(prices['close'].iloc[i])
            results['signal'].append(signal_info['signal'])
            results['log_likelihood'].append(signal_info['log_likelihood'])
            results['in_distribution'].append(signal_info['in_distribution'])
            results['regime'].append(signal_info['regime'])
            results['position'].append(position)
            results['pnl'].append(pnl)
            results['cumulative_pnl'].append(cumulative_pnl)

        return pd.DataFrame(results)

    def calculate_metrics(self, results: pd.DataFrame) -> dict:
        """
        Calculate comprehensive performance metrics
        """
        returns = results['pnl']

        total_return = results['cumulative_pnl'].iloc[-1]

        # Sharpe Ratio (annualized)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino Ratio
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = returns.mean() / downside.std() * np.sqrt(252)
        else:
            sortino = 0.0

        # Maximum Drawdown
        cumulative = results['cumulative_pnl']
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative - rolling_max
        max_drawdown = drawdown.min()

        # Win Rate
        trading_returns = returns[returns != 0]
        if len(trading_returns) > 0:
            win_rate = (trading_returns > 0).mean()
        else:
            win_rate = 0.0

        # Regime distribution
        regime_dist = results['regime'].value_counts(normalize=True).to_dict()

        # In-distribution ratio
        in_dist_ratio = results['in_distribution'].mean()

        # Average log-likelihood
        avg_log_likelihood = results['log_likelihood'].mean()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'in_distribution_ratio': in_dist_ratio,
            'avg_log_likelihood': avg_log_likelihood,
            'regime_distribution': regime_dist
        }
```

### Training Pipeline

```python
def train_glow_model(
    train_data: np.ndarray,
    val_data: np.ndarray,
    num_features: int,
    num_levels: int = 3,
    num_steps: int = 4,
    hidden_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4
):
    """
    Complete training pipeline for GLOW model
    """
    # Create model
    model = GLOW(
        num_features=num_features,
        num_levels=num_levels,
        num_steps=num_steps,
        hidden_dim=hidden_dim
    )

    # Create data loaders
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    val_tensor = torch.tensor(val_data, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(
        train_tensor, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_tensor, batch_size=batch_size, shuffle=False
    )

    # Create trainer
    trainer = GLOWTrainer(model, lr=lr)

    best_val_log_prob = float('-inf')
    best_state = None

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_log_prob = trainer.evaluate(val_loader)

        if val_log_prob > best_val_log_prob:
            best_val_log_prob = val_log_prob
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train NLL={train_loss:.4f}, "
                  f"Val Log-Prob={val_log_prob:.4f}")

    model.load_state_dict(best_state)
    return model
```

## Data Requirements

```
Historical OHLCV Data:
├── Minimum: 1 year of data for training
├── Recommended: 2+ years for robust learning
├── Frequency: 1-hour to daily
└── Source: Bybit, Binance, or other exchanges

Required Fields:
├── timestamp
├── open, high, low, close
├── volume
└── Optional: funding rate, open interest

Preprocessing:
├── Feature computation: 15-20 features per timestep
├── Normalization: Z-score per feature
├── Outlier handling: Clip to ±5 std
├── Train/Val/Test split: 70/15/15
└── Ensure feature dimension is power of 2 (for splits)
```

## Key Metrics

- **Negative Log-Likelihood (NLL)**: Training objective (lower is better)
- **Bits per Dimension**: NLL / (num_features * log(2))
- **Log-Likelihood**: Model quality indicator (higher is better)
- **Regime Accuracy**: If regimes are known
- **VaR/CVaR**: From generated scenarios
- **Sharpe Ratio**: Risk-adjusted trading returns
- **Maximum Drawdown**: Largest peak-to-trough decline

## Dependencies

```python
# Core
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.10.0

# Deep Learning
torch>=2.0.0

# Market Data
ccxt>=4.0.0

# Clustering
scikit-learn>=1.2.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
```

## Expected Outcomes

1. **Density Estimation**: GLOW captures multimodal market distributions
2. **Anomaly Detection**: Low likelihood flags unusual conditions
3. **Regime Identification**: Latent clustering reveals market states
4. **Scenario Generation**: Realistic samples for risk management
5. **Risk Assessment**: VaR/CVaR from generated scenarios
6. **Backtest Results**: Expected Sharpe Ratio 0.7-1.3 with proper tuning

## Comparison with Other Methods

| Method | Likelihood | Sampling | Latent Space | Training |
|--------|-----------|----------|--------------|----------|
| GLOW | Exact | Fast | Deterministic | Stable |
| VAE | Lower Bound | Fast | Stochastic | Stable |
| GAN | None | Fast | N/A | Unstable |
| Score Matching | Approximate | MCMC | N/A | Moderate |
| Diffusion | Approximate | Slow | N/A | Stable |

## References

1. **Glow: Generative Flow with Invertible 1x1 Convolutions** (Kingma & Dhariwal, 2018)
   - URL: https://arxiv.org/abs/1807.03039

2. **Density Estimation Using Real-NVP** (Dinh et al., 2016)
   - URL: https://arxiv.org/abs/1605.08803

3. **NICE: Non-linear Independent Components Estimation** (Dinh et al., 2014)
   - URL: https://arxiv.org/abs/1410.8516

4. **Normalizing Flows for Probabilistic Modeling and Inference** (Papamakarios et al., 2019)
   - URL: https://arxiv.org/abs/1912.02762

5. **Normalizing Flows: An Introduction and Review** (Kobyzev et al., 2020)
   - URL: https://arxiv.org/abs/1908.09257

## Rust Implementation

This chapter includes a complete Rust implementation for high-performance GLOW-based trading on cryptocurrency data from Bybit. See `rust/` directory.

### Features:
- Real-time data fetching from Bybit API
- GLOW model with ActNorm, Invertible 1x1 Conv, and Affine Coupling
- Exact log-likelihood computation
- Scenario generation for risk analysis
- Backtesting framework with comprehensive metrics
- Modular and extensible design

## Difficulty Level

⭐⭐⭐⭐⭐ (Expert)

Requires understanding of: Probability Theory, Change of Variables, Linear Algebra (Jacobians), Neural Networks, Flow-Based Models, Trading Systems
