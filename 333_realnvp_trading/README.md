# Chapter 333: RealNVP Trading — Invertible Transformations for Market Distribution Learning

## Overview

RealNVP (Real-valued Non-Volume Preserving) is a powerful normalizing flow model that learns invertible transformations between complex data distributions and simple base distributions. In trading, RealNVP enables exact density estimation of market states, anomaly detection, and generation of realistic market scenarios for risk analysis and strategy development.

This chapter explores how to apply RealNVP to cryptocurrency trading, using its bijective transformations to model market dynamics, detect regime changes, and generate probabilistic trading signals.

## Core Concepts

### What is RealNVP?

RealNVP is a normalizing flow model that transforms data through a sequence of invertible affine coupling layers:

```
Normalizing Flow: z = f(x)

Where:
├── x = complex data (market states)
├── z = simple distribution (usually Gaussian)
├── f = sequence of invertible transformations
└── Key property: exact density via change of variables

p(x) = p(z) |det(∂f/∂x)|

The Jacobian determinant is tractable due to coupling layer design!
```

### Why RealNVP for Trading?

1. **Exact Density Estimation**: Compute exact probability of any market state
2. **Bijective Mapping**: Perfect reconstruction — no information loss
3. **Efficient Sampling**: Generate realistic market scenarios
4. **Tractable Training**: No MCMC or variational bounds needed
5. **Anomaly Detection**: Low probability indicates unusual conditions
6. **Interpretable Latent Space**: Learned z-space reveals market structure

### Affine Coupling Layers

```
Coupling Layer Transformation:
├── Split input x into (x₁, x₂)
├── Compute scale s = s_θ(x₁) and translation t = t_θ(x₁)
├── Transform: y₂ = x₂ ⊙ exp(s) + t
├── Pass through: y₁ = x₁
└── Output: y = (y₁, y₂)

Inverse (trivial to compute):
├── x₂ = (y₂ - t) ⊙ exp(-s)
├── x₁ = y₁
└── Jacobian: det = exp(sum(s))

Key insight: Half the variables condition the transformation of the other half
```

## Trading Strategy

**Strategy Overview:** Use RealNVP to learn the joint distribution of market features. Trading signals are generated based on:
1. Probability density of current market state
2. Direction in latent space indicating market regime
3. Generated scenarios for risk assessment

### Signal Generation

```
1. Feature Extraction:
   - Compute market features: returns, volatility, volume ratios
   - Normalize features to standard scale

2. Flow Transformation:
   - Transform market state x → z via learned flow
   - Compute log probability: log p(x) = log p(z) + sum(log|det J|)

3. Signal Interpretation:
   - High density → normal conditions → trend following
   - Low density → anomaly → reduce exposure or mean reversion
   - Latent direction → regime indicator → adjust strategy

4. Scenario Generation:
   - Sample z ~ N(0, I)
   - Inverse transform: x = f⁻¹(z)
   - Analyze distribution of future scenarios
```

### Entry Signals

- **Long Signal**: High density + positive momentum in latent space
- **Short Signal**: High density + negative momentum in latent space
- **Exit Signal**: Low density (unusual market state) → reduce position

### Risk Management

- **Density Threshold**: Only trade when log p(x) > threshold
- **Scenario Analysis**: Use generated scenarios for VaR estimation
- **Regime Detection**: Monitor latent space clusters for regime changes

## Technical Specification

### Mathematical Foundation

#### Change of Variables Formula

For a bijective function f: X → Z with inverse g = f⁻¹:

```
p_X(x) = p_Z(f(x)) |det(∂f/∂x)|

For composition of flows f = f_K ∘ f_{K-1} ∘ ... ∘ f_1:

log p(x) = log p(z) + Σ_{k=1}^K log|det(∂f_k/∂h_{k-1})|

Where h_k = f_k(h_{k-1}) and h_0 = x, h_K = z
```

#### Affine Coupling Layer

```
Given input x ∈ R^d, split into x = (x_{1:d/2}, x_{d/2+1:d})

Forward transformation:
├── y_{1:d/2} = x_{1:d/2}  (identity)
├── y_{d/2+1:d} = x_{d/2+1:d} ⊙ exp(s(x_{1:d/2})) + t(x_{1:d/2})
│
├── Where s, t: R^{d/2} → R^{d/2} are neural networks
└── ⊙ denotes element-wise product

Jacobian:
├── ∂y/∂x is lower triangular with structure:
│   [I    0   ]
│   [*  diag(exp(s))]
├── det = Π exp(s_i) = exp(Σ s_i)
└── log|det| = Σ s_i  (simple sum!)
```

#### Multi-Scale Architecture

```
RealNVP Multi-Scale Architecture:
├── Level 1: Full resolution
│   ├── Coupling layers (x4)
│   ├── Squeeze operation
│   └── Split: factor out half dimensions
│
├── Level 2: Half resolution
│   ├── Coupling layers (x4)
│   ├── Squeeze operation
│   └── Split: factor out half dimensions
│
└── Level 3: Quarter resolution
    └── Final coupling layers

Split dimensions go directly to z, rest continues to next level
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
                           ▼ Market State x ∈ R^d
            ┌─────────────────────────────┐
            │      RealNVP Flow           │
            │                             │
            │  ┌───────────────────────┐  │
            │  │  Coupling Layer 1     │  │
            │  │  ├── Split x→(x₁,x₂)  │  │
            │  │  ├── s,t = NN(x₁)     │  │
            │  │  └── y₂ = x₂⊙eˢ + t   │  │
            │  └───────────┬───────────┘  │
            │              ▼              │
            │  ┌───────────────────────┐  │
            │  │  Coupling Layer 2     │  │
            │  │  (alternating split)  │  │
            │  └───────────┬───────────┘  │
            │              ▼              │
            │          ... x K            │
            │              ▼              │
            │  ┌───────────────────────┐  │
            │  │  Latent Space z       │  │
            │  │  z ~ N(0, I)          │  │
            │  └───────────────────────┘  │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼              ▼              ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │  Log Prob   │ │   Latent    │ │  Generated  │
     │  log p(x)   │ │  Direction  │ │  Scenarios  │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
            ┌─────────────────────────────┐
            │     Trading Decision        │
            │  ├── Density filter         │
            │  ├── Regime indicator       │
            │  ├── Risk assessment        │
            │  └── Position sizing        │
            └─────────────────────────────┘
```

### Feature Engineering for RealNVP

```python
import numpy as np
import pandas as pd

def compute_market_state(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """
    Create market state vector for RealNVP
    """
    features = {}

    # Multi-scale returns
    returns = df['close'].pct_change()
    for period in [1, 5, 10, 20]:
        features[f'return_{period}'] = returns.rolling(period).sum().iloc[-1]

    # Volatility features
    features['volatility'] = returns.rolling(lookback).std().iloc[-1]
    features['volatility_ratio'] = (
        returns.rolling(5).std().iloc[-1] /
        returns.rolling(20).std().iloc[-1]
    )

    # Momentum
    features['momentum'] = df['close'].iloc[-1] / df['close'].iloc[-lookback] - 1

    # Volume features
    volume_ma = df['volume'].rolling(lookback).mean()
    features['volume_ratio'] = df['volume'].iloc[-1] / volume_ma.iloc[-1]

    # Price position
    high_20 = df['high'].rolling(lookback).max().iloc[-1]
    low_20 = df['low'].rolling(lookback).min().iloc[-1]
    features['price_position'] = (df['close'].iloc[-1] - low_20) / (high_20 - low_20 + 1e-8)

    # RSI-like feature
    gains = returns.clip(lower=0).rolling(14).mean().iloc[-1]
    losses = (-returns.clip(upper=0)).rolling(14).mean().iloc[-1]
    features['rsi'] = gains / (gains + losses + 1e-8)

    return np.array(list(features.values()))
```

### Coupling Layer Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CouplingLayer(nn.Module):
    """
    Affine coupling layer for RealNVP

    Transforms input x by:
    - Splitting into (x1, x2)
    - Computing scale s and translation t from x1
    - Transforming: y2 = x2 * exp(s) + t, y1 = x1
    """

    def __init__(self, dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, mask_type: str = 'even'):
        super().__init__()

        self.dim = dim
        self.mask_type = mask_type

        # Create mask
        if mask_type == 'even':
            self.register_buffer('mask', torch.arange(dim) % 2 == 0)
        else:  # 'odd'
            self.register_buffer('mask', torch.arange(dim) % 2 == 1)

        self.n_masked = self.mask.sum().item()
        self.n_unmasked = dim - self.n_masked

        # Scale and translation networks
        layers = [nn.Linear(self.n_masked, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        self.shared_net = nn.Sequential(*layers)
        self.scale_net = nn.Linear(hidden_dim, self.n_unmasked)
        self.translation_net = nn.Linear(hidden_dim, self.n_unmasked)

        # Initialize scale output to zero for stable training
        nn.init.zeros_(self.scale_net.weight)
        nn.init.zeros_(self.scale_net.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: x → y

        Returns:
            y: transformed output
            log_det: log determinant of Jacobian
        """
        x_masked = x[:, self.mask]
        x_unmasked = x[:, ~self.mask]

        h = self.shared_net(x_masked)
        s = self.scale_net(h)
        t = self.translation_net(h)

        # Clamp scale for numerical stability
        s = torch.clamp(s, -5, 5)

        # Transform unmasked part
        y_unmasked = x_unmasked * torch.exp(s) + t

        # Combine
        y = torch.zeros_like(x)
        y[:, self.mask] = x_masked
        y[:, ~self.mask] = y_unmasked

        # Log determinant
        log_det = s.sum(dim=-1)

        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: y → x
        """
        y_masked = y[:, self.mask]
        y_unmasked = y[:, ~self.mask]

        h = self.shared_net(y_masked)
        s = self.scale_net(h)
        t = self.translation_net(h)

        s = torch.clamp(s, -5, 5)

        # Inverse transform
        x_unmasked = (y_unmasked - t) * torch.exp(-s)

        # Combine
        x = torch.zeros_like(y)
        x[:, self.mask] = y_masked
        x[:, ~self.mask] = x_unmasked

        return x


class RealNVP(nn.Module):
    """
    RealNVP normalizing flow model

    Composed of alternating coupling layers with different masks
    """

    def __init__(self, dim: int, hidden_dim: int = 128,
                 num_coupling_layers: int = 8, num_hidden_layers: int = 2):
        super().__init__()

        self.dim = dim

        # Alternating coupling layers
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(
                dim, hidden_dim, num_hidden_layers,
                mask_type='even' if i % 2 == 0 else 'odd'
            )
            for i in range(num_coupling_layers)
        ])

        # Base distribution
        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_std', torch.ones(dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: x → z

        Returns:
            z: latent representation
            log_det: total log determinant
        """
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)

        for layer in self.coupling_layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det

        return z, total_log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: z → x
        """
        x = z

        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x)

        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of x
        """
        z, log_det = self.forward(x)

        # Log probability under base distribution (standard Gaussian)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)

        # Change of variables
        log_px = log_pz + log_det

        return log_px

    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate samples by sampling z and inverting
        """
        z = torch.randn(n_samples, self.dim, device=device)
        x = self.inverse(z)
        return x
```

### Batch Normalization for Flows

```python
class ActNorm(nn.Module):
    """
    Activation Normalization layer

    Data-dependent initialization for stable training
    """

    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim
        self.initialized = False

        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def initialize(self, x: torch.Tensor):
        """Initialize with first batch statistics"""
        with torch.no_grad():
            mean = x.mean(dim=0)
            std = x.std(dim=0) + 1e-6

            self.bias.data = -mean
            self.log_scale.data = -torch.log(std)

        self.initialized = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized and self.training:
            self.initialize(x)

        y = (x + self.bias) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum().expand(x.shape[0])

        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * torch.exp(-self.log_scale) - self.bias


class RealNVPWithActNorm(nn.Module):
    """
    RealNVP with ActNorm layers for better training stability
    """

    def __init__(self, dim: int, hidden_dim: int = 128,
                 num_coupling_layers: int = 8):
        super().__init__()

        self.dim = dim
        self.layers = nn.ModuleList()

        for i in range(num_coupling_layers):
            self.layers.append(ActNorm(dim))
            self.layers.append(CouplingLayer(
                dim, hidden_dim,
                mask_type='even' if i % 2 == 0 else 'odd'
            ))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)

        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det

        return z, total_log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det

    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        z = torch.randn(n_samples, self.dim, device=device)
        return self.inverse(z)
```

### RealNVP Trading System

```python
class RealNVPTrader:
    """
    Trading system based on RealNVP density estimation
    """

    def __init__(self,
                 model: RealNVP,
                 feature_dim: int,
                 log_prob_threshold: float = -10.0,
                 position_scale: float = 1.0):
        self.model = model
        self.feature_dim = feature_dim
        self.log_prob_threshold = log_prob_threshold
        self.position_scale = position_scale

        # Feature statistics for normalization
        self.feature_mean = None
        self.feature_std = None

    def fit_normalizer(self, features: np.ndarray):
        """Fit feature normalizer"""
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-8

    def normalize(self, features: np.ndarray) -> torch.Tensor:
        """Normalize features"""
        normalized = (features - self.feature_mean) / self.feature_std
        return torch.tensor(normalized, dtype=torch.float32)

    def generate_signal(self, market_state: np.ndarray) -> dict:
        """
        Generate trading signal from current market state

        Args:
            market_state: Current market features

        Returns:
            dict with signal, confidence, and analysis
        """
        self.model.eval()

        x = self.normalize(market_state).unsqueeze(0)

        with torch.no_grad():
            # Get log probability and latent representation
            z, log_det = self.model.forward(x)
            log_prob = self.model.log_prob(x).item()

            # Latent space analysis
            z_np = z.squeeze().numpy()

            # Generate scenarios
            scenarios = self.model.sample(100)
            scenario_returns = scenarios[:, 0].numpy()  # Assuming first feature is return

        # Determine signal based on density and latent direction
        in_distribution = log_prob > self.log_prob_threshold

        if not in_distribution:
            # Unusual market conditions - reduce exposure
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'log_prob': log_prob,
                'in_distribution': False,
                'latent': z_np,
                'scenario_mean': scenario_returns.mean(),
                'scenario_std': scenario_returns.std()
            }

        # Use latent direction for signal
        # First component often correlates with returns
        latent_signal = np.sign(z_np[0]) if abs(z_np[0]) > 0.5 else 0.0

        # Confidence based on density (higher is better)
        # Normalize to [0, 1] range
        confidence = np.clip((log_prob - self.log_prob_threshold) / 10.0, 0, 1)

        return {
            'signal': latent_signal * confidence * self.position_scale,
            'confidence': confidence,
            'log_prob': log_prob,
            'in_distribution': True,
            'latent': z_np,
            'scenario_mean': scenario_returns.mean(),
            'scenario_std': scenario_returns.std()
        }

    def generate_scenarios(self, n_scenarios: int = 1000) -> np.ndarray:
        """
        Generate market scenarios for risk analysis

        Returns:
            Array of generated market states
        """
        self.model.eval()

        with torch.no_grad():
            scenarios = self.model.sample(n_scenarios)

        # Denormalize
        scenarios_np = scenarios.numpy()
        scenarios_np = scenarios_np * self.feature_std + self.feature_mean

        return scenarios_np

    def compute_var(self, current_state: np.ndarray,
                   alpha: float = 0.05,
                   n_scenarios: int = 1000) -> float:
        """
        Compute Value at Risk using generated scenarios
        """
        scenarios = self.generate_scenarios(n_scenarios)

        # Assume first feature is return
        scenario_returns = scenarios[:, 0]

        # VaR is the alpha quantile of losses
        var = np.percentile(scenario_returns, alpha * 100)

        return -var if var < 0 else 0
```

### Training Loop

```python
def train_realnvp(
    model: RealNVP,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5
):
    """
    Train RealNVP model by maximizing log likelihood
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Shuffle data
        perm = torch.randperm(len(train_data))

        for i in range(0, len(train_data), batch_size):
            batch = train_data[perm[i:i+batch_size]]

            optimizer.zero_grad()

            # Negative log likelihood
            log_prob = model.log_prob(batch)
            loss = -log_prob.mean()

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_log_prob = model.log_prob(val_data)
            val_loss = -val_log_prob.mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train NLL={total_loss/n_batches:.4f}, "
                  f"Val NLL={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model
```

### Backtesting Framework

```python
class RealNVPBacktest:
    """
    Backtest RealNVP trading strategy
    """

    def __init__(self,
                 trader: RealNVPTrader,
                 lookback: int = 20):
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
            'confidence': [],
            'log_prob': [],
            'in_distribution': [],
            'position': [],
            'pnl': [],
            'cumulative_pnl': []
        }

        position = 0.0
        cumulative_pnl = 0.0

        for i in range(warmup, len(prices)):
            window = prices.iloc[i-self.lookback:i]
            state = compute_market_state(window)

            # Get signal
            signal_info = self.trader.generate_signal(state)

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
            results['confidence'].append(signal_info['confidence'])
            results['log_prob'].append(signal_info['log_prob'])
            results['in_distribution'].append(signal_info['in_distribution'])
            results['position'].append(position)
            results['pnl'].append(pnl)
            results['cumulative_pnl'].append(cumulative_pnl)

        return pd.DataFrame(results)

    def calculate_metrics(self, results: pd.DataFrame) -> dict:
        """
        Calculate performance metrics
        """
        returns = results['pnl']

        total_return = results['cumulative_pnl'].iloc[-1]

        # Sharpe Ratio
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

        # In-distribution ratio
        in_dist_ratio = results['in_distribution'].mean()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'in_distribution_ratio': in_dist_ratio,
            'avg_log_prob': results['log_prob'].mean(),
            'avg_confidence': results['confidence'].mean()
        }
```

## Data Requirements

```
Historical OHLCV Data:
├── Minimum: 1 year of hourly data
├── Recommended: 2+ years for density learning
├── Frequency: 1-hour to daily
└── Source: Bybit, Binance, or other exchanges

Required Fields:
├── timestamp
├── open, high, low, close
├── volume
└── Optional: funding rate, open interest

Preprocessing:
├── Normalization: Z-score per feature
├── Outlier handling: Clip to ±5 std
├── Missing data: Forward fill then drop
└── Train/Val/Test split: 70/15/15
```

## Key Metrics

- **Negative Log Likelihood (NLL)**: Training objective, lower is better
- **Bits per Dimension**: NLL / (dim * log(2)), normalized metric
- **Log Probability**: Density estimate for each sample
- **In-Distribution Ratio**: Fraction of trading days with high density
- **Sharpe Ratio**: Risk-adjusted returns
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

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0

# Utilities
scikit-learn>=1.2.0
tqdm>=4.65.0
```

## Expected Outcomes

1. **Density Learning**: Model accurately captures market state distribution
2. **Anomaly Detection**: Low probability regions flag unusual conditions
3. **Scenario Generation**: Realistic market scenarios for risk analysis
4. **Probabilistic Signals**: Trading signals with calibrated confidence
5. **Backtest Results**: Expected Sharpe Ratio 0.8-1.5 with proper tuning

## References

1. **Density Estimation Using Real-NVP** (Dinh et al., 2016)
   - URL: https://arxiv.org/abs/1605.08803

2. **NICE: Non-linear Independent Components Estimation** (Dinh et al., 2014)
   - URL: https://arxiv.org/abs/1410.8516

3. **Glow: Generative Flow with Invertible 1×1 Convolutions** (Kingma & Dhariwal, 2018)
   - URL: https://arxiv.org/abs/1807.03039

4. **Normalizing Flows for Probabilistic Modeling and Inference** (Papamakarios et al., 2019)
   - URL: https://arxiv.org/abs/1912.02762

5. **Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design** (Ho et al., 2019)
   - URL: https://arxiv.org/abs/1902.00275

## Rust Implementation

This chapter includes a complete Rust implementation for high-performance RealNVP trading on cryptocurrency data from Bybit. See `rust/` directory.

### Features:
- Real-time data fetching from Bybit API
- Affine coupling layer implementation
- RealNVP flow with invertible transformations
- Exact log probability computation
- Sample generation for scenario analysis
- Backtesting framework with comprehensive metrics
- Modular and extensible design

## Difficulty Level

⭐⭐⭐⭐⭐ (Expert)

Requires understanding of: Probability Theory, Change of Variables, Neural Networks, Normalizing Flows, Invertible Transformations, Trading Systems
