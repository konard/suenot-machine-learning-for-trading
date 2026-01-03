# Chapter 335: Neural Spline Flows — Flexible Density Estimation for Trading

## Overview

Neural Spline Flows (NSF) represent a state-of-the-art approach to normalizing flows that use monotonic rational-quadratic splines as coupling layer transformations. Unlike simpler affine transformations, spline-based flows can model arbitrary complex distributions with high fidelity, making them ideal for capturing the heavy-tailed, skewed, and multi-modal nature of financial return distributions.

In trading, accurate density estimation is crucial for:
- **Risk Management**: Understanding tail risks and Value-at-Risk (VaR)
- **Regime Detection**: Identifying shifts in market distribution
- **Anomaly Detection**: Flagging unusual market conditions
- **Option Pricing**: More accurate implied volatility modeling
- **Portfolio Optimization**: Better covariance estimation

This chapter explores how to implement Neural Spline Flows for cryptocurrency trading, using data from Bybit exchange.

## Core Concepts

### What are Normalizing Flows?

Normalizing flows transform a simple base distribution (like a Gaussian) into a complex target distribution through a series of invertible transformations:

```
Normalizing Flow:
├── Base distribution: z ~ N(0, I)
├── Transformation: x = f(z)
├── Inverse: z = f⁻¹(x)
└── Density: p(x) = p(z) |det(∂f⁻¹/∂x)|

Key properties:
├── Bijective (invertible) transformations
├── Tractable Jacobian determinant
├── Exact likelihood computation
└── Efficient sampling
```

### Why Neural Spline Flows?

Traditional coupling flows use affine transformations:
```
Affine Coupling: y = x ⊙ exp(s) + t
├── Simple and fast
├── Limited expressiveness
└── Requires many layers for complex distributions
```

Neural Spline Flows use monotonic rational-quadratic splines:
```
Spline Coupling: y = RQS(x; w, h, d)
├── Highly expressive single layer
├── Captures multi-modal distributions
├── Better tail behavior
└── Fewer parameters needed
```

### Rational-Quadratic Splines

The core innovation of NSF is the rational-quadratic spline (RQS):

```
RQS Definition:
├── Domain: [x₀, xₖ] divided into K bins
├── Knot positions: (xₖ, yₖ) for k = 0, ..., K
├── Derivatives at knots: dₖ > 0 (ensures monotonicity)
└── Within each bin: rational-quadratic interpolation

For input ξ ∈ [0, 1] within bin k:
y = RQS(x) = [yₖ(1-ξ)² + yₖ₊₁ξ² + 2yₘξ(1-ξ)] / [(1-ξ)² + ξ² + 2ξ(1-ξ)sₖ]

Where:
├── yₘ = (yₖ + yₖ₊₁)/2 + (dₖ₊₁ - dₖ)wₖ/8
├── sₖ = (yₖ₊₁ - yₖ)/(xₖ₊₁ - xₖ)
├── wₖ = bin width
└── ξ = (x - xₖ)/wₖ
```

### Why NSF for Trading?

1. **Heavy Tails**: Financial returns have fat tails; splines model them accurately
2. **Skewness**: Markets are often asymmetric; splines capture skew naturally
3. **Multi-modality**: Different regimes create multi-modal distributions
4. **Exact Likelihood**: Enables precise probability calculations
5. **Fast Sampling**: Generate scenarios efficiently for stress testing

## Trading Strategy

**Strategy Overview:** Use Neural Spline Flows to learn the true distribution of market features. Trading signals are generated based on probability density, tail risk measures, and regime detection.

### Signal Generation Pipeline

```
1. Feature Extraction:
   - Multi-timeframe returns
   - Volatility measures
   - Volume patterns
   - Technical indicators

2. Flow Transformation:
   - Transform features through learned NSF
   - Compute log-likelihood of current state
   - Estimate density in latent space

3. Signal Generation:
   - High density + positive expected return → Long
   - High density + negative expected return → Short
   - Low density → Reduce exposure (unusual conditions)

4. Risk Management:
   - VaR/CVaR from learned distribution
   - Position sizing based on tail risk
   - Regime-aware stop losses
```

### Entry Signals

- **Long Signal**: Current state has high probability density AND flow transformation indicates positive return momentum
- **Short Signal**: Current state has high probability density AND flow transformation indicates negative return momentum
- **No Trade**: Low density indicates out-of-distribution market conditions

### Risk Management

- **Tail Risk**: Use inverse CDF to compute VaR at any confidence level
- **Density Threshold**: Only trade when log-likelihood exceeds threshold
- **Regime Detection**: Track density evolution for regime change signals
- **Dynamic Sizing**: Scale positions inversely to estimated tail risk

## Technical Specification

### Mathematical Foundation

#### Coupling Layer Architecture

```
For input x = [x₁, x₂], split into two parts:
├── x₁: unchanged (identity)
└── x₂: transformed based on x₁

Forward pass:
├── θ = NN(x₁)  // Neural network outputs spline parameters
├── y₁ = x₁
└── y₂ = RQS(x₂; θ)

Inverse pass:
├── x₁ = y₁
├── θ = NN(y₁)
└── x₂ = RQS⁻¹(y₂; θ)

Log-determinant:
└── log|det(J)| = Σ log|d RQS/dx₂|
```

#### Spline Parameters

For K bins, the neural network outputs:
```
Parameters per dimension:
├── K bin widths (sum to interval width)
├── K bin heights (sum to interval height)
└── K+1 derivative values at knots

Total parameters: 3K + 1 per transformed dimension

Constraints:
├── Widths: softmax normalization
├── Heights: softmax normalization
└── Derivatives: softplus + 1 (ensures positivity)
```

#### Multi-Scale Architecture

```
                    Input Features
                          │
                          ▼
            ┌─────────────────────────────┐
            │     Input Normalization     │
            │   (Running mean/std)        │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                              ▼
    ┌───────────────┐              ┌───────────────┐
    │  Split: x₁    │              │  Split: x₂    │
    └───────┬───────┘              └───────┬───────┘
            │                              │
            │    ┌──────────────────┐      │
            └───►│  Conditioner NN  │      │
                 │  (MLP / ResNet)  │      │
                 └────────┬─────────┘      │
                          │                │
                          ▼                │
            ┌─────────────────────────────┐│
            │  Spline Parameters          ││
            │  ├── Widths (K)             ││
            │  ├── Heights (K)            ││
            │  └── Derivatives (K+1)      ││
            └──────────────┬──────────────┘│
                           │               │
                           ▼               │
            ┌─────────────────────────────┐│
            │  Rational-Quadratic Spline  │◄┘
            │  y₂ = RQS(x₂; params)       │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                              ▼
       y₁ = x₁                        y₂ = RQS(x₂)
            │                              │
            └──────────────┬───────────────┘
                           ▼
                    ┌─────────────┐
                    │  Permute    │
                    └──────┬──────┘
                           │
                           ▼
                  (Next Coupling Layer)
                           │
                        × L layers
                           │
                           ▼
                    Latent Space z
```

### Coupling Flow Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class RationalQuadraticSpline(nn.Module):
    """
    Rational Quadratic Spline transformation

    Based on "Neural Spline Flows" (Durkan et al., 2019)
    """

    def __init__(self,
                 num_bins: int = 8,
                 bound: float = 3.0,
                 min_derivative: float = 1e-3):
        super().__init__()
        self.num_bins = num_bins
        self.bound = bound
        self.min_derivative = min_derivative

    def forward(self, x: torch.Tensor,
                params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spline transformation

        Args:
            x: (batch, dim) input
            params: (batch, dim, 3*num_bins + 1) spline parameters

        Returns:
            y: transformed output
            log_det: log determinant of Jacobian
        """
        # Split parameters
        W = params[..., :self.num_bins]
        H = params[..., self.num_bins:2*self.num_bins]
        D = params[..., 2*self.num_bins:]

        # Normalize widths and heights
        W = F.softmax(W, dim=-1) * 2 * self.bound
        H = F.softmax(H, dim=-1) * 2 * self.bound
        D = F.softplus(D) + self.min_derivative

        # Compute cumulative widths and heights
        cumwidths = torch.cumsum(W, dim=-1)
        cumheights = torch.cumsum(H, dim=-1)

        # Prepend zeros
        cumwidths = F.pad(cumwidths, (1, 0), value=-self.bound)
        cumheights = F.pad(cumheights, (1, 0), value=-self.bound)

        # Find bin for each input
        x_clamped = x.clamp(-self.bound, self.bound)
        bin_idx = torch.searchsorted(cumwidths[..., 1:], x_clamped.unsqueeze(-1))
        bin_idx = bin_idx.squeeze(-1).clamp(0, self.num_bins - 1)

        # Gather bin parameters
        input_cumwidths = cumwidths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_bin_widths = W.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_cumheights = cumheights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_bin_heights = H.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_delta = input_bin_heights / input_bin_widths
        input_derivatives = D.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_derivatives_plus = D.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

        # Compute spline
        xi = (x_clamped - input_cumwidths) / input_bin_widths
        xi_squared = xi ** 2
        one_minus_xi = 1 - xi
        one_minus_xi_squared = one_minus_xi ** 2

        numerator = input_bin_heights * (
            input_delta * xi_squared +
            input_derivatives * xi * one_minus_xi
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus - 2 * input_delta) *
            xi * one_minus_xi
        )

        y = input_cumheights + numerator / denominator

        # Compute log determinant
        derivative_numerator = input_delta ** 2 * (
            input_derivatives_plus * xi_squared +
            2 * input_delta * xi * one_minus_xi +
            input_derivatives * one_minus_xi_squared
        )
        log_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return y, log_det.sum(dim=-1)

    def inverse(self, y: torch.Tensor,
                params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse spline transformation
        """
        # Similar structure but solving quadratic for xi
        # Split parameters
        W = params[..., :self.num_bins]
        H = params[..., self.num_bins:2*self.num_bins]
        D = params[..., 2*self.num_bins:]

        W = F.softmax(W, dim=-1) * 2 * self.bound
        H = F.softmax(H, dim=-1) * 2 * self.bound
        D = F.softplus(D) + self.min_derivative

        cumwidths = torch.cumsum(W, dim=-1)
        cumheights = torch.cumsum(H, dim=-1)
        cumwidths = F.pad(cumwidths, (1, 0), value=-self.bound)
        cumheights = F.pad(cumheights, (1, 0), value=-self.bound)

        y_clamped = y.clamp(-self.bound, self.bound)
        bin_idx = torch.searchsorted(cumheights[..., 1:], y_clamped.unsqueeze(-1))
        bin_idx = bin_idx.squeeze(-1).clamp(0, self.num_bins - 1)

        # Gather and solve quadratic
        input_cumwidths = cumwidths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_bin_widths = W.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_cumheights = cumheights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_bin_heights = H.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_delta = input_bin_heights / input_bin_widths
        input_derivatives = D.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_derivatives_plus = D.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

        # Solve for xi using quadratic formula
        a = input_bin_heights * (input_delta - input_derivatives)
        a = a + (y_clamped - input_cumheights) * (
            input_derivatives + input_derivatives_plus - 2 * input_delta
        )
        b = input_bin_heights * input_derivatives
        b = b - (y_clamped - input_cumheights) * (
            input_derivatives + input_derivatives_plus - 2 * input_delta
        )
        c = -input_delta * (y_clamped - input_cumheights)

        discriminant = b ** 2 - 4 * a * c
        xi = (2 * c) / (-b - torch.sqrt(discriminant))

        x = xi * input_bin_widths + input_cumwidths

        # Compute log det (negative of forward)
        xi_squared = xi ** 2
        one_minus_xi = 1 - xi
        one_minus_xi_squared = one_minus_xi ** 2

        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus - 2 * input_delta) *
            xi * one_minus_xi
        )
        derivative_numerator = input_delta ** 2 * (
            input_derivatives_plus * xi_squared +
            2 * input_delta * xi * one_minus_xi +
            input_derivatives * one_minus_xi_squared
        )
        log_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return x, -log_det.sum(dim=-1)


class CouplingLayer(nn.Module):
    """
    Coupling layer with neural spline transformation
    """

    def __init__(self,
                 dim: int,
                 hidden_dim: int = 128,
                 num_bins: int = 8,
                 num_hidden_layers: int = 2):
        super().__init__()

        self.dim = dim
        self.split_dim = dim // 2
        self.num_bins = num_bins

        # Output dimension: widths + heights + derivatives for each transformed dim
        output_dim = (dim - self.split_dim) * (3 * num_bins + 1)

        # Conditioner network
        layers = [nn.Linear(self.split_dim, hidden_dim), nn.GELU()]
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.conditioner = nn.Sequential(*layers)
        self.spline = RationalQuadraticSpline(num_bins=num_bins)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[..., :self.split_dim], x[..., self.split_dim:]

        # Get spline parameters from conditioner
        params = self.conditioner(x1)
        params = params.reshape(*x2.shape, 3 * self.num_bins + 1)

        # Apply spline
        y2, log_det = self.spline(x2, params)

        y = torch.cat([x1, y2], dim=-1)

        return y, log_det

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y1, y2 = y[..., :self.split_dim], y[..., self.split_dim:]

        params = self.conditioner(y1)
        params = params.reshape(*y2.shape, 3 * self.num_bins + 1)

        x2, log_det = self.spline.inverse(y2, params)

        x = torch.cat([y1, x2], dim=-1)

        return x, log_det


class NeuralSplineFlow(nn.Module):
    """
    Complete Neural Spline Flow model

    Transforms complex data distribution to simple base distribution
    """

    def __init__(self,
                 dim: int,
                 num_layers: int = 4,
                 hidden_dim: int = 128,
                 num_bins: int = 8):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers

        # Create coupling layers with alternating masks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                CouplingLayer(dim, hidden_dim, num_bins)
            )

        # Permutation matrices (learnable or fixed)
        self.register_buffer(
            'permutations',
            torch.stack([
                torch.randperm(dim) for _ in range(num_layers)
            ])
        )
        self.register_buffer(
            'inverse_permutations',
            torch.stack([
                torch.argsort(self.permutations[i])
                for i in range(num_layers)
            ])
        )

        # Running statistics for input normalization
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))
        self.register_buffer('num_batches', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform from data space to latent space

        Args:
            x: (batch, dim) data samples

        Returns:
            z: (batch, dim) latent samples
            log_det: (batch,) log determinant of Jacobian
        """
        # Update running statistics during training
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            self.num_batches += 1

            # Exponential moving average
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var

        # Normalize input
        z = (x - self.running_mean) / (self.running_var.sqrt() + 1e-6)
        log_det_normalization = -0.5 * torch.log(self.running_var + 1e-6).sum()

        total_log_det = log_det_normalization.expand(x.shape[0])

        for i, layer in enumerate(self.layers):
            # Apply coupling layer
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det

            # Apply permutation
            z = z[..., self.permutations[i]]

        return z, total_log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform from latent space to data space
        """
        total_log_det = torch.zeros(z.shape[0], device=z.device)
        x = z

        for i in reversed(range(self.num_layers)):
            # Inverse permutation
            x = x[..., self.inverse_permutations[i]]

            # Inverse coupling
            x, log_det = self.layers[i].inverse(x)
            total_log_det = total_log_det + log_det

        # Denormalize
        x = x * (self.running_var.sqrt() + 1e-6) + self.running_mean
        log_det_normalization = 0.5 * torch.log(self.running_var + 1e-6).sum()
        total_log_det = total_log_det + log_det_normalization

        return x, total_log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of data under the model
        """
        z, log_det = self.forward(x)

        # Standard normal log probability
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)

        return log_pz + log_det

    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate samples from the learned distribution
        """
        z = torch.randn(num_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x
```

### Feature Engineering for NSF

```python
def compute_market_features(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """
    Create feature vector for Neural Spline Flow

    Features are designed to capture:
    - Return dynamics at multiple scales
    - Volatility regime
    - Volume patterns
    - Price momentum
    """
    features = {}

    # Returns at multiple scales
    returns = df['close'].pct_change()
    for period in [1, 5, 10, 20]:
        features[f'return_{period}d'] = returns.rolling(period).sum().iloc[-1]

    # Volatility features
    features['volatility_20d'] = returns.rolling(lookback).std().iloc[-1]
    features['volatility_5d'] = returns.rolling(5).std().iloc[-1]
    features['vol_ratio'] = features['volatility_5d'] / (features['volatility_20d'] + 1e-8)

    # Higher moments
    features['skewness'] = returns.rolling(lookback).skew().iloc[-1]
    features['kurtosis'] = returns.rolling(lookback).kurt().iloc[-1]

    # Volume features
    volume_ma = df['volume'].rolling(lookback).mean()
    features['volume_ratio'] = df['volume'].iloc[-1] / (volume_ma.iloc[-1] + 1e-8)
    features['volume_trend'] = (
        df['volume'].rolling(5).mean().iloc[-1] /
        df['volume'].rolling(20).mean().iloc[-1]
    )

    # Price position in range
    high_20 = df['high'].rolling(lookback).max().iloc[-1]
    low_20 = df['low'].rolling(lookback).min().iloc[-1]
    features['price_position'] = (df['close'].iloc[-1] - low_20) / (high_20 - low_20 + 1e-8)

    # Momentum indicators
    features['rsi'] = compute_rsi(df['close'], 14)
    features['momentum'] = df['close'].iloc[-1] / df['close'].iloc[-lookback] - 1

    return np.array(list(features.values()))


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute Relative Strength Index"""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]

    if loss == 0:
        return 100.0
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

### NSF Trading System

```python
class NSFTrader:
    """
    Trading system using Neural Spline Flows
    """

    def __init__(self,
                 model: NeuralSplineFlow,
                 feature_dim: int,
                 return_feature_idx: int = 0,
                 density_threshold: float = -10.0,
                 var_confidence: float = 0.95):
        self.model = model
        self.feature_dim = feature_dim
        self.return_idx = return_feature_idx
        self.density_threshold = density_threshold
        self.var_confidence = var_confidence

    def compute_var(self, num_samples: int = 10000) -> Tuple[float, float]:
        """
        Compute Value-at-Risk from learned distribution
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(num_samples)
            returns = samples[:, self.return_idx].numpy()

        var = np.percentile(returns, (1 - self.var_confidence) * 100)
        cvar = returns[returns <= var].mean()

        return var, cvar

    def generate_signal(self, market_state: torch.Tensor) -> dict:
        """
        Generate trading signal from current market state
        """
        self.model.eval()
        x = market_state.unsqueeze(0)

        with torch.no_grad():
            # Compute log probability
            log_prob = self.model.log_prob(x).item()

            # Transform to latent space
            z, _ = self.model.forward(x)
            z = z.squeeze()

            # Return component in latent space
            return_z = z[self.return_idx].item()

            # Generate conditional samples for expected return
            samples = self.model.sample(1000)
            expected_return = samples[:, self.return_idx].mean().item()
            return_std = samples[:, self.return_idx].std().item()

        # Determine if in distribution
        in_distribution = log_prob > self.density_threshold

        # Signal based on expected return and confidence
        if not in_distribution:
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'log_prob': log_prob,
                'in_distribution': False,
                'expected_return': expected_return,
                'return_std': return_std,
                'reason': 'Out of distribution'
            }

        # Signal strength based on z-score
        confidence = min(abs(return_z) / 2.0, 1.0)

        if expected_return > 0 and return_z > 0.5:
            signal = confidence
        elif expected_return < 0 and return_z < -0.5:
            signal = -confidence
        else:
            signal = 0.0

        return {
            'signal': signal,
            'confidence': confidence,
            'log_prob': log_prob,
            'in_distribution': True,
            'expected_return': expected_return,
            'return_std': return_std,
            'latent_return': return_z
        }

    def position_size(self, signal_info: dict,
                      max_position: float = 1.0) -> float:
        """
        Compute position size based on signal and risk
        """
        if not signal_info['in_distribution']:
            return 0.0

        var, cvar = self.compute_var()

        # Scale by inverse of tail risk
        risk_scale = 1.0 / (abs(cvar) + 0.01)

        position = signal_info['signal'] * risk_scale * max_position

        return np.clip(position, -max_position, max_position)
```

### Training Pipeline

```python
def train_nsf_model(
    model: NeuralSplineFlow,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5
) -> NeuralSplineFlow:
    """
    Train Neural Spline Flow using maximum likelihood
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')
    best_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Shuffle data
        perm = torch.randperm(len(train_data))

        for i in range(0, len(train_data), batch_size):
            batch = train_data[perm[i:i+batch_size]]

            optimizer.zero_grad()

            # Negative log-likelihood loss
            log_prob = model.log_prob(batch)
            loss = -log_prob.mean()

            loss.backward()
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
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train NLL={total_loss/n_batches:.4f}, "
                  f"Val NLL={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model
```

### Backtesting Framework

```python
class NSFBacktest:
    """
    Backtesting framework for Neural Spline Flow trading
    """

    def __init__(self,
                 trader: NSFTrader,
                 lookback: int = 20):
        self.trader = trader
        self.lookback = lookback

    def run(self, prices: pd.DataFrame, warmup: int = 50) -> pd.DataFrame:
        """
        Run backtest on historical price data
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
            state = compute_market_features(window)
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Generate signal
            signal_info = self.trader.generate_signal(state_tensor)

            # Calculate PnL
            if i > warmup:
                daily_return = prices['close'].iloc[i] / prices['close'].iloc[i-1] - 1
                pnl = position * daily_return
                cumulative_pnl += pnl
            else:
                pnl = 0.0

            # Update position
            position = self.trader.position_size(signal_info)

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
        Calculate comprehensive performance metrics
        """
        returns = results['pnl']

        # Basic metrics
        total_return = results['cumulative_pnl'].iloc[-1]

        # Risk-adjusted returns
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = returns.mean() / downside.std() * np.sqrt(252)
        else:
            sortino = 0.0

        # Drawdown analysis
        cumulative = results['cumulative_pnl']
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative - rolling_max
        max_drawdown = drawdown.min()

        # Win statistics
        trading_returns = returns[returns != 0]
        if len(trading_returns) > 0:
            win_rate = (trading_returns > 0).mean()
            avg_win = trading_returns[trading_returns > 0].mean() if (trading_returns > 0).any() else 0
            avg_loss = trading_returns[trading_returns < 0].mean() if (trading_returns < 0).any() else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0

        # Distribution metrics
        in_dist_ratio = results['in_distribution'].mean()
        avg_log_prob = results['log_prob'].mean()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'in_distribution_ratio': in_dist_ratio,
            'avg_log_probability': avg_log_prob,
            'num_trades': (results['position'].diff().abs() > 0.01).sum()
        }
```

## Data Requirements

```
Historical OHLCV Data:
├── Minimum: 1 year of data
├── Recommended: 2+ years for robust density estimation
├── Frequency: 1-hour to daily
└── Source: Bybit exchange

Required Fields:
├── timestamp
├── open, high, low, close
├── volume
└── Optional: funding rate, open interest

Preprocessing:
├── Handle missing values (forward fill)
├── Remove outliers (> 5 std)
├── Normalize features (z-score)
└── Train/Val/Test split: 70/15/15
```

## Key Metrics

### Model Quality Metrics
- **Log-Likelihood**: Average log probability on test data
- **Bits per Dimension**: Normalized likelihood measure
- **KL Divergence**: Distance from true distribution (if known)

### Trading Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### Distribution Metrics
- **In-Distribution Ratio**: Fraction of days with high log-probability
- **Tail Coverage**: How well model captures extreme events
- **Calibration**: Probability estimates vs. observed frequencies

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

1. **Accurate Density Estimation**: Model captures heavy tails, skewness, and multi-modality of return distributions
2. **Regime Detection**: Log-probability changes indicate market regime shifts
3. **Risk Quantification**: Precise VaR/CVaR estimates from learned distribution
4. **Trading Performance**: Expected Sharpe Ratio 0.8-1.5 with proper calibration
5. **Anomaly Detection**: Low probability events flagged before they impact portfolio

## Comparison with Other Methods

| Method | Flexibility | Exact Likelihood | Sampling Speed | Training Stability |
|--------|-------------|------------------|----------------|-------------------|
| **Neural Spline Flows** | ⭐⭐⭐⭐⭐ | ✅ Yes | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Affine Flows | ⭐⭐⭐ | ✅ Yes | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| VAE | ⭐⭐⭐⭐ | ❌ ELBO | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| GAN | ⭐⭐⭐⭐⭐ | ❌ No | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Diffusion | ⭐⭐⭐⭐⭐ | ❌ Approximate | ⭐⭐ | ⭐⭐⭐⭐ |

## References

1. **Neural Spline Flows** (Durkan et al., 2019)
   - URL: https://arxiv.org/abs/1906.04032
   - Key contribution: Rational-quadratic splines for coupling layers

2. **Normalizing Flows for Probabilistic Modeling and Inference** (Papamakarios et al., 2019)
   - URL: https://arxiv.org/abs/1912.02762
   - Comprehensive review of normalizing flows

3. **Density Estimation Using Real-NVP** (Dinh et al., 2017)
   - URL: https://arxiv.org/abs/1605.08803
   - Foundation of coupling-based flows

4. **NICE: Non-linear Independent Components Estimation** (Dinh et al., 2015)
   - URL: https://arxiv.org/abs/1410.8516
   - Original coupling layer idea

5. **Glow: Generative Flow with Invertible 1x1 Convolutions** (Kingma & Dhariwal, 2018)
   - URL: https://arxiv.org/abs/1807.03039
   - Introduced invertible 1x1 convolutions

## Rust Implementation

This chapter includes a complete Rust implementation for high-performance Neural Spline Flow trading on cryptocurrency data from Bybit. See `rust/` directory.

### Features:
- Real-time data fetching from Bybit API
- Neural Spline Flow implementation with rational-quadratic splines
- Maximum likelihood training
- Density estimation and sampling
- VaR/CVaR risk metrics
- Backtesting framework with comprehensive metrics
- Modular and extensible design

### Module Structure:
```
rust/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── api/
│   │   ├── mod.rs
│   │   └── bybit.rs
│   ├── flow/
│   │   ├── mod.rs
│   │   ├── spline.rs
│   │   ├── coupling.rs
│   │   └── nsf.rs
│   ├── trading/
│   │   ├── mod.rs
│   │   ├── signals.rs
│   │   └── risk.rs
│   ├── backtest/
│   │   └── mod.rs
│   └── utils/
│       └── mod.rs
└── examples/
    ├── basic_nsf.rs
    ├── bybit_trading.rs
    └── backtest.rs
```

## Difficulty Level

⭐⭐⭐⭐⭐ (Expert)

Requires understanding of: Probability Theory, Normalizing Flows, Change of Variables, Neural Networks, Spline Theory, Risk Management, Trading Systems
