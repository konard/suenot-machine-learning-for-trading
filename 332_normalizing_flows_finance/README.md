# Chapter 332: Normalizing Flows for Finance

## Overview

Normalizing Flows are a class of deep generative models that learn complex probability distributions by transforming a simple base distribution (like Gaussian) through a sequence of invertible, differentiable transformations. Unlike other generative models (VAEs, GANs), normalizing flows provide **exact likelihood computation**, making them ideal for financial applications where accurate density estimation is crucial for risk management.

## Why Normalizing Flows for Finance?

### The Problem with Traditional Approaches

Financial returns are notoriously **non-Gaussian**:

- **Fat tails**: Extreme events happen more frequently than normal distributions predict
- **Skewness**: Returns are often asymmetric (larger drops than gains)
- **Time-varying volatility**: The distribution shape changes over time
- **Multimodality**: Multiple market regimes create complex distributions

Traditional risk models (VaR, CVaR) assume Gaussian returns, leading to:
- Underestimation of tail risk
- Poor hedging decisions
- Unexpected losses during market stress

### Normalizing Flow Solution

Normalizing Flows learn the **true distribution** of returns:

```
Traditional: Assume X ~ N(μ, σ²) → Underestimate tail risk

Normalizing Flow: Learn p(X) directly → Accurate density for any shape
  z ~ N(0, I)     [Simple base distribution]
  x = f(z)        [Invertible transformation]
  p(x) = p(z)|det(∂f⁻¹/∂x)|  [Exact likelihood via change of variables]
```

## Mathematical Foundation

### Change of Variables Formula

The core principle of normalizing flows is the **change of variables formula**:

Given:
- Base distribution: z ~ p_z(z) (typically standard normal)
- Invertible transformation: x = f(z), so z = f⁻¹(x)
- Target distribution: p_x(x)

The density transformation is:

```
p_x(x) = p_z(f⁻¹(x)) |det(J_{f⁻¹}(x))|

where J_{f⁻¹}(x) = ∂f⁻¹(x)/∂x is the Jacobian matrix
```

For a sequence of K transformations:

```
z₀ → f₁ → z₁ → f₂ → z₂ → ... → f_K → x

log p(x) = log p(z₀) - Σᵢ log|det(J_{fᵢ})|
```

### Why Jacobian Matters

The Jacobian determinant accounts for how the transformation **stretches or compresses** space:

```
┌────────────────────────────────────────────────────────────┐
│                     JACOBIAN INTUITION                      │
├────────────────────────────────────────────────────────────┤
│                                                             │
│   Base Distribution (z)       Target Distribution (x)      │
│                                                             │
│      ┌─────────┐                   ┌──────────────┐        │
│      │  ***    │      f(z)        │    ***       │        │
│      │ *****   │  ──────────►     │ ***    **    │        │
│      │  ***    │                   │  ****  ***   │        │
│      └─────────┘                   └──────────────┘        │
│                                                             │
│   Gaussian blob               Complex distribution         │
│   (easy to sample)            (hard to model directly)     │
│                                                             │
│   Jacobian = How much volume changes at each point         │
│   |det(J)| > 1 → Space expands → Density decreases        │
│   |det(J)| < 1 → Space contracts → Density increases      │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## Types of Normalizing Flows

### 1. Affine Coupling Flows (RealNVP)

**Key Idea**: Split the input and apply simple transformations that have tractable Jacobians.

```
┌─────────────────────────────────────────────────────────────┐
│                    AFFINE COUPLING LAYER                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input: x = [x₁, x₂]  (split into two parts)              │
│                                                             │
│   Transformation:                                           │
│     y₁ = x₁                     (unchanged)                 │
│     y₂ = x₂ ⊙ exp(s(x₁)) + t(x₁)  (affine transform)       │
│                                                             │
│   where s() and t() are neural networks                     │
│                                                             │
│   Jacobian is TRIANGULAR → det = ∏ exp(s(x₁)) = exp(Σs)   │
│   Very efficient! O(D) instead of O(D³)                     │
│                                                             │
│        ┌──────────┐                                         │
│   x₁ ──┤ Identity ├──────────────────────────────► y₁      │
│        └──────────┘                                         │
│             │                                               │
│             ▼                                               │
│        ┌────────┐        ┌─────────────────────┐           │
│   x₂ ──┤ Neural ├──s,t──►│ y₂ = x₂·exp(s) + t │──► y₂     │
│        │Networks│        └─────────────────────┘           │
│        └────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. Autoregressive Flows (MAF/IAF)

**Masked Autoregressive Flow (MAF)**: Each dimension depends on previous dimensions.

```
x₁ = z₁ · σ₁ + μ₁
x₂ = z₂ · σ₂(x₁) + μ₂(x₁)
x₃ = z₃ · σ₃(x₁,x₂) + μ₃(x₁,x₂)
...

Jacobian is LOWER TRIANGULAR → det = ∏ σᵢ
```

**Inverse Autoregressive Flow (IAF)**: Opposite direction for faster sampling.

```
MAF: Fast density, slow sampling
IAF: Fast sampling, slow density

┌────────────────────────────────────────┐
│              MAF vs IAF                │
├────────────────────────────────────────┤
│ Operation     │  MAF    │   IAF       │
├───────────────┼─────────┼─────────────┤
│ log p(x)      │ O(1)    │ O(D)        │
│ Sampling      │ O(D)    │ O(1)        │
│ Training      │ Fast    │ Slow        │
│ Generation    │ Slow    │ Fast        │
└────────────────────────────────────────┘
```

### 3. Continuous Normalizing Flows (Neural ODE)

**Key Idea**: Instead of discrete transformations, define a continuous flow via an ODE:

```
dz/dt = f(z, t; θ)

Log-likelihood change:
d log p(z)/dt = -tr(∂f/∂z)

Solved via numerical integration (adjoint method)
```

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              NORMALIZING FLOW FOR FINANCIAL RETURNS             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT LAYER                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Financial Returns Data:                                   │   │
│  │   - Daily/hourly returns                                  │   │
│  │   - Multi-asset returns (portfolio)                       │   │
│  │   - Conditional features (volatility, volume, etc.)       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  PREPROCESSING                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Standardization: (x - μ) / σ                              │   │
│  │ Winsorization: clip extreme values                        │   │
│  │ Optional: Add context features for conditional flow       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  FLOW BLOCKS (×N)                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Affine Coupling Layer 1                             │   │   │
│  │ │   x₁ unchanged, x₂ transformed via s(x₁), t(x₁)    │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Permutation / Shuffling                             │   │   │
│  │ │   Ensure all dimensions get transformed             │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Affine Coupling Layer 2                             │   │   │
│  │ │   Opposite split (x₂ unchanged, x₁ transformed)    │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Batch Normalization (optional)                      │   │   │
│  │ │   Stabilize training                                │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  BASE DISTRIBUTION                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Standard Gaussian: z ~ N(0, I)                            │   │
│  │ Or Student-t for heavier tails: z ~ t(ν, 0, I)           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  OUTPUT                                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ log p(x) = log p(z) + Σ log|det(J_k)|                    │   │
│  │ Samples: z ~ p(z) → x = f(z)                             │   │
│  │ Density: x → z = f⁻¹(x) → p(x)                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Financial Applications

### 1. Density Estimation for Returns

```python
def estimate_return_density(model, returns):
    """
    Estimate the probability density of returns.

    Args:
        model: Trained normalizing flow
        returns: Array of return values

    Returns:
        log_prob: Log probability density at each point
    """
    # Transform returns to latent space
    z, log_det = model.inverse(returns)

    # Compute base distribution log probability
    log_pz = -0.5 * (z**2 + np.log(2*np.pi)).sum(dim=-1)

    # Total log probability via change of variables
    log_prob = log_pz + log_det

    return log_prob
```

### 2. Value at Risk (VaR) with Learned Densities

Traditional VaR assumes Gaussian returns. With normalizing flows:

```
┌─────────────────────────────────────────────────────────────┐
│                    VaR COMPARISON                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Gaussian VaR (underestimates tail risk):                  │
│   VaR_α = μ + σ · Φ⁻¹(α)                                   │
│                                                             │
│   Normalizing Flow VaR (accurate):                          │
│   VaR_α = quantile from learned distribution p(x)           │
│   Found via: ∫_{-∞}^{VaR} p(x)dx = α                       │
│                                                             │
│   Monte Carlo approach:                                     │
│   1. Sample N points from flow: xᵢ ~ p(x)                   │
│   2. Sort samples                                           │
│   3. VaR_α = x_{⌊αN⌋}                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. Conditional Value at Risk (CVaR / Expected Shortfall)

```python
def compute_cvar(model, alpha=0.05, n_samples=100000):
    """
    Compute CVaR using normalizing flow samples.

    CVaR_α = E[X | X ≤ VaR_α]
    """
    # Generate samples from learned distribution
    z = torch.randn(n_samples, model.dim)
    samples = model.forward(z)

    # Find VaR threshold
    var = np.percentile(samples, alpha * 100)

    # Average of samples below VaR
    cvar = samples[samples <= var].mean()

    return var, cvar
```

### 4. Synthetic Data Generation

Generate realistic return scenarios for:
- Stress testing
- Backtesting on more data
- Training other models
- Monte Carlo simulations

```python
def generate_synthetic_returns(model, n_scenarios, conditioning=None):
    """
    Generate synthetic return scenarios from learned distribution.
    """
    # Sample from base distribution
    z = torch.randn(n_scenarios, model.dim)

    # Transform through flow
    if conditioning is not None:
        # Conditional generation (e.g., high volatility regime)
        synthetic_returns = model.forward(z, conditioning)
    else:
        synthetic_returns = model.forward(z)

    return synthetic_returns
```

### 5. Tail Risk Modeling

```
┌─────────────────────────────────────────────────────────────┐
│               TAIL RISK COMPARISON                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Probability of -10% daily return:                         │
│                                                             │
│   Gaussian (σ=2%):  P(X < -10%) = 3 × 10⁻⁷  (very rare)   │
│   Historical:       P(X < -10%) ≈ 0.1%      (happens!)     │
│   Normalizing Flow: P(X < -10%) ≈ 0.08%     (accurate!)    │
│                                                             │
│   The flow learns the TRUE tail behavior!                   │
│                                                             │
│                    Gaussian                                  │
│          │    ***       vs        Learned Flow              │
│          │  *******                   ***                   │
│     P(x) │ *********                ******                  │
│          │***********              ********                 │
│          │           *            **********                │
│          └──────────────┘         ──────────┘              │
│             thin tails              fat tails               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### Network Architecture for Scale/Translation Networks

```python
class CouplingNetwork(nn.Module):
    """
    Neural network for computing scale and translation in coupling layers.
    """
    def __init__(self, input_dim, hidden_dim=256, n_layers=3):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        # Output scale and translation
        self.net = nn.Sequential(*layers)
        self.scale_net = nn.Linear(hidden_dim, input_dim)
        self.translation_net = nn.Linear(hidden_dim, input_dim)

        # Initialize to identity transform
        nn.init.zeros_(self.scale_net.weight)
        nn.init.zeros_(self.scale_net.bias)
        nn.init.zeros_(self.translation_net.weight)
        nn.init.zeros_(self.translation_net.bias)

    def forward(self, x):
        h = self.net(x)
        s = self.scale_net(h)
        t = self.translation_net(h)
        return s, t
```

### Affine Coupling Layer

```python
class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer as used in RealNVP.
    """
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.register_buffer('mask', mask)
        self.coupling_net = CouplingNetwork(dim // 2, hidden_dim=256)

    def forward(self, x):
        """Forward pass: data space -> latent space"""
        x_masked = x * self.mask
        s, t = self.coupling_net(x_masked)

        # Apply transformation to unmasked part
        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)

        # Log determinant of Jacobian
        log_det = (s * (1 - self.mask)).sum(dim=-1)

        return y, log_det

    def inverse(self, y):
        """Inverse pass: latent space -> data space"""
        y_masked = y * self.mask
        s, t = self.coupling_net(y_masked)

        # Inverse transformation
        x = y_masked + (1 - self.mask) * (y - t) * torch.exp(-s)

        # Log determinant (negative for inverse)
        log_det = -(s * (1 - self.mask)).sum(dim=-1)

        return x, log_det
```

### Complete Normalizing Flow Model

```python
class NormalizingFlow(nn.Module):
    """
    Complete normalizing flow for density estimation.
    """
    def __init__(self, dim, n_layers=8, hidden_dim=256):
        super().__init__()
        self.dim = dim

        # Create alternating masks
        masks = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[:dim//2] = 1.0 if i % 2 == 0 else 0.0
            mask[dim//2:] = 0.0 if i % 2 == 0 else 1.0
            masks.append(mask)

        # Stack coupling layers
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim, masks[i])
            for i in range(n_layers)
        ])

        # Base distribution
        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_std', torch.ones(dim))

    def forward(self, z):
        """Transform from latent space to data space"""
        x = z
        for layer in self.layers:
            x, _ = layer.inverse(x)
        return x

    def inverse(self, x):
        """Transform from data space to latent space"""
        z = x
        total_log_det = 0
        for layer in reversed(self.layers):
            z, log_det = layer(z)
            total_log_det += log_det
        return z, total_log_det

    def log_prob(self, x):
        """Compute log probability of data"""
        z, log_det = self.inverse(x)
        log_pz = -0.5 * (z**2 + np.log(2*np.pi)).sum(dim=-1)
        return log_pz + log_det

    def sample(self, n_samples):
        """Generate samples from learned distribution"""
        z = torch.randn(n_samples, self.dim)
        return self.forward(z)
```

### Training Configuration

```yaml
model:
  dim: 1  # Univariate returns (or portfolio dimension)
  n_layers: 8
  hidden_dim: 256
  activation: "relu"
  use_batch_norm: true

training:
  batch_size: 256
  learning_rate: 0.0001
  weight_decay: 0.0001
  max_epochs: 500
  early_stopping_patience: 20
  gradient_clip: 1.0

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  lookback_window: 252  # 1 year of daily data
  returns_type: "log"  # log returns
  standardize: true
```

## Risk Metrics with Normalizing Flows

### VaR Calculation

```python
def compute_var_flow(model, alpha_levels=[0.01, 0.05, 0.10], n_samples=100000):
    """
    Compute Value at Risk at multiple confidence levels.
    """
    # Generate samples
    samples = model.sample(n_samples).detach().numpy().flatten()

    var_results = {}
    for alpha in alpha_levels:
        var = np.percentile(samples, alpha * 100)
        var_results[f'VaR_{int((1-alpha)*100)}'] = var

    return var_results
```

### CVaR/Expected Shortfall

```python
def compute_cvar_flow(model, alpha=0.05, n_samples=100000):
    """
    Compute Conditional VaR (Expected Shortfall).
    """
    samples = model.sample(n_samples).detach().numpy().flatten()
    var = np.percentile(samples, alpha * 100)
    cvar = samples[samples <= var].mean()
    return var, cvar
```

### Tail Probability

```python
def compute_tail_probability(model, threshold, n_samples=100000):
    """
    Compute probability of returns below threshold.
    P(X < threshold)
    """
    samples = model.sample(n_samples).detach().numpy().flatten()
    tail_prob = (samples < threshold).mean()
    return tail_prob
```

## Trading Strategy Integration

### Signal Generation Based on Density

```python
def generate_density_signals(model, current_return, historical_returns):
    """
    Generate trading signals based on return density position.

    If current return is in low-probability region, expect mean reversion.
    """
    # Compute log probability of current return
    log_prob = model.log_prob(torch.tensor([[current_return]])).item()

    # Compute percentile of current return
    samples = model.sample(100000).numpy().flatten()
    percentile = (samples < current_return).mean()

    # Signal logic
    if percentile < 0.05:  # Extreme low return
        return Signal("LONG", confidence=1 - percentile,
                      reason="Extreme negative return, expect bounce")
    elif percentile > 0.95:  # Extreme high return
        return Signal("SHORT", confidence=percentile,
                      reason="Extreme positive return, expect pullback")
    else:
        return Signal("NEUTRAL", confidence=0.5)
```

### Portfolio Risk Management

```python
class FlowBasedRiskManager:
    """
    Risk manager using normalizing flow for position sizing.
    """
    def __init__(self, flow_model, max_var_pct=0.02):
        self.model = flow_model
        self.max_var = max_var_pct

    def compute_position_size(self, capital, confidence=0.99):
        """
        Size position so that 99% VaR doesn't exceed max_var_pct.
        """
        # Get VaR from flow
        var_99, _ = compute_cvar_flow(self.model, alpha=1-confidence)

        # Position size such that loss at VaR = max_var_pct of capital
        position_size = (self.max_var * capital) / abs(var_99)

        return position_size
```

## Key Metrics

### Model Performance

- **Negative Log-Likelihood (NLL)**: Lower is better (measures density fit)
- **Bits per Dimension (BPD)**: NLL / (dim * log(2))
- **Kolmogorov-Smirnov Test**: Compare learned vs empirical distribution
- **QQ Plot**: Visual check of distribution fit

### Risk Metric Accuracy

- **VaR Backtesting**: Count violations (should match confidence level)
- **CVaR Accuracy**: Compare predicted vs realized tail losses
- **Kupiec Test**: Statistical test for VaR accuracy

### Trading Performance

- **Sharpe Ratio**: Risk-adjusted returns (target > 1.5)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return / Max Drawdown

## Comparison with Other Methods

| Aspect | Gaussian | GARCH | Historical Sim | Normalizing Flow |
|--------|----------|-------|----------------|------------------|
| Fat tails | No | Partial | Yes | Yes |
| Skewness | No | No | Yes | Yes |
| Multimodality | No | No | Limited | Yes |
| Generalization | Poor | Moderate | Poor | Good |
| Exact density | Yes | Approximation | No | Yes |
| Synthetic data | Easy | Moderate | Limited | Easy |
| Computational cost | Low | Low | Low | Medium |

## Advanced Topics

### 1. Conditional Normalizing Flows

Condition the flow on external factors (volatility regime, market conditions):

```python
class ConditionalFlow(nn.Module):
    def __init__(self, dim, cond_dim, n_layers=8):
        # Conditioning network
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Coupling networks take conditioning as input
        self.layers = nn.ModuleList([
            ConditionalCouplingLayer(dim, cond_dim=64)
            for _ in range(n_layers)
        ])
```

### 2. Multivariate Flows for Portfolio

Model joint distribution of multiple assets:

```python
# Instead of modeling each asset separately
# Model full covariance structure
flow = NormalizingFlow(dim=10)  # 10 assets

# Joint samples capture correlations
joint_samples = flow.sample(1000)  # [1000, 10]

# Portfolio VaR accounts for diversification
portfolio_returns = joint_samples @ weights
portfolio_var = np.percentile(portfolio_returns, 5)
```

### 3. Time-Varying Flows

Update flow parameters as market conditions change:

```python
class AdaptiveFlow:
    def __init__(self, base_flow, adaptation_rate=0.01):
        self.flow = base_flow
        self.rate = adaptation_rate

    def update(self, new_data):
        """Online update with new observations"""
        loss = -self.flow.log_prob(new_data).mean()
        loss.backward()

        with torch.no_grad():
            for param in self.flow.parameters():
                param -= self.rate * param.grad
                param.grad.zero_()
```

## Production Considerations

```
Inference Pipeline:
├── Data Collection (Bybit via CCXT)
│   └── Real-time OHLCV data
├── Return Computation
│   └── Log returns with rolling statistics
├── Model Inference
│   └── Density evaluation / sample generation
├── Risk Calculation
│   └── VaR, CVaR, tail probabilities
├── Signal Generation
│   └── Based on density position
└── Execution
    └── Position sizing from risk model

Latency Budget:
├── Data fetch: ~50ms (REST API)
├── Preprocessing: ~1ms
├── Flow inference: ~5ms (GPU)
├── Risk calculation: ~10ms (MC sampling)
├── Signal generation: ~1ms
└── Total: ~70ms
```

## Directory Structure

```
332_normalizing_flows_finance/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Beginner-friendly explanation
├── readme.simple.ru.md          # Russian beginner version
├── python/                      # Python implementation
│   ├── __init__.py
│   ├── flows.py                 # Normalizing flow models
│   ├── layers.py                # Coupling layers
│   ├── risk_metrics.py          # VaR, CVaR calculations
│   ├── data_fetcher.py          # Bybit data via CCXT
│   ├── training.py              # Training loop
│   └── examples/
│       ├── density_estimation.py
│       ├── var_calculation.py
│       └── synthetic_generation.py
└── rust_normalizing_flows/      # Rust implementation
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs
    │   ├── api/                 # Bybit API client
    │   ├── flows/               # Flow implementations
    │   ├── risk/                # Risk metrics
    │   └── utils/               # Utilities
    └── examples/
        ├── fetch_data.rs
        ├── train_flow.rs
        └── compute_var.rs
```

## References

1. **Variational Inference with Normalizing Flows** (Rezende & Mohamed, 2015)
   - https://arxiv.org/abs/1505.05770

2. **Density Estimation using Real-NVP** (Dinh et al., 2016)
   - https://arxiv.org/abs/1605.08803

3. **Masked Autoregressive Flow for Density Estimation** (Papamakarios et al., 2017)
   - https://arxiv.org/abs/1705.07057

4. **Neural Spline Flows** (Durkan et al., 2019)
   - https://arxiv.org/abs/1906.04032

5. **Normalizing Flows for Probabilistic Modeling and Inference** (Papamakarios et al., 2019)
   - https://arxiv.org/abs/1912.02762

## Difficulty Level

**Advanced** - Requires understanding of:
- Probability theory and density estimation
- Change of variables formula
- Jacobian determinants
- Deep learning fundamentals
- Financial risk metrics (VaR, CVaR)

## Disclaimer

This chapter is for **educational purposes only**. Cryptocurrency trading involves substantial risk. The strategies and risk models described here should be thoroughly validated before any real-world application. Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.
