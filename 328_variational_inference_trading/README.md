# Chapter 328: Variational Inference Trading

## Overview

Variational Inference (VI) is a powerful technique for scalable Bayesian inference that transforms intractable probability distributions into optimization problems. Instead of sampling (like MCMC), VI finds the best approximation to the true posterior from a family of simpler distributions. This makes VI orders of magnitude faster while still capturing uncertainty - a critical advantage for real-time trading applications.

## Why Variational Inference for Trading?

### The Problem with Point Estimates

Traditional ML models for trading produce **point estimates** - single predictions without uncertainty:

- **Linear Regression**: "The price will go up by 2%"
- **Neural Network**: "70% probability of upward movement"
- **Random Forest**: "Buy signal"

But markets are **inherently uncertain**:
- What if the model is 70% confident with high uncertainty vs. 70% confident with low uncertainty?
- How should position sizing differ based on prediction uncertainty?
- When should we abstain from trading due to uncertainty?

### Variational Inference Solution

VI provides **full posterior distributions** over predictions:

```
Point Estimate: E[r] = 2.5%

Variational Inference:
p(r|data) ≈ N(μ=2.5%, σ=1.2%)

This tells us:
- Expected return: 2.5%
- 68% confidence interval: [1.3%, 3.7%]
- 95% confidence interval: [0.1%, 4.9%]
- Probability of positive return: 98%
```

## Technical Foundations

### 1. The Evidence Lower Bound (ELBO)

The core of VI is maximizing the Evidence Lower Bound:

```
ELBO(q) = E_q[log p(x,z)] - E_q[log q(z)]
        = E_q[log p(x|z)] - KL(q(z) || p(z))

where:
  q(z) = approximate posterior (what we optimize)
  p(z) = prior distribution
  p(x|z) = likelihood
  p(z|x) = true posterior (intractable)
```

**Intuition**: ELBO balances two objectives:
1. **Likelihood term**: Make predictions that fit the data
2. **KL term**: Keep the approximate posterior close to the prior (regularization)

### 2. KL Divergence

KL divergence measures how one distribution differs from another:

```
KL(q || p) = E_q[log q(z) - log p(z)]
           = ∫ q(z) log(q(z)/p(z)) dz

Properties:
- KL(q || p) ≥ 0
- KL(q || p) = 0 iff q = p
- NOT symmetric: KL(q || p) ≠ KL(p || q)
```

For Gaussians:
```
KL(N(μ₁, σ₁²) || N(μ₂, σ₂²)) =
  log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
```

### 3. Mean-Field Approximation

The simplest VI approach assumes factorized posterior:

```
True posterior: p(z₁, z₂, ..., zₙ | x)  [complex dependencies]

Mean-field: q(z) = ∏ᵢ q(zᵢ)  [independent factors]

For trading:
q(regime, returns, volatility) ≈ q(regime) × q(returns) × q(volatility)
```

### 4. Reparameterization Trick

Key innovation for training VAEs with gradient descent:

```python
# Problem: Cannot backprop through sampling
z = sample(q(z|x))  # Non-differentiable!

# Solution: Reparameterize
ε ~ N(0, 1)
z = μ + σ × ε  # Now differentiable w.r.t. μ, σ

# In code:
class Reparameterize:
    def forward(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
```

## Model Architecture

### Variational Autoencoder for Market Data

```
┌─────────────────────────────────────────────────────────────────┐
│                    MARKET VAE ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT LAYER                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Market Features (per timestep):                           │   │
│  │   - OHLCV data (normalized)                               │   │
│  │   - Technical indicators (RSI, MACD, BB)                  │   │
│  │   - Volume profile                                        │   │
│  │   - Order flow metrics                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ENCODER NETWORK                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LSTM/Transformer layers                                   │   │
│  │   - Captures temporal dependencies                        │   │
│  │   - Outputs: μ_z, log(σ²_z)                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  LATENT SPACE (z)                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ z = μ + σ × ε  (reparameterization trick)                │   │
│  │                                                           │   │
│  │ Latent dimensions represent:                              │   │
│  │   - Market regime (bull/bear/sideways)                   │   │
│  │   - Volatility regime (low/medium/high)                  │   │
│  │   - Trend strength                                        │   │
│  │   - Mean reversion tendency                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  DECODER NETWORK                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Reconstructs market features from latent code             │   │
│  │   - Predicts next-step returns                           │   │
│  │   - Predicts volatility                                   │   │
│  │   - Outputs uncertainty estimates                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  OUTPUT HEADS                                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Reconstruction: p(x|z) - reconstructed features          │   │
│  │ Returns: μ_r, σ_r - predicted return distribution        │   │
│  │ Regime: p(regime|z) - market regime probabilities        │   │
│  │ Confidence: model uncertainty estimate                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Latent Space Regime Detection

The VAE learns to organize latent space by market regimes:

```
Latent Space Visualization:

                    Bull Market
                        ↑
                   *  *   *
                  *  *  *  *
               * * * * * * * *
              *  *  *  *  *  *
    Sideways ←  *  *  *  *  *  * → Trending
              *  *  *  *  *  *
               * * * * * * * *
                  *  *  *  *
                   *  *   *
                        ↓
                    Bear Market

Each point represents a market state
Clustering emerges automatically during training
```

## Trading Strategy

### Signal Generation with Uncertainty

```python
def generate_signals(vae_model, market_data):
    # Encode current market state
    mu, log_var = vae_model.encode(market_data)

    # Sample multiple times for uncertainty estimation
    n_samples = 100
    predictions = []

    for _ in range(n_samples):
        z = reparameterize(mu, log_var)
        pred = vae_model.predict_returns(z)
        predictions.append(pred)

    # Aggregate predictions
    mean_return = np.mean(predictions)
    std_return = np.std(predictions)

    # Probability of positive return
    prob_positive = np.mean([p > 0 for p in predictions])

    # Generate signal with confidence
    if prob_positive > 0.7 and std_return < threshold:
        return Signal("LONG", confidence=prob_positive, uncertainty=std_return)
    elif prob_positive < 0.3 and std_return < threshold:
        return Signal("SHORT", confidence=1-prob_positive, uncertainty=std_return)
    else:
        return Signal("HOLD", confidence=0.5, uncertainty=std_return)
```

### Variational Bayes Portfolio Optimization

```python
def variational_portfolio(expected_returns, uncertainties, risk_aversion=1.0):
    """
    Portfolio optimization accounting for parameter uncertainty

    Instead of: max w'μ - λ/2 w'Σw  (standard mean-variance)

    We use: max E[w'r] - λ/2 Var[w'r] - κ * uncertainty_penalty

    where uncertainty_penalty accounts for model uncertainty
    """
    n_assets = len(expected_returns)

    # Build covariance matrix with uncertainty inflation
    cov_matrix = estimate_covariance(returns)

    # Inflate covariance based on prediction uncertainty
    uncertainty_matrix = np.diag(uncertainties ** 2)
    adjusted_cov = cov_matrix + uncertainty_matrix

    # Solve portfolio optimization
    def objective(w):
        portfolio_return = w @ expected_returns
        portfolio_var = w @ adjusted_cov @ w
        return -(portfolio_return - risk_aversion * portfolio_var)

    # Constraints: weights sum to 1, no shorting
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    ]
    bounds = [(0, 1) for _ in range(n_assets)]

    result = minimize(objective, x0=np.ones(n_assets)/n_assets,
                     constraints=constraints, bounds=bounds)

    return result.x
```

## Key Components

### 1. Stochastic Variational Inference

```python
class StochasticVI:
    """
    Mini-batch VI for large datasets

    Instead of computing ELBO on full dataset:
    ELBO = Σᵢ E_q[log p(xᵢ|z)] - KL(q(z) || p(z))

    We use stochastic estimates:
    ELBO ≈ (N/M) Σⱼ∈batch E_q[log p(xⱼ|z)] - KL(q(z) || p(z))
    """

    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=learning_rate)

    def compute_elbo(self, x_batch, beta=1.0):
        # Encode
        mu, log_var = self.model.encode(x_batch)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        x_recon = self.model.decode(z)

        # Reconstruction loss (negative log-likelihood)
        recon_loss = F.mse_loss(x_recon, x_batch, reduction='sum')

        # KL divergence (analytical for Gaussians)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # ELBO = -recon_loss - beta * kl_loss
        elbo = -recon_loss - beta * kl_loss

        return elbo, recon_loss, kl_loss

    def train_step(self, x_batch, beta=1.0):
        self.optimizer.zero_grad()
        elbo, recon_loss, kl_loss = self.compute_elbo(x_batch, beta)
        loss = -elbo  # Minimize negative ELBO
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

### 2. Amortized Inference

```python
class AmortizedEncoder(nn.Module):
    """
    Instead of optimizing q(z) for each datapoint separately,
    learn a single encoder network that maps x → q(z|x)

    This amortizes the cost of inference:
    - Training: O(N) to train encoder
    - Inference: O(1) per new datapoint
    """

    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 128]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        log_var = self.logvar_layer(h)
        return mu, log_var
```

### 3. Beta-VAE for Disentangled Representations

```python
class BetaVAE(nn.Module):
    """
    Beta-VAE adds a coefficient β to KL term:

    L = E_q[log p(x|z)] - β * KL(q(z|x) || p(z))

    β > 1: More disentangled latent factors
    β < 1: Better reconstruction

    For trading:
    - Higher β: Cleaner regime separation
    - Lower β: More accurate price reconstruction
    """

    def __init__(self, input_dim, latent_dim, beta=4.0):
        super().__init__()
        self.beta = beta
        self.encoder = AmortizedEncoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def loss_function(self, x, x_recon, mu, log_var):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + self.beta * kl_loss
```

## Implementation Details

### Data Requirements

```
Cryptocurrency Market Data:
├── OHLCV data (1-minute resolution minimum)
│   └── Multiple assets (BTC, ETH, SOL, ...)
├── Technical indicators
│   ├── RSI, MACD, Bollinger Bands
│   ├── Moving averages (SMA, EMA)
│   └── Volatility metrics (ATR, historical vol)
├── Volume features
│   ├── Volume profile
│   ├── Buy/sell volume ratio
│   └── VWAP deviation
└── Order flow (optional)
    ├── Order book imbalance
    └── Trade flow metrics

Preprocessing:
├── Normalization (z-score or min-max)
├── Stationarity (differencing for prices)
├── Sequence formation (sliding windows)
└── Train/validation/test split (temporal)
```

### Training Configuration

```yaml
model:
  input_dim: 64           # Number of input features
  latent_dim: 16          # Latent space dimension
  hidden_dims: [256, 128, 64]
  beta: 4.0               # KL weight for disentanglement
  dropout: 0.1

training:
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0001
  max_epochs: 200
  early_stopping_patience: 20

  # Beta annealing (start with reconstruction, gradually add KL)
  beta_start: 0.0
  beta_end: 4.0
  beta_warmup_epochs: 50

data:
  sequence_length: 60     # 1 hour of 1-min data
  prediction_horizon: 5   # 5 minutes ahead
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

## Key Metrics

### Model Performance

- **ELBO**: Evidence Lower Bound (higher is better)
- **Reconstruction MSE**: How well model reconstructs inputs
- **KL Divergence**: Posterior vs prior distance
- **Latent Traversal Quality**: Visual inspection of learned factors

### Trading Performance

- **Sharpe Ratio**: Risk-adjusted returns (target > 2.0)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: % of profitable trades
- **Calibration**: Do predicted probabilities match reality?

## Advantages of Variational Inference

| Aspect | Point Estimates | Variational Inference |
|--------|----------------|----------------------|
| Speed | Fast | Fast (vs MCMC) |
| Uncertainty | None | Full posterior |
| Position sizing | Fixed | Uncertainty-adjusted |
| Regime detection | Separate model | Built-in (latent space) |
| Overfitting | High risk | Regularized by prior |
| Out-of-distribution | No detection | Detectable via KL |

## Comparison with Other Approaches

### vs. MCMC (Markov Chain Monte Carlo)

- **MCMC**: Exact posterior, but slow (hours for complex models)
- **VI**: Approximate posterior, but fast (minutes/seconds)

### vs. Dropout Uncertainty

- **Dropout**: Easy to implement, but ad-hoc uncertainty
- **VI**: Principled uncertainty with theoretical guarantees

### vs. Ensemble Methods

- **Ensemble**: Multiple models, high memory/compute
- **VI**: Single model, efficient inference

### vs. Gaussian Processes

- **GP**: Exact uncertainty, but O(N³) complexity
- **VI**: Scalable to millions of datapoints

## Production Considerations

```
Inference Pipeline:
├── Data Collection (Bybit WebSocket)
│   └── Real-time OHLCV + indicators
├── Feature Engineering
│   └── Compute input features (normalized)
├── VAE Inference
│   └── μ, σ = encoder(features)
│   └── Sample z multiple times
├── Prediction Aggregation
│   └── Mean, std, confidence intervals
├── Signal Generation
│   └── Apply trading rules with uncertainty
└── Risk Management
    └── Position sizing based on uncertainty

Latency Budget:
├── Data collection: ~10ms
├── Feature computation: ~5ms
├── VAE forward pass: ~2ms
├── Sampling (100 samples): ~10ms
├── Signal generation: ~1ms
└── Total: ~30ms
```

## Directory Structure

```
328_variational_inference_trading/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Beginner-friendly explanation
├── readme.simple.ru.md          # Russian beginner version
├── python/                      # Python implementation
│   ├── requirements.txt
│   ├── config.py
│   ├── data_fetcher.py          # Bybit CCXT data fetching
│   ├── vae_model.py             # VAE implementation
│   ├── trainer.py               # Training loop
│   ├── strategy.py              # Trading strategy
│   ├── backtest.py              # Backtesting engine
│   └── main.py                  # Main entry point
└── rust_variational/            # Rust implementation
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               # Library entry point
    │   ├── api/                 # Bybit API client
    │   ├── variational/         # VI implementations
    │   ├── features/            # Feature engineering
    │   ├── strategy/            # Trading strategy
    │   └── backtest/            # Backtesting engine
    └── examples/
        ├── fetch_data.rs
        ├── train_vae.rs
        └── live_signals.rs
```

## References

1. **Variational Inference: A Review for Statisticians**
   - https://arxiv.org/abs/1601.00670
   - Blei, Kucukelbir, McAuliffe (2017)

2. **Auto-Encoding Variational Bayes**
   - https://arxiv.org/abs/1312.6114
   - Kingma & Welling (2014)

3. **beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework**
   - https://openreview.net/forum?id=Sy2fzU9gl
   - Higgins et al. (2017)

4. **Stochastic Variational Inference**
   - https://arxiv.org/abs/1206.7051
   - Hoffman et al. (2013)

5. **Deep Variational Portfolio**
   - Applications of VAE to portfolio optimization

## Difficulty Level

**Advanced** - Requires understanding of:
- Probability theory (KL divergence, variational calculus)
- Deep learning (autoencoders, backpropagation)
- Bayesian inference concepts
- PyTorch/tensor operations
- Financial time series

## Disclaimer

This chapter is for **educational purposes only**. Cryptocurrency trading involves substantial risk. The strategies described here have not been validated in live trading and should be thoroughly tested before any real-world application. Past performance does not guarantee future results.
