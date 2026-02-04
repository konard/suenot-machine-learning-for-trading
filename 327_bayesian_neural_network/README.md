# Chapter 327: Bayesian Neural Networks for Trading

## Overview

Bayesian Neural Networks (BNNs) represent a paradigm shift from traditional neural networks by treating network weights as probability distributions rather than fixed values. This fundamental change enables **uncertainty quantification** - the ability to know not just what the model predicts, but how confident it is in that prediction. For trading, this capability is invaluable: we can distinguish between high-confidence signals worth acting on and uncertain predictions that warrant caution.

## Why Bayesian Neural Networks for Trading?

### The Problem with Traditional Neural Networks

Standard neural networks produce **point estimates** - single predictions without any measure of confidence:

```
Traditional NN: Input → Prediction (65% chance price goes up)
               "How confident are you?" → "I don't know"
```

This is problematic for trading because:

1. **All predictions look equally confident** - even when the model is guessing
2. **No distinction between known and unknown** - extrapolation looks the same as interpolation
3. **Overfitting is invisible** - overconfident predictions on novel market conditions
4. **Position sizing is arbitrary** - no principled way to scale bets by confidence

### The Bayesian Solution

BNNs maintain **distributions over weights**, enabling:

```
Bayesian NN: Input → Prediction Distribution
            Mean: 65% up, Std: 15%
            "High uncertainty - reduce position size"

            vs.

            Mean: 68% up, Std: 3%
            "High confidence - full position size"
```

## Theoretical Foundations

### Weight Uncertainty in Neural Networks

Traditional neural networks learn point estimates for weights:

```
Standard NN: w = argmax P(D|w)  (Maximum Likelihood)

Bayesian NN: P(w|D) ∝ P(D|w) P(w)  (Posterior over weights)
```

Where:
- `P(w|D)` = Posterior distribution over weights given data
- `P(D|w)` = Likelihood of data given weights
- `P(w)` = Prior distribution over weights

### Prior Distributions

The prior `P(w)` encodes our beliefs about weights before seeing data:

```
Common Priors:

1. Standard Normal:
   P(w) = N(0, 1)
   - Encourages small weights
   - Acts as L2 regularization

2. Spike-and-Slab:
   P(w) = π·δ(0) + (1-π)·N(0, σ²)
   - Encourages sparsity
   - Some weights exactly zero

3. Hierarchical Prior:
   P(w|σ) = N(0, σ²)
   P(σ) = InverseGamma(α, β)
   - Learns appropriate regularization
   - Automatic relevance determination

4. Mixture of Gaussians:
   P(w) = Σᵢ πᵢ N(μᵢ, σᵢ²)
   - Flexible prior shapes
   - Can capture multi-modal beliefs
```

### Bayes by Backprop

Since computing the true posterior is intractable, we use **Variational Inference** to approximate it:

```
True Posterior: P(w|D) - intractable
Approximate:    q(w|θ) - parameterized distribution

Goal: Find θ that minimizes KL divergence:
      KL[q(w|θ) || P(w|D)]
```

This leads to the **Evidence Lower Bound (ELBO)**:

```
ELBO(θ) = E_q[log P(D|w)] - KL[q(w|θ) || P(w)]
        = Data Fit Term    - Complexity Term

Maximize ELBO ≈ Minimize KL divergence to true posterior
```

### Reparameterization Trick

To enable gradient-based optimization, we reparameterize the weight sampling:

```
Instead of: w ~ q(w|θ) = N(μ, σ²)

We use:     ε ~ N(0, 1)
            w = μ + σ·ε

This allows gradients to flow through μ and σ!
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BAYESIAN NEURAL NETWORK                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT LAYER                                                         │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ Market Features:                                            │     │
│  │   - Price returns (1m, 5m, 15m, 1h, 4h, 1d)               │     │
│  │   - Volume metrics (ratio, VWAP deviation)                  │     │
│  │   - Order book features (spread, imbalance, depth)          │     │
│  │   - Technical indicators (RSI, MACD, Bollinger, ATR)       │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              ↓                                       │
│  BAYESIAN DENSE LAYERS                                               │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ Layer 1: BayesianLinear(input_dim, 256)                    │     │
│  │   - Weight Distribution: N(μ_w, σ_w²)                       │     │
│  │   - Bias Distribution: N(μ_b, σ_b²)                         │     │
│  │   - Local Reparameterization for efficiency                 │     │
│  │   - Activation: LeakyReLU                                   │     │
│  │   - Dropout: 0.2                                            │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ Layer 2: BayesianLinear(256, 128)                          │     │
│  │   - Same Bayesian treatment                                 │     │
│  │   - Activation: LeakyReLU                                   │     │
│  │   - Dropout: 0.2                                            │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ Layer 3: BayesianLinear(128, 64)                           │     │
│  │   - Same Bayesian treatment                                 │     │
│  │   - Activation: LeakyReLU                                   │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              ↓                                       │
│  OUTPUT HEADS                                                        │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ Direction Head: BayesianLinear(64, 3)                       │     │
│  │   - Output: P(up), P(neutral), P(down)                      │     │
│  │   - Softmax activation                                      │     │
│  │                                                             │     │
│  │ Return Head: BayesianLinear(64, 2)                          │     │
│  │   - Output: μ_return, σ_return (predicted return dist.)     │     │
│  │   - Heteroscedastic uncertainty                             │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  UNCERTAINTY QUANTIFICATION                                          │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ Monte Carlo Sampling (N=100 forward passes):                │     │
│  │   - Epistemic Uncertainty: Variance from weight sampling    │     │
│  │   - Aleatoric Uncertainty: Predicted variance (data noise)  │     │
│  │   - Total Uncertainty: Epistemic + Aleatoric                │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Types of Uncertainty

### Epistemic Uncertainty (Model Uncertainty)

**What the model doesn't know** - reducible with more data:

```
Sources:
- Limited training data
- Model misspecification
- Out-of-distribution inputs

In BNNs: Captured by variance in weight distributions
         High weight variance → High epistemic uncertainty

Trading implication:
- New market regime → High epistemic uncertainty → Reduce exposure
- More training data → Lower epistemic uncertainty
```

### Aleatoric Uncertainty (Data Uncertainty)

**Inherent randomness in data** - irreducible noise:

```
Sources:
- Market microstructure noise
- News events
- Random fluctuations

In BNNs: Captured by heteroscedastic output layer
         Predicts both mean AND variance of return

Trading implication:
- High volatility periods → High aleatoric uncertainty
- Cannot reduce with more data
- Accept or hedge against
```

### Separating the Two

```python
# Monte Carlo inference
predictions = []
for _ in range(100):
    pred_mean, pred_var = model.forward(x, sample=True)  # Sample weights
    predictions.append((pred_mean, pred_var))

# Epistemic uncertainty: variance of means
epistemic = np.var([p[0] for p in predictions])

# Aleatoric uncertainty: mean of variances
aleatoric = np.mean([p[1] for p in predictions])

# Total uncertainty
total = epistemic + aleatoric
```

## Bayesian Layer Implementation

### BayesianLinear Layer

```python
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()

        # Weight parameters (mean and log-variance)
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.ones(out_features, in_features) * -3)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.ones(out_features) * -3)

        # Prior
        self.prior_sigma = prior_sigma

        # Initialize
        nn.init.kaiming_normal_(self.weight_mu)

    def forward(self, x, sample=True):
        if sample:
            # Reparameterization trick
            weight_sigma = F.softplus(self.weight_rho)
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)

            bias_sigma = F.softplus(self.bias_rho)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            # Use mean weights (no sampling)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """KL divergence from posterior to prior"""
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        # KL for weights
        kl_weight = self._kl_normal(self.weight_mu, weight_sigma, 0, self.prior_sigma)

        # KL for bias
        kl_bias = self._kl_normal(self.bias_mu, bias_sigma, 0, self.prior_sigma)

        return kl_weight + kl_bias

    def _kl_normal(self, mu1, sigma1, mu2, sigma2):
        """KL divergence between two normal distributions"""
        return 0.5 * torch.sum(
            2 * torch.log(sigma2 / sigma1)
            + (sigma1**2 + (mu1 - mu2)**2) / sigma2**2
            - 1
        )
```

## Training with ELBO

### Loss Function

```python
def elbo_loss(model, x, y, n_samples=1, beta=1.0):
    """
    Evidence Lower Bound Loss

    Args:
        model: BNN model
        x: Input features
        y: Target labels
        n_samples: Number of MC samples
        beta: KL weighting factor (for KL annealing)

    Returns:
        loss: -ELBO (to minimize)
    """
    # Likelihood term (data fit)
    log_likelihood = 0
    for _ in range(n_samples):
        output = model(x, sample=True)
        log_likelihood += F.cross_entropy(output, y, reduction='sum')
    log_likelihood /= n_samples

    # KL divergence term (complexity penalty)
    kl_div = model.kl_divergence()

    # ELBO = E[log p(D|w)] - KL[q(w|θ) || p(w)]
    # We minimize negative ELBO
    loss = log_likelihood + beta * kl_div

    return loss
```

### KL Annealing

```python
# Gradually increase KL weight during training
# Prevents posterior collapse early in training

def get_beta(epoch, warmup_epochs=10):
    """KL annealing schedule"""
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0
```

## Uncertainty-Aware Trading Strategy

### Signal Generation with Confidence

```python
def generate_signals(model, features, threshold=0.6, uncertainty_scale=True):
    """
    Generate trading signals with uncertainty-based position sizing

    Args:
        model: Trained BNN
        features: Market features
        threshold: Minimum confidence for signal
        uncertainty_scale: Scale position by uncertainty

    Returns:
        signals: List of trading signals
    """
    # Monte Carlo inference
    n_samples = 100
    predictions = []

    for _ in range(n_samples):
        logits = model(features, sample=True)
        probs = F.softmax(logits, dim=-1)
        predictions.append(probs)

    predictions = torch.stack(predictions)

    # Mean prediction
    mean_probs = predictions.mean(dim=0)

    # Epistemic uncertainty (variance of predictions)
    epistemic_unc = predictions.var(dim=0).sum(dim=-1)

    signals = []
    for i in range(len(features)):
        prob_up = mean_probs[i, 0].item()
        prob_down = mean_probs[i, 2].item()
        uncertainty = epistemic_unc[i].item()

        # Convert uncertainty to confidence
        confidence = 1.0 / (1.0 + uncertainty)

        if prob_up > threshold:
            position_size = confidence if uncertainty_scale else 1.0
            signals.append(Signal(
                direction="LONG",
                probability=prob_up,
                uncertainty=uncertainty,
                position_size=position_size
            ))
        elif prob_down > threshold:
            position_size = confidence if uncertainty_scale else 1.0
            signals.append(Signal(
                direction="SHORT",
                probability=prob_down,
                uncertainty=uncertainty,
                position_size=position_size
            ))

    return signals
```

### Position Sizing by Uncertainty

```python
def kelly_criterion_bayesian(prob_win, uncertainty, win_size, loss_size):
    """
    Modified Kelly Criterion accounting for uncertainty

    Standard Kelly: f = (p*b - q) / b
    Bayesian Kelly: f = (p*b - q) / b * (1 - uncertainty_penalty)

    Args:
        prob_win: Estimated probability of winning
        uncertainty: Model uncertainty (0 to 1)
        win_size: Expected win amount
        loss_size: Expected loss amount

    Returns:
        fraction: Optimal position size (0 to 1)
    """
    # Standard Kelly
    b = win_size / loss_size  # Odds
    q = 1 - prob_win
    kelly = (prob_win * b - q) / b

    # Uncertainty penalty
    # Higher uncertainty → smaller position
    uncertainty_penalty = uncertainty ** 0.5  # Sqrt for gradual scaling
    adjusted_kelly = kelly * (1 - uncertainty_penalty)

    # Constrain to [0, 0.25] (never bet more than 25%)
    return max(0, min(0.25, adjusted_kelly))
```

## Hyperparameter Selection

### Prior Selection

```yaml
prior_options:
  # For regularization similar to L2
  standard_gaussian:
    sigma: 1.0
    use_case: "Default, general purpose"

  # For sparse networks
  spike_and_slab:
    spike_prob: 0.5
    slab_sigma: 1.0
    use_case: "Feature selection, interpretability"

  # For adaptive regularization
  hierarchical:
    alpha: 1.0
    beta: 1.0
    use_case: "Automatic relevance determination"
```

### Architecture Considerations

```yaml
architecture:
  # Fewer layers than standard NN (each layer has 2x parameters)
  num_layers: 3-4

  # Moderate width (uncertainty estimation scales with params)
  hidden_dims: [256, 128, 64]

  # Output configuration
  output_type: "heteroscedastic"  # Predict mean and variance

  # MC samples for inference
  mc_samples_train: 1-5
  mc_samples_inference: 50-200
```

### Training Configuration

```yaml
training:
  # Lower learning rate (more parameters to estimate)
  learning_rate: 0.0001

  # Longer training (convergence is slower)
  epochs: 200

  # KL annealing prevents posterior collapse
  kl_warmup_epochs: 20

  # Final KL weight
  kl_weight: 1.0

  # Batch size (larger helps with gradient noise)
  batch_size: 128
```

## Comparison with Alternatives

### BNN vs. MC Dropout

| Aspect | BNN | MC Dropout |
|--------|-----|------------|
| Parameters | 2x (mean + variance) | 1x |
| Training | More complex | Standard + dropout |
| Inference | Sample weights | Keep dropout active |
| Uncertainty quality | Better calibrated | Often overconfident |
| Computational cost | Higher | Lower |
| Implementation | Complex | Simple |

### BNN vs. Deep Ensembles

| Aspect | BNN | Deep Ensembles |
|--------|-----|----------------|
| Parameters | 2x per network | Nx per network (N models) |
| Diversity | Weight distributions | Different initializations |
| Training | Single model | N separate trainings |
| Uncertainty quality | Good | Very good |
| Memory | 2x | Nx |
| Interpretability | Posterior analysis | Vote counting |

### When to Use BNNs

```
Use BNNs when:
✓ Uncertainty quantification is critical
✓ Data is limited
✓ Novel/unusual inputs are expected
✓ Position sizing needs to be principled
✓ Understanding model confidence matters

Consider alternatives when:
✗ Computational resources are limited
✗ Large datasets available (less uncertainty benefit)
✗ Speed is critical (inference is slower)
✗ Implementation simplicity is priority
```

## Production Deployment

### Inference Pipeline

```
Real-time Trading Pipeline:
├── Data Collection
│   └── Bybit WebSocket → OHLCV + Order Book
├── Feature Engineering
│   └── Technical indicators + Market features
├── Bayesian Inference
│   └── MC Sampling (100 forward passes)
├── Uncertainty Computation
│   └── Epistemic + Aleatoric decomposition
├── Signal Generation
│   └── Direction + Confidence + Position size
├── Risk Management
│   └── Uncertainty-adjusted position sizing
└── Order Execution
    └── Size based on confidence
```

### Latency Considerations

```
Latency Budget:
├── Data collection: ~10ms
├── Feature computation: ~5ms
├── MC Inference (100 samples): ~50-100ms
│   └── Can parallelize on GPU
├── Uncertainty computation: ~5ms
├── Signal generation: ~2ms
└── Total: ~70-120ms

Optimization strategies:
- Use fewer MC samples in production (50 vs 200)
- Batch multiple assets
- Cache posterior approximations
- Use mean prediction for screening, full MC for final signals
```

## Key Metrics

### Model Quality

- **ELBO**: Training objective (higher is better)
- **Calibration Error**: Does predicted uncertainty match actual error?
- **Negative Log Likelihood**: Prediction quality with uncertainty

### Uncertainty Quality

- **Coverage**: % of true values within predicted intervals
- **Sharpness**: Tightness of prediction intervals
- **Proper Scoring Rules**: Brier score, CRPS

### Trading Performance

- **Sharpe Ratio**: Risk-adjusted returns
- **Uncertainty-Adjusted Sharpe**: Sharpe accounting for prediction confidence
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate by Confidence**: Win rate stratified by uncertainty level

## Directory Structure

```
327_bayesian_neural_network/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── README.specify.md            # Original specification
├── readme.simple.md             # Beginner-friendly explanation
├── readme.simple.ru.md          # Russian beginner version
├── python/                      # Python implementation
│   ├── requirements.txt
│   ├── bnn/
│   │   ├── __init__.py
│   │   ├── layers.py            # Bayesian layers
│   │   ├── model.py             # BNN model
│   │   ├── loss.py              # ELBO loss
│   │   └── inference.py         # MC inference
│   ├── data/
│   │   ├── __init__.py
│   │   └── bybit_fetcher.py     # CCXT data fetching
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py       # Feature engineering
│   ├── strategy/
│   │   ├── __init__.py
│   │   └── trading.py           # Trading strategy
│   └── examples/
│       ├── __init__.py
│       ├── train_bnn.py
│       ├── backtest.py
│       └── fetch_data.py
└── rust_bnn/                    # Rust implementation
    ├── Cargo.toml
    ├── README.md
    ├── src/
    │   ├── lib.rs
    │   ├── api/                 # Bybit API client
    │   ├── bnn/                 # BNN implementation
    │   ├── features/            # Feature engineering
    │   ├── strategy/            # Trading strategy
    │   └── backtest/            # Backtesting
    └── examples/
        ├── fetch_data.rs
        ├── train_bnn.rs
        └── backtest.rs
```

## References

1. **Weight Uncertainty in Neural Networks** (Blundell et al., 2015)
   - https://arxiv.org/abs/1505.05424
   - Original Bayes by Backprop paper

2. **Dropout as a Bayesian Approximation** (Gal & Ghahramani, 2016)
   - https://arxiv.org/abs/1506.02142
   - MC Dropout as approximate inference

3. **What Uncertainties Do We Need in Bayesian Deep Learning?** (Kendall & Gal, 2017)
   - https://arxiv.org/abs/1703.04977
   - Epistemic vs. aleatoric uncertainty

4. **Practical Variational Inference for Neural Networks** (Graves, 2011)
   - https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks
   - Early work on variational BNNs

5. **Deep Ensemble** (Lakshminarayanan et al., 2017)
   - https://arxiv.org/abs/1612.01474
   - Comparison approach for uncertainty

## Difficulty Level

**Advanced** - Requires understanding of:
- Bayesian statistics and probability theory
- Neural network architectures
- Variational inference
- Monte Carlo methods
- Financial risk management

## Disclaimer

This chapter is for **educational purposes only**. Cryptocurrency trading involves substantial risk of loss. The strategies and models described here have not been validated for live trading and should be thoroughly tested before any real-world application. Past performance does not guarantee future results. Always trade responsibly and never risk more than you can afford to lose.
