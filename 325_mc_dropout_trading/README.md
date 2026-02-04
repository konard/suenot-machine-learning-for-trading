# Chapter 325: MC Dropout Trading

## Overview

Monte Carlo Dropout (MC Dropout) is a powerful technique that transforms standard neural networks with dropout into Bayesian approximators, enabling uncertainty estimation in deep learning models. For trading applications, this uncertainty quantification is crucial for making informed decisions about position sizing, risk management, and trade selection.

## Why MC Dropout for Trading?

### The Problem with Point Predictions

Traditional neural networks output a single prediction (e.g., "price will go up 2%"). But in trading, we need to know:
- **How confident** is the model in this prediction?
- **What's the range** of possible outcomes?
- **Should we trust** this particular prediction?

### MC Dropout Solution

MC Dropout provides:
- **Predictive uncertainty** - Know when the model is confident vs. uncertain
- **Risk-adjusted decisions** - Size positions based on confidence
- **Epistemic uncertainty** - Detect out-of-distribution samples (unusual market conditions)

```
Traditional NN:
  Input → Model → Single Prediction: "Price +2%"

MC Dropout:
  Input → Model (dropout ON) → Forward Pass 1: +2.5%
                             → Forward Pass 2: +1.8%
                             → Forward Pass 3: +3.2%
                             → ...
                             → Forward Pass N: +1.5%

  Result: Mean = +2.2%, Std = 0.6% (confidence interval!)
```

## Theoretical Foundation

### Gal & Ghahramani (2016): Dropout as Bayesian Approximation

The seminal paper "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" proved that:

1. **Dropout at training time** can be viewed as approximate Bayesian inference
2. **Dropout at inference time** (MC Dropout) approximates the posterior predictive distribution
3. The **variance of predictions** across multiple forward passes represents model uncertainty

### Mathematical Formulation

For a neural network with weights `W` and dropout applied, each forward pass samples a different sub-network. Given input `x*`:

```
Predictive Mean:
  E[y*] ≈ (1/T) Σ f(x*, W_t)   where t = 1, ..., T

Predictive Variance:
  Var[y*] ≈ τ^(-1) + (1/T) Σ f(x*, W_t)^2 - E[y*]^2

where:
  T = number of forward passes
  W_t = weights with dropout mask t
  τ = model precision (related to weight decay)
```

### Types of Uncertainty

MC Dropout captures two types of uncertainty:

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNCERTAINTY TYPES                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EPISTEMIC (Model Uncertainty)                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ "What the model doesn't know"                            │   │
│  │ - Can be reduced with more training data                 │   │
│  │ - High in out-of-distribution regions                    │   │
│  │ - MC Dropout variance captures this                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ALEATORIC (Data Uncertainty)                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ "Inherent noise in the data"                             │   │
│  │ - Cannot be reduced (fundamental randomness)             │   │
│  │ - Market microstructure noise, news events               │   │
│  │ - Requires heteroscedastic modeling                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Technical Architecture

### Model Architecture with Dropout

```
┌─────────────────────────────────────────────────────────────────┐
│                    MC DROPOUT MODEL                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT LAYER                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Market Features:                                          │   │
│  │   - Price returns (multi-timeframe)                       │   │
│  │   - Volume indicators                                      │   │
│  │   - Technical indicators (RSI, MACD, Bollinger)           │   │
│  │   - Order book features                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  HIDDEN LAYERS (with Dropout)                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Layer 1: Linear(input, 256) → ReLU → Dropout(p=0.1)       │   │
│  │ Layer 2: Linear(256, 128) → ReLU → Dropout(p=0.1)         │   │
│  │ Layer 3: Linear(128, 64) → ReLU → Dropout(p=0.1)          │   │
│  │ Layer 4: Linear(64, 32) → ReLU → Dropout(p=0.2)           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  OUTPUT LAYER                                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Linear(32, output_dim)                                    │   │
│  │ Output: Return prediction or Direction probabilities       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  MC DROPOUT INFERENCE (Dropout stays ON)                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Forward Pass 1 ─┐                                         │   │
│  │ Forward Pass 2 ─┼──→ Aggregate → Mean, Variance           │   │
│  │ Forward Pass 3 ─┤                                         │   │
│  │ ...            ─┤                                         │   │
│  │ Forward Pass T ─┘                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Forward Passes Trade-off

The number of forward passes `T` involves a trade-off:

| Forward Passes (T) | Variance Estimate | Latency | Use Case |
|-------------------|-------------------|---------|----------|
| 10-20 | Rough estimate | ~10-20ms | Quick screening |
| 30-50 | Good estimate | ~30-50ms | Standard trading |
| 100+ | Accurate estimate | ~100ms+ | Critical decisions |

```python
# Error of variance estimate decreases as 1/sqrt(T)
# Standard error of mean: σ / sqrt(T)

T=10:  Standard error ~ 0.316σ
T=50:  Standard error ~ 0.141σ
T=100: Standard error ~ 0.100σ
```

## Implementation

### Basic MC Dropout Layer

```python
import torch
import torch.nn as nn

class MCDropoutModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)  # Dropout after each layer
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_with_uncertainty(self, x, n_samples=50):
        """
        MC Dropout inference: run multiple forward passes with dropout ON
        """
        self.train()  # Keep dropout active!

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)  # [n_samples, batch, output]

        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        std = predictions.std(dim=0)

        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'samples': predictions
        }
```

### Trading Strategy with Uncertainty

```python
class MCDropoutTradingStrategy:
    def __init__(self, model, confidence_threshold=1.5, uncertainty_threshold=0.02):
        self.model = model
        self.confidence_threshold = confidence_threshold  # z-score threshold
        self.uncertainty_threshold = uncertainty_threshold

    def generate_signal(self, features, n_samples=50):
        """Generate trading signal with uncertainty-based filtering"""

        result = self.model.predict_with_uncertainty(features, n_samples)

        mean_pred = result['mean'].item()
        std_pred = result['std'].item()

        # Calculate z-score (how many std away from zero)
        z_score = abs(mean_pred) / std_pred if std_pred > 0 else 0

        # Filter by uncertainty
        if std_pred > self.uncertainty_threshold:
            return Signal(
                direction='HOLD',
                confidence=0.0,
                reason='Uncertainty too high'
            )

        # Filter by confidence
        if z_score < self.confidence_threshold:
            return Signal(
                direction='HOLD',
                confidence=z_score,
                reason='Prediction not confident enough'
            )

        # Generate signal
        direction = 'LONG' if mean_pred > 0 else 'SHORT'

        return Signal(
            direction=direction,
            confidence=z_score,
            predicted_return=mean_pred,
            uncertainty=std_pred
        )
```

## Concrete Dropout

### The Problem with Fixed Dropout Rate

Standard dropout requires manually tuning the dropout probability `p`. Too high leads to underfitting, too low doesn't provide enough regularization.

### Concrete Dropout Solution (Gal et al., 2017)

Concrete Dropout learns the optimal dropout rate during training:

```python
class ConcreteDropout(nn.Module):
    """
    Concrete Dropout layer that learns the dropout probability
    """
    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        super().__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        # Learnable parameter for dropout probability
        self.p_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def p(self):
        """Get dropout probability from logit"""
        return torch.sigmoid(self.p_logit)

    def forward(self, x, layer):
        """
        Apply concrete dropout with relaxed Bernoulli sampling
        """
        p = self.p

        if self.training:
            # Concrete/Gumbel-Softmax relaxation
            eps = 1e-7
            temp = 0.1

            unif_noise = torch.rand_like(x)
            drop_prob = (
                torch.log(p + eps)
                - torch.log(1 - p + eps)
                + torch.log(unif_noise + eps)
                - torch.log(1 - unif_noise + eps)
            )
            drop_prob = torch.sigmoid(drop_prob / temp)

            mask = 1 - drop_prob
            x = x * mask / (1 - p)

        # Apply the layer
        out = layer(x)

        # Calculate regularization terms
        weight_reg = self.weight_regularizer * torch.sum(layer.weight ** 2)
        dropout_reg = self.dropout_regularizer * (
            p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)
        )

        return out, weight_reg + dropout_reg
```

### Benefits of Concrete Dropout

| Aspect | Fixed Dropout | Concrete Dropout |
|--------|--------------|------------------|
| Dropout rate | Manually tuned | Learned automatically |
| Uncertainty calibration | Often over/underconfident | Better calibrated |
| Different layers | Same rate everywhere | Layer-specific rates |
| Adaptation | Static | Adapts to data complexity |

## Comparison with Other Uncertainty Methods

### Methods Overview

| Method | Uncertainty Type | Computational Cost | Calibration |
|--------|-----------------|-------------------|-------------|
| **MC Dropout** | Epistemic | Low (T forward passes) | Good |
| **Deep Ensembles** | Both | High (train N models) | Excellent |
| **Variational Inference** | Epistemic | Medium | Good |
| **Bayesian NN** | Both | Very High | Excellent |
| **Evidential NN** | Both | Low (single pass) | Variable |
| **Quantile Regression** | Aleatoric | Low | Good for targets |

### MC Dropout vs Deep Ensembles

```
MC Dropout:
┌─────────────────────┐
│   Single Model      │ → Train once
│   with Dropout      │ → T forward passes
└─────────────────────┘
Cost: 1 × training + T × inference

Deep Ensembles:
┌─────┐ ┌─────┐ ┌─────┐
│ M_1 │ │ M_2 │ │ M_N │ → Train N times
└─────┘ └─────┘ └─────┘ → N forward passes
Cost: N × training + N × inference

Trade-off:
- Ensembles are more accurate but expensive to train
- MC Dropout is faster to train, comparable inference cost
- Ensembles capture functional diversity better
```

## Position Sizing with Uncertainty

### Kelly Criterion with Uncertainty

The Kelly criterion for optimal position sizing can be enhanced with uncertainty:

```python
def uncertainty_adjusted_kelly(predicted_return, uncertainty, risk_free_rate=0.0):
    """
    Calculate position size using uncertainty-adjusted Kelly criterion

    Args:
        predicted_return: Expected return from MC Dropout mean
        uncertainty: Standard deviation from MC Dropout
        risk_free_rate: Risk-free rate

    Returns:
        Optimal position fraction (0 to 1)
    """
    # Standard Kelly fraction
    excess_return = predicted_return - risk_free_rate

    if uncertainty <= 0:
        return 0.0

    # Kelly fraction: f = (μ - r) / σ²
    kelly_fraction = excess_return / (uncertainty ** 2)

    # Apply uncertainty penalty
    # Higher uncertainty → smaller position
    confidence = 1.0 / (1.0 + uncertainty)
    adjusted_fraction = kelly_fraction * confidence

    # Cap at full Kelly and ensure non-negative
    return max(0.0, min(1.0, adjusted_fraction))


def generate_position_sizes(predictions, uncertainties, max_leverage=1.0):
    """
    Generate position sizes for portfolio

    Returns position sizes that sum to max_leverage
    """
    positions = {}

    for symbol, (pred, unc) in zip(predictions.keys(),
                                    zip(predictions.values(), uncertainties.values())):
        # Skip high uncertainty predictions
        if unc > 0.05:  # 5% uncertainty threshold
            positions[symbol] = 0.0
            continue

        kelly = uncertainty_adjusted_kelly(pred, unc)
        positions[symbol] = kelly

    # Normalize to max leverage
    total = sum(abs(p) for p in positions.values())
    if total > max_leverage:
        scale = max_leverage / total
        positions = {k: v * scale for k, v in positions.items()}

    return positions
```

### Risk Management Rules

```python
class UncertaintyBasedRiskManager:
    def __init__(self,
                 max_position_size=0.1,      # Max 10% per position
                 uncertainty_threshold=0.03,  # Skip if std > 3%
                 confidence_threshold=2.0,    # Need 2-sigma confidence
                 max_portfolio_uncertainty=0.05):
        self.max_position_size = max_position_size
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold
        self.max_portfolio_uncertainty = max_portfolio_uncertainty

    def filter_signals(self, signals):
        """Filter signals based on uncertainty"""
        filtered = []

        for signal in signals:
            # Check individual uncertainty
            if signal.uncertainty > self.uncertainty_threshold:
                continue

            # Check confidence (z-score)
            z_score = abs(signal.predicted_return) / signal.uncertainty
            if z_score < self.confidence_threshold:
                continue

            # Adjust position size based on uncertainty
            signal.position_size = min(
                self.max_position_size,
                signal.position_size * (1 - signal.uncertainty / self.uncertainty_threshold)
            )

            filtered.append(signal)

        return filtered

    def check_portfolio_uncertainty(self, positions, uncertainties):
        """
        Calculate portfolio-level uncertainty using error propagation
        """
        weights = np.array(list(positions.values()))
        stds = np.array(list(uncertainties.values()))

        # Simplified: assume uncorrelated
        portfolio_variance = np.sum((weights * stds) ** 2)
        portfolio_std = np.sqrt(portfolio_variance)

        return portfolio_std < self.max_portfolio_uncertainty
```

## Key Metrics

### Uncertainty Quality Metrics

```python
def evaluate_uncertainty_quality(predictions, uncertainties, actual_returns):
    """
    Evaluate how well the uncertainty estimates are calibrated
    """
    metrics = {}

    # 1. Coverage: How often does the true value fall in confidence interval?
    for confidence_level in [0.5, 0.9, 0.95]:
        z = stats.norm.ppf((1 + confidence_level) / 2)
        lower = predictions - z * uncertainties
        upper = predictions + z * uncertainties

        coverage = np.mean((actual_returns >= lower) & (actual_returns <= upper))
        metrics[f'coverage_{int(confidence_level*100)}'] = coverage

    # 2. Negative Log-Likelihood (NLL)
    nll = 0.5 * np.mean(
        np.log(2 * np.pi * uncertainties**2) +
        (actual_returns - predictions)**2 / (uncertainties**2)
    )
    metrics['nll'] = nll

    # 3. Calibration Error
    # Expected calibration error for uncertainty estimates
    metrics['calibration_error'] = abs(metrics['coverage_95'] - 0.95)

    # 4. Sharpness (average uncertainty - lower is better if calibrated)
    metrics['sharpness'] = np.mean(uncertainties)

    return metrics
```

### Trading Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Sharpe Ratio | Risk-adjusted returns | > 2.0 |
| Sortino Ratio | Downside risk-adjusted | > 2.5 |
| Maximum Drawdown | Largest peak-to-trough | < 15% |
| Win Rate | % of profitable trades | > 55% |
| Profit Factor | Gross profit / Gross loss | > 1.5 |
| Uncertainty Coverage | Actual coverage at 95% CI | ~95% |
| Uncertainty Correlation | Correlation with error | > 0.3 |

## Production Considerations

### Inference Pipeline

```
MC Dropout Inference Pipeline:
├── Data Collection (Bybit WebSocket)
│   └── Real-time OHLCV + order book updates
├── Feature Computation
│   └── Technical indicators, returns, volume
├── MC Dropout Inference
│   ├── Forward Pass 1 with dropout
│   ├── Forward Pass 2 with dropout
│   ├── ...
│   └── Forward Pass T with dropout
├── Uncertainty Aggregation
│   ├── Calculate mean prediction
│   └── Calculate standard deviation
├── Signal Generation
│   ├── Apply confidence threshold
│   └── Apply uncertainty threshold
└── Order Execution
    └── Position sizing based on uncertainty

Latency Budget:
├── Data collection: ~10ms (WebSocket)
├── Feature computation: ~5ms
├── MC Dropout (50 passes): ~25ms (GPU) / ~100ms (CPU)
├── Signal generation: ~1ms
└── Total: ~40ms (GPU) / ~120ms (CPU)
```

### Optimization Strategies

```python
# Batch multiple forward passes for GPU efficiency
def efficient_mc_dropout(model, x, n_samples=50, batch_size=10):
    """
    Run MC Dropout efficiently by batching forward passes
    """
    model.train()

    all_predictions = []

    # Replicate input for parallel forward passes
    x_batched = x.repeat(batch_size, 1)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            current_batch = min(batch_size, n_samples - i)
            x_batch = x_batched[:current_batch]

            predictions = model(x_batch)
            all_predictions.append(predictions)

    all_predictions = torch.cat(all_predictions, dim=0)

    return {
        'mean': all_predictions.mean(dim=0),
        'std': all_predictions.std(dim=0)
    }
```

## Directory Structure

```
325_mc_dropout_trading/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Beginner-friendly explanation
├── readme.simple.ru.md          # Russian beginner version
├── python/                      # Python implementation
│   ├── requirements.txt
│   ├── mc_dropout_model.py      # MC Dropout model
│   ├── concrete_dropout.py      # Concrete Dropout implementation
│   ├── trading_strategy.py      # Trading strategy with uncertainty
│   ├── bybit_client.py          # Bybit API via CCXT
│   ├── backtest.py              # Backtesting engine
│   └── main.py                  # Main entry point
└── rust_mc_dropout/             # Rust implementation
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               # Library entry point
    │   ├── api/                 # Bybit API client
    │   ├── model/               # MC Dropout implementation
    │   ├── features/            # Feature engineering
    │   ├── strategy/            # Trading strategy
    │   └── backtest/            # Backtesting engine
    └── examples/
        ├── fetch_market_data.rs
        ├── mc_dropout_inference.rs
        ├── trading_signals.rs
        └── backtest.rs
```

## References

1. **Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning** (Gal & Ghahramani, 2016)
   - https://arxiv.org/abs/1506.02142
   - Foundational paper proving dropout approximates Bayesian inference

2. **Concrete Dropout** (Gal et al., 2017)
   - https://arxiv.org/abs/1705.07832
   - Learning dropout probability during training

3. **What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?** (Kendall & Gal, 2017)
   - https://arxiv.org/abs/1703.04977
   - Separating epistemic and aleatoric uncertainty

4. **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles** (Lakshminarayanan et al., 2017)
   - https://arxiv.org/abs/1612.01474
   - Comparison method: deep ensembles

5. **Uncertainty Quantification for Deep Learning in Finance** (Various)
   - Applications of uncertainty estimation to trading

## Difficulty Level

**Intermediate to Advanced** - Requires understanding of:
- Neural networks and dropout
- Bayesian inference basics
- Uncertainty quantification
- Trading strategy design
- Risk management principles

## Disclaimer

This chapter is for **educational purposes only**. Cryptocurrency trading involves substantial risk. The strategies described here have not been validated in live trading and should be thoroughly tested before any real-world application. Past performance does not guarantee future results.
