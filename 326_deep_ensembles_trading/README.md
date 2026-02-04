# Chapter 326: Deep Ensembles Trading

## Overview

Deep Ensembles is a simple yet powerful approach for uncertainty quantification in deep learning. By training multiple neural networks with different random initializations, Deep Ensembles capture model uncertainty (epistemic uncertainty) and provide robust predictions that are crucial for risk-aware trading decisions.

## Why Deep Ensembles for Trading?

### The Problem with Single Models

A single neural network provides point predictions without any measure of confidence:

```
Single Model: Price → Neural Network → Prediction: +2.5%
                                       (But how confident is this?)
```

In trading, this is dangerous because:
- **Overconfident predictions** lead to oversized positions
- **No risk assessment** during market regime changes
- **Model failures** are silent and catastrophic

### Deep Ensembles Solution

Deep Ensembles train M independent networks and aggregate their predictions:

```
Deep Ensemble: Price → [NN₁, NN₂, NN₃, ..., NNₘ] → Mean: +2.5%, Std: 0.8%
                                                    (Now we know uncertainty!)
```

**Key insight**: Disagreement between ensemble members indicates uncertainty.

## Core Concepts

### 1. Ensemble Diversity Through Random Initialization

Neural networks are non-convex. Different random initializations find different local minima:

```
Loss Landscape:
    ↓ Init 1         ↓ Init 2         ↓ Init 3
     \               |                 /
      \              |                /
       ↘            ↓              ↙
        Minimum A   Minimum B    Minimum C

Each minimum gives a different (but valid) model!
```

**Why this works:**
- Neural networks with different initializations converge to different solutions
- These solutions make different errors on different inputs
- Averaging reduces individual errors (variance reduction)
- Disagreement reveals areas where models are uncertain

### 2. Uncertainty Decomposition

Deep Ensembles naturally decompose uncertainty into two types:

```
Total Uncertainty = Epistemic Uncertainty + Aleatoric Uncertainty

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  EPISTEMIC (Model Uncertainty)        ALEATORIC (Data Noise)    │
│  ────────────────────────────        ─────────────────────────  │
│  • Reducible with more data          • Irreducible noise        │
│  • High when models disagree         • Inherent randomness      │
│  • Indicates "I don't know"          • Indicates "Market chaos" │
│                                                                  │
│  Captured by: Variance of means      Captured by: Mean variance │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Mathematical formulation:**

For an ensemble of M models predicting Gaussian distributions:
- Each model m outputs: μₘ(x), σₘ²(x)

```
Total Variance = (1/M) Σ σₘ²(x)  +  (1/M) Σ (μₘ(x) - μ̄(x))²
                 └─────────────┘     └────────────────────────┘
                  Aleatoric            Epistemic
                  (mean variance)      (variance of means)
```

### 3. Proper Scoring Rules and NLL Loss

Deep Ensembles use **Negative Log-Likelihood (NLL)** loss for proper uncertainty estimation:

```python
# For Gaussian output
def nll_loss(mu, sigma, target):
    """
    NLL = 0.5 * log(2π) + log(σ) + (y - μ)²/(2σ²)
    """
    return 0.5 * np.log(2 * np.pi) + np.log(sigma) + 0.5 * ((target - mu) / sigma) ** 2
```

**Why NLL is important:**
- Proper scoring rule: Incentivizes honest uncertainty estimates
- Model learns both mean AND variance
- Overconfident predictions are penalized
- Underconfident predictions are penalized

### 4. Ensemble Disagreement

Ensemble disagreement is a key signal for trading decisions:

```
High Disagreement                    Low Disagreement
(Don't trade!)                       (Trade with confidence!)

Model 1: +5%                         Model 1: +2.4%
Model 2: -3%        ← CONFLICT       Model 2: +2.6%      ← AGREEMENT
Model 3: +1%                         Model 3: +2.5%
Model 4: -2%                         Model 4: +2.5%
Model 5: +4%                         Model 5: +2.4%

Mean: +1%, Std: 3.2%                Mean: +2.5%, Std: 0.08%
```

### 5. Hyperparameter Ensembles

Beyond random initialization, we can ensemble across hyperparameters:

```
Hyperparameter Diversity:
├── Architecture: [64-32, 128-64, 256-128-64]
├── Learning rate: [0.001, 0.0005, 0.0001]
├── Dropout: [0.1, 0.2, 0.3]
├── Activation: [ReLU, GELU, SiLU]
└── Batch size: [32, 64, 128]

Result: More diverse ensemble = Better uncertainty estimates
```

### 6. Parallelization

Deep Ensembles are embarrassingly parallel:

```
┌─────────────────────────────────────────────────────────────┐
│                    PARALLEL TRAINING                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GPU 0: Train Model 1   ────────────────→  Model 1 weights  │
│  GPU 1: Train Model 2   ────────────────→  Model 2 weights  │
│  GPU 2: Train Model 3   ────────────────→  Model 3 weights  │
│  GPU 3: Train Model 4   ────────────────→  Model 4 weights  │
│  GPU 4: Train Model 5   ────────────────→  Model 5 weights  │
│                                                              │
│  All trained simultaneously! No communication needed.        │
└─────────────────────────────────────────────────────────────┘
```

**Inference parallelization:**

```
Data batch → Parallel forward pass on all models → Aggregate predictions
             (Can use model parallelism or batch splitting)
```

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEEP ENSEMBLE MODEL                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Market Features [batch, features]                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  ENSEMBLE MEMBERS (M independent networks)                  │ │
│  │                                                             │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐     ┌──────────┐  │ │
│  │  │ Model 1  │ │ Model 2  │ │ Model 3  │ ... │ Model M  │  │ │
│  │  │          │ │          │ │          │     │          │  │ │
│  │  │ Linear   │ │ Linear   │ │ Linear   │     │ Linear   │  │ │
│  │  │ ReLU     │ │ ReLU     │ │ ReLU     │     │ ReLU     │  │ │
│  │  │ Dropout  │ │ Dropout  │ │ Dropout  │     │ Dropout  │  │ │
│  │  │ Linear   │ │ Linear   │ │ Linear   │     │ Linear   │  │ │
│  │  │ ReLU     │ │ ReLU     │ │ ReLU     │     │ ReLU     │  │ │
│  │  │ Linear   │ │ Linear   │ │ Linear   │     │ Linear   │  │ │
│  │  │ (μ, σ)   │ │ (μ, σ)   │ │ (μ, σ)   │     │ (μ, σ)   │  │ │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘     └────┬─────┘  │ │
│  │       │            │            │                │        │ │
│  └───────┼────────────┼────────────┼────────────────┼────────┘ │
│          │            │            │                │          │
│          ▼            ▼            ▼                ▼          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │               AGGREGATION LAYER                             │ │
│  │                                                             │ │
│  │   μ_ensemble = (1/M) Σ μₘ                                  │ │
│  │   σ²_epistemic = (1/M) Σ (μₘ - μ_ensemble)²               │ │
│  │   σ²_aleatoric = (1/M) Σ σₘ²                              │ │
│  │   σ²_total = σ²_epistemic + σ²_aleatoric                  │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  OUTPUT: (μ_ensemble, σ_total, σ_epistemic, σ_aleatoric)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Trading Strategy

### Signal Generation with Uncertainty

```python
def generate_signals(ensemble, features, threshold=0.6):
    """
    Generate trading signals with uncertainty-aware sizing.
    """
    # Get ensemble predictions
    mean_pred, total_std, epistemic_std, aleatoric_std = ensemble.predict(features)

    # Calculate signal strength
    signal_strength = mean_pred / total_std  # Sharpe-like ratio

    signals = []
    for i in range(len(mean_pred)):
        # Check prediction confidence
        if epistemic_std[i] > threshold * total_std[i]:
            # High model uncertainty - don't trade
            signals.append(Signal("HOLD", confidence=0))
            continue

        # Generate signal based on prediction
        if signal_strength[i] > 1.5:
            confidence = min(0.95, 1 - epistemic_std[i] / total_std[i])
            signals.append(Signal("LONG", confidence=confidence))
        elif signal_strength[i] < -1.5:
            confidence = min(0.95, 1 - epistemic_std[i] / total_std[i])
            signals.append(Signal("SHORT", confidence=confidence))
        else:
            signals.append(Signal("HOLD", confidence=0))

    return signals
```

### Position Sizing Based on Uncertainty

```python
def calculate_position_size(signal, uncertainty, max_position=0.1):
    """
    Position size inversely proportional to uncertainty.

    Kelly-inspired sizing with uncertainty adjustment.
    """
    if signal.type == "HOLD":
        return 0

    # Base Kelly fraction
    kelly_fraction = signal.confidence / (1 + uncertainty["total_std"])

    # Reduce position if epistemic uncertainty is high
    epistemic_penalty = 1 - min(1, uncertainty["epistemic_std"] / uncertainty["total_std"])

    # Final position
    position = kelly_fraction * epistemic_penalty * max_position

    return position if signal.type == "LONG" else -position
```

### Regime Detection Using Uncertainty

```python
def detect_regime(ensemble_predictions, window=20):
    """
    Detect market regime changes using uncertainty dynamics.
    """
    recent_epistemic = epistemic_std[-window:]
    recent_aleatoric = aleatoric_std[-window:]

    epistemic_trend = np.polyfit(range(window), recent_epistemic, 1)[0]
    aleatoric_trend = np.polyfit(range(window), recent_aleatoric, 1)[0]

    if epistemic_trend > 0.1:
        return "REGIME_CHANGE"  # Model becoming uncertain
    elif aleatoric_trend > 0.1:
        return "VOLATILITY_SPIKE"  # Market becoming chaotic
    elif np.mean(recent_epistemic) < 0.1 and np.mean(recent_aleatoric) < 0.1:
        return "STABLE"
    else:
        return "NORMAL"
```

## Implementation Details

### Training Configuration

```yaml
ensemble:
  num_models: 5
  architecture:
    hidden_dims: [256, 128, 64]
    activation: "relu"
    dropout: 0.2
    output_type: "gaussian"  # Output (μ, σ)

training:
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  max_epochs: 100
  early_stopping_patience: 10
  loss: "nll"  # Negative log-likelihood

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  sequence_length: 60
  prediction_horizon: 5

features:
  - returns_1m
  - returns_5m
  - returns_15m
  - returns_1h
  - volume_ratio
  - volatility
  - rsi_14
  - macd_signal
  - spread_bps
```

### Feature Engineering

```python
features = {
    # Price features
    'returns_1m': log_return(close, 1),
    'returns_5m': log_return(close, 5),
    'returns_15m': log_return(close, 15),
    'returns_1h': log_return(close, 60),
    'returns_4h': log_return(close, 240),

    # Volatility features
    'volatility_1h': rolling_std(returns, 60),
    'volatility_24h': rolling_std(returns, 1440),
    'volatility_ratio': volatility_1h / volatility_24h,

    # Volume features
    'volume_ratio': volume / volume_ma_20,
    'vwap_deviation': (close - vwap) / vwap,

    # Technical indicators
    'rsi_14': rsi(close, 14),
    'macd_signal': macd(close) - macd_signal(close),
    'bb_position': (close - bb_lower) / (bb_upper - bb_lower),

    # Order book features
    'spread_bps': (ask - bid) / mid * 10000,
    'depth_imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth),
}
```

## Key Metrics

### Model Performance

| Metric | Description | Target |
|--------|-------------|--------|
| NLL | Negative log-likelihood | Lower is better |
| Calibration Error | How well uncertainty matches actual errors | < 0.1 |
| Sharpness | Average predicted uncertainty | Balance with calibration |
| CRPS | Continuous Ranked Probability Score | Lower is better |

### Trading Performance

| Metric | Description | Target |
|--------|-------------|--------|
| Sharpe Ratio | Risk-adjusted returns | > 2.0 |
| Sortino Ratio | Downside risk-adjusted | > 2.5 |
| Max Drawdown | Largest peak-to-trough | < 10% |
| Win Rate | % profitable trades | > 55% |
| Profit Factor | Gross profit / Gross loss | > 1.5 |
| Uncertainty-Adjusted Return | Return / Mean uncertainty | Higher is better |

### Calibration Metrics

```python
def calibration_error(predictions, actuals, num_bins=10):
    """
    Expected Calibration Error (ECE)

    For well-calibrated predictions, X% of outcomes should fall
    within X% confidence intervals.
    """
    confidences = 1 - predictions['std'] / predictions['std'].max()
    accuracies = np.abs(predictions['mean'] - actuals) < predictions['std']

    ece = 0
    for bin_lower in np.linspace(0, 1, num_bins):
        bin_upper = bin_lower + 1/num_bins
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * in_bin.mean()

    return ece
```

## Advantages of Deep Ensembles

| Aspect | Single Model | Deep Ensemble |
|--------|--------------|---------------|
| Uncertainty quantification | None | Built-in |
| Robustness | Prone to overfitting | More robust |
| Calibration | Often overconfident | Better calibrated |
| Performance | Single optimum | Multiple optima aggregated |
| Interpretability | Black box | Disagreement = uncertainty |
| Scalability | N/A | Embarrassingly parallel |

## Comparison with Other Methods

### vs. Bayesian Neural Networks (BNN)

```
BNN:
+ Principled uncertainty from posterior
- Difficult to train
- Expensive inference
- Requires approximations (VI, MCMC)

Deep Ensembles:
+ Simple to implement
+ Parallel training
+ Often better calibration
- Requires M forward passes
- No posterior interpretation
```

### vs. MC Dropout

```
MC Dropout:
+ Single model, multiple passes
+ Memory efficient
- Underestimates uncertainty
- Depends on dropout rate

Deep Ensembles:
+ Better uncertainty estimates
+ More diverse predictions
- More memory (M models)
- More training time
```

### vs. Single Model with Softmax Calibration

```
Softmax Calibration:
+ Single model
+ Post-hoc calibration
- Only classification
- Limited uncertainty types

Deep Ensembles:
+ Works for regression
+ Epistemic/Aleatoric split
- Higher computational cost
```

## Production Considerations

```
Inference Pipeline:
├── Data Collection (Bybit WebSocket)
│   └── Real-time OHLCV + order book updates
├── Feature Computation
│   └── Rolling window calculations
├── Ensemble Inference
│   ├── Parallel forward pass (M models)
│   ├── Aggregation (mean, std)
│   └── Uncertainty decomposition
├── Signal Generation
│   ├── Confidence filtering
│   └── Position sizing
└── Order Execution
    └── Risk management integration

Latency Budget:
├── Data collection: ~10ms (WebSocket)
├── Feature computation: ~5ms
├── Ensemble inference: ~20ms (parallel, GPU)
├── Signal generation: ~2ms
└── Total: ~40ms (excluding execution)
```

## Directory Structure

```
326_deep_ensembles_trading/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Beginner-friendly explanation
├── readme.simple.ru.md          # Russian beginner version
├── python/                      # Python implementation
│   ├── requirements.txt         # Python dependencies
│   ├── deep_ensemble.py         # Core ensemble model
│   ├── data_fetcher.py          # Bybit data fetching & feature engineering
│   ├── strategy.py              # Trading strategy with uncertainty
│   ├── backtest.py              # Backtesting framework
│   └── example.py               # Complete example
└── rust_deep_ensembles/         # Rust implementation
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               # Library entry point
    │   ├── api/                 # Bybit API client
    │   ├── ensemble/            # Deep Ensemble implementation
    │   ├── features/            # Feature engineering
    │   ├── strategy/            # Trading strategy
    │   └── backtest/            # Backtesting engine
    └── examples/
        ├── fetch_data.rs
        ├── train_ensemble.rs
        ├── backtest.rs
        └── live_signals.rs
```

## References

1. **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles** (Lakshminarayanan et al., 2017)
   - https://arxiv.org/abs/1612.01474

2. **Deep Ensembles: A Loss Landscape Perspective** (Fort et al., 2019)
   - https://arxiv.org/abs/1912.02757

3. **Uncertainty Quantification in Deep Learning** (Abdar et al., 2021)
   - https://arxiv.org/abs/2011.06225

4. **Can You Trust Your Model's Uncertainty?** (Ovadia et al., 2019)
   - https://arxiv.org/abs/1906.02530

5. **Hyperparameter Ensembles** (Wenzel et al., 2020)
   - https://arxiv.org/abs/2006.13570

## Difficulty Level

**Intermediate** - Requires understanding of:
- Neural network training
- Probability distributions (Gaussian)
- Uncertainty quantification basics
- Trading fundamentals

## Disclaimer

This chapter is for **educational purposes only**. Cryptocurrency trading involves substantial risk. The strategies described here have not been validated in live trading and should be thoroughly tested before any real-world application. Past performance does not guarantee future results.
