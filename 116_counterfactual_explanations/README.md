# Chapter 116: Counterfactual Explanations for Trading

## Overview

Counterfactual explanations answer the question: "What would need to change for a different prediction?" In trading applications, this technique provides actionable insights into model decisions by identifying minimal changes to market conditions that would flip a prediction from "sell" to "buy" or vice versa. This explainability approach is crucial for understanding black-box models and building trust in algorithmic trading systems.

<p align="center">
<img src="https://i.imgur.com/8vZH3mL.png" width="70%">
</p>

## Table of Contents

1. [Introduction](#introduction)
   * [What are Counterfactual Explanations?](#what-are-counterfactual-explanations)
   * [Why Use Them in Trading?](#why-use-them-in-trading)
2. [Mathematical Foundation](#mathematical-foundation)
   * [Formal Definition](#formal-definition)
   * [Optimization Objective](#optimization-objective)
   * [Constraints and Regularization](#constraints-and-regularization)
3. [Architecture](#architecture)
   * [Counterfactual Generator](#counterfactual-generator)
   * [Validity and Proximity](#validity-and-proximity)
   * [Full System Architecture](#full-system-architecture)
4. [Implementation](#implementation)
   * [Rust Implementation](#rust-implementation)
   * [Python Reference](#python-reference)
5. [Trading Application](#trading-application)
   * [Market Data Processing](#market-data-processing)
   * [Feature Engineering](#feature-engineering)
   * [Actionable Insights](#actionable-insights)
6. [Backtesting](#backtesting)
7. [Resources](#resources)

## Introduction

### What are Counterfactual Explanations?

A counterfactual explanation describes the smallest change to input features that would result in a different model prediction. Unlike other explainability methods that tell you *why* a model made a decision, counterfactuals tell you *what would need to change* for a different outcome.

**Example in Trading:**

```
Original Input:
- RSI: 75 (overbought)
- MACD: -0.5 (bearish)
- Volume: 1.2x average
- Model Prediction: SELL (80% confidence)

Counterfactual Explanation:
"If RSI were 45 instead of 75, the model would predict BUY"

OR

"If MACD were +0.3 instead of -0.5, the model would predict HOLD"
```

This provides actionable insight — traders can understand what market conditions would need to change for a different signal.

### Why Use Them in Trading?

1. **Risk Management:** Understand how close the market is to a signal flip
2. **Transparency:** Explain model decisions to stakeholders and regulators
3. **Strategy Refinement:** Identify which features most influence predictions
4. **Confidence Assessment:** Measure stability of predictions via counterfactual distance
5. **Debugging:** Find edge cases where models behave unexpectedly

```
Traditional Explainability:        Counterfactual Explainability:
"RSI contributed 40% to SELL"     "If RSI drops by 30 points → BUY"
"Volume contributed 20%"          "OR if volume drops 50% → HOLD"

↓                                 ↓
Understanding WHY                 Understanding WHAT-IF
```

## Mathematical Foundation

### Formal Definition

Given a classifier `f: X → Y` and an input instance `x` with prediction `f(x) = y`, a counterfactual `x'` satisfies:

```
f(x') = y'  where y' ≠ y
```

The goal is to find `x'` that:
1. **Validity:** Results in the desired different prediction
2. **Proximity:** Is minimally different from the original `x`
3. **Plausibility:** Represents a realistic data point

### Optimization Objective

The counterfactual generation problem is typically formulated as:

```
x' = argmin L(x, x')
     subject to f(x') = y_target

where L(x, x') = λ₁ · d(x, x') + λ₂ · loss(f(x'), y_target) + λ₃ · plausibility(x')
```

**Components:**

- `d(x, x')`: Distance metric (L1, L2, or domain-specific)
- `loss(f(x'), y_target)`: Classification loss to ensure valid counterfactual
- `plausibility(x')`: Ensures counterfactual is realistic

```python
# Conceptual illustration
def counterfactual_loss(x, x_cf, model, target_class, lambda1=1.0, lambda2=1.0, lambda3=0.1):
    """
    Compute counterfactual optimization loss

    Args:
        x: Original input
        x_cf: Counterfactual candidate
        model: Classifier model
        target_class: Desired prediction class
        lambda1: Weight for proximity term
        lambda2: Weight for validity term
        lambda3: Weight for plausibility term

    Returns:
        Total loss value
    """
    # Proximity: How different is the counterfactual?
    proximity_loss = torch.norm(x - x_cf, p=1)  # L1 distance

    # Validity: Does it achieve the target class?
    logits = model(x_cf)
    validity_loss = F.cross_entropy(logits, target_class)

    # Plausibility: Is it realistic? (e.g., within data distribution)
    plausibility_loss = mahalanobis_distance(x_cf, data_mean, data_cov)

    return lambda1 * proximity_loss + lambda2 * validity_loss + lambda3 * plausibility_loss
```

### Constraints and Regularization

**Actionability Constraints:**

In trading, some features cannot be changed:
- Historical prices (immutable)
- Past volume (already happened)
- External events (news, regulations)

We apply masks to ensure counterfactuals only modify actionable features:

```python
# Only allow changes to forward-looking features
actionable_mask = torch.tensor([
    0,  # past_price (immutable)
    0,  # past_volume (immutable)
    1,  # rsi (can change with price movement)
    1,  # macd (can change)
    1,  # bollinger_position (can change)
    0,  # days_since_event (immutable)
])

x_cf = x + actionable_mask * delta  # Only modify actionable features
```

**Sparsity Regularization:**

To generate interpretable explanations, we prefer counterfactuals that change few features:

```
L_sparse = ||x - x'||_0  ≈  Σᵢ (1 - exp(-|xᵢ - x'ᵢ|/τ))
```

This encourages explanations like "only RSI needs to change" rather than "RSI, MACD, volume, and Bollinger width all need to change slightly."

## Architecture

### Counterfactual Generator

```
┌─────────────────────────────────────────────────────────────┐
│              Counterfactual Generator Network                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Input x ∈ ℝ^d                                            │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ Encoder │ → Latent representation z                    │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────────────────┐                                  │
│    │ Target Class        │                                  │
│    │ Conditioning        │ → z' = z ⊕ target_embedding      │
│    └────┬────────────────┘                                  │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ Decoder │ → x_cf candidate                             │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────────────────┐                                  │
│    │ Projection          │                                  │
│    │ (actionability +    │ → Valid counterfactual x'        │
│    │  plausibility)      │                                  │
│    └────┬────────────────┘                                  │
│         │                                                   │
│    Output x' ∈ ℝ^d                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Validity and Proximity

**Validity Check:**

A counterfactual is valid if the model predicts the target class:

```python
def is_valid(model, x_cf, target_class, threshold=0.5):
    """Check if counterfactual achieves target prediction"""
    with torch.no_grad():
        probs = F.softmax(model(x_cf), dim=-1)
        return probs[target_class] > threshold
```

**Proximity Metrics:**

| Metric | Formula | Use Case |
|--------|---------|----------|
| L1 (Manhattan) | Σ\|xᵢ - x'ᵢ\| | Sparse changes |
| L2 (Euclidean) | √Σ(xᵢ - x'ᵢ)² | Smooth changes |
| L0 (Count) | Σ𝟙[xᵢ ≠ x'ᵢ] | Minimal features |
| Mahalanobis | √((x-x')ᵀΣ⁻¹(x-x')) | Distribution-aware |

### Full System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│        Counterfactual Explanation System for Trading            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐                                              │
│  │ Market Data  │──┐                                           │
│  │ (OHLCV)      │  │                                           │
│  └──────────────┘  │    ┌─────────────────┐                    │
│                    ├───→│ Feature         │                    │
│  ┌──────────────┐  │    │ Engineering     │                    │
│  │ Technical    │──┤    └────────┬────────┘                    │
│  │ Indicators   │  │             │                             │
│  └──────────────┘  │    ┌────────┴────────┐                    │
│                    │    │ Trading Model   │                    │
│  ┌──────────────┐  │    │ (Black Box)     │                    │
│  │ Sentiment    │──┘    └────────┬────────┘                    │
│  └──────────────┘               │                              │
│                         ┌───────┴───────┐                      │
│                         │   Prediction  │                      │
│                         │   (BUY/SELL)  │                      │
│                         └───────┬───────┘                      │
│                                 │                              │
│                         ┌───────┴───────────────┐              │
│                         │ Counterfactual        │              │
│                         │ Generator             │              │
│                         └───────┬───────────────┘              │
│                                 │                              │
│                         ┌───────┴───────────────┐              │
│                         │ Explanations:         │              │
│                         │ "If RSI → 45, then    │              │
│                         │  prediction = BUY"    │              │
│                         └───────────────────────┘              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Implementation

### Rust Implementation

The [rust_counterfactual](rust_counterfactual/) directory contains a modular Rust implementation:

```
rust_counterfactual/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                  # Main library module
│   ├── api/
│   │   ├── mod.rs              # API module
│   │   └── bybit.rs            # Bybit API client
│   ├── data/
│   │   ├── mod.rs              # Data module
│   │   ├── loader.rs           # Data loading
│   │   └── features.rs         # Feature engineering
│   ├── model/
│   │   ├── mod.rs              # Model module
│   │   ├── classifier.rs       # Trading classifier
│   │   └── config.rs           # Model configuration
│   ├── counterfactual/
│   │   ├── mod.rs              # Counterfactual module
│   │   ├── generator.rs        # CF generator
│   │   ├── optimizer.rs        # Optimization algorithms
│   │   ├── constraints.rs      # Actionability constraints
│   │   └── metrics.rs          # Proximity metrics
│   └── strategy/
│       ├── mod.rs              # Strategy module
│       ├── signals.rs          # Trading signals
│       └── backtest.rs         # Backtesting framework
└── examples/
    ├── fetch_data.rs           # Fetch Bybit data
    ├── train_classifier.rs     # Train trading model
    ├── generate_cf.rs          # Generate counterfactuals
    └── backtest.rs             # Strategy backtest
```

### Quick Start with Rust

```bash
# Navigate to the Rust project
cd 116_counterfactual_explanations/rust_counterfactual

# Fetch cryptocurrency data from Bybit
cargo run --example fetch_data

# Train trading classifier
cargo run --example train_classifier

# Generate counterfactual explanations
cargo run --example generate_cf

# Run a full backtest
cargo run --example backtest
```

### Python Reference

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CounterfactualGenerator(nn.Module):
    """
    Neural network-based counterfactual generator
    """
    def __init__(self, input_dim, hidden_dim=128, num_classes=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, hidden_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, target_class):
        # Encode input
        z = self.encoder(x)

        # Get target class embedding
        class_emb = self.class_embedding(target_class)

        # Concatenate and decode
        z_combined = torch.cat([z, class_emb], dim=-1)
        delta = self.decoder(z_combined)

        # Generate counterfactual as perturbation
        x_cf = x + delta

        return x_cf


class CounterfactualOptimizer:
    """
    Gradient-based counterfactual optimizer
    """
    def __init__(self, model, lambda_proximity=1.0, lambda_validity=1.0,
                 lambda_sparsity=0.1, actionable_mask=None):
        self.model = model
        self.lambda_proximity = lambda_proximity
        self.lambda_validity = lambda_validity
        self.lambda_sparsity = lambda_sparsity
        self.actionable_mask = actionable_mask

    def generate(self, x, target_class, num_steps=100, lr=0.01):
        """
        Generate counterfactual via gradient descent

        Args:
            x: Original input tensor (batch, features)
            target_class: Desired prediction class
            num_steps: Optimization steps
            lr: Learning rate

        Returns:
            x_cf: Counterfactual explanation
            info: Dictionary with optimization info
        """
        x = x.clone().detach()
        x_cf = x.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x_cf], lr=lr)

        best_cf = None
        best_loss = float('inf')

        for step in range(num_steps):
            optimizer.zero_grad()

            # Compute losses
            loss, loss_dict = self._compute_loss(x, x_cf, target_class)

            # Backpropagate
            loss.backward()

            # Apply actionability mask to gradients
            if self.actionable_mask is not None:
                x_cf.grad.data *= self.actionable_mask

            optimizer.step()

            # Track best valid counterfactual
            if self._is_valid(x_cf, target_class) and loss.item() < best_loss:
                best_loss = loss.item()
                best_cf = x_cf.clone().detach()

        return best_cf if best_cf is not None else x_cf.detach(), {
            'final_loss': loss.item(),
            'is_valid': self._is_valid(x_cf, target_class),
            'num_features_changed': self._count_changes(x, x_cf)
        }

    def _compute_loss(self, x, x_cf, target_class):
        """Compute combined loss for counterfactual optimization"""
        # Proximity loss (L1)
        proximity = torch.norm(x - x_cf, p=1)

        # Validity loss (cross-entropy to target)
        logits = self.model(x_cf)
        validity = F.cross_entropy(logits, target_class)

        # Sparsity loss (approximate L0)
        sparsity = torch.sum(1 - torch.exp(-torch.abs(x - x_cf) / 0.1))

        total_loss = (
            self.lambda_proximity * proximity +
            self.lambda_validity * validity +
            self.lambda_sparsity * sparsity
        )

        return total_loss, {
            'proximity': proximity.item(),
            'validity': validity.item(),
            'sparsity': sparsity.item()
        }

    def _is_valid(self, x_cf, target_class, threshold=0.5):
        """Check if counterfactual achieves target class"""
        with torch.no_grad():
            probs = F.softmax(self.model(x_cf), dim=-1)
            return probs[0, target_class].item() > threshold

    def _count_changes(self, x, x_cf, threshold=0.01):
        """Count number of features that changed"""
        return (torch.abs(x - x_cf) > threshold).sum().item()


class TradingClassifier(nn.Module):
    """
    Simple trading classifier (the model to explain)
    """
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)
```

## Trading Application

### Market Data Processing

For cryptocurrency trading with Bybit:

```python
CRYPTO_UNIVERSE = {
    'major': ['BTCUSDT', 'ETHUSDT'],
    'large_cap': ['SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT'],
    'mid_cap': ['AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT'],
}

FEATURES = {
    'price': ['close', 'returns', 'log_returns'],
    'technical': ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower'],
    'volume': ['volume', 'volume_sma', 'volume_ratio'],
    'derived': ['volatility', 'momentum', 'trend_strength']
}
```

### Feature Engineering

```python
def prepare_features(df, lookback=20):
    """
    Prepare features for trading model

    Args:
        df: OHLCV DataFrame
        lookback: Period for technical indicators

    Returns:
        X: Feature matrix
        feature_names: List of feature names
    """
    features = {}

    # Price features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close']).diff()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    features['rsi'] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()

    # Bollinger Bands
    sma = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    features['bb_position'] = (df['close'] - sma) / (2 * std)

    # Volume
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # Volatility
    features['volatility'] = features['returns'].rolling(window=20).std()

    # Combine
    X = pd.DataFrame(features).dropna()

    return X.values, list(features.keys())
```

### Actionable Insights

```python
def explain_prediction(model, x, feature_names, target_class=None):
    """
    Generate counterfactual explanation for a trading prediction

    Args:
        model: Trained classifier
        x: Input features (1, num_features)
        feature_names: List of feature names
        target_class: Desired alternative class (None = flip prediction)

    Returns:
        explanation: Human-readable explanation
        counterfactual: The counterfactual instance
    """
    # Get original prediction
    with torch.no_grad():
        orig_probs = F.softmax(model(x), dim=-1)
        orig_class = orig_probs.argmax().item()

    # Determine target class
    class_names = ['SELL', 'HOLD', 'BUY']
    if target_class is None:
        target_class = (orig_class + 1) % 3  # Flip to different class

    # Generate counterfactual
    optimizer = CounterfactualOptimizer(model, actionable_mask=None)
    x_cf, info = optimizer.generate(x, torch.tensor([target_class]))

    # Find changed features
    changes = []
    for i, (orig, cf, name) in enumerate(zip(x[0], x_cf[0], feature_names)):
        diff = cf - orig
        if abs(diff) > 0.01:
            direction = "increase" if diff > 0 else "decrease"
            changes.append(f"  - {name}: {orig:.3f} → {cf:.3f} ({direction} by {abs(diff):.3f})")

    explanation = f"""
Counterfactual Explanation
==========================
Original prediction: {class_names[orig_class]} ({orig_probs[0, orig_class]:.1%} confidence)
Target prediction: {class_names[target_class]}

To change prediction from {class_names[orig_class]} to {class_names[target_class]}:
{chr(10).join(changes) if changes else "  No valid counterfactual found"}

Number of features changed: {info['num_features_changed']}
Counterfactual valid: {info['is_valid']}
"""

    return explanation, x_cf
```

**Example Output:**

```
Counterfactual Explanation
==========================
Original prediction: SELL (78.5% confidence)
Target prediction: BUY

To change prediction from SELL to BUY:
  - rsi: 72.500 → 45.200 (decrease by 27.300)
  - macd: -0.450 → 0.120 (increase by 0.570)

Number of features changed: 2
Counterfactual valid: True
```

## Backtesting

### Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Counterfactual Validity** | % of CFs that achieve target class | > 90% |
| **Sparsity** | Average features changed | < 3 |
| **Proximity** | Average L1 distance | Depends on scale |
| **Plausibility** | % CFs within data distribution | > 80% |
| **Stability** | Consistency across similar inputs | High |

### Expected Results

| Method | Validity | Sparsity | Proximity | Plausibility |
|--------|----------|----------|-----------|--------------|
| Random Search | 45% | 8.2 | 5.4 | 30% |
| Gradient Descent | 85% | 4.1 | 2.1 | 65% |
| **Neural CF Generator** | **92%** | **2.3** | **1.5** | **82%** |

### Trading Strategy with Counterfactual Insights

```
Entry Rules:
├── Model prediction confidence > 60%
├── Counterfactual distance > threshold (stable prediction)
├── Key features not near flip boundary
└── No conflicting signals in correlated assets

Exit Rules:
├── Model prediction flips
├── Counterfactual distance drops below threshold
├── Stop loss: -2%
├── Take profit: +4%
└── Time-based: exit after 12 hours if no clear direction

Risk Management:
├── Counterfactual distance indicates prediction stability
├── Lower distance → smaller position size
├── Track which features are closest to boundaries
└── Alert when market conditions approach flip points
```

## Resources

### Academic Papers

1. **Counterfactual Explanations without Opening the Black Box**
   - arXiv: [1711.00399](https://arxiv.org/abs/1711.00399)
   - Key ideas: Minimal perturbation counterfactuals

2. **Diverse Counterfactual Explanations for Anomaly Detection**
   - Multiple counterfactuals for comprehensive understanding

3. **Actionable Recourse in Machine Learning**
   - Focusing on realistic, actionable changes

### Books

- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) (Christoph Molnar)
- [Explainable AI: Interpreting and Explaining ML Models](https://www.springer.com/gp/book/9783030289539)

### Related Chapters

- [Chapter 115: SHAP Values](../115_shap_values) — Feature importance explanations
- [Chapter 117: LIME Explanations](../117_lime_explanations) — Local surrogate models
- [Chapter 118: Integrated Gradients](../118_integrated_gradients) — Attribution methods

## Dependencies

### Rust

```toml
ndarray = "0.16"
ndarray-linalg = "0.16"
reqwest = "0.12"
tokio = "1.0"
serde = "1.0"
serde_json = "1.0"
chrono = "0.4"
rand = "0.8"
anyhow = "1.0"
```

### Python

```python
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
shap>=0.42.0  # For comparison
```

## Difficulty Level

**Intermediate**

**Required knowledge:**
- Basic machine learning concepts
- Gradient-based optimization
- Time series analysis
- Risk management principles

---

*This material is for educational purposes. Cryptocurrency trading involves significant risk. Past performance does not guarantee future results.*
