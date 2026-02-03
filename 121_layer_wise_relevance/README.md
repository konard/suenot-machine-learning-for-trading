# Chapter 121: Layer-wise Relevance Propagation (LRP)

This chapter explores **Layer-wise Relevance Propagation (LRP)**, a powerful technique for explaining neural network predictions by decomposing the output back to input features. Unlike black-box models, LRP provides interpretability crucial for understanding trading decisions and building trust in algorithmic strategies.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LRP PROPAGATION FLOW                         │
│                                                                 │
│  Input Features          Hidden Layers         Output          │
│  ┌─────────┐             ┌─────────┐          ┌─────────┐      │
│  │ Price   │────────────▶│ Layer 1 │─────────▶│         │      │
│  │ Volume  │◀────────────│ Layer 2 │◀─────────│ Predict │      │
│  │ RSI     │  Relevance  │ Layer 3 │ Relevance│  f(x)   │      │
│  └─────────┘  Backward   └─────────┘  Backward└─────────┘      │
│                                                                 │
│  R_input = decomposed relevance showing feature importance      │
└─────────────────────────────────────────────────────────────────┘
```

## Contents

1. [Introduction to LRP](#introduction-to-lrp)
    * [Why Explainability Matters in Trading](#why-explainability-matters-in-trading)
    * [Key Advantages](#key-advantages)
    * [Comparison with Other Explanation Methods](#comparison-with-other-explanation-methods)
2. [Mathematical Foundation](#mathematical-foundation)
    * [Conservation Principle](#conservation-principle)
    * [LRP Rules](#lrp-rules)
    * [Propagation Through Layers](#propagation-through-layers)
3. [LRP Variants](#lrp-variants)
    * [LRP-0 (Basic Rule)](#lrp-0-basic-rule)
    * [LRP-ε (Epsilon Rule)](#lrp-ε-epsilon-rule)
    * [LRP-γ (Gamma Rule)](#lrp-γ-gamma-rule)
    * [Composite Rules](#composite-rules)
4. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: LRP Implementation](#02-lrp-implementation)
    * [03: Model Training with Explanations](#03-model-training-with-explanations)
    * [04: Trading Signal Analysis](#04-trading-signal-analysis)
    * [05: Backtesting with LRP Insights](#05-backtesting-with-lrp-insights)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to LRP

Layer-wise Relevance Propagation (LRP) is an explanation technique introduced by Bach et al. (2015) that decomposes neural network predictions into contributions from individual input features. The method satisfies a conservation property: the sum of all input relevances equals the network output.

### Why Explainability Matters in Trading

In algorithmic trading, understanding **why** a model makes predictions is often as important as the predictions themselves:

1. **Regulatory Compliance**: Financial regulators increasingly require model explainability
2. **Risk Management**: Understanding which features drive predictions helps identify model risks
3. **Strategy Validation**: Verify that models learn meaningful patterns, not spurious correlations
4. **Debug & Improve**: Identify when models focus on wrong features
5. **Trust Building**: Traders need confidence in automated decisions

```
Trading Decision Flow with LRP:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Market Data  ──▶  Neural Network  ──▶  Prediction            │
│   (OHLCV, etc.)     (Black Box)         (Buy/Sell/Hold)        │
│                          │                    │                 │
│                          ▼                    ▼                 │
│                        LRP              Relevance Scores        │
│                          │                    │                 │
│                          ▼                    ▼                 │
│                    "Price momentum (45%)                        │
│                     Volume spike (30%)                          │
│                     RSI oversold (25%)"                         │
│                                                                 │
│   ═══════════════════════════════════════════════════════════   │
│   Result: Interpretable trading decisions with full attribution │
└─────────────────────────────────────────────────────────────────┘
```

### Key Advantages

1. **Conservation Property**
   - Total relevance equals output value
   - No relevance is created or destroyed
   - Mathematically principled decomposition

2. **Layer-wise Interpretation**
   - Understand contribution at each layer
   - Visualize information flow through network
   - Debug layer-specific issues

3. **Sign-aware Attribution**
   - Positive relevance: supports prediction
   - Negative relevance: contradicts prediction
   - Rich interpretation beyond simple importance

4. **Architecture Agnostic**
   - Works with MLPs, CNNs, RNNs, Transformers
   - Applicable to any differentiable architecture
   - Consistent interpretation across models

### Comparison with Other Explanation Methods

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **LRP** | Relevance propagation | Conservation, layer-wise | Rule selection complexity |
| SHAP | Shapley values | Game-theoretic foundation | Computationally expensive |
| LIME | Local surrogates | Model-agnostic | Instability, approximation |
| Integrated Gradients | Path integration | Axiomatic | Baseline selection |
| Attention Weights | Direct from model | Easy to extract | Only for attention models |
| Gradient × Input | Simple gradient | Fast computation | Noisy, no conservation |

## Mathematical Foundation

### Conservation Principle

The fundamental principle of LRP is **relevance conservation**:

```
Conservation Law:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  For each layer l:  Σ_i R_i^(l) = Σ_j R_j^(l+1)                │
│                                                                 │
│  At output:         R_output = f(x)  (network prediction)       │
│                                                                 │
│  At input:          Σ_i R_i^input = f(x)                       │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│  Total relevance is conserved through all layers                │
└─────────────────────────────────────────────────────────────────┘
```

For a network with layers L₁, L₂, ..., Lₙ:

```
f(x) = R^(n) = Σ R^(n-1) = Σ R^(n-2) = ... = Σ R^(0) = Σ R_input
```

### LRP Rules

LRP defines how to redistribute relevance from layer l+1 to layer l:

```python
# General LRP rule for neuron j receiving relevance R_j:
# Distribute to contributing neurons i based on weighted activations

R_i←j = (a_i * w_ij / Σ_k a_k * w_kj) * R_j

# Where:
# a_i = activation of neuron i in previous layer
# w_ij = weight connecting neuron i to neuron j
# R_j = relevance at neuron j (to be distributed)
# R_i←j = relevance passed from j to i
```

The total relevance at neuron i is the sum of all contributions:

```
R_i = Σ_j R_i←j
```

### Propagation Through Layers

```
Layer-wise Propagation:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Layer L (output)     Layer L-1           Layer L-2    ...     │
│  ┌─────┐              ┌─────┐             ┌─────┐              │
│  │ R_1 │ ──────────▶  │ R_1 │ ──────────▶ │ R_1 │              │
│  │ R_2 │   LRP Rule   │ R_2 │   LRP Rule  │ R_2 │              │
│  │ R_3 │ ──────────▶  │ R_3 │ ──────────▶ │ R_3 │              │
│  │ ... │              │ ... │             │ ... │              │
│  └─────┘              └─────┘             └─────┘              │
│                                                                 │
│  Start: R = [f(x), 0, 0, ...]  (only predicted class)          │
│  End: R_input = feature relevances summing to f(x)              │
└─────────────────────────────────────────────────────────────────┘
```

## LRP Variants

### LRP-0 (Basic Rule)

The simplest rule distributes relevance proportionally to weighted activations:

```python
def lrp_0(a, w, R_next):
    """
    LRP-0: Basic relevance propagation rule.

    R_i←j = (a_i * w_ij) / (Σ_k a_k * w_kj) * R_j

    Args:
        a: Activations from previous layer [batch, in_features]
        w: Weight matrix [in_features, out_features]
        R_next: Relevance from next layer [batch, out_features]

    Returns:
        Relevance for current layer [batch, in_features]
    """
    z = a.unsqueeze(-1) * w.unsqueeze(0)  # [batch, in, out]
    z_sum = z.sum(dim=1, keepdim=True)     # [batch, 1, out]

    # Avoid division by zero
    z_sum = z_sum + 1e-9 * (z_sum == 0).float()

    # Proportion of contribution
    s = z / z_sum  # [batch, in, out]

    # Distribute relevance
    R = (s * R_next.unsqueeze(1)).sum(dim=-1)  # [batch, in]

    return R
```

**Properties:**
- Simple and intuitive
- Can be unstable with small denominators
- No handling of negative contributions

### LRP-ε (Epsilon Rule)

Adds small stabilizer ε to denominator for numerical stability:

```python
def lrp_epsilon(a, w, R_next, epsilon=0.01):
    """
    LRP-ε: Epsilon-stabilized rule.

    R_i←j = (a_i * w_ij) / (Σ_k a_k * w_kj + ε * sign(Σ_k a_k * w_kj)) * R_j

    Args:
        a: Activations [batch, in_features]
        w: Weights [in_features, out_features]
        R_next: Relevance [batch, out_features]
        epsilon: Stabilization term

    Returns:
        Relevance [batch, in_features]
    """
    z = a.unsqueeze(-1) * w.unsqueeze(0)  # [batch, in, out]
    z_sum = z.sum(dim=1, keepdim=True)     # [batch, 1, out]

    # Add epsilon with sign preservation
    z_sum_stabilized = z_sum + epsilon * torch.sign(z_sum)
    z_sum_stabilized = torch.where(
        z_sum == 0,
        torch.ones_like(z_sum) * epsilon,
        z_sum_stabilized
    )

    s = z / z_sum_stabilized
    R = (s * R_next.unsqueeze(1)).sum(dim=-1)

    return R
```

**Properties:**
- More stable than LRP-0
- ε absorbs some relevance (weak conservation)
- Good default choice for most layers

### LRP-γ (Gamma Rule)

Emphasizes positive contributions over negative ones:

```python
def lrp_gamma(a, w, R_next, gamma=0.25):
    """
    LRP-γ: Gamma rule emphasizing positive contributions.

    w+ = max(w, 0),  w- = min(w, 0)
    R_i←j = (a_i * (w_ij + γ*w_ij+)) / (Σ_k a_k * (w_kj + γ*w_kj+)) * R_j

    Args:
        a: Activations [batch, in_features]
        w: Weights [in_features, out_features]
        R_next: Relevance [batch, out_features]
        gamma: Emphasis factor for positive weights

    Returns:
        Relevance [batch, in_features]
    """
    w_positive = torch.clamp(w, min=0)
    w_modified = w + gamma * w_positive

    z = a.unsqueeze(-1) * w_modified.unsqueeze(0)
    z_sum = z.sum(dim=1, keepdim=True) + 1e-9

    s = z / z_sum
    R = (s * R_next.unsqueeze(1)).sum(dim=-1)

    return R
```

**Properties:**
- Focuses on excitatory (positive) evidence
- γ > 0 increases weight of positive contributions
- Useful for classification tasks

### Composite Rules

Best practice is to use different rules for different layer types:

```python
class CompositeLRP:
    """
    Composite LRP strategy using different rules per layer.

    Recommended configuration:
    - Lower layers (near input): LRP-γ (γ=0.25)
    - Middle layers: LRP-ε (ε=0.25)
    - Upper layers (near output): LRP-0

    This combination provides:
    - Stable explanations (ε in middle)
    - Focus on positive evidence (γ at input)
    - Precise attribution (0 at output)
    """

    def __init__(self, model, rules=None):
        self.model = model
        self.rules = rules or self._default_rules()

    def _default_rules(self):
        """Default composite rule assignment."""
        num_layers = len(list(self.model.modules()))
        rules = {}

        for i, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                position = i / num_layers
                if position < 0.33:
                    rules[name] = ('gamma', 0.25)
                elif position < 0.66:
                    rules[name] = ('epsilon', 0.25)
                else:
                    rules[name] = ('zero', None)

        return rules
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

def prepare_lrp_data(
    symbols: List[str],
    lookback: int = 60,
    horizon: int = 1,
    features: List[str] = None
) -> Dict:
    """
    Prepare financial data for LRP-analyzed trading model.

    Args:
        symbols: Trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        lookback: Historical window size
        horizon: Prediction horizon
        features: Feature names to use

    Returns:
        Dictionary with X (features), y (targets), feature_names
    """
    if features is None:
        features = [
            'log_return', 'volume_ratio', 'volatility_20',
            'rsi_14', 'macd', 'bb_position', 'atr_14'
        ]

    all_data = []

    for symbol in symbols:
        df = load_market_data(symbol)  # From Bybit or other source

        # Calculate features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility_20'] = df['log_return'].rolling(20).std()
        df['rsi_14'] = compute_rsi(df['close'], 14)
        df['macd'] = compute_macd(df['close'])
        df['bb_position'] = compute_bollinger_position(df['close'])
        df['atr_14'] = compute_atr(df, 14)

        # Target: next period return direction
        df['target'] = (df['log_return'].shift(-horizon) > 0).astype(int)

        all_data.append(df[features + ['target']].dropna())

    combined = pd.concat(all_data)

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(combined) - horizon):
        X.append(combined[features].iloc[i-lookback:i].values)
        y.append(combined['target'].iloc[i])

    return {
        'X': np.array(X),
        'y': np.array(y),
        'feature_names': features,
        'lookback': lookback,
        'horizon': horizon
    }


class TradingDataset(Dataset):
    """PyTorch Dataset for trading with LRP analysis."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

### 02: LRP Implementation

See [python/model.py](python/model.py) for complete implementation.

```python
# Core LRP module

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class LRPRule(Enum):
    """Available LRP rules."""
    ZERO = "zero"
    EPSILON = "epsilon"
    GAMMA = "gamma"
    ALPHA_BETA = "alpha_beta"


@dataclass
class LRPConfig:
    """Configuration for LRP analysis."""
    epsilon: float = 0.01
    gamma: float = 0.25
    alpha: float = 2.0
    beta: float = 1.0
    default_rule: LRPRule = LRPRule.EPSILON


class LRPLinear(nn.Module):
    """Linear layer with LRP support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: LRPConfig = None
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.config = config or LRPConfig()
        self.activations = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.activations = x.detach().clone()
        return self.linear(x)

    def lrp(
        self,
        R: torch.Tensor,
        rule: LRPRule = None
    ) -> torch.Tensor:
        """
        Propagate relevance through this layer.

        Args:
            R: Relevance from next layer
            rule: LRP rule to apply

        Returns:
            Relevance for previous layer
        """
        rule = rule or self.config.default_rule
        a = self.activations
        w = self.linear.weight.T  # [in, out]

        if rule == LRPRule.ZERO:
            return self._lrp_zero(a, w, R)
        elif rule == LRPRule.EPSILON:
            return self._lrp_epsilon(a, w, R)
        elif rule == LRPRule.GAMMA:
            return self._lrp_gamma(a, w, R)
        elif rule == LRPRule.ALPHA_BETA:
            return self._lrp_alpha_beta(a, w, R)

    def _lrp_zero(self, a, w, R):
        z = a.unsqueeze(-1) * w.unsqueeze(0)
        z_sum = z.sum(dim=1, keepdim=True) + 1e-9
        s = z / z_sum
        return (s * R.unsqueeze(1)).sum(dim=-1)

    def _lrp_epsilon(self, a, w, R):
        z = a.unsqueeze(-1) * w.unsqueeze(0)
        z_sum = z.sum(dim=1, keepdim=True)
        z_sum = z_sum + self.config.epsilon * torch.sign(z_sum)
        z_sum = torch.where(z_sum == 0, torch.ones_like(z_sum) * self.config.epsilon, z_sum)
        s = z / z_sum
        return (s * R.unsqueeze(1)).sum(dim=-1)

    def _lrp_gamma(self, a, w, R):
        w_pos = torch.clamp(w, min=0)
        w_mod = w + self.config.gamma * w_pos
        z = a.unsqueeze(-1) * w_mod.unsqueeze(0)
        z_sum = z.sum(dim=1, keepdim=True) + 1e-9
        s = z / z_sum
        return (s * R.unsqueeze(1)).sum(dim=-1)

    def _lrp_alpha_beta(self, a, w, R):
        alpha, beta = self.config.alpha, self.config.beta

        z_pos = torch.clamp(a.unsqueeze(-1) * w.unsqueeze(0), min=0)
        z_neg = torch.clamp(a.unsqueeze(-1) * w.unsqueeze(0), max=0)

        z_pos_sum = z_pos.sum(dim=1, keepdim=True) + 1e-9
        z_neg_sum = z_neg.sum(dim=1, keepdim=True) - 1e-9

        s_pos = alpha * z_pos / z_pos_sum
        s_neg = beta * z_neg / z_neg_sum

        return ((s_pos + s_neg) * R.unsqueeze(1)).sum(dim=-1)


class LRPNetwork(nn.Module):
    """
    Neural network with built-in LRP explanation capability.

    Architecture:
    - Multiple LRP-enabled linear layers
    - ReLU activations
    - Dropout for regularization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        config: LRPConfig = None
    ):
        super().__init__()
        self.config = config or LRPConfig()

        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(
                LRPLinear(dims[i], dims[i+1], config=self.config)
            )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass storing activations for LRP."""
        # Flatten if needed
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.layers[-1](x)
        return x

    def explain(
        self,
        x: torch.Tensor,
        target_class: int = None
    ) -> torch.Tensor:
        """
        Compute LRP explanations for predictions.

        Args:
            x: Input tensor [batch, ...]
            target_class: Class to explain (default: predicted class)

        Returns:
            Relevance scores [batch, input_dim]
        """
        # Forward pass
        output = self.forward(x)

        # Initialize relevance at output
        if target_class is not None:
            R = torch.zeros_like(output)
            R[:, target_class] = output[:, target_class]
        else:
            # Use predicted class
            pred = output.argmax(dim=1)
            R = torch.zeros_like(output)
            R[torch.arange(len(pred)), pred] = output[torch.arange(len(pred)), pred]

        # Backward propagation through layers
        for layer in reversed(self.layers):
            R = layer.lrp(R)

        return R.view(x.shape)
```

### 03: Model Training with Explanations

```python
# python/03_train_with_lrp.py

import torch
import torch.nn as nn
from model import LRPNetwork, LRPConfig

def train_explainable_model(
    train_loader,
    val_loader,
    input_dim: int,
    num_epochs: int = 100,
    learning_rate: float = 1e-3
) -> Tuple[LRPNetwork, Dict]:
    """
    Train neural network with periodic LRP explanation analysis.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        input_dim: Input feature dimension
        num_epochs: Training epochs
        learning_rate: Learning rate

    Returns:
        Trained model and training history with explanations
    """
    config = LRPConfig(epsilon=0.01, gamma=0.25)
    model = LRPNetwork(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        output_dim=2,
        dropout=0.2,
        config=config
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'feature_importance': []
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

                pred = output.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += len(batch_y)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step(val_loss)

        # Compute LRP explanations periodically
        if epoch % 10 == 0:
            importance = compute_feature_importance(model, val_loader)
            history['feature_importance'].append(importance)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            print(f"  Top features: {importance[:3]}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    return model, history


def compute_feature_importance(
    model: LRPNetwork,
    data_loader,
    feature_names: List[str] = None
) -> List[Tuple[str, float]]:
    """
    Compute average feature importance using LRP.

    Args:
        model: Trained LRP-enabled model
        data_loader: Data to analyze
        feature_names: Names of features

    Returns:
        List of (feature_name, importance) sorted by importance
    """
    model.eval()
    total_relevance = None
    count = 0

    with torch.no_grad():
        for batch_x, _ in data_loader:
            relevance = model.explain(batch_x)

            # Average over batch and time
            batch_relevance = relevance.abs().mean(dim=0)
            if len(batch_relevance.shape) > 1:
                batch_relevance = batch_relevance.mean(dim=0)

            if total_relevance is None:
                total_relevance = batch_relevance
            else:
                total_relevance += batch_relevance
            count += 1

    avg_relevance = total_relevance / count
    avg_relevance = avg_relevance / avg_relevance.sum()  # Normalize

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(avg_relevance))]

    importance = list(zip(feature_names, avg_relevance.tolist()))
    importance.sort(key=lambda x: x[1], reverse=True)

    return importance
```

### 04: Trading Signal Analysis

```python
# python/04_signal_analysis.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def analyze_trading_signal(
    model: LRPNetwork,
    sample: torch.Tensor,
    feature_names: List[str],
    threshold: float = 0.1
) -> Dict:
    """
    Analyze a single trading signal with LRP explanation.

    Args:
        model: Trained LRP model
        sample: Single input sample [1, seq_len, features]
        feature_names: Names of features
        threshold: Minimum relevance to report

    Returns:
        Dictionary with prediction, confidence, and explanations
    """
    model.eval()

    with torch.no_grad():
        # Get prediction
        output = model(sample)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        # Get LRP explanation
        relevance = model.explain(sample, target_class=pred_class)

        # Aggregate relevance over time dimension
        if len(relevance.shape) == 3:
            relevance = relevance.mean(dim=1)  # [1, features]

        relevance = relevance[0].numpy()

        # Normalize to percentages
        total = np.abs(relevance).sum()
        relevance_pct = relevance / total * 100

        # Create explanation
        explanations = []
        for name, rel in zip(feature_names, relevance_pct):
            if abs(rel) >= threshold * 100:
                direction = "supports" if rel > 0 else "contradicts"
                explanations.append({
                    'feature': name,
                    'relevance_pct': rel,
                    'direction': direction
                })

        # Sort by absolute relevance
        explanations.sort(key=lambda x: abs(x['relevance_pct']), reverse=True)

    signal = "BUY" if pred_class == 1 else "SELL/HOLD"

    return {
        'signal': signal,
        'confidence': confidence,
        'explanations': explanations,
        'raw_relevance': relevance,
        'feature_names': feature_names
    }


def visualize_explanation(analysis: Dict, save_path: str = None):
    """
    Visualize LRP explanation as a bar chart.

    Args:
        analysis: Output from analyze_trading_signal
        save_path: Path to save figure
    """
    explanations = analysis['explanations']

    features = [e['feature'] for e in explanations]
    relevances = [e['relevance_pct'] for e in explanations]
    colors = ['green' if r > 0 else 'red' for r in relevances]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(features))
    ax.barh(y_pos, relevances, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Relevance (%)')
    ax.set_title(f"Signal: {analysis['signal']} (Confidence: {analysis['confidence']:.2%})")

    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Add value labels
    for i, (v, c) in enumerate(zip(relevances, colors)):
        ax.text(v + 0.5 if v > 0 else v - 0.5, i, f'{v:.1f}%',
                va='center', ha='left' if v > 0 else 'right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def generate_trading_report(
    model: LRPNetwork,
    data_loader,
    feature_names: List[str]
) -> str:
    """
    Generate a trading report with LRP insights.

    Args:
        model: Trained model
        data_loader: Data to analyze
        feature_names: Feature names

    Returns:
        Markdown-formatted report
    """
    model.eval()

    buy_signals = []
    sell_signals = []

    with torch.no_grad():
        for batch_x, _ in data_loader:
            for i in range(len(batch_x)):
                sample = batch_x[i:i+1]
                analysis = analyze_trading_signal(model, sample, feature_names)

                if analysis['signal'] == 'BUY':
                    buy_signals.append(analysis)
                else:
                    sell_signals.append(analysis)

    # Aggregate insights
    report = "# Trading Signal Analysis Report\n\n"

    report += f"## Summary\n"
    report += f"- Total BUY signals: {len(buy_signals)}\n"
    report += f"- Total SELL/HOLD signals: {len(sell_signals)}\n\n"

    report += "## Feature Importance (BUY signals)\n"
    if buy_signals:
        buy_importance = aggregate_explanations(buy_signals)
        for feat, rel in buy_importance[:5]:
            report += f"- **{feat}**: {rel:.1f}%\n"

    report += "\n## Feature Importance (SELL signals)\n"
    if sell_signals:
        sell_importance = aggregate_explanations(sell_signals)
        for feat, rel in sell_importance[:5]:
            report += f"- **{feat}**: {rel:.1f}%\n"

    return report


def aggregate_explanations(analyses: List[Dict]) -> List[Tuple[str, float]]:
    """Aggregate explanations across multiple analyses."""
    feature_relevance = {}

    for analysis in analyses:
        for exp in analysis['explanations']:
            feat = exp['feature']
            rel = abs(exp['relevance_pct'])
            feature_relevance[feat] = feature_relevance.get(feat, 0) + rel

    # Average
    for feat in feature_relevance:
        feature_relevance[feat] /= len(analyses)

    return sorted(feature_relevance.items(), key=lambda x: x[1], reverse=True)
```

### 05: Backtesting with LRP Insights

```python
# python/05_backtest.py

import torch
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    initial_capital: float = 100000
    transaction_cost: float = 0.001
    confidence_threshold: float = 0.6
    max_position: float = 1.0
    stop_loss: float = 0.02
    take_profit: float = 0.04


def backtest_with_lrp(
    model: LRPNetwork,
    test_data,
    feature_names: List[str],
    config: BacktestConfig = None
) -> Dict:
    """
    Backtest trading strategy with LRP-based position sizing.

    The strategy uses LRP confidence to adjust position sizes:
    - High confidence in key features = larger positions
    - Contradicting signals from features = smaller positions

    Args:
        model: Trained LRP model
        test_data: Test DataLoader
        feature_names: Feature names
        config: Backtest configuration

    Returns:
        Backtest results with LRP insights
    """
    config = config or BacktestConfig()
    model.eval()

    capital = config.initial_capital
    position = 0.0

    history = {
        'capital': [capital],
        'positions': [],
        'returns': [],
        'signals': [],
        'explanations': []
    }

    with torch.no_grad():
        for batch_x, batch_y in test_data:
            for i in range(len(batch_x)):
                sample = batch_x[i:i+1]
                actual_direction = batch_y[i].item()

                # Get prediction and explanation
                output = model(sample)
                probs = torch.softmax(output, dim=1)
                pred_class = output.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()

                # Get LRP explanation
                relevance = model.explain(sample, target_class=pred_class)
                if len(relevance.shape) == 3:
                    relevance = relevance.mean(dim=1)
                relevance = relevance[0].numpy()

                # Calculate explanation coherence
                coherence = calculate_coherence(relevance, feature_names)

                # Determine position based on prediction and confidence
                if confidence >= config.confidence_threshold:
                    target_position = 1.0 if pred_class == 1 else -1.0
                    # Scale by confidence and coherence
                    target_position *= min(confidence * coherence, config.max_position)
                else:
                    target_position = 0.0

                # Calculate trading costs
                position_change = abs(target_position - position)
                costs = position_change * config.transaction_cost * capital

                # Simulate return (using actual outcome)
                actual_return = 0.01 if actual_direction == 1 else -0.01  # Simplified
                pnl = position * actual_return * capital - costs

                # Update state
                capital += pnl
                position = target_position

                # Record history
                history['capital'].append(capital)
                history['positions'].append(position)
                history['returns'].append(pnl / (capital - pnl))
                history['signals'].append({
                    'pred': pred_class,
                    'actual': actual_direction,
                    'confidence': confidence,
                    'coherence': coherence
                })
                history['explanations'].append(relevance)

    # Calculate metrics
    returns = np.array(history['returns'])

    results = {
        'total_return': (capital - config.initial_capital) / config.initial_capital,
        'sharpe_ratio': calculate_sharpe(returns),
        'sortino_ratio': calculate_sortino(returns),
        'max_drawdown': calculate_max_drawdown(history['capital']),
        'win_rate': calculate_win_rate(history['signals']),
        'avg_confidence': np.mean([s['confidence'] for s in history['signals']]),
        'avg_coherence': np.mean([s['coherence'] for s in history['signals']]),
        'history': history,
        'feature_importance': calculate_backtest_importance(
            history['explanations'], feature_names
        )
    }

    return results


def calculate_coherence(relevance: np.ndarray, feature_names: List[str]) -> float:
    """
    Calculate explanation coherence score.

    High coherence = few dominant features
    Low coherence = spread across many features

    Uses normalized entropy as inverse coherence measure.
    """
    abs_rel = np.abs(relevance)
    probs = abs_rel / (abs_rel.sum() + 1e-9)

    # Entropy (lower = more focused)
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = np.log(len(relevance))

    # Coherence = 1 - normalized entropy
    coherence = 1 - (entropy / max_entropy)

    return float(coherence)


def calculate_sharpe(returns: np.ndarray, rf: float = 0) -> float:
    """Calculate annualized Sharpe ratio."""
    excess = returns - rf / 252
    return np.sqrt(252) * excess.mean() / (excess.std() + 1e-9)


def calculate_sortino(returns: np.ndarray, target: float = 0) -> float:
    """Calculate Sortino ratio."""
    downside = returns[returns < target]
    downside_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-9
    return np.sqrt(252) * (returns.mean() - target) / downside_std


def calculate_max_drawdown(capital_history: List[float]) -> float:
    """Calculate maximum drawdown."""
    peak = capital_history[0]
    max_dd = 0

    for capital in capital_history:
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak
        max_dd = max(max_dd, dd)

    return max_dd


def calculate_win_rate(signals: List[Dict]) -> float:
    """Calculate prediction accuracy."""
    correct = sum(1 for s in signals if s['pred'] == s['actual'])
    return correct / len(signals) if signals else 0


def calculate_backtest_importance(
    explanations: List[np.ndarray],
    feature_names: List[str]
) -> List[Tuple[str, float]]:
    """Calculate aggregate feature importance from backtest."""
    total = np.zeros(len(feature_names))

    for exp in explanations:
        total += np.abs(exp)

    total = total / total.sum()

    importance = list(zip(feature_names, total.tolist()))
    importance.sort(key=lambda x: x[1], reverse=True)

    return importance
```

## Rust Implementation

See [rust_lrp](rust_lrp/) for complete Rust implementation.

```
rust_lrp/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── api/                # Exchange API client
│   │   ├── mod.rs
│   │   ├── client.rs       # Bybit/Exchange HTTP client
│   │   └── types.rs        # API response types
│   ├── data/               # Data processing
│   │   ├── mod.rs
│   │   ├── loader.rs       # Data loading utilities
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset for training
│   ├── model/              # LRP model
│   │   ├── mod.rs
│   │   ├── config.rs       # Model configuration
│   │   ├── linear.rs       # LRP-enabled linear layer
│   │   ├── network.rs      # Complete network
│   │   └── lrp.rs          # LRP propagation rules
│   └── strategy/           # Trading strategy
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting engine
└── examples/
    ├── fetch_data.rs       # Download market data
    ├── train.rs            # Train model
    ├── explain.rs          # Generate explanations
    └── backtest.rs         # Run backtest
```

### Quick Start (Rust)

```bash
# Navigate to Rust project
cd rust_lrp

# Fetch data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT --interval 1h

# Train model
cargo run --example train -- --epochs 100 --batch-size 32

# Generate explanations
cargo run --example explain -- --model model.bin --input latest_data.json

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── __init__.py             # Package initialization
├── model.py                # LRP network implementation
├── data.py                 # Data loading and preprocessing
├── strategy.py             # Trading strategy and backtesting
├── example_usage.py        # Complete example
└── requirements.txt        # Dependencies
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete example
python example_usage.py

# Or use as library
python -c "from model import LRPNetwork; print('LRP ready!')"
```

## Best Practices

### When to Use LRP

**Ideal use cases:**
- Regulatory compliance requiring model explainability
- Risk management and model debugging
- Building trust in trading decisions
- Feature importance analysis
- Model validation and sanity checking

**Consider alternatives for:**
- Real-time predictions (LRP adds overhead)
- Simple linear models (coefficients are already interpretable)
- When only aggregate importance is needed (use permutation importance)

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `epsilon` | 0.01 - 0.1 | Smaller = more precise, larger = more stable |
| `gamma` | 0.1 - 0.5 | Higher = more focus on positive evidence |
| `alpha/beta` | 2/1 | Standard α-β rule values |
| Composite rules | Yes | Different rules for different layers |

### Common Pitfalls

1. **Using single rule everywhere**: Composite rules work better
2. **Ignoring numerical stability**: Always use epsilon stabilization
3. **Not normalizing relevances**: Compare relative, not absolute values
4. **Forgetting to store activations**: Required for backward LRP pass
5. **Misinterpreting negative relevance**: It means contradiction, not unimportance

### LRP vs Alternatives

```
Decision Guide:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Need theoretical guarantees?                                   │
│  └── Yes ──▶ LRP (conservation) or SHAP (Shapley axioms)       │
│  └── No ──▶ Gradient methods (faster)                          │
│                                                                 │
│  Need layer-wise analysis?                                      │
│  └── Yes ──▶ LRP                                               │
│  └── No ──▶ SHAP or Integrated Gradients                       │
│                                                                 │
│  Computational budget tight?                                    │
│  └── Yes ──▶ LRP or Gradient × Input                           │
│  └── No ──▶ SHAP (most comprehensive)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Resources

### Papers

- [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) — Original LRP paper (Bach et al., 2015)
- [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10) — Comprehensive survey (2019)
- [Explaining NonLinear Classification Decisions with Deep Taylor Decomposition](https://arxiv.org/abs/1512.02479) — Theoretical foundations
- [Methods for Interpreting and Understanding Deep Neural Networks](https://arxiv.org/abs/1711.07104) — Review including LRP

### Implementations

- [LRP Tutorial (Heatmapping.org)](https://heatmapping.org/) — Official tutorial and software
- [Captum (PyTorch)](https://captum.ai/) — Facebook's interpretability library with LRP
- [iNNvestigate](https://github.com/albermax/innvestigate) — Keras/TensorFlow LRP implementation
- [Zennit](https://github.com/chr5tphr/zennit) — PyTorch LRP library

### Related Chapters

- [Chapter 120: SHAP Trading Analysis](../120_shap_trading) — Shapley value explanations
- [Chapter 122: Integrated Gradients](../122_integrated_gradients) — Path-based attribution
- [Chapter 123: Attention Visualization](../123_attention_visualization) — Transformer interpretability
- [Chapter 124: Feature Importance](../124_feature_importance) — Permutation-based methods

---

## Difficulty Level

**Intermediate**

Prerequisites:
- Neural network fundamentals
- Backpropagation understanding
- PyTorch/Rust ML programming
- Basic linear algebra

Learning path:
1. Start with LRP-ε (most stable)
2. Understand conservation property
3. Experiment with different rules
4. Apply composite strategies
5. Integrate with trading strategies
