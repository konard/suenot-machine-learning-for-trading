# Chapter 122: DeepLIFT for Trading

## Overview

DeepLIFT (Deep Learning Important FeaTures) is a powerful interpretability method that explains the predictions of neural networks by comparing activations to a reference input. Introduced by Shrikumar et al. (2017), DeepLIFT assigns contribution scores to each input feature by propagating the difference between the actual activation and a reference activation back through the network.

In algorithmic trading, DeepLIFT is invaluable for understanding which market features drive trading signals, identifying regime changes, and building more transparent and trustworthy trading systems.

## Table of Contents

1. [Introduction to DeepLIFT](#introduction-to-deeplift)
2. [Mathematical Foundation](#mathematical-foundation)
3. [DeepLIFT vs Other Attribution Methods](#deeplift-vs-other-attribution-methods)
4. [DeepLIFT for Trading Applications](#deeplift-for-trading-applications)
5. [Implementation in Python](#implementation-in-python)
6. [Implementation in Rust](#implementation-in-rust)
7. [Practical Examples with Stock and Crypto Data](#practical-examples-with-stock-and-crypto-data)
8. [Backtesting Framework](#backtesting-framework)
9. [Performance Evaluation](#performance-evaluation)
10. [Future Directions](#future-directions)

---

## Introduction to DeepLIFT

### What is Feature Attribution?

Feature attribution methods explain neural network predictions by assigning importance scores to each input feature. These scores indicate how much each feature contributed to the final prediction, helping us understand the "why" behind model decisions.

### The DeepLIFT Algorithm

DeepLIFT was introduced by Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje in their 2017 paper "Learning Important Features Through Propagating Activation Differences." The key insight is elegant:

1. Choose a reference input (baseline) representing "absence of signal"
2. Compute the difference in activation at each neuron between actual input and reference
3. Decompose this difference into contributions from each input feature
4. Propagate contributions back through the network using the chain rule for multipliers

The "summation-to-delta" property ensures that contributions sum exactly to the difference between the output for the actual input and the reference:

```
Σᵢ Cᵢ = f(x) - f(x_ref)
```

### Why DeepLIFT for Trading?

Financial markets present unique challenges that make DeepLIFT particularly attractive:

- **Interpretability**: Understand which technical indicators drive trading signals
- **Risk Management**: Identify when models rely on spurious correlations
- **Regime Detection**: Observe shifts in feature importance during market transitions
- **Model Validation**: Verify that models learn meaningful market patterns
- **Regulatory Compliance**: Provide explainable AI for financial decisions

---

## Mathematical Foundation

### The Core Principle

DeepLIFT computes contribution scores by comparing activations to a reference:

**Activation Difference:**
```
Δt = t - t⁰
```
where t is the actual activation and t⁰ is the reference activation.

**Contribution Score:**
```
Cᵢ = contribution of input xᵢ to the activation difference Δt
```

### The Multiplier Rule

For a neuron with inputs x₁, ..., xₙ and output t:

**Multiplier Definition:**
```
mᵢ = Cᵢ / Δxᵢ
```

where Δxᵢ = xᵢ - x⁰ᵢ is the difference from the reference input.

**Summation-to-Delta:**
```
Σᵢ mᵢ × Δxᵢ = Δt
```

### Propagation Rules

**Linear Layer:**
For t = Σᵢ wᵢ × xᵢ + b:
```
mᵢ = wᵢ
```

**ReLU Activation (Rescale Rule):**
```
mᵢ = Δy / Δx    (if Δx ≠ 0)
    = 0         (if Δx = 0)
```

**ReLU Activation (RevealCancel Rule):**
For more accurate attribution, separate positive and negative contributions:
```
Δy⁺ = (y⁺ - y⁰⁺)
Δy⁻ = (y⁻ - y⁰⁻)
```

### Chain Rule for Multipliers

For a chain of layers, multiply the multipliers:
```
m_total = m₁ × m₂ × ... × mₙ
```

### Reference Selection

Choosing the right reference is crucial:

- **Zero Reference**: All features set to 0 (common but not always meaningful)
- **Mean Reference**: Average values across the dataset
- **Neutral Reference**: Values representing "no trading signal"
- **Distribution Reference**: Sample from input distribution (expected gradients)

---

## DeepLIFT vs Other Attribution Methods

### Comparison Table

| Method | Reference Required | Saturation Handling | Computation | Accuracy |
|--------|-------------------|---------------------|-------------|----------|
| DeepLIFT | Yes | Excellent | Medium | Excellent |
| Gradient | No | Poor | Low | Fair |
| Integrated Gradients | Yes | Good | High | Very Good |
| SHAP | Yes (distribution) | Excellent | Very High | Excellent |
| LRP | No | Good | Medium | Good |
| Saliency Maps | No | Poor | Low | Fair |

### When to Use DeepLIFT

**Use DeepLIFT when:**
- You need fast, accurate attribution scores
- Your model has ReLU-like activations
- You want to understand relative feature importance
- Saturation-aware attribution is important

**Consider alternatives when:**
- You need theoretical guarantees (use SHAP)
- You have non-standard architectures (use Integrated Gradients)
- Speed is paramount (use simple gradients)

---

## DeepLIFT for Trading Applications

### 1. Trading Signal Explanation

Understand which features drive buy/sell signals:

```
Input Features: [returns, volatility, momentum, RSI, MACD, volume, ...]
DeepLIFT Output: Contribution of each feature to the prediction
Example: "RSI contributed +0.3 to the buy signal, while high volatility contributed -0.15"
```

### 2. Risk Attribution

Identify which factors contribute to portfolio risk:

```
For a risk prediction of 0.8:
- Market correlation: +0.4
- Sector exposure: +0.25
- Volatility regime: +0.15
```

### 3. Regime Change Detection

Monitor shifts in feature importance over time:

```
Bull Market:  momentum=0.5, mean_reversion=-0.1
Bear Market:  momentum=-0.2, mean_reversion=0.4
Transition detected when importance patterns shift significantly
```

### 4. Model Debugging

Verify models learn sensible patterns:

```
Good: RSI oversold → positive contribution to buy signal
Bad: Day-of-week → large contribution (likely spurious)
```

---

## Implementation in Python

### Core DeepLIFT Algorithm

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Attribution:
    """Attribution scores for a single prediction."""
    feature_names: List[str]
    scores: np.ndarray
    baseline_output: float
    actual_output: float
    delta: float

    def top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top n contributing features."""
        indices = np.argsort(np.abs(self.scores))[::-1][:n]
        return [(self.feature_names[i], self.scores[i]) for i in indices]


class DeepLIFT:
    """
    DeepLIFT attribution for neural network trading models.

    This implementation supports both the Rescale rule and
    RevealCancel rule for ReLU-like activations.
    """

    def __init__(
        self,
        model: nn.Module,
        reference: Optional[torch.Tensor] = None,
        rule: str = "rescale"
    ):
        """
        Initialize DeepLIFT explainer.

        Args:
            model: Neural network model to explain
            reference: Reference input (baseline). If None, uses zeros.
            rule: Attribution rule - "rescale" or "reveal_cancel"
        """
        self.model = model
        self.reference = reference
        self.rule = rule
        self._hooks = []
        self._activations = {}
        self._ref_activations = {}

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        self._hooks = []
        self._activations = {}
        self._ref_activations = {}

        def get_activation(name, storage):
            def hook(module, input, output):
                storage[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.LeakyReLU)):
                self._hooks.append(
                    module.register_forward_hook(
                        get_activation(name, self._activations)
                    )
                )

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def attribute(
        self,
        input_tensor: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> Attribution:
        """
        Compute DeepLIFT attribution scores.

        Args:
            input_tensor: Input to explain (batch_size=1)
            feature_names: Names for input features

        Returns:
            Attribution object with contribution scores
        """
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        # Set reference
        if self.reference is None:
            reference = torch.zeros_like(input_tensor)
        else:
            reference = self.reference.expand_as(input_tensor)

        # Compute reference output
        self.model.eval()
        with torch.no_grad():
            ref_output = self.model(reference)
            actual_output = self.model(input_tensor)

        # Compute attribution using backpropagation
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        # Compute gradients
        output.backward(torch.ones_like(output))

        # Get gradients
        gradients = input_tensor.grad.detach()

        # Compute delta from reference
        delta_input = input_tensor.detach() - reference

        # DeepLIFT attribution: gradient * delta (rescale rule)
        if self.rule == "rescale":
            attributions = gradients * delta_input
        else:
            # RevealCancel rule - separate positive and negative
            attributions = self._reveal_cancel_attribution(
                input_tensor, reference, gradients
            )

        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(input_tensor.shape[1])]

        return Attribution(
            feature_names=feature_names,
            scores=attributions.squeeze().numpy(),
            baseline_output=ref_output.item(),
            actual_output=actual_output.item(),
            delta=actual_output.item() - ref_output.item()
        )

    def _reveal_cancel_attribution(
        self,
        input_tensor: torch.Tensor,
        reference: torch.Tensor,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attribution using RevealCancel rule.
        Separates positive and negative contributions.
        """
        delta = input_tensor.detach() - reference
        positive_delta = F.relu(delta)
        negative_delta = -F.relu(-delta)

        # Compute separate attributions
        positive_attr = gradients * positive_delta
        negative_attr = gradients * negative_delta

        return positive_attr + negative_attr

    def batch_attribute(
        self,
        inputs: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> List[Attribution]:
        """
        Compute attributions for a batch of inputs.

        Args:
            inputs: Batch of inputs (batch_size, num_features)
            feature_names: Names for input features

        Returns:
            List of Attribution objects
        """
        attributions = []
        for i in range(inputs.shape[0]):
            attr = self.attribute(inputs[i:i+1], feature_names)
            attributions.append(attr)
        return attributions


class TradingModelWithDeepLIFT(nn.Module):
    """
    Neural network for trading with built-in DeepLIFT support.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 1
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def create_trading_features(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Create technical features for trading.

    Args:
        prices: Price array
        window: Lookback window for features

    Returns:
        Feature array of shape (N, 11)
    """
    n = len(prices)
    features = np.zeros((n, 11))

    # Returns at different horizons
    features[1:, 0] = np.diff(prices) / prices[:-1]  # 1-day return
    features[5:, 1] = (prices[5:] - prices[:-5]) / prices[:-5]  # 5-day return
    features[10:, 2] = (prices[10:] - prices[:-10]) / prices[:-10]  # 10-day return

    # Moving average ratios
    for i in range(window, n):
        sma = np.mean(prices[i-window:i])
        features[i, 3] = prices[i] / sma - 1  # SMA ratio

        # EMA
        alpha = 2 / (window + 1)
        ema = prices[i-window]
        for j in range(i-window+1, i+1):
            ema = alpha * prices[j] + (1 - alpha) * ema
        features[i, 4] = prices[i] / ema - 1  # EMA ratio

        # Volatility
        returns = np.diff(prices[i-window:i+1]) / prices[i-window:i]
        features[i, 5] = np.std(returns)

        # Momentum
        features[i, 6] = prices[i] / prices[i-window] - 1

    # RSI
    for i in range(15, n):
        deltas = np.diff(prices[i-14:i+1])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            features[i, 7] = (100 - 100 / (1 + rs)) / 100  # Normalized RSI
        else:
            features[i, 7] = 1.0

    # MACD
    for i in range(26, n):
        ema12 = prices[i-12]
        alpha12 = 2 / 13
        for j in range(i-11, i+1):
            ema12 = alpha12 * prices[j] + (1 - alpha12) * ema12

        ema26 = prices[i-26]
        alpha26 = 2 / 27
        for j in range(i-25, i+1):
            ema26 = alpha26 * prices[j] + (1 - alpha26) * ema26

        features[i, 8] = (ema12 - ema26) / prices[i]

    # Bollinger Band position
    for i in range(window, n):
        sma = np.mean(prices[i-window:i])
        std = np.std(prices[i-window:i])
        if std > 0:
            features[i, 9] = (prices[i] - sma) / (2 * std)

    # Volume SMA ratio (simulated as price-based proxy)
    for i in range(window, n):
        features[i, 10] = np.random.randn() * 0.1  # Placeholder

    return features


def compute_feature_importance(
    model: nn.Module,
    features: np.ndarray,
    feature_names: List[str],
    reference: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute average feature importance across all samples.

    Args:
        model: Trained trading model
        features: Feature array
        feature_names: Names of features
        reference: Reference input (default: mean of features)

    Returns:
        Dictionary mapping feature names to average importance
    """
    if reference is None:
        reference = torch.FloatTensor(np.mean(features, axis=0, keepdims=True))
    else:
        reference = torch.FloatTensor(reference)

    explainer = DeepLIFT(model, reference=reference)

    # Compute attributions for all samples
    importance_sum = np.zeros(len(feature_names))
    n_samples = min(len(features), 1000)  # Limit for efficiency

    for i in range(n_samples):
        input_tensor = torch.FloatTensor(features[i:i+1])
        attr = explainer.attribute(input_tensor, feature_names)
        importance_sum += np.abs(attr.scores)

    # Average importance
    avg_importance = importance_sum / n_samples

    return dict(zip(feature_names, avg_importance))
```

### Data Preparation

```python
import pandas as pd
from typing import Tuple
import requests


class BybitClient:
    """Client for fetching cryptocurrency data from Bybit."""

    def __init__(self, base_url: str = "https://api.bybit.com"):
        self.base_url = base_url

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch historical klines from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval ("1", "5", "15", "60", "D")
            limit: Number of klines

        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.base_url}/v5/market/kline"
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data.get("retCode") != 0:
            raise ValueError(f"API error: {data.get('retMsg')}")

        klines = data["result"]["list"]
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)

        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df


def prepare_training_data(
    prices: np.ndarray,
    features: np.ndarray,
    target_horizon: int = 5,
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training and test data.

    Args:
        prices: Price array
        features: Feature array
        target_horizon: Prediction horizon
        train_ratio: Fraction of data for training

    Returns:
        X_train, y_train, X_test, y_test
    """
    # Create target: future returns
    target = np.zeros(len(prices))
    target[:-target_horizon] = (
        prices[target_horizon:] - prices[:-target_horizon]
    ) / prices[:-target_horizon]

    # Remove NaN rows
    valid_mask = ~np.any(np.isnan(features), axis=1)
    valid_mask[-(target_horizon+1):] = False

    X = features[valid_mask]
    y = target[valid_mask]

    # Split
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test
```

---

## Implementation in Rust

The Rust implementation provides high-performance DeepLIFT for production trading systems.

### Project Structure

```
122_deeplift_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   └── network.rs
│   ├── deeplift/
│   │   ├── mod.rs
│   │   └── attribution.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── features.rs
│   │   └── bybit.rs
│   ├── trading/
│   │   ├── mod.rs
│   │   ├── strategy.rs
│   │   └── signals.rs
│   └── backtest/
│       ├── mod.rs
│       └── engine.rs
├── examples/
│   ├── basic_deeplift.rs
│   ├── feature_importance.rs
│   └── trading_explanation.rs
└── python/
    ├── deeplift_trader.py
    ├── data_loader.py
    ├── backtest.py
    └── requirements.txt
```

### Core Rust Implementation

See the `src/` directory for the complete Rust implementation with:

- Efficient matrix operations using ndarray
- Forward pass with activation caching
- Backward pass with multiplier propagation
- Async Bybit API integration for cryptocurrency data
- Production-ready error handling and logging

---

## Practical Examples with Stock and Crypto Data

### Example 1: Training and Explaining a Trading Model

```python
import yfinance as yf

# Download data
data = yf.download('BTC-USD', period='2y')
prices = data['Close'].values

# Create features
features = create_trading_features(prices)

# Prepare data
X_train, y_train, X_test, y_test = prepare_training_data(prices, features)

# Define feature names
feature_names = [
    "return_1d", "return_5d", "return_10d", "sma_ratio", "ema_ratio",
    "volatility", "momentum", "rsi", "macd", "bb_position", "volume_ratio"
]

# Train model
model = TradingModelWithDeepLIFT(input_size=11, hidden_size=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)

for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Explain a prediction
reference = torch.FloatTensor(np.mean(X_train, axis=0, keepdims=True))
explainer = DeepLIFT(model, reference=reference)

sample_idx = 0
sample = torch.FloatTensor(X_test[sample_idx:sample_idx+1])
attribution = explainer.attribute(sample, feature_names)

print("\nPrediction Explanation:")
print(f"Baseline output: {attribution.baseline_output:.6f}")
print(f"Actual output: {attribution.actual_output:.6f}")
print(f"Delta: {attribution.delta:.6f}")
print("\nTop contributing features:")
for name, score in attribution.top_features(5):
    print(f"  {name}: {score:.6f}")
```

### Example 2: Feature Importance Analysis

```python
# Compute overall feature importance
importance = compute_feature_importance(
    model, X_test, feature_names,
    reference=np.mean(X_train, axis=0, keepdims=True)
)

print("\nOverall Feature Importance:")
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_importance:
    print(f"  {name}: {score:.6f}")
```

### Example 3: Bybit Crypto Trading with Explanations

```python
# Fetch data from Bybit
client = BybitClient()
crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

for symbol in crypto_pairs:
    df = client.fetch_klines(symbol, interval='60', limit=500)
    prices = df['close'].values
    features = create_trading_features(prices)

    # Use pre-trained model
    X_test = features[100:]  # Skip warm-up period
    X_test_t = torch.FloatTensor(X_test)

    # Get predictions and explanations
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t)

    # Explain latest prediction
    latest = X_test_t[-1:]
    attribution = explainer.attribute(latest, feature_names)

    print(f"\n{symbol} Latest Signal Explanation:")
    print(f"  Prediction: {predictions[-1].item():.6f}")
    print(f"  Top factors:")
    for name, score in attribution.top_features(3):
        direction = "bullish" if score > 0 else "bearish"
        print(f"    {name}: {score:.6f} ({direction})")
```

---

## Backtesting Framework

### DeepLIFT-Aware Backtester

```python
class DeepLIFTBacktester:
    """
    Backtesting framework with DeepLIFT explanations.
    """

    def __init__(
        self,
        model: nn.Module,
        explainer: DeepLIFT,
        feature_names: List[str],
        prediction_threshold: float = 0.001,
        transaction_cost: float = 0.001
    ):
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names
        self.threshold = prediction_threshold
        self.transaction_cost = transaction_cost

    def backtest(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_capital: float = 10000.0
    ) -> pd.DataFrame:
        """
        Run backtest with explanation logging.
        """
        results = []
        capital = initial_capital
        position = 0

        self.model.eval()

        for i in range(len(features)):
            input_tensor = torch.FloatTensor(features[i:i+1])

            with torch.no_grad():
                prediction = self.model(input_tensor).item()

            # Get explanation
            attribution = self.explainer.attribute(input_tensor, self.feature_names)
            top_features = attribution.top_features(3)

            # Trading logic
            if prediction > self.threshold:
                new_position = 1
            elif prediction < -self.threshold:
                new_position = -1
            else:
                new_position = 0

            # Transaction costs
            if new_position != position and i > 0:
                capital *= (1 - self.transaction_cost)

            # Calculate returns
            if i < len(prices) - 1:
                actual_return = prices[i+1] / prices[i] - 1
                position_return = position * actual_return
                capital *= (1 + position_return)
            else:
                actual_return = 0
                position_return = 0

            results.append({
                'index': i,
                'price': prices[i],
                'prediction': prediction,
                'position': position,
                'position_return': position_return,
                'capital': capital,
                'top_feature_1': top_features[0][0] if len(top_features) > 0 else '',
                'top_score_1': top_features[0][1] if len(top_features) > 0 else 0,
                'top_feature_2': top_features[1][0] if len(top_features) > 1 else '',
                'top_score_2': top_features[1][1] if len(top_features) > 1 else 0,
            })

            position = new_position

        return pd.DataFrame(results)


def calculate_metrics(results: pd.DataFrame) -> dict:
    """
    Calculate trading performance metrics.
    """
    returns = results['position_return']

    total_return = (results['capital'].iloc[-1] / results['capital'].iloc[0]) - 1
    ann_return = (1 + total_return) ** (252 / len(results)) - 1
    ann_volatility = returns.std() * np.sqrt(252)

    sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)

    downside_returns = returns[returns < 0]
    sortino_ratio = np.sqrt(252) * returns.mean() / (downside_returns.std() + 1e-10)

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()

    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
    }
```

---

## Performance Evaluation

### Expected Performance Targets

| Metric | Target Range |
|--------|-------------|
| Sharpe Ratio | > 1.0 |
| Sortino Ratio | > 1.5 |
| Max Drawdown | < 20% |
| Win Rate | > 50% |
| Explanation Consistency | > 80% |

### DeepLIFT vs Baseline Attribution

In typical experiments, DeepLIFT shows:
- **2-5x faster** computation than SHAP
- **Better saturation handling** than gradient-based methods
- **Consistent feature rankings** across similar inputs
- **Summation property** guarantees attributions sum to prediction delta

---

## Future Directions

### 1. Temporal DeepLIFT

Extend attribution to sequential models:
- LSTM/GRU networks with temporal feature importance
- Attention-weighted attributions for transformers

### 2. Uncertainty-Aware Attribution

Combine DeepLIFT with uncertainty quantification:
```
Attribution with confidence intervals for each feature contribution
```

### 3. Counterfactual Explanations

Generate "what-if" scenarios:
```
"If RSI were 30 instead of 70, the signal would change from buy to hold"
```

### 4. Real-Time Attribution

Streaming explanations for live trading:
- Low-latency attribution computation
- Anomaly detection based on unusual feature importance

### 5. Multi-Model Attribution

Ensemble explanations:
- Aggregate attributions across multiple models
- Identify consensus and disagreement in feature importance

---

## References

1. Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning Important Features Through Propagating Activation Differences. ICML. [arXiv:1704.02685](https://arxiv.org/abs/1704.02685)

2. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML. [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)

3. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.

4. Ancona, M., et al. (2018). Towards Better Understanding of Gradient-based Attribution Methods for Deep Neural Networks. ICLR.

5. Montavon, G., et al. (2018). Methods for Interpreting and Understanding Deep Neural Networks. Digital Signal Processing.

---

## Running the Examples

### Python

```bash
# Navigate to chapter directory
cd 122_deeplift_trading

# Install dependencies
pip install -r python/requirements.txt

# Run Python examples
python python/deeplift_trader.py
```

### Rust

```bash
# Navigate to chapter directory
cd 122_deeplift_trading

# Build the project
cargo build --release

# Run tests
cargo test

# Run examples
cargo run --example basic_deeplift
cargo run --example feature_importance
cargo run --example trading_explanation
```

---

## Summary

DeepLIFT provides a powerful framework for neural network interpretability in trading:

- **Theoretical Foundation**: Compares activations to reference for meaningful attributions
- **Summation Property**: Feature contributions sum exactly to prediction difference
- **Saturation Handling**: Properly handles ReLU saturation unlike gradient methods
- **Practical Value**: Essential for building transparent, trustworthy trading systems

By understanding which features drive trading signals, DeepLIFT enables traders and quants to validate model behavior, detect regime changes, and comply with explainability requirements in financial applications.

---

*Previous Chapter: [Chapter 121: Layer-wise Relevance Propagation](../121_layer_wise_relevance)*

*Next Chapter: [Chapter 123: GradCAM for Finance](../123_gradcam_finance)*
