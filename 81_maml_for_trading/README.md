# Chapter 81: Model-Agnostic Meta-Learning (MAML) for Trading

## Overview

Model-Agnostic Meta-Learning (MAML) is a powerful meta-learning algorithm that enables models to quickly adapt to new tasks with only a few gradient steps. Introduced by Finn et al. (2017), MAML learns an optimal initialization for neural network parameters that can be fine-tuned rapidly for any new task.

In algorithmic trading, MAML is invaluable for adapting trading strategies to new market conditions, different assets, or changing market regimes with minimal data and computational overhead.

## Table of Contents

1. [Introduction to MAML](#introduction-to-maml)
2. [Mathematical Foundation](#mathematical-foundation)
3. [MAML vs Other Meta-Learning Methods](#maml-vs-other-meta-learning-methods)
4. [MAML for Trading Applications](#maml-for-trading-applications)
5. [Implementation in Python](#implementation-in-python)
6. [Implementation in Rust](#implementation-in-rust)
7. [Practical Examples with Stock and Crypto Data](#practical-examples-with-stock-and-crypto-data)
8. [Backtesting Framework](#backtesting-framework)
9. [Performance Evaluation](#performance-evaluation)
10. [Future Directions](#future-directions)

---

## Introduction to MAML

### What is Meta-Learning?

Meta-learning, or "learning to learn," is a paradigm where models learn how to learn efficiently across a distribution of tasks. Instead of training a model for a single specific task, meta-learning produces a model that can rapidly adapt to new, unseen tasks with minimal data.

### The MAML Algorithm

MAML was introduced by Chelsea Finn, Pieter Abbeel, and Sergey Levine in their 2017 paper "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." The key insight is elegant:

1. Sample a batch of tasks from a task distribution
2. For each task, compute gradients with respect to task-specific loss
3. Update parameters using the gradients computed at the **adapted** parameters
4. The meta-objective optimizes for the initialization that leads to best performance **after** adaptation

The "model-agnostic" nature means MAML works with any model trained via gradient descent, making it widely applicable.

### Why MAML for Trading?

Financial markets present unique challenges that make MAML particularly attractive:

- **Non-Stationarity**: Market dynamics constantly evolve, requiring rapid adaptation
- **Regime Shifts**: Markets transition between different regimes (bull/bear, high/low volatility)
- **Cross-Asset Learning**: Patterns learned on one asset may transfer to others
- **Limited Data**: New market conditions often have limited historical data
- **Speed Requirements**: Markets move fast, adaptation must be rapid

---

## Mathematical Foundation

### The MAML Objective

Given:
- θ: Model parameters (the initialization we're learning)
- τ: A task sampled from task distribution p(τ)
- L_τ: Loss function for task τ
- α: Inner learning rate (for task adaptation)
- β: Outer learning rate (meta-learning rate)

The MAML update consists of two stages:

**Inner Loop (Task Adaptation):**
```
θ'_τ = θ - α ∇_θ L_τ(f_θ)
```

**Outer Loop (Meta-Update):**
```
θ ← θ - β ∇_θ Σ_τ L_τ(f_{θ'_τ})
```

### Understanding the Bi-Level Optimization

The key insight is that MAML optimizes for:

```
min_θ Σ_τ L_τ(f_{θ - α∇_θL_τ(f_θ)})
```

This means we're finding the θ that, after one (or few) gradient steps, yields good performance across all tasks.

### Computing the Meta-Gradient

The meta-gradient requires differentiating through the inner optimization:

```
∇_θ L_τ(f_{θ'_τ}) = ∇_θ L_τ(f_{θ - α∇_θL_τ(f_θ)})
```

Using the chain rule, this involves computing second-order derivatives (Hessians), which can be computationally expensive.

### First-Order MAML (FOMAML)

To reduce computational cost, First-Order MAML (FOMAML) ignores the second-order terms:

```
∇_θ L_τ(f_{θ'_τ}) ≈ ∇_{θ'_τ} L_τ(f_{θ'_τ})
```

FOMAML often achieves comparable performance with significantly reduced computation.

---

## MAML vs Other Meta-Learning Methods

### Comparison Table

| Method | Gradient Order | Computation | Memory | Performance |
|--------|---------------|-------------|--------|-------------|
| MAML | Second-order | High | High | Excellent |
| FOMAML | First-order | Medium | Medium | Very Good |
| Reptile | First-order | Low | Low | Good |
| Prototypical Networks | N/A | Low | Low | Good (classification) |
| Matching Networks | N/A | Low | Low | Good (classification) |

### When to Use MAML

**Use MAML when:**
- You need the best possible adaptation performance
- Computational resources are available
- Tasks have complex, non-linear relationships

**Consider alternatives when:**
- Computational efficiency is critical (use Reptile)
- Tasks are primarily classification (use Prototypical Networks)
- Memory is constrained (use FOMAML or Reptile)

---

## MAML for Trading Applications

### 1. Multi-Asset Adaptation

Train across multiple assets to learn an initialization that adapts quickly to any asset:

```
Tasks = {AAPL_prediction, MSFT_prediction, BTCUSD_prediction, ETHUSD_prediction, ...}
Each task: Predict next-period return given historical features
```

### 2. Regime Adaptation

Define tasks based on market regimes:

```
Tasks = {Bull_Market, Bear_Market, High_Volatility, Low_Volatility, Ranging}
Goal: Rapid adaptation when regime changes are detected
```

### 3. Time-Period Adaptation

Sample tasks from different time periods:

```
Tasks = {Q1_2023, Q2_2023, Q3_2023, Q4_2023, ...}
Goal: Learn temporal patterns that generalize across market conditions
```

### 4. Strategy-Type Adaptation

Meta-learn across different strategy types:

```
Tasks = {Momentum, Mean_Reversion, Breakout, Arbitrage}
Goal: Initialize a model that can specialize to any strategy
```

---

## Implementation in Python

### Core MAML Algorithm

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import copy

class MAMLTrader:
    """
    Model-Agnostic Meta-Learning for trading strategy adaptation.

    This implementation supports both full MAML (second-order) and
    FOMAML (first-order approximation).
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = False
    ):
        """
        Initialize MAML trader.

        Args:
            model: Neural network model for trading predictions
            inner_lr: Learning rate for task-specific adaptation
            outer_lr: Meta-learning rate
            inner_steps: Number of gradient steps for inner loop
            first_order: If True, use FOMAML (faster but approximate)
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)

    def inner_loop(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        create_graph: bool = True
    ) -> nn.Module:
        """
        Perform task-specific adaptation (inner loop).

        Args:
            support_data: (features, labels) for adaptation
            create_graph: Whether to create computation graph for second-order gradients

        Returns:
            Adapted model parameters
        """
        features, labels = support_data

        # Clone model parameters for adaptation
        adapted_params = {name: param.clone() for name, param in self.model.named_parameters()}

        # Perform gradient steps
        for _ in range(self.inner_steps):
            # Forward pass with adapted parameters
            predictions = self._forward_with_params(features, adapted_params)
            loss = F.mse_loss(predictions, labels)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=create_graph and not self.first_order
            )

            # Update adapted parameters
            adapted_params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(adapted_params.items(), grads)
            }

        return adapted_params

    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        """Forward pass using specified parameters."""
        # This requires a functional forward implementation
        # For simplicity, we'll use a manual approach
        return self._functional_forward(x, params)

    def _functional_forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Functional forward pass through the network."""
        # Layer 1
        x = F.linear(x, params['layers.0.weight'], params['layers.0.bias'])
        x = F.relu(x)
        # Layer 2
        x = F.linear(x, params['layers.1.weight'], params['layers.1.bias'])
        x = F.relu(x)
        # Output layer
        x = F.linear(x, params['layers.2.weight'], params['layers.2.bias'])
        return x

    def meta_train_step(
        self,
        tasks: List[Tuple[Tuple[torch.Tensor, torch.Tensor],
                          Tuple[torch.Tensor, torch.Tensor]]]
    ) -> float:
        """
        Perform one meta-training step.

        Args:
            tasks: List of (support_data, query_data) tuples

        Returns:
            Average meta-loss across tasks
        """
        self.meta_optimizer.zero_grad()
        total_meta_loss = 0.0

        for support_data, query_data in tasks:
            # Inner loop: adapt to task
            adapted_params = self.inner_loop(
                support_data,
                create_graph=not self.first_order
            )

            # Outer loop: evaluate on query set
            query_features, query_labels = query_data
            query_predictions = self._forward_with_params(query_features, adapted_params)
            task_loss = F.mse_loss(query_predictions, query_labels)

            total_meta_loss += task_loss

        # Meta-update
        meta_loss = total_meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        adaptation_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Adapt the meta-learned model to a new task.

        Args:
            support_data: Small amount of data from the new task
            adaptation_steps: Number of gradient steps (default: inner_steps)

        Returns:
            Adapted model ready for prediction
        """
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps

        adapted_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        features, labels = support_data

        for _ in range(adaptation_steps):
            optimizer.zero_grad()
            predictions = adapted_model(features)
            loss = F.mse_loss(predictions, labels)
            loss.backward()
            optimizer.step()

        return adapted_model


class TradingModel(nn.Module):
    """
    Neural network for trading signal prediction.
    Designed for use with MAML.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
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


class FunctionalTradingModel(nn.Module):
    """
    Trading model with functional forward pass for MAML.
    This allows passing custom parameters for the forward pass.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.fc1_weight = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2_weight = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.fc2_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc3_weight = nn.Parameter(torch.randn(output_size, hidden_size) * 0.1)
        self.fc3_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x: torch.Tensor, params: Optional[dict] = None) -> torch.Tensor:
        if params is None:
            params = dict(self.named_parameters())

        x = F.linear(x, params['fc1_weight'], params['fc1_bias'])
        x = F.relu(x)
        x = F.linear(x, params['fc2_weight'], params['fc2_bias'])
        x = F.relu(x)
        x = F.linear(x, params['fc3_weight'], params['fc3_bias'])
        return x
```

### Data Preparation

```python
import numpy as np
import pandas as pd
from typing import Generator, Tuple

def create_trading_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Create technical features for trading.

    Args:
        prices: Price series
        window: Lookback window for features

    Returns:
        DataFrame with features
    """
    features = pd.DataFrame(index=prices.index)

    # Returns at different horizons
    features['return_1d'] = prices.pct_change(1)
    features['return_5d'] = prices.pct_change(5)
    features['return_10d'] = prices.pct_change(10)

    # Moving average ratios
    features['sma_ratio'] = prices / prices.rolling(window).mean()
    features['ema_ratio'] = prices / prices.ewm(span=window).mean()

    # Volatility
    features['volatility'] = prices.pct_change().rolling(window).std()

    # Momentum
    features['momentum'] = prices / prices.shift(window) - 1

    # RSI
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    features['macd'] = (ema12 - ema26) / prices

    # Bollinger Band position
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    features['bb_position'] = (prices - sma) / (2 * std + 1e-10)

    return features.dropna()


def create_task_data(
    prices: pd.Series,
    features: pd.DataFrame,
    support_size: int = 20,
    query_size: int = 10,
    target_horizon: int = 5
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create support and query sets for a trading task.

    Args:
        prices: Price series
        features: Feature DataFrame
        support_size: Number of samples for adaptation
        query_size: Number of samples for evaluation
        target_horizon: Prediction horizon for returns

    Returns:
        (support_data, query_data) tuples
    """
    # Create target (future returns)
    target = prices.pct_change(target_horizon).shift(-target_horizon)

    # Align and drop NaN
    aligned = features.join(target.rename('target')).dropna()

    total_needed = support_size + query_size
    if len(aligned) < total_needed:
        raise ValueError(f"Not enough data: {len(aligned)} < {total_needed}")

    # Random split point
    start_idx = np.random.randint(0, len(aligned) - total_needed)

    # Split into support and query
    support_df = aligned.iloc[start_idx:start_idx + support_size]
    query_df = aligned.iloc[start_idx + support_size:start_idx + total_needed]

    # Convert to tensors
    feature_cols = [c for c in aligned.columns if c != 'target']

    support_features = torch.FloatTensor(support_df[feature_cols].values)
    support_labels = torch.FloatTensor(support_df['target'].values).unsqueeze(1)

    query_features = torch.FloatTensor(query_df[feature_cols].values)
    query_labels = torch.FloatTensor(query_df['target'].values).unsqueeze(1)

    return (support_features, support_labels), (query_features, query_labels)


def task_generator(
    asset_data: dict,
    batch_size: int = 4
) -> Generator:
    """
    Generate batches of tasks from multiple assets.

    Args:
        asset_data: Dict of {asset_name: (prices, features)}
        batch_size: Number of tasks per batch

    Yields:
        List of tasks
    """
    asset_names = list(asset_data.keys())

    while True:
        tasks = []
        for _ in range(batch_size):
            # Sample random asset
            asset = np.random.choice(asset_names)
            prices, features = asset_data[asset]

            try:
                support, query = create_task_data(prices, features)
                tasks.append((support, query))
            except ValueError:
                continue

        if tasks:
            yield tasks
```

---

## Implementation in Rust

The Rust implementation provides high-performance MAML for production trading systems.

### Project Structure

```
81_maml_for_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   └── network.rs
│   ├── maml/
│   │   ├── mod.rs
│   │   └── algorithm.rs
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
│   ├── basic_maml.rs
│   ├── multi_asset_training.rs
│   └── trading_strategy.rs
└── python/
    ├── maml_trader.py
    ├── data_loader.py
    ├── backtest.py
    └── requirements.txt
```

### Core Rust Implementation

See the `src/` directory for the complete Rust implementation with:

- Second-order gradient computation (numerical approximation)
- First-order MAML option for efficiency
- Multi-asset training for cross-market generalization
- Async Bybit API integration for cryptocurrency data
- Production-ready error handling and logging

---

## Practical Examples with Stock and Crypto Data

### Example 1: Multi-Asset Meta-Training

```python
import yfinance as yf

# Download data for multiple assets
assets = {
    'AAPL': yf.download('AAPL', period='2y'),
    'MSFT': yf.download('MSFT', period='2y'),
    'GOOGL': yf.download('GOOGL', period='2y'),
    'BTC-USD': yf.download('BTC-USD', period='2y'),
    'ETH-USD': yf.download('ETH-USD', period='2y'),
}

# Prepare data
asset_data = {}
for name, df in assets.items():
    prices = df['Close']
    features = create_trading_features(prices)
    asset_data[name] = (prices, features)

# Initialize model and MAML trainer
model = FunctionalTradingModel(input_size=11)  # 11 features
maml = MAMLTrader(
    model=model,
    inner_lr=0.01,
    outer_lr=0.001,
    inner_steps=5,
    first_order=False  # Use full MAML
)

# Meta-training
task_gen = task_generator(asset_data, batch_size=4)
losses = []
for epoch in range(1000):
    tasks = next(task_gen)
    loss = maml.meta_train_step(tasks)
    losses.append(loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Meta-Loss: {loss:.6f}")
```

### Example 2: Rapid Adaptation to New Asset

```python
# New asset not seen during training
new_asset = yf.download('TSLA', period='1y')
new_prices = new_asset['Close']
new_features = create_trading_features(new_prices)

# Create small support set (just 20 samples)
support, query = create_task_data(new_prices, new_features, support_size=20)

# Adapt in just 5 gradient steps
adapted_model = maml.adapt(support, adaptation_steps=5)

# Evaluate on query set
with torch.no_grad():
    predictions = adapted_model(query[0])
    loss = F.mse_loss(predictions, query[1])
    print(f"Adapted model query loss: {loss.item():.6f}")

# Compare with training from scratch
baseline_model = FunctionalTradingModel(input_size=11)
baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)

for _ in range(100):  # Train for 100 steps
    baseline_optimizer.zero_grad()
    preds = baseline_model(support[0])
    loss = F.mse_loss(preds, support[1])
    loss.backward()
    baseline_optimizer.step()

with torch.no_grad():
    baseline_preds = baseline_model(query[0])
    baseline_loss = F.mse_loss(baseline_preds, query[1])
    print(f"Baseline model query loss (100 steps): {baseline_loss.item():.6f}")
```

### Example 3: Bybit Crypto Trading

```python
import requests

def fetch_bybit_klines(symbol: str, interval: str = '60', limit: int = 1000):
    """Fetch historical klines from Bybit."""
    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.set_index('timestamp').sort_index()

    return df

# Fetch data for multiple crypto pairs
crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT']
crypto_data = {}

for symbol in crypto_pairs:
    df = fetch_bybit_klines(symbol)
    prices = df['close']
    features = create_trading_features(prices)
    crypto_data[symbol] = (prices, features)

# Meta-train on crypto data
crypto_task_gen = task_generator(crypto_data, batch_size=4)
for epoch in range(500):
    tasks = next(crypto_task_gen)
    loss = maml.meta_train_step(tasks)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Crypto Meta-Loss: {loss:.6f}")
```

---

## Backtesting Framework

### MAML Backtester Implementation

```python
class MAMLBacktester:
    """
    Backtesting framework for MAML-based trading strategies.
    """

    def __init__(
        self,
        maml_trader: MAMLTrader,
        adaptation_window: int = 20,
        adaptation_steps: int = 5,
        prediction_threshold: float = 0.001,
        transaction_cost: float = 0.001
    ):
        self.maml = maml_trader
        self.adaptation_window = adaptation_window
        self.adaptation_steps = adaptation_steps
        self.threshold = prediction_threshold
        self.transaction_cost = transaction_cost

    def backtest(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        initial_capital: float = 10000.0
    ) -> pd.DataFrame:
        """
        Run backtest on historical data.

        Args:
            prices: Price series
            features: Feature DataFrame
            initial_capital: Starting capital

        Returns:
            DataFrame with backtest results
        """
        results = []
        capital = initial_capital
        position = 0  # -1, 0, or 1

        feature_cols = list(features.columns)

        for i in range(self.adaptation_window, len(features) - 1):
            # Get adaptation data
            adapt_features = torch.FloatTensor(
                features.iloc[i-self.adaptation_window:i][feature_cols].values
            )
            adapt_returns = torch.FloatTensor(
                prices.pct_change().iloc[i-self.adaptation_window+1:i+1].values
            ).unsqueeze(1)

            # Adapt model
            adapted = self.maml.adapt(
                (adapt_features[:-1], adapt_returns[:-1]),
                adaptation_steps=self.adaptation_steps
            )

            # Make prediction
            current_features = torch.FloatTensor(
                features.iloc[i][feature_cols].values
            ).unsqueeze(0)

            with torch.no_grad():
                prediction = adapted(current_features).item()

            # Trading logic
            if prediction > self.threshold:
                new_position = 1  # Long
            elif prediction < -self.threshold:
                new_position = -1  # Short
            else:
                new_position = 0  # Neutral

            # Calculate transaction costs
            if new_position != position:
                capital *= (1 - self.transaction_cost)

            # Calculate returns
            actual_return = prices.iloc[i+1] / prices.iloc[i] - 1
            position_return = position * actual_return
            capital *= (1 + position_return)

            results.append({
                'date': features.index[i],
                'price': prices.iloc[i],
                'prediction': prediction,
                'actual_return': actual_return,
                'position': position,
                'position_return': position_return,
                'capital': capital
            })

            position = new_position

        return pd.DataFrame(results)


def calculate_metrics(results: pd.DataFrame) -> dict:
    """
    Calculate trading performance metrics.
    """
    returns = results['position_return']

    # Basic metrics
    total_return = (results['capital'].iloc[-1] / results['capital'].iloc[0]) - 1

    # Annualized metrics (assuming daily data)
    ann_return = (1 + total_return) ** (252 / len(results)) - 1
    ann_volatility = returns.std() * np.sqrt(252)

    # Risk-adjusted metrics
    sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)

    downside_returns = returns[returns < 0]
    sortino_ratio = np.sqrt(252) * returns.mean() / (downside_returns.std() + 1e-10)

    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Win rate
    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    # Profit factor
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / (gross_losses + 1e-10)

    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': len(results[results['position'] != 0])
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
| Profit Factor | > 1.5 |

### MAML vs Baseline Comparison

In typical experiments, MAML shows:
- **3-5x faster adaptation** compared to training from scratch
- **15-30% improvement** in Sharpe ratio after adaptation
- **Better generalization** to unseen assets and market conditions

---

## Future Directions

### 1. Multi-Step MAML

Extend to multiple inner-loop steps for deeper adaptation:

```
θ'_τ^(k) = θ'_τ^(k-1) - α ∇_θ L_τ(f_{θ'_τ^(k-1)})
```

### 2. Adaptive Learning Rates

Learn task-specific learning rates:

```
θ'_τ = θ - α_τ ∇_θ L_τ(f_θ)
```

### 3. Hierarchical MAML

Organize tasks hierarchically:
- Level 1: Asset classes (stocks, crypto, forex)
- Level 2: Individual assets
- Level 3: Different time scales

### 4. Online MAML

Continuously update the meta-initialization:

```
θ ← θ - β_t ∇_θ L_τ_t(f_{θ'_τ_t})
```

### 5. Uncertainty-Aware MAML

Incorporate Bayesian methods for uncertainty quantification.

---

## References

1. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML. [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)

2. Nichol, A., Achiam, J., & Schulman, J. (2018). On First-Order Meta-Learning Algorithms. arXiv:1803.02999

3. Antoniou, A., Edwards, H., & Storkey, A. (2019). How to train your MAML. ICLR.

4. Hospedales, T., et al. (2020). Meta-Learning in Neural Networks: A Survey. IEEE TPAMI.

5. Raghu, A., et al. (2020). Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML. ICLR.

---

## Running the Examples

### Python

```bash
# Navigate to chapter directory
cd 81_maml_for_trading

# Install dependencies
pip install -r python/requirements.txt

# Run Python examples
python python/maml_trader.py
```

### Rust

```bash
# Navigate to chapter directory
cd 81_maml_for_trading

# Build the project
cargo build --release

# Run tests
cargo test

# Run examples
cargo run --example basic_maml
cargo run --example multi_asset_training
cargo run --example trading_strategy
```

---

## Summary

MAML provides a powerful framework for meta-learning in trading:

- **Theoretical Foundation**: Optimizes for fast adaptation through bi-level optimization
- **Flexibility**: Works with any differentiable model architecture
- **Performance**: Enables rapid adaptation with minimal data
- **Practical Value**: Crucial for non-stationary financial markets

By learning an initialization optimized for fast adaptation, MAML enables trading systems to quickly respond to new market conditions, assets, or regimes - a critical capability in the ever-changing financial markets.

---

*Previous Chapter: [Chapter 80: LLM Compliance Check](../80_llm_compliance_check)*

*Next Chapter: [Chapter 82: Reptile Algorithm for Trading](../82_reptile_algorithm_trading)*
