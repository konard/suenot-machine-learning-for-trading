# Chapter 93: Multi-Task Learning for Trading

## Overview

Multi-Task Learning (MTL) is a machine learning paradigm where a single model is trained to solve multiple related tasks simultaneously, sharing representations across tasks to improve generalization. Introduced by Caruana (1997), MTL leverages inductive transfer between auxiliary tasks to learn richer feature representations than single-task models.

In algorithmic trading, MTL is a natural fit: financial data contains interconnected signals for return prediction, volatility estimation, trend classification, and volume forecasting. By jointly training on these tasks, MTL models discover shared structure in market data that single-task approaches miss.

## Table of Contents

1. [Introduction to Multi-Task Learning](#introduction-to-multi-task-learning)
2. [Mathematical Foundation](#mathematical-foundation)
3. [MTL Architectures](#mtl-architectures)
4. [MTL for Trading Applications](#mtl-for-trading-applications)
5. [Implementation in Python](#implementation-in-python)
6. [Implementation in Rust](#implementation-in-rust)
7. [Practical Examples with Stock and Crypto Data](#practical-examples-with-stock-and-crypto-data)
8. [Backtesting Framework](#backtesting-framework)
9. [Performance Evaluation](#performance-evaluation)
10. [Future Directions](#future-directions)

---

## Introduction to Multi-Task Learning

### What is Multi-Task Learning?

Multi-Task Learning trains a model on several related tasks at the same time, sharing learned representations. Rather than building separate models for each task (e.g., one for return prediction, another for volatility forecasting), MTL builds a single model that handles all tasks through a shared backbone and task-specific output heads.

### Key Insight

The fundamental premise is that tasks that share common underlying structure can benefit from being learned together. By training on multiple tasks:

- The model is forced to learn representations that generalize across tasks
- Auxiliary tasks act as a form of regularization, reducing overfitting
- Shared features capture more robust patterns in the data

### Why MTL for Trading?

Financial markets present compelling reasons for multi-task learning:

- **Correlated Signals**: Return, volatility, and volume share underlying market dynamics
- **Regularization**: Multiple objectives prevent the model from overfitting to noise in any single target
- **Efficiency**: One model handles multiple predictions, reducing inference cost
- **Cross-Asset Transfer**: Patterns learned from one asset class inform predictions for another
- **Richer Representations**: Shared features capture market microstructure that single-task models miss

---

## Mathematical Foundation

### The MTL Objective

Given K tasks with individual loss functions, the MTL objective is:

```
L_total = Σ_{k=1}^{K} w_k * L_k(θ_shared, θ_k)
```

Where:
- θ_shared: Parameters of the shared representation
- θ_k: Task-specific parameters for task k
- w_k: Weight for task k
- L_k: Loss function for task k

### Hard Parameter Sharing

In hard parameter sharing, all tasks share a common set of hidden layers (the shared backbone), with task-specific output heads branching off at the top:

```
Input → [Shared Layers] → Task 1 Head → Output 1
                        → Task 2 Head → Output 2
                        → Task 3 Head → Output 3
```

This is the most common MTL architecture and provides strong regularization because the shared layers must learn features useful for all tasks.

### Soft Parameter Sharing

In soft parameter sharing, each task has its own model, but a regularization term encourages parameters across models to be similar:

```
L_total = Σ_k L_k(θ_k) + λ * Σ_{i≠j} ||θ_i - θ_j||²
```

This is more flexible but introduces additional hyperparameters.

### Task Weighting Strategies

Balancing task losses is critical. Common approaches:

**1. Uncertainty Weighting (Kendall et al., 2018)**
```
L_total = Σ_k (1/(2σ_k²)) * L_k + log(σ_k)
```
Where σ_k is a learned uncertainty parameter for each task.

**2. GradNorm (Chen et al., 2018)**
Dynamically adjusts task weights to balance gradient magnitudes:
```
w_k(t+1) = w_k(t) * (||∇L_k|| / E[||∇L_k||])^α
```

**3. Dynamic Weight Average (Liu et al., 2019)**
Uses the rate of change of task losses to adjust weights:
```
w_k(t) = K * exp(r_k(t-1) / T) / Σ_j exp(r_j(t-1) / T)
```
Where r_k is the ratio of consecutive losses.

---

## MTL Architectures

### 1. Shared-Bottom Architecture

The simplest and most widely used MTL architecture:

```
Input Features
      │
  [Shared Encoder]
      │
  ┌───┼───┐
  │   │   │
[H1] [H2] [H3]    ← Task-specific heads
  │   │   │
 O1  O2  O3       ← Task outputs
```

### 2. Cross-Stitch Networks

Allow the model to learn linear combinations of task-specific features at each layer:

```
Task A Layer i    Task B Layer i
    │                  │
    └──── Cross ───────┘
          Stitch
    ┌──── Units ───────┐
    │                  │
Task A Layer i+1  Task B Layer i+1
```

### 3. Multi-Gate Mixture-of-Experts (MMoE)

Uses multiple expert sub-networks with task-specific gating:

```
Input → Expert 1 ─┐
Input → Expert 2 ─┼→ Gate(Task k) → Weighted Sum → Task k Head
Input → Expert 3 ─┘
```

### 4. Progressive Layered Extraction (PLE)

Extends MMoE with both shared and task-specific experts:

```
Shared Experts ─────┐
Task A Experts ──┐  ├→ Gate(A) → Task A
Task B Experts ──┤  │
                 └──┤
                    └→ Gate(B) → Task B
```

---

## MTL for Trading Applications

### Trading Tasks for MTL

A typical MTL trading model jointly predicts:

| Task | Type | Loss Function | Description |
|------|------|--------------|-------------|
| Return Prediction | Regression | MSE | Predict next-period returns |
| Direction Classification | Classification | BCE | Predict price direction (up/down) |
| Volatility Estimation | Regression | MSE | Estimate future volatility |
| Volume Prediction | Regression | MSE | Predict trading volume |

### Cross-Asset MTL

Train simultaneously across multiple assets:

```
Tasks = {
    (AAPL, return), (AAPL, volatility), (AAPL, direction),
    (MSFT, return), (MSFT, volatility), (MSFT, direction),
    (BTCUSDT, return), (BTCUSDT, volatility), (BTCUSDT, direction),
}
```

### Multi-Horizon MTL

Predict at multiple time horizons simultaneously:

```
Tasks = {
    1-hour return, 4-hour return, 1-day return,
    1-hour volatility, 4-hour volatility, 1-day volatility
}
```

---

## Implementation in Python

### Core Multi-Task Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class SharedEncoder(nn.Module):
    """Shared feature encoder for multi-task learning."""

    def __init__(self, input_size: int, hidden_sizes: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = h
        self.encoder = nn.Sequential(*layers)
        self.output_size = hidden_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TaskHead(nn.Module):
    """Task-specific output head."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 task_type: str = "regression"):
        super().__init__()
        self.task_type = task_type
        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(x)
        if self.task_type == "classification":
            out = torch.sigmoid(out)
        return out


class MultiTaskTradingModel(nn.Module):
    """
    Multi-Task Learning model for trading.

    Jointly predicts returns, volatility, direction, and volume
    through a shared encoder and task-specific heads.
    """

    def __init__(self, input_size: int, shared_hidden: List[int],
                 head_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = SharedEncoder(input_size, shared_hidden, dropout)
        enc_out = self.encoder.output_size

        self.task_heads = nn.ModuleDict({
            "return": TaskHead(enc_out, head_hidden, 1, "regression"),
            "volatility": TaskHead(enc_out, head_hidden, 1, "regression"),
            "direction": TaskHead(enc_out, head_hidden, 1, "classification"),
            "volume": TaskHead(enc_out, head_hidden, 1, "regression"),
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.encoder(x)
        return {name: head(shared) for name, head in self.task_heads.items()}


class UncertaintyWeighting(nn.Module):
    """Learnable task weighting via homoscedastic uncertainty."""

    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=losses[0].device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-2 * self.log_sigma[i])
            total += precision * loss + self.log_sigma[i]
        return total


class MTLTrainer:
    """
    Trainer for multi-task trading models.

    Supports uncertainty weighting and standard fixed-weight strategies.
    """

    def __init__(self, model: MultiTaskTradingModel, lr: float = 1e-3,
                 use_uncertainty_weighting: bool = True):
        self.model = model
        self.use_uncertainty = use_uncertainty_weighting

        params = list(model.parameters())
        if use_uncertainty_weighting:
            self.weighting = UncertaintyWeighting(len(model.task_heads))
            params += list(self.weighting.parameters())
        else:
            self.weighting = None

        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.loss_fns = {
            "return": nn.MSELoss(),
            "volatility": nn.MSELoss(),
            "direction": nn.BCELoss(),
            "volume": nn.MSELoss(),
        }

    def train_step(self, features: torch.Tensor,
                   targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        predictions = self.model(features)

        task_losses = []
        loss_values = {}
        for name in predictions:
            if name in targets:
                loss = self.loss_fns[name](predictions[name], targets[name])
                task_losses.append(loss)
                loss_values[name] = loss.item()

        if self.use_uncertainty and self.weighting is not None:
            total_loss = self.weighting(task_losses)
        else:
            total_loss = sum(task_losses)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_values["total"] = total_loss.item()
        return loss_values
```

### Data Preparation

```python
import pandas as pd
import requests

def create_mtl_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """Create technical features for multi-task learning."""
    features = pd.DataFrame(index=prices.index)

    features['return_1d'] = prices.pct_change(1)
    features['return_5d'] = prices.pct_change(5)
    features['return_10d'] = prices.pct_change(10)
    features['sma_ratio'] = prices / prices.rolling(window).mean()
    features['ema_ratio'] = prices / prices.ewm(span=window).mean()
    features['volatility'] = prices.pct_change().rolling(window).std()
    features['momentum'] = prices / prices.shift(window) - 1

    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    features['macd'] = (ema12 - ema26) / prices

    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    features['bb_position'] = (prices - sma) / (2 * std + 1e-10)

    return features.dropna()


def create_mtl_targets(prices: pd.Series, horizon: int = 5) -> pd.DataFrame:
    """Create multi-task targets."""
    targets = pd.DataFrame(index=prices.index)

    future_return = prices.pct_change(horizon).shift(-horizon)
    targets['return'] = future_return
    targets['direction'] = (future_return > 0).astype(float)
    targets['volatility'] = prices.pct_change().rolling(horizon).std().shift(-horizon)
    targets['volume'] = 0.0  # Placeholder; use actual volume in production

    return targets


def fetch_bybit_klines(symbol: str, interval: str = '60', limit: int = 1000) -> pd.DataFrame:
    """Fetch historical klines from Bybit."""
    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }
    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.set_index('timestamp').sort_index()

    return df
```

---

## Implementation in Rust

The Rust implementation provides high-performance multi-task learning for production trading systems.

### Project Structure

```
93_multi_task_learning_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   └── network.rs
│   ├── mtl/
│   │   ├── mod.rs
│   │   └── trainer.rs
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
│   ├── basic_mtl.rs
│   ├── multi_asset.rs
│   └── trading_strategy.rs
└── python/
    ├── __init__.py
    ├── mtl_model.py
    ├── data_loader.py
    ├── backtest.py
    └── requirements.txt
```

### Core Rust Implementation

See the `src/` directory for the complete Rust implementation with:

- Multi-task neural network with shared encoder and task-specific heads
- Uncertainty-based task weighting
- Async Bybit API integration for cryptocurrency data
- Feature engineering pipeline
- Backtesting engine with transaction cost modeling
- Production-ready error handling and logging

---

## Practical Examples with Stock and Crypto Data

### Example 1: Multi-Task Training on Stock Data

```python
import yfinance as yf

assets = {
    'AAPL': yf.download('AAPL', period='2y'),
    'MSFT': yf.download('MSFT', period='2y'),
    'GOOGL': yf.download('GOOGL', period='2y'),
    'BTC-USD': yf.download('BTC-USD', period='2y'),
    'ETH-USD': yf.download('ETH-USD', period='2y'),
}

# Prepare features and targets
all_features, all_targets = [], []
for name, df in assets.items():
    prices = df['Close']
    feats = create_mtl_features(prices)
    tgts = create_mtl_targets(prices)
    aligned = feats.join(tgts, how='inner').dropna()
    all_features.append(aligned[feats.columns])
    all_targets.append(aligned[tgts.columns])

features_df = pd.concat(all_features)
targets_df = pd.concat(all_targets)

# Convert to tensors
X = torch.FloatTensor(features_df.values)
y = {col: torch.FloatTensor(targets_df[col].values).unsqueeze(1) for col in targets_df.columns}

# Train
model = MultiTaskTradingModel(input_size=X.shape[1], shared_hidden=[64, 32])
trainer = MTLTrainer(model, lr=1e-3, use_uncertainty_weighting=True)

for epoch in range(200):
    losses = trainer.train_step(X, y)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: {losses}")
```

### Example 2: Bybit Crypto Multi-Task Training

```python
crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT']
crypto_features, crypto_targets = [], []

for symbol in crypto_pairs:
    df = fetch_bybit_klines(symbol)
    prices = df['close']
    feats = create_mtl_features(prices)
    tgts = create_mtl_targets(prices)
    aligned = feats.join(tgts, how='inner').dropna()
    crypto_features.append(aligned[feats.columns])
    crypto_targets.append(aligned[tgts.columns])

X_crypto = torch.FloatTensor(pd.concat(crypto_features).values)
y_crypto = {
    col: torch.FloatTensor(pd.concat(crypto_targets)[col].values).unsqueeze(1)
    for col in crypto_targets[0].columns
}

crypto_model = MultiTaskTradingModel(input_size=X_crypto.shape[1], shared_hidden=[128, 64])
crypto_trainer = MTLTrainer(crypto_model, lr=1e-3)

for epoch in range(300):
    losses = crypto_trainer.train_step(X_crypto, y_crypto)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: {losses}")
```

---

## Backtesting Framework

### MTL Backtester Implementation

```python
class MTLBacktester:
    """Backtesting framework for multi-task learning trading strategies."""

    def __init__(self, model: MultiTaskTradingModel,
                 prediction_threshold: float = 0.001,
                 transaction_cost: float = 0.001):
        self.model = model
        self.threshold = prediction_threshold
        self.transaction_cost = transaction_cost

    def backtest(self, prices: pd.Series, features: pd.DataFrame,
                 initial_capital: float = 10000.0) -> pd.DataFrame:
        self.model.eval()
        results = []
        capital = initial_capital
        position = 0

        feature_tensor = torch.FloatTensor(features.values)

        with torch.no_grad():
            predictions = self.model(feature_tensor)

        pred_return = predictions['return'].numpy().flatten()
        pred_direction = predictions['direction'].numpy().flatten()

        for i in range(len(features) - 1):
            # Combine return prediction and direction confidence
            signal = pred_return[i] * pred_direction[i]

            if signal > self.threshold:
                new_position = 1
            elif signal < -self.threshold:
                new_position = -1
            else:
                new_position = 0

            if new_position != position:
                capital *= (1 - self.transaction_cost)

            actual_return = prices.iloc[i + 1] / prices.iloc[i] - 1
            position_return = position * actual_return
            capital *= (1 + position_return)

            results.append({
                'date': features.index[i],
                'price': prices.iloc[i],
                'signal': signal,
                'position': position,
                'position_return': position_return,
                'capital': capital,
            })

            position = new_position

        return pd.DataFrame(results)


def calculate_metrics(results: pd.DataFrame) -> dict:
    """Calculate trading performance metrics."""
    returns = results['position_return']
    total_return = (results['capital'].iloc[-1] / results['capital'].iloc[0]) - 1
    ann_return = (1 + total_return) ** (252 / len(results)) - 1
    ann_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)

    downside = returns[returns < 0]
    sortino_ratio = np.sqrt(252) * returns.mean() / (downside.std() + 1e-10)

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    max_drawdown = (cumulative / rolling_max - 1).min()

    wins = (returns > 0).sum()
    losses_count = (returns < 0).sum()
    win_rate = wins / (wins + losses_count) if (wins + losses_count) > 0 else 0

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
        'num_trades': len(results[results['position'] != 0]),
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

### MTL vs Single-Task Comparison

In typical experiments, multi-task models show:

- **10-25% improvement** in Sharpe ratio over single-task baselines
- **Better calibration** of volatility estimates when jointly trained with return prediction
- **Reduced overfitting** due to the regularization effect of auxiliary tasks
- **More stable performance** across market regimes

---

## Future Directions

### 1. Mixture-of-Experts MTL

Use gating networks to dynamically route inputs to specialized expert sub-networks:

```
Input → Expert 1 ─┐
Input → Expert 2 ─┼→ Gate(Task k) → Weighted Sum → Task k
Input → Expert 3 ─┘
```

### 2. Hierarchical MTL

Organize tasks in a hierarchy reflecting their relationships:

- Level 1: Low-level features (returns, volume)
- Level 2: Mid-level tasks (volatility regimes, trend strength)
- Level 3: High-level decisions (position sizing, entry/exit signals)

### 3. Gradient Surgery

Resolve conflicting gradients between tasks by projecting conflicting gradients onto the normal plane of other task gradients (Yu et al., 2020).

### 4. Task Curriculum Learning

Gradually increase task complexity during training:

1. Start with simpler tasks (direction classification)
2. Introduce harder tasks (return prediction) after shared features stabilize
3. Fine-tune with all tasks jointly

### 5. Auxiliary Task Discovery

Automatically discover useful auxiliary tasks from the data through learned task construction.

---

## References

1. Caruana, R. (1997). Multitask Learning. Machine Learning, 28(1), 41-75.

2. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. CVPR.

3. Chen, Z., et al. (2018). GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. ICML.

4. Liu, S., Johns, E., & Davison, A. J. (2019). End-to-End Multi-Task Learning with Attention. CVPR.

5. Ma, J., et al. (2018). Modeling Task Relationships in Multi-Task Learning with Multi-Gate Mixture-of-Experts. KDD.

6. Yu, T., et al. (2020). Gradient Surgery for Multi-Task Learning. NeurIPS.

---

## Running the Examples

### Python

```bash
cd 93_multi_task_learning_trading

pip install -r python/requirements.txt

python python/mtl_model.py
```

### Rust

```bash
cd 93_multi_task_learning_trading

cargo build --release

cargo test

cargo run --example basic_mtl
cargo run --example multi_asset
cargo run --example trading_strategy
```

---

## Summary

Multi-Task Learning provides a powerful paradigm for trading model development:

- **Shared Representations**: A common encoder learns features useful across multiple trading objectives
- **Regularization**: Auxiliary tasks prevent overfitting to noise in any single target
- **Efficiency**: One model produces multiple predictions in a single forward pass
- **Robustness**: Joint training yields more stable performance across market conditions

By simultaneously learning to predict returns, volatility, direction, and volume, MTL models capture the interconnected nature of financial markets more effectively than isolated single-task approaches.

---

*Previous Chapter: [Chapter 92: Domain Adaptation for Finance](../92_domain_adaptation_finance)*

*Next Chapter: [Chapter 94: QuantNet Transfer Trading](../94_quantnet_transfer_trading)*
