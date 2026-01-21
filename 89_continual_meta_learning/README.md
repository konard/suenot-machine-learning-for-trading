# Chapter 89: Continual Meta-Learning for Algorithmic Trading

## Overview

Continual Meta-Learning (CML) combines the rapid adaptation capabilities of meta-learning with the ability to continuously learn from new experiences without forgetting previously acquired knowledge. In trading, this is crucial because markets evolve over time, and a model must adapt to new market regimes while retaining knowledge of historical patterns that may recur.

Unlike traditional meta-learning approaches that assume a fixed distribution of tasks, CML operates in a non-stationary environment where the task distribution itself changes over time. This makes it particularly suitable for financial markets, where market regimes (bull, bear, high volatility, low volatility) shift dynamically.

## Table of Contents

1. [Introduction to Continual Meta-Learning](#introduction-to-continual-meta-learning)
2. [The Challenge of Catastrophic Forgetting](#the-challenge-of-catastrophic-forgetting)
3. [Mathematical Foundation](#mathematical-foundation)
4. [CML Algorithms for Trading](#cml-algorithms-for-trading)
5. [Implementation in Python](#implementation-in-python)
6. [Implementation in Rust](#implementation-in-rust)
7. [Practical Examples with Stock and Crypto Data](#practical-examples-with-stock-and-crypto-data)
8. [Backtesting Framework](#backtesting-framework)
9. [Performance Evaluation](#performance-evaluation)
10. [Future Directions](#future-directions)

---

## Introduction to Continual Meta-Learning

### What is Continual Learning?

Continual Learning (CL), also known as Lifelong Learning, is the ability of a machine learning model to:

1. **Learn new tasks** sequentially over time
2. **Retain knowledge** from previously learned tasks
3. **Transfer knowledge** between related tasks

### What is Meta-Learning?

Meta-Learning, or "learning to learn," focuses on:

1. **Fast adaptation** to new tasks with limited data
2. **Learning a prior** that generalizes across tasks
3. **Task-agnostic representations** that transfer well

### Combining Both: Continual Meta-Learning

CML merges these paradigms to create models that:

- Quickly adapt to new market conditions (meta-learning)
- Remember past market regimes (continual learning)
- Transfer knowledge across different assets and time periods

### Why CML for Trading?

Financial markets present unique challenges:

1. **Non-Stationarity**: Market dynamics change over time
2. **Regime Shifts**: Bull/bear markets, volatility clusters
3. **Limited Data per Regime**: Each market condition has limited historical examples
4. **Recurring Patterns**: Similar market conditions may recur years apart

Traditional approaches fail because:
- Fine-tuning on new data causes forgetting of old patterns
- Static meta-learning assumes the task distribution doesn't change
- Pure continual learning doesn't provide fast adaptation

---

## The Challenge of Catastrophic Forgetting

### What is Catastrophic Forgetting?

When neural networks learn new tasks, they often "forget" previously learned tasks. This is called catastrophic forgetting.

```
Before: Model knows Tasks A, B, C
After learning Task D: Model only knows Task D well, forgets A, B, C!
```

### Why This Matters in Trading

Consider a model that has learned:
- Task A: Bull market patterns (2020-2021)
- Task B: Bear market patterns (2022)
- Task C: Sideways market patterns (2023)

When it learns Task D (new 2024 patterns), it might forget how to trade in bear markets - which could be catastrophic when bear markets return!

### Solutions: Three Approaches to CML

#### 1. Regularization-Based Methods

Add constraints to prevent parameter changes that would hurt past task performance.

**Elastic Weight Consolidation (EWC):**
```
L_total = L_new_task + λ Σᵢ Fᵢ (θᵢ - θ*ᵢ)²
```
Where Fᵢ is the Fisher information measuring parameter importance.

#### 2. Memory-Based Methods

Store examples from past tasks and replay them during new learning.

**Experience Replay:**
- Maintain a memory buffer of past experiences
- Interleave old and new data during training
- Can be combined with meta-learning objectives

#### 3. Architecture-Based Methods

Dedicate different network parts to different tasks.

**Progressive Neural Networks:**
- Add new columns for new tasks
- Keep old columns frozen
- Allow lateral connections for transfer

---

## Mathematical Foundation

### The CML Objective

We want to find parameters θ that minimize:

```
L(θ) = E_{τ~p(τ,t)} [L_τ(f_θ)]
```

Where p(τ,t) is the task distribution that changes over time t.

### Online Meta-Learning with Memory

Given:
- θ: Meta-learned initialization
- M: Memory buffer of past task data
- τ_new: New task

The CML update:

```
1. Inner loop (fast adaptation):
   θ'_new = θ - α ∇_θ L_τ_new(f_θ)

2. Memory replay:
   For τ_mem ~ M:
     θ'_mem = θ - α ∇_θ L_τ_mem(f_θ)

3. Meta update (with consolidation):
   θ ← θ + ε [ (θ'_new - θ) + β Σ_mem (θ'_mem - θ) + λ R(θ, θ_old) ]
```

Where:
- α: Inner learning rate
- ε: Meta learning rate
- β: Memory weight
- λ: Regularization strength
- R(θ, θ_old): Regularization term preventing forgetting

### Fisher Information for EWC

The Fisher Information Matrix approximates parameter importance:

```
F = E[(∇_θ log p(x|θ))²]
```

In practice, computed as:

```python
F_i = (1/N) Σ_n (∂L/∂θ_i)²
```

### Memory Selection Strategies

**Reservoir Sampling:** Maintain uniform distribution over all seen examples
**Gradient-Based:** Keep examples with highest gradient magnitude
**Diversity-Based:** Maximize coverage of feature space
**Loss-Based:** Keep examples model struggles with most

---

## CML Algorithms for Trading

### 1. Online Meta-Learning (OML)

A simple approach that uses recent tasks as the task distribution:

```python
def oml_update(model, new_data, memory_buffer):
    # Sample recent tasks from memory
    memory_tasks = sample_tasks(memory_buffer, k=4)

    # Combine with new task
    all_tasks = memory_tasks + [new_data]

    # Standard meta-learning update (e.g., Reptile)
    for task in all_tasks:
        adapted_params = inner_loop(model, task)
        update_meta_params(model, adapted_params)

    # Add new task to memory
    memory_buffer.add(new_data)
```

### 2. Meta-Continual Learning (Meta-CL)

Explicitly model task transitions:

```python
def meta_cl_update(model, task_t, task_t_minus_1):
    # Learn transition dynamics
    transition = learn_transition(task_t_minus_1, task_t)

    # Condition adaptation on transition
    adapted_params = conditioned_inner_loop(model, task_t, transition)

    # Update with transition-aware regularization
    update_with_transition(model, adapted_params, transition)
```

### 3. Gradient Episodic Memory (GEM) for Meta-Learning

Project gradients to avoid forgetting:

```python
def gem_meta_update(model, new_task, memory):
    # Compute gradient on new task
    g_new = compute_meta_gradient(model, new_task)

    # Compute gradients on memory tasks
    g_mem = [compute_meta_gradient(model, task) for task in memory]

    # Project g_new to not increase loss on memory tasks
    g_projected = project_gradient(g_new, g_mem)

    # Apply projected gradient
    model.params -= lr * g_projected
```

### 4. Elastic Meta-Learning (EML)

Combine EWC with meta-learning:

```python
def eml_update(model, tasks, fisher_info, importance_weight):
    # Standard meta-learning update
    meta_loss = compute_meta_loss(model, tasks)

    # Add EWC regularization
    ewc_loss = importance_weight * sum(
        fisher_info[p] * (model.params[p] - old_params[p])**2
        for p in model.params
    )

    # Combined update
    total_loss = meta_loss + ewc_loss
    total_loss.backward()
```

---

## Implementation in Python

### Core Continual Meta-Learner

```python
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import copy
import numpy as np
from collections import deque


class ContinualMetaLearner:
    """
    Continual Meta-Learning algorithm for trading strategy adaptation.

    Combines meta-learning's rapid adaptation with continual learning's
    ability to retain knowledge across changing market regimes.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        memory_size: int = 100,
        ewc_lambda: float = 0.4,
        replay_batch_size: int = 4
    ):
        """
        Initialize Continual Meta-Learner.

        Args:
            model: Neural network model for trading predictions
            inner_lr: Learning rate for task-specific adaptation
            outer_lr: Meta-learning rate
            inner_steps: Number of SGD steps per task
            memory_size: Maximum number of tasks to store in memory
            ewc_lambda: Strength of elastic weight consolidation
            replay_batch_size: Number of past tasks to replay per update
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.memory_size = memory_size
        self.ewc_lambda = ewc_lambda
        self.replay_batch_size = replay_batch_size

        # Memory buffer for past tasks
        self.memory_buffer: deque = deque(maxlen=memory_size)

        # Fisher information for EWC
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

        # Track market regimes
        self.regime_history: List[str] = []

        self.device = next(model.parameters()).device

    def compute_fisher_information(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor]]
    ):
        """
        Compute Fisher Information Matrix for EWC.

        The Fisher information approximates parameter importance by
        measuring the curvature of the loss surface.
        """
        self.model.eval()
        fisher = {name: torch.zeros_like(param)
                  for name, param in self.model.named_parameters()}

        for features, labels in tasks:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.model.zero_grad()
            predictions = self.model(features)
            loss = nn.MSELoss()(predictions, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

        # Average over tasks
        for name in fisher:
            fisher[name] /= len(tasks)
            # Online update: blend with previous Fisher info
            if name in self.fisher_info:
                fisher[name] = 0.5 * fisher[name] + 0.5 * self.fisher_info[name]

        self.fisher_info = fisher
        self.optimal_params = {name: param.clone()
                               for name, param in self.model.named_parameters()}

    def ewc_penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty to prevent forgetting.

        Returns:
            Penalty term based on Fisher-weighted parameter deviation
        """
        penalty = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.optimal_params:
                penalty += (self.fisher_info[name] *
                           (param - self.optimal_params[name]) ** 2).sum()

        return penalty

    def inner_loop(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        query_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[nn.Module, float]:
        """
        Perform task-specific adaptation (inner loop).

        Args:
            support_data: (features, labels) for adaptation
            query_data: (features, labels) for evaluation

        Returns:
            Adapted model and query loss
        """
        adapted_model = copy.deepcopy(self.model)
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr
        )

        features, labels = support_data
        features = features.to(self.device)
        labels = labels.to(self.device)

        adapted_model.train()
        for _ in range(self.inner_steps):
            inner_optimizer.zero_grad()
            predictions = adapted_model(features)
            loss = nn.MSELoss()(predictions, labels)
            loss.backward()
            inner_optimizer.step()

        # Evaluate on query set
        adapted_model.eval()
        with torch.no_grad():
            query_features, query_labels = query_data
            query_features = query_features.to(self.device)
            query_labels = query_labels.to(self.device)
            query_predictions = adapted_model(query_features)
            query_loss = nn.MSELoss()(query_predictions, query_labels).item()

        return adapted_model, query_loss

    def meta_train_step(
        self,
        new_task: Tuple[Tuple[torch.Tensor, torch.Tensor],
                        Tuple[torch.Tensor, torch.Tensor]],
        regime: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Perform one continual meta-training step.

        This combines:
        1. Meta-learning on the new task
        2. Experience replay from memory
        3. EWC regularization to prevent forgetting

        Args:
            new_task: (support_data, query_data) for new task
            regime: Optional market regime label

        Returns:
            Dictionary with loss metrics
        """
        # Store original parameters
        original_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }

        # Accumulate parameter updates
        param_updates = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }

        # Process new task
        support_data, query_data = new_task
        adapted_model, new_task_loss = self.inner_loop(support_data, query_data)

        with torch.no_grad():
            for (name, param), (_, adapted_param) in zip(
                self.model.named_parameters(),
                adapted_model.named_parameters()
            ):
                param_updates[name] += adapted_param - original_params[name]

        # Experience replay from memory
        replay_losses = []
        if len(self.memory_buffer) > 0:
            # Sample from memory
            replay_size = min(self.replay_batch_size, len(self.memory_buffer))
            replay_indices = np.random.choice(
                len(self.memory_buffer), replay_size, replace=False
            )

            for idx in replay_indices:
                # Reset to original parameters
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        param.copy_(original_params[name])

                mem_support, mem_query = self.memory_buffer[idx]
                adapted_model, replay_loss = self.inner_loop(mem_support, mem_query)
                replay_losses.append(replay_loss)

                with torch.no_grad():
                    for (name, param), (_, adapted_param) in zip(
                        self.model.named_parameters(),
                        adapted_model.named_parameters()
                    ):
                        param_updates[name] += adapted_param - original_params[name]

        # Apply meta update with EWC regularization
        total_tasks = 1 + len(replay_losses)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Meta update
                new_param = (original_params[name] +
                            self.outer_lr * param_updates[name] / total_tasks)

                # EWC regularization pull towards optimal params
                if name in self.optimal_params:
                    new_param = (new_param -
                                self.ewc_lambda * self.outer_lr *
                                (new_param - self.optimal_params[name]))

                param.copy_(new_param)

        # Add new task to memory
        self.memory_buffer.append(new_task)

        # Track regime if provided
        if regime:
            self.regime_history.append(regime)

        # Update Fisher information periodically
        if len(self.memory_buffer) % 10 == 0:
            recent_tasks = list(self.memory_buffer)[-20:]
            task_data = [(s[0], s[1]) for s, q in recent_tasks]
            self.compute_fisher_information(task_data)

        return {
            'new_task_loss': new_task_loss,
            'replay_loss': np.mean(replay_losses) if replay_losses else 0.0,
            'memory_size': len(self.memory_buffer)
        }

    def adapt(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        adaptation_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Adapt the meta-learned model to a new task.

        Args:
            support_data: Small amount of data from the new task
            adaptation_steps: Number of gradient steps

        Returns:
            Adapted model
        """
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps

        adapted_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        features, labels = support_data
        features = features.to(self.device)
        labels = labels.to(self.device)

        adapted_model.train()
        for _ in range(adaptation_steps):
            optimizer.zero_grad()
            predictions = adapted_model(features)
            loss = nn.MSELoss()(predictions, labels)
            loss.backward()
            optimizer.step()

        adapted_model.eval()
        return adapted_model

    def evaluate_forgetting(
        self,
        test_tasks: List[Tuple[Tuple[torch.Tensor, torch.Tensor],
                               Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Dict[str, float]:
        """
        Evaluate forgetting on held-out tasks from past regimes.

        Args:
            test_tasks: List of (support_data, query_data) tuples

        Returns:
            Dictionary with forgetting metrics
        """
        losses = []

        for support_data, query_data in test_tasks:
            adapted_model = self.adapt(support_data)

            with torch.no_grad():
                features, labels = query_data
                features = features.to(self.device)
                labels = labels.to(self.device)
                predictions = adapted_model(features)
                loss = nn.MSELoss()(predictions, labels).item()
                losses.append(loss)

        return {
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'max_loss': np.max(losses),
            'min_loss': np.min(losses)
        }


class TradingModel(nn.Module):
    """
    Neural network for trading signal prediction.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

### Data Preparation for Market Regimes

```python
import pandas as pd
from typing import Generator

def detect_market_regime(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Detect market regime based on price dynamics.

    Returns:
        Series with regime labels: 'bull', 'bear', 'high_vol', 'low_vol'
    """
    returns = prices.pct_change()
    rolling_return = returns.rolling(window).mean()
    rolling_vol = returns.rolling(window).std()

    # Define thresholds
    vol_median = rolling_vol.median()

    regimes = pd.Series(index=prices.index, dtype=str)

    for i in range(window, len(prices)):
        ret = rolling_return.iloc[i]
        vol = rolling_vol.iloc[i]

        if vol > vol_median * 1.5:
            regimes.iloc[i] = 'high_vol'
        elif vol < vol_median * 0.5:
            regimes.iloc[i] = 'low_vol'
        elif ret > 0:
            regimes.iloc[i] = 'bull'
        else:
            regimes.iloc[i] = 'bear'

    return regimes


def create_trading_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Create technical features for trading.
    """
    features = pd.DataFrame(index=prices.index)

    # Returns at different horizons
    features['return_1d'] = prices.pct_change(1)
    features['return_5d'] = prices.pct_change(5)
    features['return_10d'] = prices.pct_change(10)

    # Moving averages
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
    features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    # Normalize RSI to [-1, 1]
    features['rsi'] = (features['rsi'] - 50) / 50

    return features.dropna()


def create_regime_tasks(
    prices: pd.Series,
    features: pd.DataFrame,
    regimes: pd.Series,
    support_size: int = 20,
    query_size: int = 10,
    target_horizon: int = 5
) -> Generator:
    """
    Generate tasks organized by market regime.

    Yields tasks that can be used for continual meta-learning,
    with regime labels for tracking.
    """
    target = prices.pct_change(target_horizon).shift(-target_horizon)
    aligned = features.join(target.rename('target')).join(regimes.rename('regime')).dropna()

    feature_cols = [c for c in aligned.columns if c not in ['target', 'regime']]

    while True:
        # Sample a regime
        valid_regimes = aligned['regime'].dropna().unique()
        if len(valid_regimes) == 0:
            continue

        regime = np.random.choice(valid_regimes)
        regime_data = aligned[aligned['regime'] == regime]

        total_needed = support_size + query_size
        if len(regime_data) < total_needed:
            continue

        # Sample contiguous window from this regime
        start_idx = np.random.randint(0, len(regime_data) - total_needed)
        window = regime_data.iloc[start_idx:start_idx + total_needed]

        support_df = window.iloc[:support_size]
        query_df = window.iloc[support_size:]

        support_features = torch.FloatTensor(support_df[feature_cols].values)
        support_labels = torch.FloatTensor(support_df['target'].values).unsqueeze(1)

        query_features = torch.FloatTensor(query_df[feature_cols].values)
        query_labels = torch.FloatTensor(query_df['target'].values).unsqueeze(1)

        yield ((support_features, support_labels),
               (query_features, query_labels)), regime
```

---

## Implementation in Rust

The Rust implementation provides high-performance trading signal generation suitable for production environments.

### Project Structure

```
89_continual_meta_learning/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   └── network.rs
│   ├── continual/
│   │   ├── mod.rs
│   │   ├── memory.rs
│   │   ├── ewc.rs
│   │   └── learner.rs
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
│   ├── basic_cml.rs
│   ├── regime_adaptation.rs
│   └── trading_strategy.rs
└── python/
    ├── continual_meta_learner.py
    ├── data_loader.py
    └── backtest.py
```

### Core Rust Implementation

See the `src/` directory for the complete Rust implementation with:

- High-performance matrix operations
- Memory-efficient experience replay
- Thread-safe Fisher information computation
- Async data fetching from Bybit
- Production-ready error handling

---

## Practical Examples with Stock and Crypto Data

### Example 1: Continual Learning Across Market Regimes

```python
import yfinance as yf

# Download data
btc = yf.download('BTC-USD', period='3y')
prices = btc['Close']
features = create_trading_features(prices)
regimes = detect_market_regime(prices)

# Initialize CML
model = TradingModel(input_size=8)
cml = ContinualMetaLearner(
    model=model,
    inner_lr=0.01,
    outer_lr=0.001,
    inner_steps=5,
    memory_size=100,
    ewc_lambda=0.4,
    replay_batch_size=4
)

# Create task generator
task_gen = create_regime_tasks(prices, features, regimes)

# Continual meta-training
for epoch in range(500):
    (support, query), regime = next(task_gen)

    metrics = cml.meta_train_step(
        new_task=(support, query),
        regime=regime
    )

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Regime: {regime}, "
              f"New Loss: {metrics['new_task_loss']:.6f}, "
              f"Replay Loss: {metrics['replay_loss']:.6f}, "
              f"Memory: {metrics['memory_size']}")
```

### Example 2: Evaluating Forgetting

```python
# Create test tasks from different regimes
test_tasks_by_regime = {}
for regime in ['bull', 'bear', 'high_vol', 'low_vol']:
    regime_data = aligned[aligned['regime'] == regime]
    if len(regime_data) >= 30:
        idx = np.random.randint(0, len(regime_data) - 30)
        window = regime_data.iloc[idx:idx + 30]

        support = (
            torch.FloatTensor(window.iloc[:20][feature_cols].values),
            torch.FloatTensor(window.iloc[:20]['target'].values).unsqueeze(1)
        )
        query = (
            torch.FloatTensor(window.iloc[20:][feature_cols].values),
            torch.FloatTensor(window.iloc[20:]['target'].values).unsqueeze(1)
        )
        test_tasks_by_regime[regime] = (support, query)

# Evaluate forgetting
for regime, task in test_tasks_by_regime.items():
    metrics = cml.evaluate_forgetting([task])
    print(f"Regime {regime}: Loss = {metrics['mean_loss']:.6f}")
```

### Example 3: Bybit Crypto Trading with CML

```python
import requests

def fetch_bybit_klines(symbol: str, interval: str = '1h', limit: int = 1000):
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

# Fetch multiple crypto assets
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT']
crypto_data = {}

for symbol in symbols:
    df = fetch_bybit_klines(symbol)
    prices = df['close']
    features = create_trading_features(prices)
    regimes = detect_market_regime(prices)
    crypto_data[symbol] = (prices, features, regimes)

# Train CML across multiple crypto assets
for symbol, (prices, features, regimes) in crypto_data.items():
    task_gen = create_regime_tasks(prices, features, regimes)

    for _ in range(100):
        (support, query), regime = next(task_gen)
        cml.meta_train_step(
            new_task=(support, query),
            regime=f"{symbol}_{regime}"
        )

    print(f"Completed training on {symbol}")
```

---

## Backtesting Framework

### CML Backtester

```python
class CMLBacktester:
    """
    Backtesting framework for Continual Meta-Learning strategies.
    """

    def __init__(
        self,
        cml: ContinualMetaLearner,
        adaptation_window: int = 20,
        adaptation_steps: int = 5,
        prediction_threshold: float = 0.001,
        retraining_frequency: int = 20
    ):
        self.cml = cml
        self.adaptation_window = adaptation_window
        self.adaptation_steps = adaptation_steps
        self.threshold = prediction_threshold
        self.retraining_frequency = retraining_frequency

    def backtest(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        regimes: pd.Series,
        initial_capital: float = 10000.0
    ) -> pd.DataFrame:
        """
        Run backtest with continual meta-learning.

        The model adapts quickly to new data while retaining
        knowledge of past market regimes.
        """
        results = []
        capital = initial_capital
        position = 0

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
            adapted = self.cml.adapt(
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
                new_position = 1
            elif prediction < -self.threshold:
                new_position = -1
            else:
                new_position = 0

            # Calculate returns
            actual_return = prices.iloc[i+1] / prices.iloc[i] - 1
            position_return = position * actual_return
            capital *= (1 + position_return)

            # Periodic retraining with continual learning
            if i % self.retraining_frequency == 0:
                support = (adapt_features[:-1], adapt_returns[:-1])
                query = (adapt_features[-5:], adapt_returns[-5:])
                regime = regimes.iloc[i] if not pd.isna(regimes.iloc[i]) else 'unknown'
                self.cml.meta_train_step((support, query), regime=regime)

            results.append({
                'date': features.index[i],
                'price': prices.iloc[i],
                'prediction': prediction,
                'actual_return': actual_return,
                'position': position,
                'position_return': position_return,
                'capital': capital,
                'regime': regimes.iloc[i] if not pd.isna(regimes.iloc[i]) else 'unknown'
            })

            position = new_position

        return pd.DataFrame(results)
```

---

## Performance Evaluation

### Key Metrics

```python
def calculate_metrics(results: pd.DataFrame) -> dict:
    """
    Calculate trading performance metrics.
    """
    returns = results['position_return']

    # Basic metrics
    total_return = (results['capital'].iloc[-1] / results['capital'].iloc[0]) - 1

    # Risk-adjusted metrics
    sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
    sortino_ratio = np.sqrt(252) * returns.mean() / (returns[returns < 0].std() + 1e-8)

    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Win rate
    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    # Regime-specific performance
    regime_performance = {}
    for regime in results['regime'].unique():
        regime_data = results[results['regime'] == regime]
        regime_returns = regime_data['position_return']
        regime_performance[regime] = {
            'sharpe': np.sqrt(252) * regime_returns.mean() / (regime_returns.std() + 1e-8),
            'return': regime_returns.sum()
        }

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(results[results['position'] != 0]),
        'regime_performance': regime_performance
    }


def evaluate_forgetting_over_time(
    cml: ContinualMetaLearner,
    test_tasks_by_regime: dict,
    training_epochs: int = 100
) -> pd.DataFrame:
    """
    Track forgetting metrics over training time.
    """
    forgetting_history = []

    for epoch in range(training_epochs):
        # ... training step ...

        # Evaluate on all regime test tasks
        for regime, task in test_tasks_by_regime.items():
            metrics = cml.evaluate_forgetting([task])
            forgetting_history.append({
                'epoch': epoch,
                'regime': regime,
                'loss': metrics['mean_loss']
            })

    return pd.DataFrame(forgetting_history)
```

### Expected Performance

| Metric | Standard Meta-Learning | Continual Meta-Learning |
|--------|----------------------|------------------------|
| Sharpe Ratio | > 1.0 | > 1.0 |
| Regime Forgetting | High | Low |
| Adaptation Speed | Fast | Fast |
| Memory Efficiency | O(1) | O(memory_size) |
| Max Drawdown | < 20% | < 18% |

The key advantage of CML is maintaining consistent performance across different market regimes, even when trained sequentially.

---

## Future Directions

### 1. Task-Free Continual Meta-Learning

Automatically detect task boundaries without explicit regime labels:

```python
def detect_task_shift(features: torch.Tensor, threshold: float = 0.5):
    """Detect when distribution has shifted enough to define new task."""
    # Use statistical tests or representation-based detection
    pass
```

### 2. Hierarchical Memory

Organize memory by regime type for more efficient replay:

```python
class HierarchicalMemory:
    def __init__(self):
        self.regime_memories = {
            'bull': deque(maxlen=50),
            'bear': deque(maxlen=50),
            'high_vol': deque(maxlen=50),
            'low_vol': deque(maxlen=50)
        }
```

### 3. Meta-Learning the Forgetting Rate

Learn when to forget vs. when to remember:

```python
class AdaptiveEWC:
    def __init__(self):
        self.lambda_network = nn.Linear(feature_size, 1)

    def compute_lambda(self, task_features):
        """Learn optimal forgetting rate based on task similarity."""
        return torch.sigmoid(self.lambda_network(task_features))
```

### 4. Multi-Timescale Learning

Different adaptation speeds for different types of changes:

- Fast: Daily market noise
- Medium: Weekly regime shifts
- Slow: Long-term market structure changes

---

## References

1. Javed, K., & White, M. (2019). Meta-Learning Representations for Continual Learning. NeurIPS.
2. Riemer, M., et al. (2019). Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference. ICLR.
3. Kirkpatrick, J., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks. PNAS.
4. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML.
5. Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning. NeurIPS.
6. Parisi, G. I., et al. (2019). Continual Lifelong Learning with Neural Networks: A Review. Neural Networks.

---

## Running the Examples

### Python

```bash
# Navigate to chapter directory
cd 89_continual_meta_learning

# Install dependencies
pip install torch numpy pandas yfinance scikit-learn requests

# Run Python examples
python python/continual_meta_learner.py
```

### Rust

```bash
# Navigate to chapter directory
cd 89_continual_meta_learning

# Build the project
cargo build --release

# Run examples
cargo run --example basic_cml
cargo run --example regime_adaptation
cargo run --example trading_strategy
```

---

## Summary

Continual Meta-Learning addresses a critical challenge in algorithmic trading: how to adapt quickly to new market conditions while retaining valuable knowledge from past experiences.

Key benefits:

- **Fast Adaptation**: Learn new market regimes with minimal data
- **No Forgetting**: Retain knowledge of past regimes that may recur
- **Transfer Learning**: Knowledge transfers across assets and time periods
- **Regime Awareness**: Explicitly models market regime transitions

By combining meta-learning's rapid adaptation with continual learning's memory preservation, CML provides a robust framework for building trading systems that improve over time without losing valuable historical knowledge.

---

*Previous Chapter: [Chapter 88: Meta-RL Trading](../88_meta_rl_trading)*

*Next Chapter: [Chapter 90: Meta-Gradient Optimization](../90_meta_gradient_optimization)*
