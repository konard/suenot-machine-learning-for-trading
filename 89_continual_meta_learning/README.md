# Chapter 89: Continual Meta-Learning for Trading

## Overview

Continual Meta-Learning combines two powerful paradigms — **meta-learning** (learning to learn) and **continual learning** (learning without forgetting) — to create trading systems that can rapidly adapt to new market regimes while preserving knowledge of previously seen conditions.

Financial markets undergo **regime changes** — transitions between bull markets, bear markets, high volatility periods, and consolidation phases. A standard meta-learning approach (like MAML) can adapt quickly to a new regime, but when retrained on new data, it suffers from **catastrophic forgetting**: it loses its ability to handle older regimes. Continual meta-learning solves this by combining MAML with techniques like **Elastic Weight Consolidation (EWC)** and **experience replay**.

## Key Concepts

### 1. MAML (Model-Agnostic Meta-Learning)

MAML learns an initialization of model parameters that enables fast adaptation to new tasks with only a few gradient steps.

**Bi-level optimization:**

```
Outer loop: θ ← θ - β ∇_θ Σ_i L(f_{θ'_i}, D^query_i)
Inner loop: θ'_i = θ - α ∇_θ L(f_θ, D^support_i)
```

Where:
- `θ` = meta-parameters (shared initialization)
- `α` = inner learning rate (task adaptation)
- `β` = outer learning rate (meta-update)
- `D^support` = support set (for adaptation)
- `D^query` = query set (for evaluation)

### 2. Elastic Weight Consolidation (EWC)

EWC prevents catastrophic forgetting by adding a regularization term that penalizes changes to parameters that were important for previously learned tasks.

```
L_total = L_new + (λ/2) Σ_i F_i (θ_i - θ*_i)²
```

Where:
- `F_i` = Fisher Information for parameter `i` (importance weight)
- `θ*_i` = optimal parameter from previous training
- `λ` = regularization strength

The Fisher Information Matrix approximates how sensitive the loss is to each parameter, identifying which parameters are critical for past tasks.

### 3. Experience Replay

A replay buffer stores representative tasks from past market regimes. During meta-training on a new regime, tasks from the buffer are mixed in to reinforce old knowledge.

### 4. Combined Algorithm

```
For each new market regime:
    1. Create meta-learning tasks from the new regime data
    2. For each meta-training epoch:
        a. Sample replay tasks from the buffer
        b. Combine new + replay tasks
        c. Compute MAML loss on combined task batch
        d. Add EWC penalty to protect important parameters
        e. Update meta-parameters
    3. Update Fisher Information Matrix with new + old data
    4. Store representative tasks in the replay buffer
```

## Trading Application

### Market Regimes as Tasks

In the trading context:
- **Task** = a specific market condition (bull, bear, sideways, volatile, etc.)
- **Support set** = recent historical data for adaptation
- **Query set** = upcoming data for evaluation
- **Regime** = a collection of similar market conditions over a time period

### Sequential Regime Learning

Markets evolve through regimes over time:

```
Time →
[Bull Market] → [Correction] → [Sideways] → [Bear Market] → [Recovery]
   Regime 0       Regime 1       Regime 2       Regime 3      Regime 4
```

The continual meta-learner processes these sequentially:
1. Learns to trade in bull markets
2. Learns corrections **without forgetting** bull market strategies
3. Learns sideways markets while retaining both prior regimes
4. And so on...

### Advantages Over Standard MAML

| Feature | Standard MAML | Continual MAML |
|---------|--------------|----------------|
| Fast adaptation | Yes | Yes |
| Multi-regime | Requires all data | Sequential learning |
| Memory efficient | Stores all data | Replay buffer only |
| Forgetting | Catastrophic | Controlled (EWC + replay) |
| Online learning | Limited | Natural fit |

## Implementation

### Python

The Python implementation uses PyTorch and provides three main modules:

#### Core Module: `continual_meta_learner.py`

```python
from continual_meta_learner import ContinualMAMLTrainer, TradingModel

# Create model and trainer
model = TradingModel(input_size=11, hidden_size=64)
trainer = ContinualMAMLTrainer(
    model,
    inner_lr=0.01,       # Task adaptation rate
    outer_lr=0.001,      # Meta-learning rate
    inner_steps=5,       # Adaptation gradient steps
    first_order=True,    # Use FOMAML for stability
    ewc_lambda=100.0,    # EWC regularization strength
    replay_buffer_size=100,
    replay_ratio=0.5     # 50% replay tasks mixed in
)

# Learn regimes sequentially
for regime_id, tasks in enumerate(regime_task_list):
    losses = trainer.learn_regime(tasks, regime_id, num_epochs=20)

# Adapt to current market
adapted_model = trainer.adapt(recent_features, recent_returns)
prediction = adapted_model.predict(current_features)
```

**Key classes:**
- `TradingModel` — 3-layer neural network (ReLU + ReLU + Tanh)
- `ContinualMAMLTrainer` — MAML + EWC + replay training loop
- `EWC` — Elastic Weight Consolidation regularizer
- `ReplayBuffer` — Experience storage with regime-balanced sampling
- `TradingStrategy` — Signal generation with risk management
- `TaskData` — Support/query set container with regime ID

#### Data Module: `data_loader.py`

```python
from data_loader import BybitClient, FeatureGenerator, SimulatedDataGenerator

# Fetch real data
client = BybitClient()
klines = client.fetch_klines("BTCUSDT", interval="60", limit=500)

# Or simulate for testing
klines = SimulatedDataGenerator.generate_regime_changing_klines(1000)
regimes = SimulatedDataGenerator.generate_sequential_regimes(200)

# Compute features (11 technical indicators)
feature_gen = FeatureGenerator(window=20)
features = feature_gen.compute_features(klines)
```

**11 Technical Features:**
1. Returns (1-day, 5-day, 10-day)
2. SMA ratio (price / SMA-20)
3. EMA ratio (price / EMA-20)
4. Rolling volatility (20-period)
5. Momentum (20-period)
6. RSI (Relative Strength Index)
7. MACD (normalized)
8. Bollinger Band position
9. Volume SMA ratio

#### Backtesting Module: `backtest.py`

```python
from backtest import BacktestEngine, BacktestConfig

config = BacktestConfig(
    initial_capital=10000.0,
    transaction_cost=0.001,
    slippage=0.0005,
    threshold=0.001,
    adaptation_window=30,
    adaptation_steps=5
)

engine = BacktestEngine(config)
results = engine.run(trainer, test_klines)
print(results.summary())
```

**Metrics computed:**
- Total return, annualized return
- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Win rate, profit factor
- Full trade history and equity curve

### Rust

The Rust implementation mirrors the Python structure with performance optimizations:

```rust
use continual_meta_learning::{
    ContinualMAMLTrainer, TradingModel, FeatureGenerator,
    BacktestEngine,
};

// Create model and trainer
let model = TradingModel::new(11, 64, 1);
let mut trainer = ContinualMAMLTrainer::new(
    model, 0.01, 0.001, 5, true, 100.0, 50,
);

// Learn regime
let losses = trainer.learn_regime(&tasks, regime_id, 20, 5);

// Adapt and predict
let adapted = trainer.adapt(&features, &labels, Some(5));
let prediction = adapted.predict(&current_features);
```

**Module structure:**
- `model::network` — Neural network with numerical gradient computation
- `continual::algorithm` — ContinualMAMLTrainer with EWC and replay
- `data::bybit` — Async Bybit API client
- `data::features` — Technical indicator computation
- `trading::strategy` — Signal generation and risk management
- `backtest::engine` — Historical simulation engine

## Project Structure

```
89_continual_meta_learning/
├── README.md                  # This file
├── README.ru.md               # Russian translation
├── README.specify.md          # Technical specification
├── readme.simple.md           # Simplified explanation (English)
├── readme.simple.ru.md        # Simplified explanation (Russian)
├── Cargo.toml                 # Rust project configuration
├── python/
│   ├── __init__.py
│   ├── continual_meta_learner.py  # Core algorithm
│   ├── data_loader.py             # Data loading & features
│   ├── backtest.py                # Backtesting framework
│   └── requirements.txt           # Python dependencies
├── src/
│   ├── lib.rs                     # Crate root
│   ├── model/
│   │   ├── mod.rs
│   │   └── network.rs             # Neural network
│   ├── continual/
│   │   ├── mod.rs
│   │   └── algorithm.rs           # Continual MAML + EWC
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs               # Bybit API client
│   │   └── features.rs            # Feature engineering
│   ├── trading/
│   │   ├── mod.rs
│   │   ├── strategy.rs            # Trading strategy
│   │   └── signals.rs             # Signal types
│   └── backtest/
│       ├── mod.rs
│       └── engine.rs              # Backtest engine
└── examples/
    ├── basic_continual.rs         # Basic continual learning
    ├── regime_learning.rs         # Sequential regime acquisition
    └── trading_strategy.rs        # Full trading example
```

## Running the Code

### Python

```bash
cd 89_continual_meta_learning/python
pip install -r requirements.txt

# Run the main example
python continual_meta_learner.py

# Run the backtest
python backtest.py
```

### Rust

```bash
cd 89_continual_meta_learning

# Run examples
cargo run --example basic_continual
cargo run --example regime_learning
cargo run --example trading_strategy

# Run tests
cargo test
```

## Performance Evaluation

### Forgetting Metrics

After learning N regimes sequentially, evaluate on each regime:

| Regime | After Learning R0 | After R0+R1 | After R0+R1+R2 |
|--------|-------------------|-------------|-----------------|
| Bull   | Best              | Slight drop | Maintained      |
| Bear   | N/A               | Best        | Maintained      |
| Sideways | N/A             | N/A         | Best            |

The key metric is how much performance drops on old regimes (forgetting), measured by:
- **Average forgetting** = mean accuracy drop across old regimes
- **Backward transfer** = performance change on old regimes after learning new ones

### Trading Metrics

- **Sharpe Ratio** — Risk-adjusted return (target > 1.0)
- **Sortino Ratio** — Downside-risk-adjusted return
- **Maximum Drawdown** — Largest peak-to-trough decline
- **Win Rate** — Percentage of profitable trades
- **Profit Factor** — Gross profit / gross loss

## Hyperparameter Guide

| Parameter | Typical Range | Description |
|-----------|--------------|-------------|
| `inner_lr` | 0.001 - 0.05 | Task adaptation speed |
| `outer_lr` | 0.0001 - 0.005 | Meta-learning speed |
| `inner_steps` | 3 - 10 | Adaptation gradient steps |
| `ewc_lambda` | 10 - 1000 | Forgetting prevention strength |
| `replay_buffer_size` | 20 - 200 | Past task storage capacity |
| `replay_ratio` | 0.3 - 0.7 | Proportion of replay tasks |
| `hidden_size` | 32 - 128 | Network capacity |

## References

1. **Finn, C., Abbeel, P., & Levine, S.** (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML.

2. **Kirkpatrick, J., et al.** (2017). "Overcoming Catastrophic Forgetting in Neural Networks." PNAS.

3. **Javed, K. & White, M.** (2019). "Meta-Learning Representations for Continual Learning." NeurIPS.

4. **Riemer, M., et al.** (2019). "Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference." ICLR.

5. **Caccia, M., et al.** (2020). "Online Fast Adaptation and Knowledge Accumulation: a New Approach to Continual Learning." NeurIPS.

## Data Sources

- **Bybit Exchange API** — Cryptocurrency spot/derivatives market data
- **Simulated Data** — Built-in generators for testing with configurable regimes
- **Yahoo Finance** — Traditional stock market data (via yfinance)

## Future Directions

- **Online regime detection** — Automatically identify regime changes
- **Multi-asset continual learning** — Share knowledge across assets
- **Attention-based replay** — Prioritize important past experiences
- **Progressive networks** — Expand model capacity for new regimes
- **Meta-continual RL** — Apply to reinforcement learning trading agents
