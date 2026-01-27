# Chapter 88: Meta-Reinforcement Learning (Meta-RL) for Trading

## Overview

Meta-Reinforcement Learning (Meta-RL) combines meta-learning with reinforcement learning to create trading agents that can rapidly adapt to new market environments. Unlike standard RL, which requires extensive retraining when market conditions change, Meta-RL learns a learning algorithm itself - an agent that can quickly discover effective trading policies in novel market regimes with minimal interaction.

This chapter implements Meta-RL for adaptive trading agents using the RL^2 (Learning to Reinforcement Learn) framework, where an RNN-based agent is trained across a distribution of trading environments and learns to adapt its behavior within its hidden state.

## Table of Contents

1. [Introduction to Meta-RL](#introduction-to-meta-rl)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Meta-RL vs Standard RL and Meta-Learning](#meta-rl-vs-standard-rl-and-meta-learning)
4. [Meta-RL for Trading Applications](#meta-rl-for-trading-applications)
5. [Implementation in Python](#implementation-in-python)
6. [Implementation in Rust](#implementation-in-rust)
7. [Practical Examples with Stock and Crypto Data](#practical-examples-with-stock-and-crypto-data)
8. [Backtesting Framework](#backtesting-framework)
9. [Performance Evaluation](#performance-evaluation)
10. [Future Directions](#future-directions)

---

## Introduction to Meta-RL

### What is Meta-Reinforcement Learning?

Meta-Reinforcement Learning sits at the intersection of two powerful paradigms:

- **Meta-Learning**: Learning how to learn efficiently across a distribution of tasks
- **Reinforcement Learning**: Learning to make sequential decisions through interaction with an environment

Meta-RL produces an agent that, when placed in a new environment, can rapidly discover an effective policy without extensive retraining. The agent's "learning algorithm" is encoded in its recurrent hidden state or through learned optimization procedures.

### Key Approaches to Meta-RL

There are three main families of Meta-RL algorithms:

1. **Recurrent Meta-RL (RL^2)**: Uses an RNN whose hidden state encodes the learning algorithm. The agent learns to adapt by processing sequences of (state, action, reward, done) tuples across episodes.

2. **Gradient-Based Meta-RL (MAML-RL)**: Applies MAML to RL policy gradients, learning an initialization that adapts quickly with a few policy gradient steps.

3. **Context-Based Meta-RL**: Learns a latent context variable that captures task-specific information and conditions the policy.

This chapter focuses primarily on the RL^2 approach with elements of context-based methods, as they are most practical for trading applications.

### Why Meta-RL for Trading?

Trading presents challenges that make Meta-RL particularly compelling:

- **Non-Stationary Environments**: Market dynamics shift continuously, invalidating fixed policies
- **Regime Changes**: Bull/bear markets, volatility regimes, and structural breaks require rapid policy adaptation
- **Multi-Market Deployment**: A single agent must handle diverse assets with different statistical properties
- **Sample Efficiency**: Real trading data is limited; the agent must learn quickly from few interactions
- **Exploration vs Exploitation**: The agent must balance gathering market information with executing profitable trades

---

## Mathematical Foundation

### The Meta-RL Objective

In Meta-RL, we optimize over a distribution of MDPs (Markov Decision Processes):

**Standard RL Objective:**
```
max_π E_τ~π [Σ_t γ^t r_t]
```

**Meta-RL Objective:**
```
max_θ E_{M~p(M)} E_{τ~π_θ(M)} [Σ_t γ^t r_t]
```

Where:
- θ: Meta-learned parameters
- M: A specific MDP (market environment) sampled from distribution p(M)
- π_θ: Policy parameterized by θ
- γ: Discount factor
- r_t: Reward at time t

### RL^2: Learning to Reinforcement Learn

In the RL^2 framework (Duan et al., 2016; Wang et al., 2016), the agent is an RNN that receives the previous action, reward, and termination signal as additional inputs:

**Input at each step:**
```
x_t = [s_t, a_{t-1}, r_{t-1}, d_{t-1}]
```

**Hidden state update (GRU):**
```
h_t = GRU(x_t, h_{t-1})
```

**Policy and value outputs:**
```
π(a_t | s_t, h_t) = softmax(W_π h_t + b_π)
V(s_t, h_t) = W_v h_t + b_v
```

The key insight is that the RNN's hidden state h_t implicitly encodes a learning algorithm. Over multiple episodes in the same environment, the hidden state accumulates information about the task and adapts the policy accordingly.

### Training with PPO

The meta-policy is trained using Proximal Policy Optimization (PPO) across sampled environments:

**PPO Clipped Objective:**
```
L^CLIP(θ) = E_t [min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]
```

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio
- A_t is the advantage estimate (computed via GAE)
- ε is the clipping parameter

### Generalized Advantage Estimation (GAE)

```
A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

---

## Meta-RL vs Standard RL and Meta-Learning

### Comparison Table

| Aspect | Standard RL | Meta-Learning (MAML) | Meta-RL |
|--------|------------|---------------------|---------|
| Adaptation | None (fixed policy) | Few gradient steps | Within-episode (hidden state) |
| Task definition | Single MDP | Supervised tasks | Distribution of MDPs |
| Optimization | Policy gradient / Q-learning | Bi-level optimization | Policy gradient over MDP distribution |
| Speed of adaptation | Requires retraining | 3-5 gradient steps | Immediate (no gradient steps) |
| Exploration | Fixed strategy | N/A | Learned exploration |
| Sequential decisions | Yes | No | Yes |

### When to Use Meta-RL

**Use Meta-RL when:**
- The environment changes frequently and policies must adapt in real-time
- You need an agent that explores intelligently in new environments
- Sequential decision-making is important (position sizing, entry/exit timing)
- You want a unified agent that handles multiple market regimes

**Consider alternatives when:**
- Market conditions are relatively stable (use standard RL)
- You only need point predictions, not sequential decisions (use MAML)
- Computational resources are very limited (use simpler adaptive methods)

---

## Meta-RL for Trading Applications

### 1. Regime-Adaptive Trading Agent

The agent encounters different market regimes as separate MDPs:

```
MDPs = {Bull_Market_MDP, Bear_Market_MDP, High_Volatility_MDP, Sideways_MDP}
State: [price_features, technical_indicators, position, portfolio_value]
Actions: {Buy, Sell, Hold}
Reward: risk-adjusted returns (Sharpe-based)
```

### 2. Multi-Asset Meta-RL Agent

Each asset defines a separate MDP:

```
MDPs = {BTCUSDT_MDP, ETHUSDT_MDP, SOLUSDT_MDP, ...}
Goal: Single agent that adapts to any asset's dynamics within a few episodes
```

### 3. Cross-Timeframe Adaptation

Different timeframes as different MDPs:

```
MDPs = {1min_MDP, 5min_MDP, 1hour_MDP, Daily_MDP}
Goal: Agent learns temporal patterns that transfer across timeframes
```

### 4. Adaptive Position Sizing

Meta-RL for dynamic position management:

```
State: [market_features, current_position, unrealized_pnl, volatility]
Actions: {Increase, Decrease, Maintain, Close}
Goal: Learn position sizing rules that adapt to risk conditions
```

---

## Implementation in Python

### Core Meta-RL Agent (RL^2 with GRU)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class MetaRLAgent(nn.Module):
    """
    RL^2 Meta-Reinforcement Learning agent for trading.

    Uses a GRU to encode the learning algorithm in its hidden state.
    The agent receives (state, prev_action, prev_reward, done) as input
    and outputs a policy and value estimate.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input: state + prev_action (one-hot) + prev_reward + done
        input_size = state_dim + action_dim + 2

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        # Recurrent core (GRU for learning algorithm)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Policy head
        self.policy_head = nn.Linear(hidden_size, action_dim)

        # Value head
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        done: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Meta-RL agent.

        Args:
            state: Current state (batch, state_dim)
            prev_action: Previous action one-hot (batch, action_dim)
            prev_reward: Previous reward (batch, 1)
            done: Done flag (batch, 1)
            hidden: GRU hidden state (num_layers, batch, hidden_size)

        Returns:
            action_logits, value, new_hidden
        """
        # Concatenate inputs
        x = torch.cat([state, prev_action, prev_reward, done], dim=-1)

        # Encode
        x = self.encoder(x)

        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # GRU forward
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        gru_out, new_hidden = self.gru(x, hidden)
        gru_out = gru_out.squeeze(1)

        # Policy and value
        action_logits = self.policy_head(gru_out)
        value = self.value_head(gru_out)

        return action_logits, value, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )

    def get_action(
        self,
        state: np.ndarray,
        prev_action: int,
        prev_reward: float,
        done: bool,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[int, float, float, torch.Tensor]:
        """
        Select action using current policy.

        Returns:
            action, log_prob, value, new_hidden
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            prev_action_t = torch.zeros(1, self.action_dim)
            prev_action_t[0, prev_action] = 1.0
            prev_reward_t = torch.FloatTensor([[prev_reward]])
            done_t = torch.FloatTensor([[float(done)]])

            logits, value, new_hidden = self.forward(
                state_t, prev_action_t, prev_reward_t, done_t, hidden
            )

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item(), new_hidden
```

### Trading Environment for Meta-RL

```python
class TradingEnvironment:
    """
    Trading environment that acts as an MDP for Meta-RL training.

    Each environment instance represents a specific market regime
    or asset, providing a different MDP for meta-training.
    """

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        max_steps: int = 200
    ):
        self.prices = prices
        self.features = features
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps

        self.state_dim = features.shape[1] + 3  # features + position + pnl + time
        self.action_dim = 3  # Buy, Sell, Hold

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.step_count = 0
        self.position = 0  # -1, 0, 1
        self.capital = self.initial_capital
        self.entry_price = 0.0
        self.total_pnl = 0.0

        # Random starting point
        max_start = len(self.prices) - self.max_steps - 1
        self.start_idx = np.random.randint(0, max(1, max_start))
        self.current_idx = self.start_idx

        return self._get_state()

    def step(self, action: int):
        """
        Take action in environment.

        Args:
            action: 0=Buy, 1=Sell, 2=Hold

        Returns:
            next_state, reward, done, info
        """
        current_price = self.prices[self.current_idx]
        prev_position = self.position

        # Execute action
        if action == 0:  # Buy
            if self.position <= 0:
                if self.position == -1:
                    # Close short
                    pnl = (self.entry_price - current_price) / self.entry_price
                    self.total_pnl += pnl
                    self.capital *= (1 + pnl - self.transaction_cost)
                # Open long
                self.position = 1
                self.entry_price = current_price
                self.capital *= (1 - self.transaction_cost)

        elif action == 1:  # Sell
            if self.position >= 0:
                if self.position == 1:
                    # Close long
                    pnl = (current_price - self.entry_price) / self.entry_price
                    self.total_pnl += pnl
                    self.capital *= (1 + pnl - self.transaction_cost)
                # Open short
                self.position = -1
                self.entry_price = current_price
                self.capital *= (1 - self.transaction_cost)

        # action == 2: Hold (do nothing)

        # Advance time
        self.current_idx += 1
        self.step_count += 1

        # Check termination
        done = (self.step_count >= self.max_steps or
                self.current_idx >= len(self.prices) - 1)

        # Compute reward (risk-adjusted return)
        next_price = self.prices[self.current_idx]
        step_return = 0.0
        if self.position == 1:
            step_return = (next_price - current_price) / current_price
        elif self.position == -1:
            step_return = (current_price - next_price) / current_price

        # Penalize transaction costs for position changes
        transaction_penalty = 0.0
        if prev_position != self.position:
            transaction_penalty = self.transaction_cost

        reward = step_return - transaction_penalty

        next_state = self._get_state()
        info = {
            'capital': self.capital,
            'position': self.position,
            'total_pnl': self.total_pnl,
            'step_return': step_return
        }

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Construct state vector."""
        if self.current_idx >= len(self.features):
            market_features = np.zeros(self.features.shape[1])
        else:
            market_features = self.features[self.current_idx]

        position_feature = float(self.position)
        pnl_feature = self.total_pnl
        time_feature = self.step_count / self.max_steps

        state = np.concatenate([
            market_features,
            [position_feature, pnl_feature, time_feature]
        ])

        return state.astype(np.float32)
```

### PPO Meta-Trainer

```python
class PPOMetaTrainer:
    """
    PPO-based meta-trainer for Meta-RL agents.

    Trains the agent across a distribution of trading environments,
    enabling it to learn a general learning algorithm.
    """

    def __init__(
        self,
        agent: MetaRLAgent,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        num_episodes_per_trial: int = 3,
        max_grad_norm: float = 0.5
    ):
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.num_episodes_per_trial = num_episodes_per_trial
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    def collect_trial(self, env: TradingEnvironment):
        """
        Collect a trial (multiple episodes) in one environment.

        The hidden state persists across episodes within the same trial,
        allowing the agent to adapt to the environment.
        """
        states, actions, rewards = [], [], []
        log_probs, values, dones = [], [], []

        hidden = None
        prev_action = 0
        prev_reward = 0.0
        prev_done = False

        for episode in range(self.num_episodes_per_trial):
            state = env.reset()
            episode_done = False

            while not episode_done:
                action, log_prob, value, hidden = self.agent.get_action(
                    state, prev_action, prev_reward, prev_done, hidden
                )

                next_state, reward, done, info = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)

                prev_action = action
                prev_reward = reward
                prev_done = done
                state = next_state
                episode_done = done

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
            'dones': np.array(dones)
        }

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, batch):
        """Perform PPO update on collected batch."""
        advantages, returns = self.compute_gae(
            batch['rewards'], batch['values'], batch['dones']
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_t = torch.FloatTensor(batch['states'])
        actions_t = torch.LongTensor(batch['actions'])
        old_log_probs_t = torch.FloatTensor(batch['log_probs'])
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        # Forward pass through agent (re-compute with full sequence)
        hidden = None
        all_logits = []
        all_values = []

        prev_action_t = torch.zeros(1, self.agent.action_dim)
        prev_reward_t = torch.zeros(1, 1)
        done_t = torch.zeros(1, 1)

        for i in range(len(states_t)):
            logits, value, hidden = self.agent(
                states_t[i:i+1], prev_action_t, prev_reward_t, done_t, hidden
            )
            all_logits.append(logits)
            all_values.append(value)

            # Update prev inputs
            prev_action_t = torch.zeros(1, self.agent.action_dim)
            prev_action_t[0, actions_t[i]] = 1.0
            prev_reward_t = torch.FloatTensor([[batch['rewards'][i]]])
            done_t = torch.FloatTensor([[float(batch['dones'][i])]])

        all_logits = torch.cat(all_logits, dim=0)
        all_values = torch.cat(all_values, dim=0).squeeze(-1)

        # Compute new log probs
        dist = torch.distributions.Categorical(logits=all_logits)
        new_log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        # PPO clipped loss
        ratio = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(all_values, returns_t)

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }

    def meta_train_step(self, environments):
        """
        One step of meta-training across multiple environments.

        Args:
            environments: List of TradingEnvironment instances

        Returns:
            Dictionary of average metrics
        """
        total_metrics = {}

        for env in environments:
            batch = self.collect_trial(env)
            metrics = self.update(batch)

            for key, val in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + val

        # Average
        num_envs = len(environments)
        return {k: v / num_envs for k, v in total_metrics.items()}
```

### Data Preparation

```python
import pandas as pd
import requests

def create_trading_features(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Create technical features from price array.

    Returns array of shape (N, 11).
    """
    closes = prices
    n = len(closes)

    # Returns
    ret_1 = np.zeros(n)
    ret_5 = np.zeros(n)
    ret_10 = np.zeros(n)
    ret_1[1:] = closes[1:] / closes[:-1] - 1
    ret_5[5:] = closes[5:] / closes[:-5] - 1
    ret_10[10:] = closes[10:] / closes[:-10] - 1

    # SMA ratio
    sma = np.convolve(closes, np.ones(window)/window, mode='full')[:n]
    sma_ratio = np.where(sma > 0, closes / sma - 1, 0)

    # EMA ratio
    alpha = 2 / (window + 1)
    ema = np.zeros(n)
    ema[0] = closes[0]
    for i in range(1, n):
        ema[i] = alpha * closes[i] + (1 - alpha) * ema[i-1]
    ema_ratio = np.where(ema > 0, closes / ema - 1, 0)

    # Volatility
    log_returns = np.zeros(n)
    log_returns[1:] = np.log(closes[1:] / closes[:-1])
    volatility = np.array([
        np.std(log_returns[max(0, i-window+1):i+1]) if i >= window else 0.0
        for i in range(n)
    ])

    # Momentum
    momentum = np.zeros(n)
    momentum[window:] = closes[window:] / closes[:-window] - 1

    # RSI
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses_arr = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gains, np.ones(14)/14, mode='full')[:n]
    avg_loss = np.convolve(losses_arr, np.ones(14)/14, mode='full')[:n]
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = (100 - 100 / (1 + rs)) / 100

    # MACD
    ema12 = np.zeros(n)
    ema26 = np.zeros(n)
    ema12[0] = closes[0]
    ema26[0] = closes[0]
    for i in range(1, n):
        ema12[i] = 2/13 * closes[i] + 11/13 * ema12[i-1]
        ema26[i] = 2/27 * closes[i] + 25/27 * ema26[i-1]
    macd = np.where(closes > 0, (ema12 - ema26) / closes, 0)

    # Bollinger Band position
    bb_pos = np.zeros(n)
    for i in range(window, n):
        w = closes[i-window:i]
        mean = np.mean(w)
        std = np.std(w)
        bb_pos[i] = (closes[i] - mean) / (2 * std + 1e-10)

    # Volume SMA ratio (use price as proxy for volume in simulation)
    vol_sma = np.convolve(np.abs(log_returns), np.ones(window)/window, mode='full')[:n]
    vol_ratio = np.where(vol_sma > 0, np.abs(log_returns) / vol_sma - 1, 0)

    features = np.column_stack([
        ret_1, ret_5, ret_10, sma_ratio, ema_ratio,
        volatility, momentum, rsi, macd, bb_pos, vol_ratio
    ])

    # Remove initial NaN period
    valid_start = max(window, 26)
    return features[valid_start:]


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
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.set_index('timestamp').sort_index()

    return df
```

---

## Implementation in Rust

The Rust implementation provides a high-performance Meta-RL agent for production trading systems.

### Project Structure

```
88_meta_rl_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── agent/
│   │   ├── mod.rs
│   │   └── meta_rl.rs
│   ├── env/
│   │   ├── mod.rs
│   │   └── trading_env.rs
│   ├── trainer/
│   │   ├── mod.rs
│   │   └── ppo.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── features.rs
│   │   └── bybit.rs
│   └── backtest/
│       ├── mod.rs
│       └── engine.rs
├── examples/
│   ├── basic_meta_rl.rs
│   ├── multi_asset_training.rs
│   └── trading_strategy.rs
└── python/
    ├── __init__.py
    ├── meta_rl_trader.py
    ├── data_loader.py
    ├── backtest.py
    └── requirements.txt
```

### Core Rust Implementation

See the `src/` directory for the complete Rust implementation with:

- GRU-based recurrent meta-RL agent
- Trading environment with realistic market simulation
- PPO training loop for meta-learning across environments
- Async Bybit API integration for cryptocurrency data
- Production-ready error handling and logging
- Backtesting engine with comprehensive performance metrics

---

## Practical Examples with Stock and Crypto Data

### Example 1: Multi-Regime Meta-Training

```python
import yfinance as yf

# Download data for multiple assets (different MDPs)
assets = {
    'AAPL': yf.download('AAPL', period='2y'),
    'MSFT': yf.download('MSFT', period='2y'),
    'GOOGL': yf.download('GOOGL', period='2y'),
    'BTC-USD': yf.download('BTC-USD', period='2y'),
    'ETH-USD': yf.download('ETH-USD', period='2y'),
}

# Create environments for each asset
environments = []
for name, df in assets.items():
    prices = df['Close'].values
    features = create_trading_features(prices)
    prices_aligned = prices[len(prices) - len(features):]

    env = TradingEnvironment(
        prices=prices_aligned,
        features=features,
        initial_capital=10000.0,
        max_steps=200
    )
    environments.append(env)

# Initialize Meta-RL agent
state_dim = environments[0].state_dim
action_dim = environments[0].action_dim
agent = MetaRLAgent(state_dim, action_dim, hidden_size=128)

# Initialize PPO trainer
trainer = PPOMetaTrainer(
    agent=agent,
    lr=3e-4,
    num_episodes_per_trial=3
)

# Meta-training
for epoch in range(500):
    metrics = trainer.meta_train_step(environments)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss={metrics['total_loss']:.4f}, "
              f"entropy={metrics['entropy']:.4f}")
```

### Example 2: Adaptation to New Asset

```python
# New asset not seen during training
new_asset = yf.download('TSLA', period='1y')
new_prices = new_asset['Close'].values
new_features = create_trading_features(new_prices)
new_prices_aligned = new_prices[len(new_prices) - len(new_features):]

new_env = TradingEnvironment(
    prices=new_prices_aligned,
    features=new_features,
    initial_capital=10000.0,
    max_steps=200
)

# Run 3 episodes - the agent adapts via its hidden state
hidden = None
prev_action, prev_reward, prev_done = 0, 0.0, False

for episode in range(3):
    state = new_env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _, _, hidden = agent.get_action(
            state, prev_action, prev_reward, prev_done, hidden
        )
        state, reward, done, info = new_env.step(action)
        total_reward += reward
        prev_action, prev_reward, prev_done = action, reward, done

    print(f"Episode {episode + 1}: reward={total_reward:.4f}, "
          f"capital={info['capital']:.2f}")
```

### Example 3: Bybit Crypto Trading

```python
# Fetch data for multiple crypto pairs
crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT']
crypto_envs = []

for symbol in crypto_pairs:
    df = fetch_bybit_klines(symbol, interval='60', limit=1000)
    prices = df['close'].values
    features = create_trading_features(prices)
    prices_aligned = prices[len(prices) - len(features):]

    env = TradingEnvironment(
        prices=prices_aligned,
        features=features,
        initial_capital=10000.0,
        max_steps=200
    )
    crypto_envs.append(env)

# Meta-train on crypto environments
for epoch in range(300):
    metrics = trainer.meta_train_step(crypto_envs)
    if epoch % 30 == 0:
        print(f"Epoch {epoch}: loss={metrics['total_loss']:.4f}")
```

---

## Backtesting Framework

### Meta-RL Backtester

```python
class MetaRLBacktester:
    """
    Backtesting framework for Meta-RL trading agents.

    Runs the agent through historical data, allowing it to
    adapt via its hidden state across episodes.
    """

    def __init__(
        self,
        agent: MetaRLAgent,
        adaptation_episodes: int = 2,
        episode_length: int = 200,
        transaction_cost: float = 0.001
    ):
        self.agent = agent
        self.adaptation_episodes = adaptation_episodes
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost

    def backtest(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_capital: float = 10000.0
    ) -> dict:
        """Run backtest on historical data."""
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_capital=initial_capital,
            transaction_cost=self.transaction_cost,
            max_steps=self.episode_length
        )

        all_rewards = []
        all_capitals = []
        hidden = None
        prev_action, prev_reward, prev_done = 0, 0.0, False

        # Adaptation phase
        for ep in range(self.adaptation_episodes):
            state = env.reset()
            done = False

            while not done:
                action, _, _, hidden = self.agent.get_action(
                    state, prev_action, prev_reward, prev_done, hidden
                )
                state, reward, done, info = env.step(action)
                prev_action, prev_reward, prev_done = action, reward, done

        # Evaluation phase
        state = env.reset()
        done = False
        episode_rewards = []

        while not done:
            action, _, _, hidden = self.agent.get_action(
                state, prev_action, prev_reward, prev_done, hidden
            )
            state, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            all_capitals.append(info['capital'])
            prev_action, prev_reward, prev_done = action, reward, done

        returns = np.array(episode_rewards)
        total_return = (all_capitals[-1] / initial_capital) - 1 if all_capitals else 0.0

        # Compute metrics
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)
        downside = returns[returns < 0]
        sortino = np.sqrt(252) * returns.mean() / (downside.std() + 1e-10) if len(downside) > 0 else 0.0

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1
        max_drawdown = drawdowns.min()

        wins = (returns > 0).sum()
        losses = (returns < 0).sum()
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_steps': len(returns),
            'final_capital': all_capitals[-1] if all_capitals else initial_capital,
            'equity_curve': all_capitals
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
| Adaptation Speed | < 3 episodes |

### Meta-RL vs Baselines

In typical experiments, Meta-RL shows:
- **Immediate adaptation** to new environments (no gradient steps needed)
- **Learned exploration** that efficiently gathers market information
- **20-40% improvement** in Sharpe ratio compared to fixed RL policies
- **Better regime change handling** than standard RL agents

### Advantages Over MAML-Based Approaches

| Feature | MAML | Meta-RL (RL^2) |
|---------|------|----------------|
| Requires gradient steps for adaptation | Yes | No |
| Handles sequential decisions natively | No | Yes |
| Learns exploration strategy | No | Yes |
| Computational cost at adaptation time | Medium | Low (single forward pass) |
| Handles partial observability | No | Yes (via hidden state) |

---

## Future Directions

### 1. Transformer-Based Meta-RL

Replace GRU with a Transformer for better long-range dependencies:

```
h_t = TransformerDecoder(x_1, x_2, ..., x_t)
```

### 2. Task Inference Networks

Explicitly infer a task embedding for conditioning:

```
z_t = Encoder(history_{1:t})
π(a_t | s_t, z_t) = Decoder(s_t, z_t)
```

### 3. Hierarchical Meta-RL

Multi-level decision making:
- Level 1: Meta-policy selects strategies (momentum, mean-reversion, etc.)
- Level 2: Sub-policies execute specific strategies
- Level 3: Position sizing and risk management

### 4. Offline Meta-RL

Train from logged market data without online interaction:

```
L(θ) = E_{M~p(M)} E_{τ~D_M} [Σ_t log π_θ(a_t|s_t, h_t) * A_t]
```

### 5. Multi-Agent Meta-RL

Multiple agents that adapt to each other's strategies in competitive markets.

---

## References

1. Duan, Y., Schulman, J., Chen, X., Bartlett, P., Sutskever, I., & Abbeel, P. (2016). RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning. [arXiv:1611.02779](https://arxiv.org/abs/1611.02779)

2. Wang, J. X., et al. (2016). Learning to Reinforcement Learn. [arXiv:1611.05763](https://arxiv.org/abs/1611.05763)

3. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML. [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)

4. Rakelly, K., et al. (2019). Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables. ICML. [arXiv:1903.08254](https://arxiv.org/abs/1903.08254)

5. Beck, J., et al. (2023). A Survey of Meta-Reinforcement Learning. [arXiv:2301.08028](https://arxiv.org/abs/2301.08028)

---

## Running the Examples

### Python

```bash
# Navigate to chapter directory
cd 88_meta_rl_trading

# Install dependencies
pip install -r python/requirements.txt

# Run Python examples
python python/meta_rl_trader.py
```

### Rust

```bash
# Navigate to chapter directory
cd 88_meta_rl_trading

# Build the project
cargo build --release

# Run tests
cargo test

# Run examples
cargo run --example basic_meta_rl
cargo run --example multi_asset_training
cargo run --example trading_strategy
```

---

## Summary

Meta-Reinforcement Learning provides a powerful framework for adaptive trading:

- **Learning Algorithm Internalization**: The agent's hidden state encodes a complete learning algorithm
- **Zero-Shot Adaptation**: No gradient steps needed at deployment; adaptation happens through forward passes
- **Learned Exploration**: The agent learns how to efficiently gather information in new markets
- **Sequential Decision Making**: Naturally handles the temporal aspects of trading decisions

By training across diverse market environments, Meta-RL agents develop the ability to quickly identify market dynamics and adapt their trading behavior accordingly - a critical capability for robust algorithmic trading systems.

---

*Previous Chapter: [Chapter 87: Task-Agnostic Meta-Learning](../87_task_agnostic_meta_learning)*

*Next Chapter: [Chapter 89: Meta-SGD for Trading](../89_meta_sgd_trading)*
