"""
Meta-RL (RL^2) implementation for trading.

This module provides:
- MetaRLAgent: GRU-based RL^2 agent for adaptive trading
- TradingEnvironment: MDP environment for trading simulation
- PPOMetaTrainer: PPO-based meta-training across environments
- RolloutBuffer: Experience storage for PPO updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from copy import deepcopy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaRLAgent(nn.Module):
    """
    RL^2 Meta-Reinforcement Learning agent for trading.

    Uses a GRU to encode the learning algorithm in its hidden state.
    The agent receives (state, prev_action, prev_reward, done) as input
    and outputs a policy distribution and value estimate.

    Reference: Duan et al., 2016. "RL^2: Fast Reinforcement Learning
    via Slow Reinforcement Learning"
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

        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=0.01 if 'policy' in name else 1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)

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
        x = torch.cat([state, prev_action, prev_reward, done], dim=-1)
        x = self.encoder(x)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        gru_out, new_hidden = self.gru(x, hidden)
        gru_out = gru_out.squeeze(1)

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
                    pnl = (self.entry_price - current_price) / self.entry_price
                    self.total_pnl += pnl
                    self.capital *= (1 + pnl - self.transaction_cost)
                self.position = 1
                self.entry_price = current_price
                self.capital *= (1 - self.transaction_cost)

        elif action == 1:  # Sell
            if self.position >= 0:
                if self.position == 1:
                    pnl = (current_price - self.entry_price) / self.entry_price
                    self.total_pnl += pnl
                    self.capital *= (1 + pnl - self.transaction_cost)
                self.position = -1
                self.entry_price = current_price
                self.capital *= (1 - self.transaction_cost)

        # Advance time
        self.current_idx += 1
        self.step_count += 1

        done = (self.step_count >= self.max_steps or
                self.current_idx >= len(self.prices) - 1)

        # Compute reward
        next_price = self.prices[self.current_idx]
        step_return = 0.0
        if self.position == 1:
            step_return = (next_price - current_price) / current_price
        elif self.position == -1:
            step_return = (current_price - next_price) / current_price

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

        state = np.concatenate([
            market_features,
            [float(self.position), self.total_pnl, self.step_count / self.max_steps]
        ])

        return state.astype(np.float32)


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    prev_actions: List[int] = field(default_factory=list)
    prev_rewards: List[float] = field(default_factory=list)
    prev_dones: List[bool] = field(default_factory=list)

    def clear(self):
        """Clear all stored data."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.prev_actions.clear()
        self.prev_rewards.clear()
        self.prev_dones.clear()

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        prev_action: int,
        prev_reward: float,
        prev_done: bool
    ):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.prev_actions.append(prev_action)
        self.prev_rewards.append(prev_reward)
        self.prev_dones.append(prev_done)


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
        self.buffer = RolloutBuffer()

    def collect_trial(self, env: TradingEnvironment) -> RolloutBuffer:
        """
        Collect a trial (multiple episodes) in one environment.

        The hidden state persists across episodes within the same trial,
        allowing the agent to adapt to the environment.
        """
        buffer = RolloutBuffer()
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

                buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    log_prob=log_prob,
                    value=value,
                    done=done,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                    prev_done=prev_done
                )

                prev_action = action
                prev_reward = reward
                prev_done = done
                state = next_state
                episode_done = done

        return buffer

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            )

        returns = advantages + values
        return advantages, returns

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Perform PPO update on collected buffer."""
        rewards = np.array(buffer.rewards)
        values = np.array(buffer.values)
        dones = np.array(buffer.dones, dtype=np.float32)

        advantages, returns = self.compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(buffer.states))
        actions_t = torch.LongTensor(buffer.actions)
        old_log_probs_t = torch.FloatTensor(buffer.log_probs)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        # Forward pass through agent (re-compute with full sequence)
        hidden = None
        all_logits = []
        all_values = []

        for i in range(len(states_t)):
            prev_action_t = torch.zeros(1, self.agent.action_dim)
            prev_action_t[0, buffer.prev_actions[i]] = 1.0
            prev_reward_t = torch.FloatTensor([[buffer.prev_rewards[i]]])
            done_t = torch.FloatTensor([[float(buffer.prev_dones[i])]])

            logits, value, hidden = self.agent(
                states_t[i:i + 1], prev_action_t, prev_reward_t, done_t, hidden
            )
            all_logits.append(logits)
            all_values.append(value)

        all_logits = torch.cat(all_logits, dim=0)
        all_values = torch.cat(all_values, dim=0).squeeze(-1)

        # Compute new log probs and entropy
        dist = torch.distributions.Categorical(logits=all_logits)
        new_log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        # PPO clipped loss
        ratio = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(
            ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        ) * advantages_t
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(all_values, returns_t)

        # Total loss
        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.agent.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }

    def meta_train_step(
        self,
        environments: List[TradingEnvironment]
    ) -> Dict[str, float]:
        """
        One step of meta-training across multiple environments.

        Args:
            environments: List of TradingEnvironment instances

        Returns:
            Dictionary of average metrics
        """
        total_metrics: Dict[str, float] = {}

        for env in environments:
            buffer = self.collect_trial(env)
            metrics = self.update(buffer)

            for key, val in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + val

        num_envs = len(environments)
        return {k: v / num_envs for k, v in total_metrics.items()}

    def save(self, path: str):
        """Save agent and optimizer state."""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Saved trainer to {path}")

    def load(self, path: str):
        """Load agent and optimizer state."""
        checkpoint = torch.load(path, weights_only=True)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded trainer from {path}")


if __name__ == "__main__":
    print("=== Meta-RL Trading Example ===\n")

    # Create simulated environments
    from data_loader import SimulatedDataGenerator, FeatureGenerator

    feature_gen = FeatureGenerator(window=20)

    regimes = [
        ("Bull", 0.015, 0.0003),
        ("Bear", 0.02, -0.0003),
        ("Sideways", 0.008, 0.0),
        ("Volatile", 0.03, 0.0),
    ]

    environments = []
    for name, vol, trend in regimes:
        klines = SimulatedDataGenerator.generate_trending_klines(500, 50000.0, vol, trend)
        features = feature_gen.compute_features(klines)
        closes = np.array([k.close for k in klines])
        # Align prices with features
        prices = closes[len(closes) - len(features):]

        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_capital=10000.0,
            max_steps=100
        )
        environments.append(env)
        print(f"  Created {name} environment: {len(features)} steps")

    # Create agent
    state_dim = environments[0].state_dim
    action_dim = environments[0].action_dim
    agent = MetaRLAgent(state_dim, action_dim, hidden_size=64)
    print(f"\nCreated Meta-RL agent with {sum(p.numel() for p in agent.parameters())} parameters")

    # Create trainer
    trainer = PPOMetaTrainer(
        agent=agent,
        lr=3e-4,
        num_episodes_per_trial=2
    )
    print("Created PPO meta-trainer\n")

    # Meta-training
    print("Meta-training...")
    for epoch in range(20):
        metrics = trainer.meta_train_step(environments)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: loss={metrics['total_loss']:.4f}, "
                  f"entropy={metrics['entropy']:.4f}")

    # Test adaptation on new environment
    print("\nTesting adaptation on new environment...")
    new_klines = SimulatedDataGenerator.generate_regime_changing_klines(400, 45000.0)
    new_features = feature_gen.compute_features(new_klines)
    new_closes = np.array([k.close for k in new_klines])
    new_prices = new_closes[len(new_closes) - len(new_features):]

    new_env = TradingEnvironment(
        prices=new_prices,
        features=new_features,
        initial_capital=10000.0,
        max_steps=100
    )

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

        print(f"  Episode {episode + 1}: reward={total_reward:.4f}, "
              f"capital={info['capital']:.2f}")

    print("\n=== Example Complete ===")
