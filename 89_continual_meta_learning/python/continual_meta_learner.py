"""
Continual Meta-Learning implementation for trading.

This module provides:
- TradingModel: Neural network for price prediction
- ContinualMAMLTrainer: MAML with continual learning (EWC + replay buffer)
- TradingStrategy: Adaptive trading using continual meta-learning

Key idea: Combines MAML's fast adaptation with continual learning techniques
(Elastic Weight Consolidation + experience replay) so the model can learn
new market regimes without forgetting previously learned ones.

References:
- Finn et al., 2017. "Model-Agnostic Meta-Learning for Fast Adaptation"
- Kirkpatrick et al., 2017. "Overcoming Catastrophic Forgetting in Neural Networks" (EWC)
- Javed & White, 2019. "Meta-Learning Representations for Continual Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Deque
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingModel(nn.Module):
    """
    Neural network for trading signal prediction.

    Architecture: Input -> Hidden (ReLU) -> Hidden (ReLU) -> Output (Tanh)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict from numpy array."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            output = self.forward(x)
            return output.numpy().flatten()


@dataclass
class TaskData:
    """Data for a single meta-learning task."""
    support_features: np.ndarray  # (N_support, D)
    support_labels: np.ndarray    # (N_support,)
    query_features: np.ndarray    # (N_query, D)
    query_labels: np.ndarray      # (N_query,)
    regime_id: int = 0            # Identifies which market regime

    def to_tensors(self) -> Tuple[torch.Tensor, ...]:
        """Convert to PyTorch tensors."""
        return (
            torch.tensor(self.support_features, dtype=torch.float32),
            torch.tensor(self.support_labels, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(self.query_features, dtype=torch.float32),
            torch.tensor(self.query_labels, dtype=torch.float32).unsqueeze(-1)
        )


class ReplayBuffer:
    """
    Experience replay buffer for continual learning.

    Stores representative tasks from past market regimes to prevent
    catastrophic forgetting during meta-training on new regimes.
    """

    def __init__(self, max_size: int = 100, tasks_per_regime: int = 5):
        self.max_size = max_size
        self.tasks_per_regime = tasks_per_regime
        self.buffer: Deque[TaskData] = deque(maxlen=max_size)
        self.regime_counts: Dict[int, int] = {}

    def add(self, task: TaskData):
        """Add a task to the replay buffer."""
        regime = task.regime_id
        current_count = self.regime_counts.get(regime, 0)

        if current_count < self.tasks_per_regime:
            self.buffer.append(task)
            self.regime_counts[regime] = current_count + 1
        elif np.random.random() < 0.3:
            # Reservoir sampling: replace with probability
            self.buffer.append(task)

    def sample(self, n: int) -> List[TaskData]:
        """Sample n tasks from the buffer."""
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        return [self.buffer[i] for i in indices]

    def add_batch(self, tasks: List[TaskData]):
        """Add multiple tasks to the buffer."""
        for task in tasks:
            self.add(task)

    @property
    def size(self) -> int:
        return len(self.buffer)

    @property
    def num_regimes(self) -> int:
        return len(self.regime_counts)


class EWC:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.

    Computes the Fisher Information Matrix to identify important parameters
    and penalizes changes to them during meta-training on new regimes.

    Reference: Kirkpatrick et al., 2017
    """

    def __init__(self, model: TradingModel, ewc_lambda: float = 100.0):
        self.ewc_lambda = ewc_lambda
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self._initialized = False

    def compute_fisher(
        self,
        model: TradingModel,
        tasks: List[TaskData],
        num_samples: int = 50
    ):
        """
        Compute the Fisher Information Matrix from a set of tasks.

        Uses empirical Fisher approximation: average of squared gradients.
        """
        model.eval()

        # Initialize Fisher accumulators
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

        n_samples = 0
        for task in tasks:
            support_x, support_y, _, _ = task.to_tensors()

            for i in range(min(len(support_x), num_samples)):
                model.zero_grad()
                x = support_x[i:i+1]
                y = support_y[i:i+1]
                output = model(x)
                loss = F.mse_loss(output, y)
                loss.backward()

                for n, p in model.named_parameters():
                    if p.grad is not None:
                        fisher[n] += p.grad.data.pow(2)
                n_samples += 1

        # Average
        if n_samples > 0:
            for n in fisher:
                fisher[n] /= n_samples

        self.fisher = fisher
        self.optimal_params = {
            n: p.data.clone() for n, p in model.named_parameters()
        }
        self._initialized = True

    def penalty(self, model: TradingModel) -> torch.Tensor:
        """
        Compute the EWC penalty term.

        penalty = (lambda/2) * sum_i F_i * (theta_i - theta_i*)^2
        """
        if not self._initialized:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]).pow(2)).sum()

        return (self.ewc_lambda / 2) * loss

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class ContinualMAMLTrainer:
    """
    Continual Meta-Learning trainer combining MAML with EWC and replay.

    This trainer can learn new market regimes sequentially without
    forgetting previously learned regimes. It uses:

    1. MAML/FOMAML for fast adaptation to new tasks
    2. EWC to protect important parameters from past regimes
    3. Experience replay to revisit tasks from earlier regimes

    Reference:
    - Finn et al., 2017 (MAML)
    - Kirkpatrick et al., 2017 (EWC)
    - Javed & White, 2019 (Continual Meta-Learning)
    """

    def __init__(
        self,
        model: TradingModel,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = True,
        ewc_lambda: float = 100.0,
        replay_buffer_size: int = 100,
        replay_ratio: float = 0.5
    ):
        """
        Initialize Continual MAML trainer.

        Args:
            model: The model to meta-train
            inner_lr: Learning rate for task adaptation (alpha)
            outer_lr: Meta-learning rate (beta)
            inner_steps: Number of gradient steps per task
            first_order: Use FOMAML if True (recommended for stability)
            ewc_lambda: EWC regularization strength
            replay_buffer_size: Maximum tasks stored in replay buffer
            replay_ratio: Fraction of replay tasks mixed with new tasks
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.replay_ratio = replay_ratio

        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=outer_lr
        )

        self.ewc = EWC(model, ewc_lambda)
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.current_regime = 0
        self.regimes_seen: List[int] = []

    def _compute_loss(
        self,
        model: TradingModel,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss."""
        predictions = model(features)
        return F.mse_loss(predictions, labels)

    def _inner_loop(
        self,
        task: TaskData,
        model: Optional[TradingModel] = None
    ) -> Tuple[TradingModel, torch.Tensor]:
        """
        Perform inner loop adaptation on a single task.

        Returns:
            Tuple of (adapted_model, query_loss)
        """
        if model is None:
            model = self.model

        support_x, support_y, query_x, query_y = task.to_tensors()

        # Clone model for adaptation
        adapted_model = deepcopy(model)

        # Inner loop: adapt on support set
        for _ in range(self.inner_steps):
            loss = self._compute_loss(adapted_model, support_x, support_y)

            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=not self.first_order
            )

            with torch.no_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    param.sub_(self.inner_lr * grad)

        # Evaluate on query set
        query_loss = self._compute_loss(adapted_model, query_x, query_y)

        return adapted_model, query_loss

    def meta_train_step(self, tasks: List[TaskData]) -> float:
        """
        Perform one meta-training step with continual learning.

        Combines:
        1. Standard MAML loss on current tasks
        2. MAML loss on replayed tasks from past regimes
        3. EWC penalty to protect important parameters

        Args:
            tasks: Batch of tasks for meta-training

        Returns:
            Average query loss across all tasks
        """
        if not tasks:
            return 0.0

        self.model.train()
        self.meta_optimizer.zero_grad()

        total_loss = torch.tensor(0.0)
        all_tasks = list(tasks)

        # Mix in replay tasks from past regimes
        if self.replay_buffer.size > 0:
            n_replay = max(1, int(len(tasks) * self.replay_ratio))
            replay_tasks = self.replay_buffer.sample(n_replay)
            all_tasks.extend(replay_tasks)

        # Compute MAML loss over all tasks
        for task in all_tasks:
            _, query_loss = self._inner_loop(task)
            total_loss = total_loss + query_loss

        avg_maml_loss = total_loss / len(all_tasks)

        # Add EWC penalty
        ewc_penalty = self.ewc.penalty(self.model)
        total_meta_loss = avg_maml_loss + ewc_penalty

        # Meta-update
        total_meta_loss.backward()
        self.meta_optimizer.step()

        return avg_maml_loss.item()

    def learn_regime(
        self,
        tasks: List[TaskData],
        regime_id: int,
        num_epochs: int = 20,
        log_interval: int = 5
    ) -> List[float]:
        """
        Learn a new market regime while preserving knowledge of past regimes.

        This is the main continual learning interface:
        1. Meta-train on the new regime's tasks
        2. After training, update EWC with the new regime
        3. Store representative tasks in the replay buffer

        Args:
            tasks: Tasks from the new market regime
            regime_id: Identifier for this regime
            num_epochs: Number of meta-training epochs
            log_interval: Log every N epochs

        Returns:
            List of losses per epoch
        """
        # Tag tasks with regime ID
        for task in tasks:
            task.regime_id = regime_id

        self.current_regime = regime_id

        # Meta-train on this regime
        losses = []
        for epoch in range(num_epochs):
            loss = self.meta_train_step(tasks)
            losses.append(loss)

            if (epoch + 1) % log_interval == 0:
                replay_info = f" (replay: {self.replay_buffer.size} tasks)"
                ewc_info = " + EWC" if self.ewc.is_initialized else ""
                logger.info(
                    f"Regime {regime_id} | Epoch {epoch + 1}: "
                    f"Loss = {loss:.6f}{ewc_info}{replay_info}"
                )

        # Update EWC with combined old + new regime data
        all_fisher_tasks = list(tasks)
        if self.replay_buffer.size > 0:
            all_fisher_tasks.extend(self.replay_buffer.sample(
                min(len(tasks), self.replay_buffer.size)
            ))
        self.ewc.compute_fisher(self.model, all_fisher_tasks)

        # Store tasks in replay buffer
        self.replay_buffer.add_batch(tasks)

        if regime_id not in self.regimes_seen:
            self.regimes_seen.append(regime_id)

        logger.info(
            f"Regime {regime_id} learned. "
            f"Total regimes: {len(self.regimes_seen)}. "
            f"Replay buffer: {self.replay_buffer.size} tasks."
        )

        return losses

    def adapt(
        self,
        support_features: np.ndarray,
        support_labels: np.ndarray,
        adaptation_steps: Optional[int] = None
    ) -> TradingModel:
        """
        Adapt the meta-learned model to new data.

        Args:
            support_features: Features for adaptation
            support_labels: Labels for adaptation
            adaptation_steps: Number of steps (default: inner_steps)

        Returns:
            Adapted model ready for prediction
        """
        steps = adaptation_steps or self.inner_steps

        self.model.eval()
        adapted_model = deepcopy(self.model)

        support_x = torch.tensor(support_features, dtype=torch.float32)
        support_y = torch.tensor(support_labels, dtype=torch.float32).unsqueeze(-1)

        for _ in range(steps):
            loss = self._compute_loss(adapted_model, support_x, support_y)
            grads = torch.autograd.grad(loss, adapted_model.parameters())

            with torch.no_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    param.sub_(self.inner_lr * grad)

        adapted_model.eval()
        return adapted_model

    def evaluate_regime(
        self,
        tasks: List[TaskData],
        regime_id: int
    ) -> Dict[str, float]:
        """
        Evaluate model performance on a specific regime's tasks.

        Used to measure forgetting: after learning new regimes,
        how well does the model still perform on old regimes?

        Returns:
            Dictionary with loss and directional accuracy metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0

        for task in tasks:
            adapted_model, query_loss = self._inner_loop(task)
            total_loss += query_loss.item()

            # Measure directional accuracy on query set
            query_x = torch.tensor(task.query_features, dtype=torch.float32)
            with torch.no_grad():
                predictions = adapted_model(query_x).numpy().flatten()

            for pred, actual in zip(predictions, task.query_labels):
                if (pred > 0 and actual > 0) or (pred < 0 and actual < 0):
                    total_correct += 1
                total_predictions += 1

        avg_loss = total_loss / len(tasks) if tasks else 0.0
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0

        return {
            "regime_id": regime_id,
            "avg_loss": avg_loss,
            "directional_accuracy": accuracy,
            "num_tasks": len(tasks),
        }

    def save(self, path: str):
        """Save model and trainer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'inner_steps': self.inner_steps,
            'first_order': self.first_order,
            'regimes_seen': self.regimes_seen,
            'current_regime': self.current_regime,
        }, path)
        logger.info(f"Saved trainer to {path}")

    def load(self, path: str):
        """Load model and trainer state."""
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.inner_lr = checkpoint['inner_lr']
        self.outer_lr = checkpoint['outer_lr']
        self.inner_steps = checkpoint['inner_steps']
        self.first_order = checkpoint['first_order']
        self.regimes_seen = checkpoint.get('regimes_seen', [])
        self.current_regime = checkpoint.get('current_regime', 0)
        logger.info(f"Loaded trainer from {path}")


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    signal: int  # 1: buy, -1: sell, 0: hold
    strength: float
    prediction: float
    timestamp: int
    symbol: str

    @property
    def is_buy(self) -> bool:
        return self.signal == 1

    @property
    def is_sell(self) -> bool:
        return self.signal == -1

    @property
    def is_hold(self) -> bool:
        return self.signal == 0


class TradingStrategy:
    """
    Continual MAML-based adaptive trading strategy.

    Adapts to recent market data before making predictions,
    leveraging knowledge from all previously learned regimes.
    """

    def __init__(
        self,
        trainer: ContinualMAMLTrainer,
        threshold: float = 0.001,
        strong_threshold: float = 0.005,
        stop_loss: float = 0.02,
        take_profit: float = 0.04,
        adaptation_window: int = 20,
        adaptation_steps: int = 5
    ):
        self.trainer = trainer
        self.threshold = threshold
        self.strong_threshold = strong_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.adaptation_window = adaptation_window
        self.adaptation_steps = adaptation_steps

    def generate_signal(
        self,
        recent_features: np.ndarray,
        recent_returns: np.ndarray,
        current_features: np.ndarray,
        timestamp: int,
        symbol: str
    ) -> TradingSignal:
        """
        Generate trading signal for current market state.

        Args:
            recent_features: Recent feature history for adaptation
            recent_returns: Recent returns for adaptation
            current_features: Current features for prediction
            timestamp: Current timestamp
            symbol: Trading symbol

        Returns:
            TradingSignal with prediction and signal type
        """
        n = min(len(recent_features), len(recent_returns), self.adaptation_window)
        if n == 0:
            return TradingSignal(0, 0.0, 0.0, timestamp, symbol)

        adapt_features = recent_features[-n:]
        adapt_labels = recent_returns[-n:]

        adapted_model = self.trainer.adapt(
            adapt_features,
            adapt_labels,
            self.adaptation_steps
        )

        prediction = adapted_model.predict(current_features)[0]

        if prediction > self.strong_threshold:
            signal = 1
            strength = min(1.0, prediction / self.strong_threshold)
        elif prediction < -self.strong_threshold:
            signal = -1
            strength = min(1.0, -prediction / self.strong_threshold)
        elif prediction > self.threshold:
            signal = 1
            strength = prediction / self.strong_threshold
        elif prediction < -self.threshold:
            signal = -1
            strength = -prediction / self.strong_threshold
        else:
            signal = 0
            strength = 0.0

        return TradingSignal(signal, strength, prediction, timestamp, symbol)

    def check_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        position: int
    ) -> bool:
        """Check if stop loss should be triggered."""
        if position == 0:
            return False

        if position > 0:
            pnl_pct = (current_price / entry_price) - 1
        else:
            pnl_pct = 1 - (current_price / entry_price)

        return pnl_pct < -self.stop_loss

    def check_take_profit(
        self,
        entry_price: float,
        current_price: float,
        position: int
    ) -> bool:
        """Check if take profit should be triggered."""
        if position == 0:
            return False

        if position > 0:
            pnl_pct = (current_price / entry_price) - 1
        else:
            pnl_pct = 1 - (current_price / entry_price)

        return pnl_pct > self.take_profit


def create_tasks_from_data(
    features: np.ndarray,
    returns: np.ndarray,
    num_tasks: int = 10,
    support_size: int = 20,
    query_size: int = 10,
    regime_id: int = 0
) -> List[TaskData]:
    """
    Create meta-learning tasks from time series data.

    Args:
        features: Feature matrix (N, D)
        returns: Return vector (N,)
        num_tasks: Number of tasks to create
        support_size: Samples per support set
        query_size: Samples per query set
        regime_id: Market regime identifier

    Returns:
        List of TaskData objects
    """
    total_size = support_size + query_size
    if len(features) < total_size:
        return []

    tasks = []
    max_start = len(features) - total_size

    for _ in range(num_tasks):
        start_idx = np.random.randint(0, max_start + 1)

        support_end = start_idx + support_size
        query_end = support_end + query_size

        tasks.append(TaskData(
            support_features=features[start_idx:support_end],
            support_labels=returns[start_idx:support_end],
            query_features=features[support_end:query_end],
            query_labels=returns[support_end:query_end],
            regime_id=regime_id
        ))

    return tasks


if __name__ == "__main__":
    print("=== Continual Meta-Learning Trading Example ===\n")

    # Create model
    input_size = 11
    hidden_size = 64
    model = TradingModel(input_size, hidden_size)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters\n")

    # Create continual MAML trainer
    trainer = ContinualMAMLTrainer(
        model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        first_order=True,
        ewc_lambda=100.0,
        replay_buffer_size=50,
        replay_ratio=0.5
    )
    print("Created Continual MAML trainer (FOMAML + EWC + Replay)\n")

    # Simulate learning multiple market regimes sequentially
    regimes = [
        ("Bull Market", 0.015, 0.0003),
        ("Bear Market", 0.025, -0.0003),
        ("High Volatility", 0.035, 0.0),
    ]

    regime_tasks = {}
    for regime_id, (name, vol, trend) in enumerate(regimes):
        print(f"--- Learning Regime {regime_id}: {name} ---")

        # Generate dummy tasks for this regime
        tasks = []
        for _ in range(5):
            features = np.random.randn(30, input_size) * vol
            returns = np.random.randn(30) * vol + trend
            tasks.append(TaskData(
                support_features=features[:20],
                support_labels=returns[:20],
                query_features=features[20:],
                query_labels=returns[20:],
                regime_id=regime_id
            ))

        regime_tasks[regime_id] = tasks

        # Learn this regime
        losses = trainer.learn_regime(tasks, regime_id, num_epochs=10, log_interval=5)
        print(f"  Final loss: {losses[-1]:.6f}\n")

    # Evaluate on all regimes (measure forgetting)
    print("=== Evaluating on all regimes (forgetting test) ===")
    for regime_id, (name, _, _) in enumerate(regimes):
        metrics = trainer.evaluate_regime(regime_tasks[regime_id], regime_id)
        print(f"  Regime {regime_id} ({name}): "
              f"Loss={metrics['avg_loss']:.6f}, "
              f"Accuracy={metrics['directional_accuracy']:.1%}")

    print(f"\nRegimes learned: {trainer.regimes_seen}")
    print(f"Replay buffer size: {trainer.replay_buffer.size}")
    print("\n=== Example Complete ===")
