"""
MAML (Model-Agnostic Meta-Learning) implementation for trading.

This module provides:
- TradingModel: Simple neural network for price prediction
- MAMLTrainer: MAML and FOMAML training algorithm
- TradingStrategy: Adaptive trading using MAML
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from copy import deepcopy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingModel(nn.Module):
    """
    Simple neural network for trading signal prediction.

    Architecture: Input -> Hidden (ReLU) -> Output (Tanh)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
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

    def to_tensors(self) -> Tuple[torch.Tensor, ...]:
        """Convert to PyTorch tensors."""
        return (
            torch.tensor(self.support_features, dtype=torch.float32),
            torch.tensor(self.support_labels, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(self.query_features, dtype=torch.float32),
            torch.tensor(self.query_labels, dtype=torch.float32).unsqueeze(-1)
        )


class MAMLTrainer:
    """
    MAML meta-learning trainer.

    Implements both full MAML (with second-order gradients) and
    FOMAML (first-order approximation).

    Reference: Finn et al., 2017. "Model-Agnostic Meta-Learning
    for Fast Adaptation of Deep Networks"
    """

    def __init__(
        self,
        model: TradingModel,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = True
    ):
        """
        Initialize MAML trainer.

        Args:
            model: The model to meta-train
            inner_lr: Learning rate for task adaptation (alpha)
            outer_lr: Meta-learning rate (beta)
            inner_steps: Number of gradient steps per task
            first_order: Use FOMAML if True (recommended for stability)
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order

        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=outer_lr
        )

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

        Args:
            task: Task data with support and query sets
            model: Model to adapt (uses self.model if None)

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

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=not self.first_order
            )

            # Manual SGD update
            with torch.no_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    param.sub_(self.inner_lr * grad)

        # Evaluate on query set
        query_loss = self._compute_loss(adapted_model, query_x, query_y)

        return adapted_model, query_loss

    def meta_train_step(self, tasks: List[TaskData]) -> float:
        """
        Perform one meta-training step.

        Args:
            tasks: Batch of tasks for meta-training

        Returns:
            Average query loss across tasks
        """
        if not tasks:
            return 0.0

        self.model.train()
        self.meta_optimizer.zero_grad()

        total_loss = 0.0

        for task in tasks:
            _, query_loss = self._inner_loop(task)
            total_loss = total_loss + query_loss

        # Average loss
        avg_loss = total_loss / len(tasks)

        # Meta-update
        avg_loss.backward()
        self.meta_optimizer.step()

        return avg_loss.item()

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

    def save(self, path: str):
        """Save model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'inner_steps': self.inner_steps,
            'first_order': self.first_order
        }, path)
        logger.info(f"Saved trainer to {path}")

    def load(self, path: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.inner_lr = checkpoint['inner_lr']
        self.outer_lr = checkpoint['outer_lr']
        self.inner_steps = checkpoint['inner_steps']
        self.first_order = checkpoint['first_order']
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
    MAML-based adaptive trading strategy.

    Adapts to recent market data before making predictions.
    """

    def __init__(
        self,
        trainer: MAMLTrainer,
        threshold: float = 0.001,
        strong_threshold: float = 0.005,
        stop_loss: float = 0.02,
        take_profit: float = 0.04,
        adaptation_window: int = 20,
        adaptation_steps: int = 5
    ):
        """
        Initialize trading strategy.

        Args:
            trainer: MAML trainer with meta-learned model
            threshold: Minimum prediction for trading signal
            strong_threshold: Threshold for strong signals
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            adaptation_window: Number of recent samples for adaptation
            adaptation_steps: Gradient steps for adaptation
        """
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
        # Get adaptation data
        n = min(len(recent_features), len(recent_returns), self.adaptation_window)
        if n == 0:
            return TradingSignal(0, 0.0, 0.0, timestamp, symbol)

        adapt_features = recent_features[-n:]
        adapt_labels = recent_returns[-n:]

        # Adapt model
        adapted_model = self.trainer.adapt(
            adapt_features,
            adapt_labels,
            self.adaptation_steps
        )

        # Generate prediction
        prediction = adapted_model.predict(current_features)[0]

        # Convert to signal
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
    query_size: int = 10
) -> List[TaskData]:
    """
    Create meta-learning tasks from time series data.

    Args:
        features: Feature matrix (N, D)
        returns: Return vector (N,)
        num_tasks: Number of tasks to create
        support_size: Samples per support set
        query_size: Samples per query set

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
            query_labels=returns[support_end:query_end]
        ))

    return tasks


if __name__ == "__main__":
    # Example usage
    print("=== MAML Trading Example ===\n")

    # Create model
    input_size = 11  # Number of features
    hidden_size = 64
    model = TradingModel(input_size, hidden_size)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters\n")

    # Create trainer
    trainer = MAMLTrainer(
        model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        first_order=True
    )
    print("Created FOMAML trainer\n")

    # Generate dummy tasks
    print("Generating dummy tasks...")
    dummy_tasks = []
    for i in range(5):
        task = TaskData(
            support_features=np.random.randn(20, input_size),
            support_labels=np.random.randn(20),
            query_features=np.random.randn(10, input_size),
            query_labels=np.random.randn(10)
        )
        dummy_tasks.append(task)
    print(f"Created {len(dummy_tasks)} tasks\n")

    # Meta-training
    print("Meta-training...")
    for epoch in range(10):
        loss = trainer.meta_train_step(dummy_tasks)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {loss:.6f}")

    print("\n=== Example Complete ===")
