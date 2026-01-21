"""
Continual Meta-Learning Algorithm for Trading

This module implements the Continual Meta-Learning (CML) algorithm for algorithmic trading.
CML combines meta-learning's rapid adaptation with continual learning's ability to
retain knowledge across changing market regimes.

Reference: Javed, K., & White, M. (2019).
"Meta-Learning Representations for Continual Learning." NeurIPS.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Generator
import copy
import numpy as np
from collections import deque


class TradingModel(nn.Module):
    """
    Neural network for trading signal prediction.

    A feedforward network with dropout for regularization,
    suitable for meta-learning with the CML algorithm.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        """
        Initialize the trading model.

        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output predictions
        """
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
        """Forward pass through the network."""
        return self.network(x)


class ContinualMetaLearner:
    """
    Continual Meta-Learning algorithm for trading strategy adaptation.

    Combines meta-learning's rapid adaptation with continual learning's
    ability to retain knowledge across changing market regimes.

    Features:
    - Experience Replay: Stores and replays past tasks during training
    - Elastic Weight Consolidation (EWC): Protects important parameters
    - Fast Adaptation: Quick adaptation to new market conditions

    Example:
        >>> model = TradingModel(input_size=8)
        >>> cml = ContinualMetaLearner(model, inner_lr=0.01, outer_lr=0.001)
        >>> # Continual meta-training
        >>> for epoch in range(1000):
        ...     task, regime = next(task_generator)
        ...     metrics = cml.meta_train_step(task, regime=regime)
        >>> # Fast adaptation to new regime
        >>> adapted_model = cml.adapt(new_task_data)
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

        # Get device from model
        self.device = next(model.parameters()).device

    def compute_fisher_information(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor]]
    ):
        """
        Compute Fisher Information Matrix for EWC.

        The Fisher information approximates parameter importance by
        measuring the curvature of the loss surface.

        Args:
            tasks: List of (features, labels) tuples
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
        # Clone model for task-specific adaptation
        adapted_model = copy.deepcopy(self.model)
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr
        )

        features, labels = support_data
        features = features.to(self.device)
        labels = labels.to(self.device)

        # Perform k steps of SGD on the task
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
            support_data: Small amount of data from the new task (features, labels)
            adaptation_steps: Number of gradient steps (default: inner_steps)

        Returns:
            Adapted model ready for prediction
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

    def save(self, path: str):
        """Save the continual meta-learned model and state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'fisher_info': self.fisher_info,
            'optimal_params': self.optimal_params,
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'inner_steps': self.inner_steps,
            'memory_size': self.memory_size,
            'ewc_lambda': self.ewc_lambda,
            'replay_batch_size': self.replay_batch_size,
            'regime_history': self.regime_history,
        }, path)

    @classmethod
    def load(cls, path: str, model: nn.Module) -> 'ContinualMetaLearner':
        """Load a saved ContinualMetaLearner."""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])

        cml = cls(
            model=model,
            inner_lr=checkpoint['inner_lr'],
            outer_lr=checkpoint['outer_lr'],
            inner_steps=checkpoint['inner_steps'],
            memory_size=checkpoint['memory_size'],
            ewc_lambda=checkpoint['ewc_lambda'],
            replay_batch_size=checkpoint['replay_batch_size'],
        )
        cml.fisher_info = checkpoint['fisher_info']
        cml.optimal_params = checkpoint['optimal_params']
        cml.regime_history = checkpoint['regime_history']

        return cml


def train_continual_meta(
    cml: ContinualMetaLearner,
    task_generator: Generator,
    num_epochs: int,
    log_interval: int = 100
) -> List[Dict[str, float]]:
    """
    Continual meta-training loop.

    Args:
        cml: ContinualMetaLearner instance
        task_generator: Generator yielding (task, regime) tuples
        num_epochs: Number of training epochs
        log_interval: How often to print progress

    Returns:
        List of metrics per epoch
    """
    metrics_history = []

    for epoch in range(num_epochs):
        task, regime = next(task_generator)
        metrics = cml.meta_train_step(task, regime=regime)
        metrics_history.append(metrics)

        if epoch % log_interval == 0:
            print(f"Epoch {epoch}, Regime: {regime}, "
                  f"New Loss: {metrics['new_task_loss']:.6f}, "
                  f"Replay Loss: {metrics['replay_loss']:.6f}, "
                  f"Memory: {metrics['memory_size']}")

    return metrics_history


if __name__ == "__main__":
    # Example usage
    print("Continual Meta-Learning for Trading")
    print("=" * 50)

    # Create model
    model = TradingModel(input_size=8, hidden_size=64, output_size=1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create CML trainer
    cml = ContinualMetaLearner(
        model=model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        memory_size=100,
        ewc_lambda=0.4,
        replay_batch_size=4
    )
    print(f"Inner LR: {cml.inner_lr}")
    print(f"Outer LR: {cml.outer_lr}")
    print(f"Inner Steps: {cml.inner_steps}")
    print(f"Memory Size: {cml.memory_size}")
    print(f"EWC Lambda: {cml.ewc_lambda}")

    # Create dummy tasks for different "regimes"
    regimes = ['bull', 'bear', 'high_vol', 'low_vol']

    print("\nSimulating continual meta-training across regimes...")

    for epoch in range(100):
        regime = np.random.choice(regimes)

        # Generate dummy data for this regime
        support_features = torch.randn(20, 8)
        support_labels = torch.randn(20, 1)
        query_features = torch.randn(10, 8)
        query_labels = torch.randn(10, 1)

        task = ((support_features, support_labels),
                (query_features, query_labels))

        metrics = cml.meta_train_step(task, regime=regime)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Regime: {regime}, "
                  f"New Loss: {metrics['new_task_loss']:.4f}, "
                  f"Memory: {metrics['memory_size']}")

    # Test adaptation
    print("\nTesting fast adaptation to new data...")
    test_support = (torch.randn(20, 8), torch.randn(20, 1))
    adapted_model = cml.adapt(test_support, adaptation_steps=10)

    # Make prediction
    test_input = torch.randn(1, 8)
    with torch.no_grad():
        prediction = adapted_model(test_input)
        print(f"Prediction: {prediction.item():.6f}")

    # Evaluate forgetting
    print("\nEvaluating forgetting on past regimes...")
    test_tasks = []
    for _ in range(4):
        test_tasks.append((
            (torch.randn(20, 8), torch.randn(20, 1)),
            (torch.randn(10, 8), torch.randn(10, 1))
        ))

    forgetting_metrics = cml.evaluate_forgetting(test_tasks)
    print(f"Mean Loss: {forgetting_metrics['mean_loss']:.6f}")
    print(f"Std Loss: {forgetting_metrics['std_loss']:.6f}")

    print("\nDone!")
