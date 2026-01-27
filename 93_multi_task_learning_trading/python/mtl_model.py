"""
Multi-Task Learning model and trainer for trading.

This module provides:
- SharedEncoder: Shared feature backbone for all tasks
- TaskHead: Task-specific output head
- MultiTaskTradingModel: Complete MTL architecture
- UncertaintyWeighting: Learnable task loss weighting
- MTLTrainer: Training loop with gradient clipping and logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SharedEncoder(nn.Module):
    """
    Shared feature encoder for multi-task learning.

    Produces a common representation consumed by all task heads.
    Architecture: [Linear -> BatchNorm -> ReLU -> Dropout] x N
    """

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
    """
    Task-specific output head.

    Supports regression and classification tasks.
    """

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

    Architecture:
        Input -> SharedEncoder -> { TaskHead_return,
                                     TaskHead_volatility,
                                     TaskHead_direction,
                                     TaskHead_volume }
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

    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference from numpy arrays."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            outputs = self.forward(x)
            return {k: v.numpy().flatten() for k, v in outputs.items()}


class UncertaintyWeighting(nn.Module):
    """
    Learnable task weighting via homoscedastic uncertainty.

    Each task has a learnable log-variance parameter. The total loss
    scales each task loss by precision (1/sigma^2) and adds a
    log-barrier term.

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics", CVPR 2018.
    """

    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=losses[0].device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-2 * self.log_sigma[i])
            total += precision * loss + self.log_sigma[i]
        return total

    def get_weights(self) -> List[float]:
        """Return current effective task weights."""
        with torch.no_grad():
            return [torch.exp(-2 * ls).item() for ls in self.log_sigma]


class MTLTrainer:
    """
    Trainer for multi-task trading models.

    Supports:
    - Uncertainty-based task weighting (Kendall et al. 2018)
    - Fixed equal weighting
    - Gradient clipping
    - Per-task loss tracking
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
        """
        Perform one training step.

        Args:
            features: Input feature tensor (batch_size, num_features)
            targets: Dict mapping task name -> target tensor

        Returns:
            Dict mapping task name -> loss value (float)
        """
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

    def train_epoch(self, features: torch.Tensor,
                    targets: Dict[str, torch.Tensor],
                    batch_size: int = 64) -> Dict[str, float]:
        """
        Train for one full epoch with mini-batches.

        Args:
            features: Full feature tensor
            targets: Full target dict
            batch_size: Mini-batch size

        Returns:
            Average losses per task over the epoch
        """
        n = features.shape[0]
        indices = torch.randperm(n)
        epoch_losses: Dict[str, List[float]] = {}

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch_features = features[idx]
            batch_targets = {k: v[idx] for k, v in targets.items()}

            step_losses = self.train_step(batch_features, batch_targets)
            for k, v in step_losses.items():
                epoch_losses.setdefault(k, []).append(v)

        return {k: np.mean(v) for k, v in epoch_losses.items()}


@dataclass
class TrainingConfig:
    """Configuration for MTL training."""
    input_size: int = 10
    shared_hidden: List[int] = None
    head_hidden: int = 32
    dropout: float = 0.1
    learning_rate: float = 1e-3
    use_uncertainty_weighting: bool = True
    num_epochs: int = 200
    batch_size: int = 64

    def __post_init__(self):
        if self.shared_hidden is None:
            self.shared_hidden = [64, 32]


def main():
    """Example usage of the MTL model."""
    logger.info("Multi-Task Learning Trading Model Demo")

    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y_return = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1).astype(np.float32)
    y_vol = np.abs(y_return * 0.5 + np.random.randn(n_samples) * 0.05).astype(np.float32)
    y_dir = (y_return > 0).astype(np.float32)
    y_volume = np.abs(np.random.randn(n_samples) * 100).astype(np.float32)

    features = torch.tensor(X)
    targets = {
        "return": torch.tensor(y_return).unsqueeze(1),
        "volatility": torch.tensor(y_vol).unsqueeze(1),
        "direction": torch.tensor(y_dir).unsqueeze(1),
        "volume": torch.tensor(y_volume).unsqueeze(1),
    }

    # Create model and trainer
    model = MultiTaskTradingModel(
        input_size=n_features,
        shared_hidden=[64, 32],
        head_hidden=32,
        dropout=0.1,
    )
    trainer = MTLTrainer(model, lr=1e-3, use_uncertainty_weighting=True)

    # Train
    for epoch in range(100):
        losses = trainer.train_epoch(features, targets, batch_size=64)
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: {losses}")

    # Inference
    predictions = model.predict(X[:5])
    logger.info(f"Sample predictions: {predictions}")

    if trainer.weighting is not None:
        logger.info(f"Learned task weights: {trainer.weighting.get_weights()}")


if __name__ == "__main__":
    main()
