"""
Meta-Volatility Prediction using MAML.

Implements Model-Agnostic Meta-Learning for fast adaptation
of volatility forecasting models to new assets and market regimes.
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VolatilityNet(nn.Module):
    """Neural network for volatility prediction.

    Architecture:
        Input -> Linear(input_dim, hidden_dim) -> ReLU
              -> Linear(hidden_dim, hidden_dim) -> ReLU
              -> Linear(hidden_dim, hidden_dim // 2) -> ReLU
              -> Linear(hidden_dim // 2, 1) -> Softplus
    """

    def __init__(self, input_dim: int = 11, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            h = F.relu(layer(h))
        h = self.layers[-1](h)
        return F.softplus(h).squeeze(-1)


def forward_with_weights(
    x: torch.Tensor,
    weights: dict,
    layer_names: List[str],
) -> torch.Tensor:
    """Forward pass using explicit weight dictionaries (for MAML inner loop)."""
    h = x
    num_layers = len(layer_names) // 2  # Each layer has weight + bias
    for i in range(num_layers):
        w = weights[layer_names[i * 2]]
        b = weights[layer_names[i * 2 + 1]]
        h = F.linear(h, w, b)
        if i < num_layers - 1:
            h = F.relu(h)
    return F.softplus(h).squeeze(-1)


def volatility_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: float = 0.1,
) -> torch.Tensor:
    """Combined MSE + MAE loss for volatility prediction.

    MSE provides accuracy, MAE provides robustness to outliers.
    """
    mse = ((pred - target) ** 2).mean()
    mae = (pred - target).abs().mean()
    return mse + lam * mae


class MAMLVolatility:
    """MAML-based meta-learner for volatility prediction.

    Args:
        input_dim: Number of input features.
        hidden_dim: Hidden layer size.
        inner_lr: Learning rate for task-specific adaptation.
        outer_lr: Learning rate for meta-update.
        inner_steps: Number of gradient steps in inner loop.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        self.model = VolatilityNet(input_dim, hidden_dim)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=outer_lr
        )
        self._layer_names: Optional[List[str]] = None

    @property
    def layer_names(self) -> List[str]:
        if self._layer_names is None:
            self._layer_names = [
                name for name, _ in self.model.named_parameters()
            ]
        return self._layer_names

    def inner_update(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> dict:
        """Perform task-specific adaptation on support set.

        Returns adapted weights after inner_steps gradient updates.
        """
        fast_weights = {
            name: p.clone()
            for name, p in self.model.named_parameters()
        }

        for step in range(self.inner_steps):
            pred = forward_with_weights(
                support_x, fast_weights, self.layer_names
            )
            loss = volatility_loss(pred, support_y)
            grads = torch.autograd.grad(
                loss, fast_weights.values(), create_graph=True
            )
            fast_weights = {
                name: w - self.inner_lr * g
                for (name, w), g in zip(fast_weights.items(), grads)
            }

        return fast_weights

    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor,
                          torch.Tensor, torch.Tensor]],
    ) -> float:
        """One step of MAML meta-training across a batch of tasks.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples.

        Returns:
            Meta-loss value (float).
        """
        meta_loss = torch.tensor(0.0)

        for support_x, support_y, query_x, query_y in tasks:
            fast_weights = self.inner_update(support_x, support_y)
            pred = forward_with_weights(
                query_x, fast_weights, self.layer_names
            )
            task_loss = volatility_loss(pred, query_y)
            meta_loss = meta_loss + task_loss

        meta_loss = meta_loss / len(tasks)

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt_and_predict(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> np.ndarray:
        """Adapt to a new task and make predictions.

        This is the inference method: given a few support examples,
        adapt the model and predict on query examples.
        """
        self.model.eval()
        with torch.no_grad():
            fast_weights = {
                name: p.clone()
                for name, p in self.model.named_parameters()
            }

        # Re-enable gradients for adaptation
        for v in fast_weights.values():
            v.requires_grad_(True)

        for step in range(self.inner_steps):
            pred = forward_with_weights(
                support_x, fast_weights, self.layer_names
            )
            loss = volatility_loss(pred, support_y)
            grads = torch.autograd.grad(loss, fast_weights.values())
            fast_weights = {
                name: (w - self.inner_lr * g).detach().requires_grad_(True)
                for (name, w), g in zip(fast_weights.items(), grads)
            }

        with torch.no_grad():
            predictions = forward_with_weights(
                query_x, fast_weights, self.layer_names
            )

        return predictions.numpy()

    def save(self, path: str) -> None:
        """Save model state."""
        torch.save({
            'model_state': self.model.state_dict(),
            'inner_lr': self.inner_lr,
            'inner_steps': self.inner_steps,
        }, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model state."""
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state'])
        self.inner_lr = checkpoint['inner_lr']
        self.inner_steps = checkpoint['inner_steps']
        logger.info("Model loaded from %s", path)
