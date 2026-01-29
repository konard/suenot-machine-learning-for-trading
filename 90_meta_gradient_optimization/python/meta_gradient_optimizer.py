"""
Meta-Gradient Optimization for Trading.

This module implements meta-gradient optimization where hyperparameters
(learning rate, loss function parameters) are learned through gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict


class MetaGradientTradingModel(nn.Module):
    """
    Neural network for trading predictions, compatible with
    meta-gradient optimization (functional forward pass).
    """

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class MetaGradientOptimizer:
    """
    Meta-Gradient Optimization for trading models.

    Learns optimal hyperparameters (learning rates, loss function parameters)
    by differentiating through the inner optimization loop.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr_init: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
        learn_lr: bool = True,
        learn_loss: bool = False,
        per_param_lr: bool = False,
    ):
        self.model = model
        self.inner_steps = inner_steps
        self.learn_lr = learn_lr
        self.learn_loss = learn_loss
        self.per_param_lr = per_param_lr

        # Initialize learnable learning rates (in log space for positivity)
        if per_param_lr:
            self.log_lr = nn.ParameterDict({
                name.replace('.', '_'): nn.Parameter(
                    torch.full_like(param, np.log(inner_lr_init))
                )
                for name, param in model.named_parameters()
            })
            self._param_name_map = {
                name: name.replace('.', '_')
                for name in dict(model.named_parameters()).keys()
            }
        else:
            self.log_lr = nn.Parameter(torch.tensor(np.log(inner_lr_init)))

        # Learnable loss function parameters
        if learn_loss:
            self.loss_weight_direction = nn.Parameter(torch.tensor(1.0))
            self.loss_weight_magnitude = nn.Parameter(torch.tensor(1.0))
            self.loss_asymmetry = nn.Parameter(torch.tensor(0.0))

        # Collect all meta-parameters
        meta_params = []
        if learn_lr:
            if per_param_lr:
                meta_params.extend(self.log_lr.values())
            else:
                meta_params.append(self.log_lr)
        if learn_loss:
            meta_params.extend([
                self.loss_weight_direction,
                self.loss_weight_magnitude,
                self.loss_asymmetry,
            ])

        self.meta_optimizer = torch.optim.Adam(meta_params, lr=meta_lr)

    def get_learning_rates(self) -> dict:
        """Get current learning rates."""
        if self.per_param_lr:
            return {
                name: torch.exp(self.log_lr[mapped]).item()
                for name, mapped in self._param_name_map.items()
            }
        else:
            return {"all": torch.exp(self.log_lr).item()}

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute (possibly learned) loss function."""
        if not self.learn_loss:
            return F.mse_loss(predictions, targets)

        errors = predictions - targets
        direction_errors = (torch.sign(predictions) != torch.sign(targets)).float()
        direction_penalty = direction_errors * F.softplus(self.loss_weight_direction)

        magnitude_loss = errors.pow(2) * F.softplus(self.loss_weight_magnitude)

        asymmetry = torch.sigmoid(self.loss_asymmetry)
        asymmetric_weight = torch.where(
            errors < 0,
            1.0 + asymmetry,
            1.0 - asymmetry * 0.5,
        )

        total_loss = (magnitude_loss * asymmetric_weight + direction_penalty).mean()
        return total_loss

    def _functional_forward(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Functional forward pass using specified parameters."""
        h = F.linear(x, params['fc1.weight'], params['fc1.bias'])
        h = F.relu(h)
        h = F.linear(h, params['fc2.weight'], params['fc2.bias'])
        h = F.relu(h)
        out = F.linear(h, params['fc3.weight'], params['fc3.bias'])
        return out

    def inner_loop(
        self,
        train_features: torch.Tensor,
        train_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Perform inner loop optimization with learnable hyperparameters."""
        adapted_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }

        for _ in range(self.inner_steps):
            predictions = self._functional_forward(train_features, adapted_params)
            loss = self.compute_loss(predictions, train_targets)

            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=True,
            )

            if self.per_param_lr:
                adapted_params = {
                    name: param - torch.exp(self.log_lr[self._param_name_map[name]]) * grad
                    for (name, param), grad in zip(adapted_params.items(), grads)
                }
            else:
                lr = torch.exp(self.log_lr)
                adapted_params = {
                    name: param - lr * grad
                    for (name, param), grad in zip(adapted_params.items(), grads)
                }

        return adapted_params

    def meta_train_step(
        self,
        tasks: List[Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ]],
    ) -> float:
        """
        Perform one meta-training step.

        Args:
            tasks: List of ((train_features, train_targets), (val_features, val_targets))

        Returns:
            Average meta-loss
        """
        self.meta_optimizer.zero_grad()
        total_meta_loss = 0.0

        for (train_features, train_targets), (val_features, val_targets) in tasks:
            adapted_params = self.inner_loop(train_features, train_targets)
            val_predictions = self._functional_forward(val_features, adapted_params)
            val_loss = F.mse_loss(val_predictions, val_targets)
            total_meta_loss += val_loss

        meta_loss = total_meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt_and_predict(
        self,
        train_features: torch.Tensor,
        train_targets: torch.Tensor,
        test_features: torch.Tensor,
    ) -> torch.Tensor:
        """Adapt model on train data and predict on test data."""
        adapted_params = self.inner_loop(train_features, train_targets)
        with torch.no_grad():
            predictions = self._functional_forward(test_features, adapted_params)
        return predictions


class OnlineMetaGradientTrader:
    """
    Online trading agent that continuously adapts its learning dynamics
    using meta-gradient optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        meta_optimizer: MetaGradientOptimizer,
        adaptation_window: int = 50,
        validation_window: int = 10,
    ):
        self.model = model
        self.meta_opt = meta_optimizer
        self.adaptation_window = adaptation_window
        self.validation_window = validation_window
        self.feature_buffer: List[torch.Tensor] = []
        self.target_buffer: List[torch.Tensor] = []

    def observe(self, features: torch.Tensor, target: torch.Tensor):
        """Add new observation to buffer."""
        self.feature_buffer.append(features)
        self.target_buffer.append(target)

        max_buffer = self.adaptation_window + self.validation_window + 10
        if len(self.feature_buffer) > max_buffer:
            self.feature_buffer = self.feature_buffer[-max_buffer:]
            self.target_buffer = self.target_buffer[-max_buffer:]

    def update_and_predict(self, current_features: torch.Tensor) -> Optional[float]:
        """Update meta-parameters and predict."""
        total_needed = self.adaptation_window + self.validation_window
        if len(self.feature_buffer) < total_needed:
            return None

        train_features = torch.stack(
            self.feature_buffer[-total_needed:-self.validation_window]
        )
        train_targets = torch.stack(
            self.target_buffer[-total_needed:-self.validation_window]
        )
        val_features = torch.stack(
            self.feature_buffer[-self.validation_window:]
        )
        val_targets = torch.stack(
            self.target_buffer[-self.validation_window:]
        )

        task = [((train_features, train_targets), (val_features, val_targets))]
        self.meta_opt.meta_train_step(task)

        prediction = self.meta_opt.adapt_and_predict(
            train_features, train_targets, current_features.unsqueeze(0)
        )
        return prediction.item()


if __name__ == "__main__":
    print("Meta-Gradient Optimization for Trading")
    print("=" * 50)

    # Create model and optimizer
    model = MetaGradientTradingModel(input_size=11, hidden_size=32, output_size=1)
    meta_opt = MetaGradientOptimizer(
        model=model,
        inner_lr_init=0.01,
        meta_lr=0.001,
        inner_steps=3,
        learn_lr=True,
        learn_loss=True,
        per_param_lr=False,
    )

    print(f"Initial learning rates: {meta_opt.get_learning_rates()}")

    # Generate synthetic tasks for demonstration
    num_tasks = 4
    for epoch in range(100):
        tasks = []
        for _ in range(num_tasks):
            train_f = torch.randn(20, 11)
            train_t = torch.randn(20, 1) * 0.01
            val_f = torch.randn(10, 11)
            val_t = torch.randn(10, 1) * 0.01
            tasks.append(((train_f, train_t), (val_f, val_t)))

        loss = meta_opt.meta_train_step(tasks)

        if epoch % 10 == 0:
            lrs = meta_opt.get_learning_rates()
            print(f"Epoch {epoch:3d}: Meta-Loss={loss:.6f}, LR={lrs}")

    print(f"\nFinal learning rates: {meta_opt.get_learning_rates()}")
    print("Training complete!")
