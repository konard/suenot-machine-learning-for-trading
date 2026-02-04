"""
Concrete Dropout Implementation
================================

This module implements Concrete Dropout, which learns the optimal
dropout rate during training (Gal et al., 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


class ConcreteDropout(nn.Module):
    """
    Concrete Dropout layer that learns the dropout probability.

    Based on: "Concrete Dropout" (Gal et al., 2017)
    https://arxiv.org/abs/1705.07832
    """

    def __init__(self,
                 weight_regularizer: float = 1e-6,
                 dropout_regularizer: float = 1e-5,
                 init_min: float = 0.1,
                 init_max: float = 0.1,
                 temperature: float = 0.1):
        """
        Args:
            weight_regularizer: Weight regularization factor
            dropout_regularizer: Dropout regularization factor
            init_min: Minimum initial dropout probability
            init_max: Maximum initial dropout probability
            temperature: Temperature for concrete distribution
        """
        super().__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.temperature = temperature

        # Initialize p_logit to achieve initial dropout in [init_min, init_max]
        init_p = (init_min + init_max) / 2
        init_logit = np.log(init_p / (1 - init_p))

        self.p_logit = nn.Parameter(torch.tensor(init_logit))

    @property
    def p(self) -> torch.Tensor:
        """Get dropout probability from logit."""
        return torch.sigmoid(self.p_logit)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply concrete dropout.

        Args:
            x: Input tensor

        Returns:
            Tuple of (output, regularization_loss)
        """
        p = self.p

        if self.training:
            # Concrete/Gumbel-Softmax relaxation
            eps = 1e-7

            # Sample from uniform distribution
            unif_noise = torch.rand_like(x)

            # Apply concrete relaxation
            drop_prob = (
                torch.log(p + eps)
                - torch.log(1 - p + eps)
                + torch.log(unif_noise + eps)
                - torch.log(1 - unif_noise + eps)
            )
            drop_prob = torch.sigmoid(drop_prob / self.temperature)

            # Apply dropout mask
            mask = 1 - drop_prob
            x = x * mask / (1 - p + eps)  # Scale to maintain expected value

        # Calculate regularization (entropy term for dropout)
        eps = 1e-7
        dropout_reg = self.dropout_regularizer * (
            p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)
        )

        return x, dropout_reg


class ConcreteDropoutLinear(nn.Module):
    """
    Linear layer with Concrete Dropout.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 weight_regularizer: float = 1e-6,
                 dropout_regularizer: float = 1e-5):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.concrete_dropout = ConcreteDropout(
            weight_regularizer=weight_regularizer,
            dropout_regularizer=dropout_regularizer
        )
        self.weight_regularizer = weight_regularizer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple of (output, total_regularization)
        """
        # Apply concrete dropout before linear
        x, dropout_reg = self.concrete_dropout(x)

        # Apply linear transformation
        x = self.linear(x)

        # Weight regularization
        weight_reg = self.weight_regularizer * torch.sum(self.linear.weight ** 2) / 2

        total_reg = dropout_reg + weight_reg

        return x, total_reg

    @property
    def dropout_rate(self) -> float:
        """Get current dropout rate."""
        return self.concrete_dropout.p.item()


class ConcreteDropoutModel(nn.Module):
    """
    Neural network with Concrete Dropout for automatic dropout tuning.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int = 1,
                 weight_regularizer: float = 1e-6,
                 dropout_regularizer: float = 1e-5,
                 n_mc_samples: int = 50):
        super().__init__()

        self.n_mc_samples = n_mc_samples
        self.layers = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(
                ConcreteDropoutLinear(
                    prev_dim, hidden_dim,
                    weight_regularizer=weight_regularizer,
                    dropout_regularizer=dropout_regularizer
                )
            )
            prev_dim = hidden_dim

        # Output layer (no dropout)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple of (prediction, total_regularization)
        """
        total_reg = torch.tensor(0.0, device=x.device)

        for layer in self.layers:
            x, reg = layer(x)
            x = F.relu(x)
            total_reg = total_reg + reg

        x = self.output_layer(x)

        return x, total_reg

    def predict_with_uncertainty(self,
                                 x: torch.Tensor,
                                 n_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty using MC Dropout.

        Args:
            x: Input tensor
            n_samples: Number of MC samples

        Returns:
            Dictionary with prediction statistics
        """
        if n_samples is None:
            n_samples = self.n_mc_samples

        self.train()  # Keep dropout active

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred, _ = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)

        return {
            'mean': predictions.mean(dim=0),
            'std': predictions.std(dim=0),
            'variance': predictions.var(dim=0),
            'samples': predictions,
        }

    def get_dropout_rates(self) -> List[float]:
        """Get learned dropout rates for each layer."""
        return [layer.dropout_rate for layer in self.layers]


class ConcreteDropoutRegressor(ConcreteDropoutModel):
    """
    Concrete Dropout model for regression with learned dropout rates.
    """

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            patience: int = 10) -> Dict[str, List]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            patience: Early stopping patience

        Returns:
            Training history
        """
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)

        if X_val is not None and y_val is not None:
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        history = {
            'train_loss': [],
            'val_loss': [],
            'dropout_rates': [],
        }
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            train_losses = []

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                predictions, reg_loss = self.forward(X_batch)
                mse_loss = F.mse_loss(predictions, y_batch)
                loss = mse_loss + reg_loss
                loss.backward()
                optimizer.step()
                train_losses.append(mse_loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['dropout_rates'].append(self.get_dropout_rates())

            if X_val is not None:
                self.set_eval_mode()
                with torch.no_grad():
                    val_pred, _ = self.forward(X_val)
                    val_loss = F.mse_loss(val_pred, y_val).item()
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                if (epoch + 1) % 10 == 0:
                    dropout_str = ', '.join([f'{p:.4f}' for p in self.get_dropout_rates()])
                    logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.6f}, Val: {val_loss:.6f}, Dropout: [{dropout_str}]")

        return history

    def set_eval_mode(self):
        """Set to evaluation mode (dropout still active for MC inference)."""
        self.eval()


def create_concrete_dropout_model(input_dim: int,
                                  hidden_dims: List[int] = None,
                                  weight_regularizer: float = 1e-6,
                                  dropout_regularizer: float = 1e-5,
                                  n_mc_samples: int = 50) -> ConcreteDropoutRegressor:
    """
    Factory function to create Concrete Dropout model.

    Args:
        input_dim: Number of input features
        hidden_dims: Hidden layer dimensions
        weight_regularizer: Weight regularization factor
        dropout_regularizer: Dropout regularization factor
        n_mc_samples: Number of MC samples for inference

    Returns:
        ConcreteDropoutRegressor instance
    """
    return ConcreteDropoutRegressor(
        input_dim=input_dim,
        hidden_dims=hidden_dims or [128, 64, 32],
        output_dim=1,
        weight_regularizer=weight_regularizer,
        dropout_regularizer=dropout_regularizer,
        n_mc_samples=n_mc_samples,
    )


if __name__ == '__main__':
    print("=== Concrete Dropout Example ===\n")

    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(1000) * 0.1).astype(np.float32)

    # Split data
    train_idx = int(0.8 * len(X))
    X_train, X_val = X[:train_idx], X[train_idx:]
    y_train, y_val = y[:train_idx], y[train_idx:]

    # Create model with Concrete Dropout
    model = create_concrete_dropout_model(
        input_dim=20,
        hidden_dims=[64, 32],
        weight_regularizer=1e-6,
        dropout_regularizer=1e-5,
        n_mc_samples=50
    )

    print("Initial dropout rates:", model.get_dropout_rates())

    print("\nTraining model...")
    history = model.fit(X_train, y_train, X_val, y_val, epochs=50, patience=10)

    print("\nLearned dropout rates:", model.get_dropout_rates())

    # Make predictions
    print("\nMaking predictions with uncertainty...")
    X_test = torch.FloatTensor(X_val[:5])
    result = model.predict_with_uncertainty(X_test, n_samples=100)

    print("\nPredictions:")
    for i in range(5):
        mean = result['mean'][i].item()
        std = result['std'][i].item()
        actual = y_val[i]
        print(f"  Sample {i+1}: Predicted = {mean:.4f} +/- {std:.4f}, Actual = {actual:.4f}")
