"""
MC Dropout Model for Trading
=============================

This module implements Monte Carlo Dropout for uncertainty estimation
in trading prediction models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class MCDropoutConfig:
    """Configuration for MC Dropout model."""
    input_dim: int = 20           # Number of input features
    hidden_dims: List[int] = None  # Hidden layer dimensions
    output_dim: int = 1           # Output dimension (1 for regression)
    dropout_rate: float = 0.1     # Dropout probability
    n_mc_samples: int = 50        # Number of MC samples for inference

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


class MCDropoutModel(nn.Module):
    """
    Neural network with MC Dropout for uncertainty estimation.

    MC Dropout keeps dropout active during inference to generate
    multiple predictions and estimate uncertainty.
    """

    def __init__(self, config: MCDropoutConfig):
        super().__init__()

        self.config = config
        layers = []
        prev_dim = config.input_dim

        # Build hidden layers with dropout
        for i, hidden_dim in enumerate(config.hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=config.dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer (no dropout on output)
        layers.append(nn.Linear(prev_dim, config.output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        return self.network(x)

    def predict_with_uncertainty(self,
                                 x: torch.Tensor,
                                 n_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using MC Dropout.

        This method keeps dropout active during inference and runs
        multiple forward passes to estimate prediction uncertainty.

        Args:
            x: Input tensor of shape [batch_size, input_dim]
            n_samples: Number of MC samples (default: config.n_mc_samples)

        Returns:
            Dictionary containing:
            - mean: Mean prediction
            - std: Standard deviation (uncertainty)
            - variance: Variance
            - samples: All MC samples
        """
        if n_samples is None:
            n_samples = self.config.n_mc_samples

        # IMPORTANT: Keep model in training mode to activate dropout
        self.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        # Stack predictions: [n_samples, batch_size, output_dim]
        predictions = torch.stack(predictions, dim=0)

        # Calculate statistics
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        std = predictions.std(dim=0)

        return {
            'mean': mean,
            'std': std,
            'variance': variance,
            'samples': predictions,
        }

    def get_confidence_interval(self,
                                x: torch.Tensor,
                                confidence: float = 0.95,
                                n_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get confidence interval for predictions.

        Args:
            x: Input tensor
            confidence: Confidence level (0-1)
            n_samples: Number of MC samples

        Returns:
            Tuple of (lower_bound, upper_bound) tensors
        """
        result = self.predict_with_uncertainty(x, n_samples)

        # Calculate z-score for desired confidence level
        z = stats.norm.ppf((1 + confidence) / 2)

        lower = result['mean'] - z * result['std']
        upper = result['mean'] + z * result['std']

        return lower, upper


class MCDropoutRegressor(MCDropoutModel):
    """
    MC Dropout model for regression tasks (e.g., return prediction).
    """

    def __init__(self, config: MCDropoutConfig):
        config.output_dim = 1
        super().__init__(config)

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            weight_decay: float = 0.01,
            patience: int = 10) -> Dict[str, List[float]]:
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
            weight_decay: L2 regularization
            patience: Early stopping patience

        Returns:
            Dictionary with training history
        """
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)

        if X_val is not None and y_val is not None:
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val).unsqueeze(1)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer with weight decay (L2 regularization)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Loss function
        criterion = nn.MSELoss()

        # Training history
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.train()
            train_losses = []

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                predictions = self.forward(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation
            if X_val is not None:
                self.set_eval_mode()
                with torch.no_grad():
                    val_pred = self.forward(X_val)
                    val_loss = criterion(val_pred, y_val).item()
                history['val_loss'].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")

        return history

    def set_eval_mode(self):
        """Set model to evaluation mode (dropout off)."""
        super().eval()


class MCDropoutClassifier(MCDropoutModel):
    """
    MC Dropout model for classification tasks (e.g., direction prediction).
    """

    def __init__(self, config: MCDropoutConfig, n_classes: int = 3):
        """
        Args:
            config: Model configuration
            n_classes: Number of classes (default 3: up/neutral/down)
        """
        config.output_dim = n_classes
        super().__init__(config)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.network(x)

    def predict_proba_with_uncertainty(self,
                                       x: torch.Tensor,
                                       n_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Predict class probabilities with uncertainty.

        Args:
            x: Input tensor
            n_samples: Number of MC samples

        Returns:
            Dictionary with probability statistics
        """
        if n_samples is None:
            n_samples = self.config.n_mc_samples

        self.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

        predictions = torch.stack(predictions, dim=0)

        mean_probs = predictions.mean(dim=0)
        std_probs = predictions.std(dim=0)

        # Predictive entropy (total uncertainty)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)

        return {
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'entropy': entropy,
            'samples': predictions,
        }

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            weight_decay: float = 0.01,
            patience: int = 10) -> Dict[str, List[float]]:
        """Train the classifier."""
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)

        if X_val is not None and y_val is not None:
            X_val = torch.FloatTensor(X_val)
            y_val = torch.LongTensor(y_val)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            train_losses = []
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.forward(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                _, predicted = torch.max(logits, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            avg_train_loss = np.mean(train_losses)
            train_acc = correct / total
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)

            if X_val is not None:
                self.set_eval_mode()
                with torch.no_grad():
                    val_logits = self.forward(X_val)
                    val_loss = criterion(val_logits, y_val).item()
                    _, val_pred = torch.max(val_logits, 1)
                    val_acc = (val_pred == y_val).sum().item() / len(y_val)

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        return history

    def set_eval_mode(self):
        """Set model to evaluation mode (dropout off)."""
        super().eval()


def create_mc_dropout_model(input_dim: int,
                            hidden_dims: List[int] = None,
                            dropout_rate: float = 0.1,
                            n_mc_samples: int = 50,
                            task: str = 'regression') -> MCDropoutModel:
    """
    Factory function to create MC Dropout model.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability
        n_mc_samples: Number of MC samples for inference
        task: 'regression' or 'classification'

    Returns:
        MCDropoutModel instance
    """
    config = MCDropoutConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims or [128, 64, 32],
        dropout_rate=dropout_rate,
        n_mc_samples=n_mc_samples,
    )

    if task == 'regression':
        return MCDropoutRegressor(config)
    elif task == 'classification':
        return MCDropoutClassifier(config)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == '__main__':
    # Example usage
    print("=== MC Dropout Model Example ===\n")

    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(1000) * 0.1).astype(np.float32)

    # Split data
    train_idx = int(0.8 * len(X))
    X_train, X_val = X[:train_idx], X[train_idx:]
    y_train, y_val = y[:train_idx], y[train_idx:]

    # Create and train model
    model = create_mc_dropout_model(
        input_dim=20,
        hidden_dims=[64, 32],
        dropout_rate=0.1,
        n_mc_samples=50,
        task='regression'
    )

    print("Training model...")
    history = model.fit(X_train, y_train, X_val, y_val, epochs=50, patience=5)

    # Make predictions with uncertainty
    print("\nMaking predictions with uncertainty estimation...")
    X_test = torch.FloatTensor(X_val[:5])
    result = model.predict_with_uncertainty(X_test, n_samples=100)

    print("\nPredictions:")
    for i in range(5):
        mean = result['mean'][i].item()
        std = result['std'][i].item()
        actual = y_val[i]
        print(f"  Sample {i+1}: Predicted = {mean:.4f} +/- {std:.4f}, Actual = {actual:.4f}")

    # Get confidence interval
    lower, upper = model.get_confidence_interval(X_test, confidence=0.95)
    print("\n95% Confidence Intervals:")
    for i in range(5):
        print(f"  Sample {i+1}: [{lower[i].item():.4f}, {upper[i].item():.4f}]")
