"""
Trading classifier model for counterfactual explanations.

This module provides a simple neural network classifier that predicts
trading signals (BUY, HOLD, SELL) from technical indicators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class TradingClassifier(nn.Module):
    """
    Neural network classifier for trading signals.

    Predicts one of three classes:
    - 0: SELL
    - 1: HOLD
    - 2: BUY

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes (default: 3)
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (class indices).

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predicted class indices of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def get_class_name(self, class_idx: int) -> str:
        """Get human-readable class name."""
        class_names = ['SELL', 'HOLD', 'BUY']
        return class_names[class_idx]


class TradingClassifierTrainer:
    """
    Trainer for TradingClassifier.

    Args:
        model: TradingClassifier instance
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization weight
    """

    def __init__(
        self,
        model: TradingClassifier,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()

            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        Evaluate model on validation data.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

                total_loss += loss.item()
                predictions = logits.argmax(dim=-1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            if val_loader is not None:
                val_loss, val_accuracy = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}")

        return self.history


def create_trading_labels(
    returns: np.ndarray,
    threshold: float = 0.005
) -> np.ndarray:
    """
    Create trading labels from returns.

    Args:
        returns: Array of returns
        threshold: Threshold for buy/sell signals

    Returns:
        Array of labels (0=SELL, 1=HOLD, 2=BUY)
    """
    labels = np.ones(len(returns), dtype=np.int64)  # Default to HOLD
    labels[returns > threshold] = 2  # BUY
    labels[returns < -threshold] = 0  # SELL
    return labels
