"""
Training Utilities for Normalizing Flows

This module provides training loops, utilities, and callbacks
for training normalizing flow models on financial data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional, Callable, List, Tuple
import logging
from datetime import datetime
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class FlowTrainer:
    """
    Trainer class for normalizing flow models.

    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    - Early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        device: str = "auto",
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize the trainer.

        Args:
            model: Normalizing flow model
            optimizer: PyTorch optimizer (default: Adam)
            scheduler: Learning rate scheduler
            device: Device to train on ("auto", "cpu", "cuda", "mps")
            checkpoint_dir: Directory for saving checkpoints
        """
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir

        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def create_dataloaders(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        batch_size: int = 256,
        val_split: float = 0.15
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create PyTorch DataLoaders from numpy arrays.

        Args:
            train_data: Training data array
            val_data: Validation data array (optional)
            batch_size: Batch size
            val_split: Validation split ratio if val_data not provided

        Returns:
            train_loader, val_loader
        """
        if val_data is None:
            # Split training data
            n = len(train_data)
            n_val = int(n * val_split)
            indices = np.random.permutation(n)

            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            val_data = train_data[val_indices]
            train_data = train_data[train_indices]

        # Create tensors
        train_tensor = torch.FloatTensor(train_data)
        val_tensor = torch.FloatTensor(val_data) if val_data is not None else None

        # Create datasets
        train_dataset = TensorDataset(train_tensor)
        val_dataset = TensorDataset(val_tensor) if val_tensor is not None else None

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            x = batch[0].to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.nll_loss(x)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                loss = self.model.nll_loss(x)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Maximum number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        train_loader, val_loader = self.create_dataloaders(
            train_data, val_data, batch_size
        )

        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
            else:
                val_loss = train_loss

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.checkpoint_dir:
                    self.save_checkpoint('best_model.pt')

            # Early stopping
            if early_stopping(val_loss):
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"LR: {current_lr:.6f}"
                )

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename) if self.checkpoint_dir else filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Loaded checkpoint from {path}")


def train_flow_simple(
    model: nn.Module,
    train_data: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    device: str = "auto"
) -> nn.Module:
    """
    Simple training function for quick experiments.

    Args:
        model: Normalizing flow model
        train_data: Training data (numpy array)
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device string

    Returns:
        Trained model
    """
    trainer = FlowTrainer(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=learning_rate),
        device=device
    )

    trainer.train(
        train_data=train_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )

    return model


def cross_validate_flow(
    model_class: type,
    model_kwargs: Dict,
    data: np.ndarray,
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 256
) -> Dict:
    """
    Cross-validate a normalizing flow model.

    Args:
        model_class: Flow model class
        model_kwargs: Arguments for model construction
        data: Full dataset
        n_folds: Number of cross-validation folds
        epochs: Training epochs per fold
        batch_size: Batch size

    Returns:
        Cross-validation results
    """
    n = len(data)
    fold_size = n // n_folds
    indices = np.random.permutation(n)

    fold_results = []

    for fold in range(n_folds):
        logger.info(f"Training fold {fold + 1}/{n_folds}")

        # Split data
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size

        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([
            indices[:val_start],
            indices[val_end:]
        ])

        train_data = data[train_indices]
        val_data = data[val_indices]

        # Create and train model
        model = model_class(**model_kwargs)
        trainer = FlowTrainer(model)

        history = trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False
        )

        fold_results.append({
            'fold': fold,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'best_val_loss': min(history['val_loss'])
        })

    # Aggregate results
    val_losses = [r['final_val_loss'] for r in fold_results]

    return {
        'fold_results': fold_results,
        'mean_val_loss': np.mean(val_losses),
        'std_val_loss': np.std(val_losses),
        'best_fold': np.argmin(val_losses)
    }


def hyperparameter_search(
    model_class: type,
    data: np.ndarray,
    param_grid: Dict,
    n_trials: int = 20,
    epochs: int = 50
) -> Dict:
    """
    Random hyperparameter search for normalizing flows.

    Args:
        model_class: Flow model class
        data: Training data
        param_grid: Dictionary of parameter ranges
        n_trials: Number of random trials
        epochs: Training epochs per trial

    Returns:
        Best parameters and results
    """
    results = []

    for trial in range(n_trials):
        # Sample parameters
        params = {}
        for key, values in param_grid.items():
            if isinstance(values, list):
                params[key] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                if isinstance(values[0], int):
                    params[key] = np.random.randint(values[0], values[1])
                else:
                    params[key] = np.random.uniform(values[0], values[1])

        logger.info(f"Trial {trial + 1}/{n_trials}: {params}")

        try:
            # Create model
            model = model_class(**params)

            # Train
            trainer = FlowTrainer(model)
            history = trainer.train(
                train_data=data,
                epochs=epochs,
                verbose=False
            )

            final_loss = history['val_loss'][-1] if history['val_loss'] else history['train_loss'][-1]

            results.append({
                'params': params,
                'final_loss': final_loss,
                'history': history
            })

        except Exception as e:
            logger.warning(f"Trial failed: {e}")

    # Find best
    if results:
        best_idx = np.argmin([r['final_loss'] for r in results])
        best_result = results[best_idx]
    else:
        best_result = None

    return {
        'all_results': results,
        'best_params': best_result['params'] if best_result else None,
        'best_loss': best_result['final_loss'] if best_result else None
    }
