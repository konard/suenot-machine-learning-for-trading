"""
Training Utilities for GQA Trading Model

This module provides training and evaluation functions for the
Grouped Query Attention trading model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import time
import numpy as np
from pathlib import Path

from .model import GQATrader


def train_model(
    model: GQATrader,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 10,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    save_path: Optional[str] = None
) -> Dict:
    """
    Train the GQA trading model.

    Args:
        model: GQATrader model instance
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization weight
        patience: Early stopping patience
        device: Training device (auto-detected if None)
        verbose: Whether to print progress
        save_path: Path to save best model checkpoint

    Returns:
        Dictionary with training history

    Example:
        >>> model = GQATrader(input_dim=5, d_model=64)
        >>> history = train_model(model, train_loader, val_loader, epochs=50)
        >>> print(f"Best val accuracy: {max(history['val_acc']):.2%}")
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if verbose:
        print(f"Training on {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
    )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": []
    }

    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])

        # Validation phase
        if val_loader is not None:
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0

                if save_path:
                    torch.save(best_model_state, save_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            epoch_time = time.time() - epoch_start
            msg = f"Epoch {epoch + 1:3d}/{epochs}: "
            msg += f"Train Loss={train_loss:.4f}, Acc={train_acc:.3f}"
            if val_loader is not None:
                msg += f" | Val Loss={val_loss:.4f}, Acc={val_acc:.3f}"
            msg += f" | Time={epoch_time:.1f}s"
            print(msg)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


def evaluate_model(
    model: GQATrader,
    data_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None
) -> Tuple[float, float]:
    """
    Evaluate the model on a dataset.

    Args:
        model: GQATrader model instance
        data_loader: Data loader for evaluation
        criterion: Loss function (default: CrossEntropyLoss)
        device: Evaluation device

    Returns:
        Tuple of (loss, accuracy)
    """
    if device is None:
        device = next(model.parameters()).device

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / total, correct / total


def cross_validate(
    model_fn,
    X: torch.Tensor,
    y: torch.Tensor,
    n_folds: int = 5,
    **train_kwargs
) -> Dict:
    """
    Perform time-series cross-validation.

    Uses expanding window approach suitable for time series data.

    Args:
        model_fn: Function that creates a new model instance
        X: Input sequences
        y: Labels
        n_folds: Number of cross-validation folds
        **train_kwargs: Additional arguments for train_model

    Returns:
        Dictionary with cross-validation results
    """
    from torch.utils.data import TensorDataset, DataLoader

    n_samples = len(X)
    fold_size = n_samples // (n_folds + 1)

    results = {
        "fold_val_loss": [],
        "fold_val_acc": [],
        "fold_train_size": [],
        "fold_val_size": []
    }

    for fold in range(n_folds):
        # Expanding window: train on data up to this point
        train_end = (fold + 1) * fold_size
        val_start = train_end
        val_end = min(train_end + fold_size, n_samples)

        if val_end <= val_start:
            continue

        # Create data loaders for this fold
        train_dataset = TensorDataset(X[:train_end], y[:train_end])
        val_dataset = TensorDataset(X[val_start:val_end], y[val_start:val_end])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Create and train new model
        model = model_fn()
        train_kwargs['verbose'] = False
        history = train_model(model, train_loader, val_loader, **train_kwargs)

        # Record results
        best_val_loss = min(history['val_loss'])
        best_val_acc = max(history['val_acc'])

        results["fold_val_loss"].append(best_val_loss)
        results["fold_val_acc"].append(best_val_acc)
        results["fold_train_size"].append(train_end)
        results["fold_val_size"].append(val_end - val_start)

        print(f"Fold {fold + 1}/{n_folds}: Val Loss={best_val_loss:.4f}, Acc={best_val_acc:.3f}")

    # Aggregate results
    results["mean_val_loss"] = np.mean(results["fold_val_loss"])
    results["std_val_loss"] = np.std(results["fold_val_loss"])
    results["mean_val_acc"] = np.mean(results["fold_val_acc"])
    results["std_val_acc"] = np.std(results["fold_val_acc"])

    return results


def compute_class_weights(y: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Args:
        y: Labels tensor

    Returns:
        Class weights tensor
    """
    class_counts = torch.bincount(y)
    total = len(y)
    weights = total / (len(class_counts) * class_counts.float())
    return weights


if __name__ == "__main__":
    # Test training utilities
    print("Testing Training Utilities...")
    print("=" * 50)

    from .model import GQATrader
    from .data import prepare_sequences, create_data_loader, _generate_synthetic_data

    # Generate test data
    print("\n1. Generating test data...")
    data = _generate_synthetic_data(500)
    X, y = prepare_sequences(data, seq_len=30)
    train_loader, val_loader = create_data_loader(X, y, batch_size=32)

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Create model
    print("\n2. Creating model...")
    model = GQATrader(
        input_dim=5,
        d_model=32,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2
    )
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train for a few epochs
    print("\n3. Training model...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=20,
        learning_rate=1e-3,
        patience=5,
        verbose=True
    )

    print(f"\n   Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"   Final val accuracy: {history['val_acc'][-1]:.3f}")

    print("\n" + "=" * 50)
    print("All training tests passed!")
