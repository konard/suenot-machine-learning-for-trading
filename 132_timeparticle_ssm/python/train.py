"""
Training Utilities for TimeParticle SSM

Provides training, evaluation, and model management utilities
for TimeParticle trading models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import time


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    min_delta: float = 1e-4
    grad_clip: float = 1.0
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop


class TimeParticleTrainer:
    """
    Trainer for TimeParticle models.

    Handles training loop, validation, and model checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = 'auto'
    ):
        """
        Initialize trainer.

        Args:
            model: TimeParticle model to train
            config: Training configuration
            device: Device to train on ('auto', 'cuda', 'cpu')
        """
        self.model = model
        self.config = config

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        return None

    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_features: Training features [n_samples, seq_len, n_features]
            train_labels: Training labels [n_samples]
            val_features: Validation features
            val_labels: Validation labels
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_features),
            torch.LongTensor(train_labels)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        val_loader = None
        if val_features is not None and val_labels is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(val_features),
                torch.LongTensor(val_labels)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()

                # Early stopping check
                if early_stopping(val_loss):
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

            elapsed = time.time() - start_time

            if verbose:
                if val_loader is not None:
                    print(f"Epoch {epoch + 1}/{self.config.epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                          f"Time: {elapsed:.1f}s")
                else:
                    print(f"Epoch {epoch + 1}/{self.config.epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                          f"Time: {elapsed:.1f}s")

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * features.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            features: Test features
            labels: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        features_tensor = torch.FloatTensor(features).to(self.device)
        labels_tensor = torch.LongTensor(labels).to(self.device)

        with torch.no_grad():
            outputs = self.model(features_tensor)
            loss = self.criterion(outputs, labels_tensor)

            probs = torch.softmax(outputs, dim=-1)
            _, predicted = outputs.max(1)

        # Calculate metrics
        accuracy = (predicted == labels_tensor).float().mean().item() * 100

        # Per-class metrics
        n_classes = 3
        class_correct = [0] * n_classes
        class_total = [0] * n_classes

        for i in range(len(labels)):
            label = labels[i]
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

        class_accuracy = {
            'sell': class_correct[0] / max(class_total[0], 1) * 100,
            'hold': class_correct[1] / max(class_total[1], 1) * 100,
            'buy': class_correct[2] / max(class_total[2], 1) * 100
        }

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'predictions': predicted.cpu().numpy(),
            'probabilities': probs.cpu().numpy()
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']


def create_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Create class weights for imbalanced data.

    Args:
        labels: Array of labels

    Returns:
        Tensor of class weights
    """
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history and history['val_loss']:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train Acc')
        if 'val_acc' in history and history['val_acc']:
            axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    except ImportError:
        print("matplotlib not available for plotting")


if __name__ == "__main__":
    print("Testing Training Utilities...")

    from timeparticle_model import TimeParticleTradingModel
    from data_loader import generate_simulated_data, DataPreprocessor
    from features import prepare_multiscale_features, get_feature_columns

    # Generate sample data
    print("\n1. Generating sample data...")
    df = generate_simulated_data(500)
    features_df = prepare_multiscale_features(df)
    feature_cols = get_feature_columns(features_df)

    # Prepare sequences
    preprocessor = DataPreprocessor(lookback=50)
    X, y = preprocessor.prepare_sequences(features_df, feature_cols)

    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Create model
    print("\n2. Creating model...")
    model = TimeParticleTradingModel(
        n_features=len(feature_cols),
        d_hidden=32,
        d_state=16,
        n_particles=4,
        n_layers=2
    )

    # Training config
    config = TrainingConfig(
        epochs=5,  # Short for testing
        batch_size=16,
        learning_rate=1e-3,
        patience=3
    )

    # Create trainer
    print("\n3. Training model...")
    trainer = TimeParticleTrainer(model, config)
    history = trainer.train(X_train, y_train, X_val, y_val, verbose=True)

    # Evaluate
    print("\n4. Evaluating model...")
    results = trainer.evaluate(X_val, y_val)
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print(f"Class Accuracy: {results['class_accuracy']}")

    print("\nAll tests passed!")
