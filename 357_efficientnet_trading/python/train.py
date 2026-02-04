"""
Training Script for EfficientNet Trading Model

This script provides training utilities for the EfficientNet trading model.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch torchvision")


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model
    variant: str = "b2"
    pretrained: bool = True

    # Training
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    max_epochs: int = 100
    early_stopping_patience: int = 10

    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    image_size: int = 224
    lookback: int = 120

    # Scheduler
    use_scheduler: bool = True
    warmup_epochs: int = 5

    # Device
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"


if TORCH_AVAILABLE:

    class TradingDataset(Dataset):
        """Dataset for EfficientNet trading model."""

        def __init__(
            self,
            images: np.ndarray,
            labels: np.ndarray,
            returns: Optional[np.ndarray] = None,
        ):
            """
            Initialize dataset.

            Args:
                images: Array of shape (N, H, W, C) or (N, C, H, W)
                labels: Array of shape (N,) with class labels
                returns: Optional array of shape (N,) with actual returns
            """
            self.images = images
            self.labels = labels
            self.returns = returns if returns is not None else np.zeros(len(labels))

            # Ensure images are in (N, C, H, W) format
            if len(self.images.shape) == 4 and self.images.shape[-1] == 3:
                self.images = np.transpose(self.images, (0, 3, 1, 2))

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            image = torch.tensor(self.images[idx], dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            ret = torch.tensor(self.returns[idx], dtype=torch.float32)
            return image, label, ret


    class Trainer:
        """Trainer for EfficientNet trading model."""

        def __init__(
            self,
            model: nn.Module,
            config: Optional[TrainingConfig] = None,
        ):
            """
            Initialize trainer.

            Args:
                model: EfficientNet model
                config: Training configuration
            """
            self.model = model
            self.config = config or TrainingConfig()

            self.device = torch.device(self.config.device)
            self.model.to(self.device)

            # Loss functions
            self.direction_loss = nn.CrossEntropyLoss()
            self.magnitude_loss = nn.MSELoss()

            # Optimizer
            self.optimizer = self._setup_optimizer()

            # Scheduler
            self.scheduler = None
            if self.config.use_scheduler:
                self.scheduler = self._setup_scheduler()

            # Training state
            self.best_val_loss = float("inf")
            self.patience_counter = 0
            self.history = {
                "train_loss": [],
                "val_loss": [],
                "train_acc": [],
                "val_acc": [],
            }

        def _setup_optimizer(self) -> optim.Optimizer:
            """Set up optimizer with different learning rates for backbone and head."""
            backbone_params = []
            head_params = []

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "trading_head" in name:
                        head_params.append(param)
                    else:
                        backbone_params.append(param)

            if backbone_params:
                param_groups = [
                    {"params": backbone_params, "lr": self.config.learning_rate * 0.1},
                    {"params": head_params, "lr": self.config.learning_rate},
                ]
            else:
                param_groups = [{"params": head_params, "lr": self.config.learning_rate}]

            return optim.AdamW(param_groups, weight_decay=self.config.weight_decay)

        def _setup_scheduler(self):
            """Set up learning rate scheduler."""
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
            )

        def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
            """
            Train for one epoch.

            Returns:
                (average_loss, accuracy)
            """
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels, returns in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                returns = returns.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(images)

                # Compute loss
                if isinstance(output, tuple):
                    direction_logits, magnitude = output
                    loss_dir = self.direction_loss(direction_logits, labels)
                    loss_mag = self.magnitude_loss(magnitude.squeeze(), returns)
                    loss = loss_dir + 0.5 * loss_mag
                else:
                    direction_logits = output
                    loss = self.direction_loss(direction_logits, labels)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Statistics
                total_loss += loss.item() * images.size(0)
                _, predicted = direction_logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / total
            accuracy = correct / total

            return avg_loss, accuracy

        def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
            """
            Validate the model.

            Returns:
                (average_loss, accuracy)
            """
            self.model.train(False)
            total_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels, returns in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    returns = returns.to(self.device)

                    # Forward pass
                    output = self.model(images)

                    # Compute loss
                    if isinstance(output, tuple):
                        direction_logits, magnitude = output
                        loss_dir = self.direction_loss(direction_logits, labels)
                        loss_mag = self.magnitude_loss(magnitude.squeeze(), returns)
                        loss = loss_dir + 0.5 * loss_mag
                    else:
                        direction_logits = output
                        loss = self.direction_loss(direction_logits, labels)

                    # Statistics
                    total_loss += loss.item() * images.size(0)
                    _, predicted = direction_logits.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

            avg_loss = total_loss / total
            accuracy = correct / total

            return avg_loss, accuracy

        def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            verbose: bool = True,
        ) -> Dict:
            """
            Full training loop.

            Args:
                train_loader: Training data loader
                val_loader: Validation data loader
                verbose: Print progress

            Returns:
                Training history dictionary
            """
            for epoch in range(self.config.max_epochs):
                # Train
                train_loss, train_acc = self.train_epoch(train_loader)
                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)

                # Validate
                val_loss, val_acc = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step()

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best model weights
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    self.patience_counter += 1

                if verbose:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch+1}/{self.config.max_epochs} | "
                        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                        f"LR: {lr:.6f}"
                    )

                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Restore best weights
            if hasattr(self, "best_state"):
                self.model.load_state_dict(self.best_state)

            return self.history

        def save_model(self, path: str):
            """Save model to file."""
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "history": self.history,
            }, path)
            print(f"Model saved to {path}")

        def load_model(self, path: str):
            """Load model from file."""
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.history = checkpoint.get("history", self.history)
            print(f"Model loaded from {path}")


    def create_data_loaders(
        images: np.ndarray,
        labels: np.ndarray,
        returns: Optional[np.ndarray] = None,
        config: Optional[TrainingConfig] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.

        Args:
            images: Image array
            labels: Label array
            returns: Optional returns array
            config: Training configuration

        Returns:
            (train_loader, val_loader, test_loader)
        """
        config = config or TrainingConfig()

        # Create dataset
        dataset = TradingDataset(images, labels, returns)

        # Split dataset
        n_total = len(dataset)
        n_train = int(n_total * config.train_split)
        n_val = int(n_total * config.val_split)
        n_test = n_total - n_train - n_val

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if config.device == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, val_loader, test_loader


def train_efficientnet_model(
    images: np.ndarray,
    labels: np.ndarray,
    returns: Optional[np.ndarray] = None,
    variant: str = "b0",
    pretrained: bool = True,
    max_epochs: int = 50,
    batch_size: int = 32,
    verbose: bool = True,
):
    """
    Convenience function to train an EfficientNet trading model.

    Args:
        images: Training images (N, H, W, 3) or (N, 3, H, W)
        labels: Training labels (N,)
        returns: Optional actual returns (N,)
        variant: EfficientNet variant (b0-b7)
        pretrained: Use ImageNet pretrained weights
        max_epochs: Maximum training epochs
        batch_size: Batch size
        verbose: Print progress

    Returns:
        (trained_model, training_history)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training")

    from .model import EfficientNetTrader, ModelConfig

    # Create model
    model_config = ModelConfig(
        variant=variant,
        pretrained=pretrained,
        num_classes=3,
        freeze_blocks=3 if pretrained else 0,
    )
    model = EfficientNetTrader(model_config)

    # Create training config
    train_config = TrainingConfig(
        variant=variant,
        pretrained=pretrained,
        batch_size=batch_size,
        max_epochs=max_epochs,
    )

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        images, labels, returns, train_config
    )

    # Create trainer
    trainer = Trainer(model, train_config)

    # Train
    history = trainer.train(train_loader, val_loader, verbose=verbose)

    return model, history


if __name__ == "__main__" and TORCH_AVAILABLE:
    # Example usage with synthetic data
    print("Testing EfficientNet Training Pipeline...")

    from .model import EfficientNetTrader, ModelConfig
    from .image_transform import ImageStackGenerator

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    image_size = 64  # Small for testing

    # Create synthetic images and labels
    images = np.random.randn(n_samples, image_size, image_size, 3).astype(np.float32)
    labels = np.random.randint(0, 3, n_samples)
    returns = np.random.randn(n_samples) * 0.01

    print(f"Generated {n_samples} synthetic samples")
    print(f"Image shape: {images.shape}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Create model
    config = ModelConfig(variant="b0", pretrained=False, image_size=image_size)
    model = EfficientNetTrader(config)

    # Create training config
    train_config = TrainingConfig(
        batch_size=16,
        max_epochs=5,
        learning_rate=0.001,
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        images, labels, returns, train_config
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create trainer
    trainer = Trainer(model, train_config)

    # Train for a few epochs
    print("\nStarting training...")
    history = trainer.train(train_loader, val_loader, verbose=True)

    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
