"""
Training pipeline for ProtoPNet.

Implements training with combined loss function, prototype projection,
and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import json

from .model import ProtoPNet, ProtoPNetLoss, MarketState


class ProtoPNetTrainer:
    """
    Trainer for ProtoPNet model.

    Handles training loop, validation, prototype projection,
    and model checkpointing.
    """

    def __init__(
        self,
        model: ProtoPNet,
        device: torch.device = None,
        learning_rate: float = 1e-3,
        lambda_clst: float = 0.8,
        lambda_sep: float = -0.08,
    ):
        """
        Initialize trainer.

        Args:
            model: ProtoPNet model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            lambda_clst: Weight for clustering loss
            lambda_sep: Weight for separation loss
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = ProtoPNetLoss(
            lambda_clst=lambda_clst,
            lambda_sep=lambda_sep,
        )

        # Separate learning rates for different parts
        self.optimizer = optim.Adam([
            {'params': model.encoder.parameters(), 'lr': learning_rate},
            {'params': model.prototype_layer.parameters(), 'lr': learning_rate * 3},
            {'params': model.classifier.parameters(), 'lr': learning_rate},
        ])

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }

    def create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create DataLoader from numpy arrays."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits, similarities, distances = self.model(batch_x)

            # Compute loss
            loss, loss_components = self.criterion(
                logits,
                similarities,
                distances,
                batch_y,
                self.model.prototype_layer.prototype_class_identity,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            logits, similarities, distances = self.model(batch_x)

            loss, _ = self.criterion(
                logits,
                similarities,
                distances,
                batch_y,
                self.model.prototype_layer.prototype_class_identity,
            )

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        push_epochs: List[int] = None,
        early_stopping_patience: int = 15,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        Full training loop.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            push_epochs: Epochs at which to push prototypes
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save checkpoints

        Returns:
            Training history dictionary
        """
        train_loader = self.create_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self.create_dataloader(X_val, y_val, batch_size, shuffle=False)

        if push_epochs is None:
            # Push prototypes every 20 epochs after epoch 20
            push_epochs = list(range(20, epochs, 20))

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Prototype projection
            if epoch in push_epochs:
                print("  Pushing prototypes to nearest training examples...")
                self.model.push_prototypes(train_loader, self.device)
                # Re-validate after pushing
                val_loss, val_acc = self.validate(val_loader)
                print(f"  After push - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_dir:
                    self.save_checkpoint(save_path / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        if save_dir:
            self.load_checkpoint(save_path / 'best_model.pt')
            self.save_history(save_path / 'history.json')

        return self.history

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']

    def save_history(self, path: Path):
        """Save training history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    @torch.no_grad()
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
    ) -> Dict:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with evaluation metrics
        """
        test_loader = self.create_dataloader(X_test, y_test, batch_size, shuffle=False)
        self.model.eval()

        all_preds = []
        all_targets = []
        all_confidences = []

        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(self.device)

            preds, confs = self.model.predict(batch_x)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            all_confidences.extend(confs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_confidences = np.array(all_confidences)

        # Compute metrics
        accuracy = (all_preds == all_targets).mean()

        # Per-class metrics
        class_metrics = {}
        for class_idx in range(self.model.num_classes):
            class_mask = all_targets == class_idx
            if class_mask.sum() > 0:
                class_acc = (all_preds[class_mask] == all_targets[class_mask]).mean()
                class_metrics[MarketState(class_idx).name] = {
                    'accuracy': float(class_acc),
                    'support': int(class_mask.sum()),
                }

        # Confusion matrix
        confusion = np.zeros((self.model.num_classes, self.model.num_classes), dtype=int)
        for pred, target in zip(all_preds, all_targets):
            confusion[target, pred] += 1

        return {
            'accuracy': float(accuracy),
            'mean_confidence': float(all_confidences.mean()),
            'class_metrics': class_metrics,
            'confusion_matrix': confusion.tolist(),
            'predictions': all_preds,
            'targets': all_targets,
            'confidences': all_confidences,
        }

    @torch.no_grad()
    def get_prototype_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict:
        """
        Analyze prototype activations.

        Returns:
            Dictionary with prototype analysis
        """
        dataloader = self.create_dataloader(X, y, batch_size=64, shuffle=False)
        self.model.eval()

        all_similarities = []
        all_targets = []

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            _, similarities, _ = self.model(batch_x)
            all_similarities.append(similarities.cpu().numpy())
            all_targets.extend(batch_y.numpy())

        all_similarities = np.vstack(all_similarities)
        all_targets = np.array(all_targets)

        # Analyze each prototype
        prototype_analysis = []
        for proto_idx in range(self.model.num_prototypes):
            proto_class = self.model.prototype_layer.get_prototype_class(proto_idx)
            proto_sims = all_similarities[:, proto_idx]

            # Activation statistics
            analysis = {
                'prototype_idx': proto_idx,
                'class': MarketState(proto_class).name,
                'mean_similarity': float(proto_sims.mean()),
                'max_similarity': float(proto_sims.max()),
                'std_similarity': float(proto_sims.std()),
            }

            # Class-specific activations
            class_activations = {}
            for class_idx in range(self.model.num_classes):
                class_mask = all_targets == class_idx
                if class_mask.sum() > 0:
                    class_activations[MarketState(class_idx).name] = float(
                        proto_sims[class_mask].mean()
                    )
            analysis['class_activations'] = class_activations

            prototype_analysis.append(analysis)

        return {
            'prototype_analysis': prototype_analysis,
            'similarity_matrix': all_similarities,
        }
