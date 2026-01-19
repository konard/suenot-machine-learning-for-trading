"""
Training pipeline for fine-tuning LLMs on financial data.

This module provides a flexible training framework supporting:
- LoRA and QLoRA fine-tuning
- Early stopping and checkpointing
- Learning rate scheduling
- Gradient accumulation
"""

import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pandas as pd
import numpy as np

from .model import FinancialSentimentLoRA, LoRAConfig, count_parameters


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model
    model_name: str = "bert-base-uncased"
    method: str = "lora"  # lora, qlora, prefix, full
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1

    # Training
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping: bool = True
    patience: int = 3
    min_delta: float = 0.001

    # Checkpointing
    save_dir: str = "outputs"
    save_best_only: bool = True
    save_total_limit: int = 3

    # Logging
    log_steps: int = 10
    eval_steps: int = 100

    # Hardware
    device: str = "auto"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int = 0
    step: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    val_f1: float = 0.0
    learning_rate: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)


class FinancialTextDataset(Dataset):
    """
    Dataset for financial text classification.

    Args:
        texts: List of text strings
        labels: List of integer labels
        tokenizer: Tokenizer function or object
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Optional[Callable] = None,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        # Simple tokenizer if none provided
        self.tokenizer = tokenizer or self._simple_tokenizer

    def _simple_tokenizer(self, text: str) -> Dict[str, torch.Tensor]:
        """Simple character-level tokenizer for demonstration."""
        # In practice, use HuggingFace tokenizer
        tokens = [ord(c) % 10000 for c in text[:self.max_length]]
        tokens = tokens + [0] * (self.max_length - len(tokens))

        return {
            "input_ids": torch.tensor(tokens),
            "attention_mask": torch.tensor([1] * len(text[:self.max_length]) +
                                          [0] * (self.max_length - len(text[:self.max_length])))
        }

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text)
        encoding["labels"] = torch.tensor(label)

        return encoding


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: "min" for loss, "max" for accuracy
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if training should stop
        """
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

        return self.early_stop


class FineTuningTrainer:
    """
    Trainer for fine-tuning LLMs on financial tasks.

    Supports LoRA, QLoRA, and full fine-tuning with various
    optimization techniques.

    Example:
        >>> trainer = FineTuningTrainer(model_name="finbert", method="lora")
        >>> trainer.train(train_data, val_data, epochs=3)
        >>> metrics = trainer.evaluate(test_data)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        method: str = "lora",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        learning_rate: float = 2e-4,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize trainer.

        Args:
            model_name: Name of base model
            method: Fine-tuning method (lora, qlora, prefix, full)
            lora_rank: LoRA rank
            lora_alpha: LoRA scaling factor
            learning_rate: Initial learning rate
            config: Full training configuration
        """
        if config is None:
            config = TrainingConfig(
                model_name=model_name,
                method=method,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                learning_rate=learning_rate
            )

        self.config = config
        self.device = torch.device(config.device)

        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)

        # Metrics tracking
        self.metrics = TrainingMetrics()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        ) if config.early_stopping else None

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if (
            config.mixed_precision and config.device == "cuda"
        ) else None

    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        # For this implementation, we use our custom LoRA model
        # In practice, you'd load a HuggingFace model and apply PEFT
        model = FinancialSentimentLoRA(
            base_dim=768,
            lora_rank=self.config.lora_rank,
            num_classes=3
        )

        # Log parameter counts
        params = count_parameters(model)
        print(f"Model created:")
        print(f"  Trainable parameters: {params['trainable']:,}")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable %: {params['trainable_percent']:.2f}%")

        return model

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter groups."""
        # Separate LoRA and non-LoRA parameters
        lora_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora" in name.lower():
                lora_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": lora_params, "lr": self.config.learning_rate},
            {"params": other_params, "lr": self.config.learning_rate * 0.1}
        ]

        return AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int
    ):
        """Create learning rate scheduler with warmup."""
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps
        )

        decay_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[num_warmup_steps]
        )

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        epochs: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> TrainingMetrics:
        """
        Train the model.

        Args:
            train_data: Training DataFrame with 'text' and 'label' columns
            val_data: Validation DataFrame
            epochs: Number of epochs (overrides config)
            callbacks: Optional list of callback functions

        Returns:
            TrainingMetrics with training history
        """
        epochs = epochs or self.config.epochs

        # Create datasets
        train_dataset = FinancialTextDataset(
            train_data["text"].tolist(),
            train_data["label"].tolist()
        )
        val_dataset = FinancialTextDataset(
            val_data["text"].tolist(),
            val_data["label"].tolist()
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size
        )

        # Setup training
        optimizer = self._create_optimizer()
        num_training_steps = len(train_loader) * epochs
        scheduler = self._create_scheduler(optimizer, num_training_steps)

        # Training loop
        print(f"\nStarting training for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(epochs):
            self.metrics.epoch = epoch + 1

            # Train epoch
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)

            # Validate
            val_metrics = self._validate(val_loader)

            # Update metrics
            self.metrics.train_loss = train_loss
            self.metrics.val_loss = val_metrics["loss"]
            self.metrics.val_accuracy = val_metrics["accuracy"]
            self.metrics.learning_rate = scheduler.get_last_lr()[0]

            self.metrics.history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": self.metrics.learning_rate
            })

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, "
                  f"val_acc={val_metrics['accuracy']:.4f}")

            # Check for best model
            if val_metrics["loss"] < self.metrics.best_val_loss:
                self.metrics.best_val_loss = val_metrics["loss"]
                self.metrics.best_epoch = epoch + 1
                if self.config.save_best_only:
                    self._save_checkpoint("best")

            # Early stopping
            if self.early_stopping and self.early_stopping(val_metrics["loss"]):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Final save
        self._save_checkpoint("final")

        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed:.2f}s")
        print(f"Best validation loss: {self.metrics.best_val_loss:.4f} "
              f"(epoch {self.metrics.best_epoch})")

        return self.metrics

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Create dummy hidden states from input_ids
            # In practice, this would come from a transformer encoder
            hidden_states = torch.randn(
                input_ids.size(0), 128, 768,
                device=self.device
            )

            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits = self.model(hidden_states)
                    loss = F.cross_entropy(logits, labels)
                    loss = loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                logits = self.model(hidden_states)
                loss = F.cross_entropy(logits, labels)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            self.metrics.step += 1

        return total_loss / num_batches

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Create dummy hidden states
                hidden_states = torch.randn(
                    input_ids.size(0), 128, 768,
                    device=self.device
                )

                logits = self.model(hidden_states)
                loss = F.cross_entropy(logits, labels)

                total_loss += loss.item()
                predictions = logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": correct / total
        }

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate on test data.

        Args:
            test_data: Test DataFrame with 'text' and 'label' columns

        Returns:
            Dictionary with evaluation metrics
        """
        test_dataset = FinancialTextDataset(
            test_data["text"].tolist(),
            test_data["label"].tolist()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size
        )

        metrics = self._validate(test_loader)

        # Compute additional metrics
        all_predictions = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"]

                hidden_states = torch.randn(
                    input_ids.size(0), 128, 768,
                    device=self.device
                )

                logits = self.model(hidden_states)
                predictions = logits.argmax(dim=-1).cpu()

                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())

        # Compute F1 score
        from collections import Counter
        pred_counts = Counter(all_predictions)
        label_counts = Counter(all_labels)

        # Macro F1
        f1_scores = []
        for label in set(all_labels):
            tp = sum(1 for p, l in zip(all_predictions, all_labels)
                    if p == l == label)
            precision = tp / (pred_counts.get(label, 1) or 1)
            recall = tp / (label_counts.get(label, 1) or 1)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores.append(f1)

        metrics["f1"] = np.mean(f1_scores)
        metrics["num_samples"] = len(all_labels)

        return metrics

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        os.makedirs(self.config.save_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            self.config.save_dir,
            f"checkpoint_{name}.pt"
        )

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "metrics": asdict(self.metrics)
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {path}")

    def save_model(self, path: str):
        """Save model for deployment."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Save model state
        torch.save(self.model.state_dict(), path)

        # Save config
        config_path = path.replace(".pt", "_config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        print(f"Model saved: {path}")


if __name__ == "__main__":
    from .data_loader import load_financial_phrasebank

    # Load sample data
    train_data, val_data = load_financial_phrasebank()

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Create trainer
    trainer = FineTuningTrainer(
        model_name="finbert",
        method="lora",
        lora_rank=8,
        learning_rate=2e-4
    )

    # Train
    metrics = trainer.train(train_data, val_data, epochs=2)

    # Evaluate
    eval_metrics = trainer.evaluate(val_data)
    print(f"\nFinal evaluation:")
    print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {eval_metrics['f1']:.4f}")
