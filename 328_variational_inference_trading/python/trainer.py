"""
Training loop for Variational Autoencoder
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import json

from config import TrainingConfig, ModelConfig
from vae_model import MarketVAE, create_model


class Trainer:
    """
    Trainer for MarketVAE model

    Handles:
    - Training loop with beta annealing
    - Validation
    - Early stopping
    - Model checkpointing
    - Logging
    """

    def __init__(
        self,
        model: MarketVAE,
        config: Optional[TrainingConfig] = None,
        model_dir: str = "./models"
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        self.device = self._get_device()
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Setup scheduler
        if self.config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.lr_min
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'train_return_loss': [],
            'learning_rate': [],
            'beta': []
        }

    def _get_device(self) -> torch.device:
        """Determine the device to use"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def _get_beta(self, epoch: int) -> float:
        """Get beta value for current epoch (with annealing)"""
        if epoch < self.config.beta_warmup_epochs:
            # Linear annealing
            progress = epoch / self.config.beta_warmup_epochs
            return self.config.beta_start + progress * (self.config.beta_end - self.config.beta_start)
        return self.config.beta_end

    def create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders from numpy arrays

        Args:
            X_train: Training features (N, seq_len, n_features)
            y_train: Training targets (N,)
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, beta: float) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            beta: Current beta value for KL term

        Returns:
            Dictionary of average losses
        """
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_return_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            x_recon, return_mu, return_logvar, regime_logits, mu, log_var = self.model(batch_x)

            # Compute loss
            loss, loss_dict = self.model.compute_loss(
                batch_x, x_recon, return_mu, return_logvar,
                mu, log_var, batch_y, beta
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()

            # Accumulate losses
            total_loss += loss_dict['total_loss']
            total_recon_loss += loss_dict['recon_loss']
            total_kl_loss += loss_dict['kl_loss']
            total_return_loss += loss_dict['return_loss']
            n_batches += 1

        return {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'kl_loss': total_kl_loss / n_batches,
            'return_loss': total_return_loss / n_batches
        }

    def validate(self, val_loader: DataLoader, beta: float) -> Dict[str, float]:
        """
        Validate the model

        Args:
            val_loader: Validation data loader
            beta: Current beta value

        Returns:
            Dictionary of average validation losses
        """
        self.model.set_to_eval_mode()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                x_recon, return_mu, return_logvar, regime_logits, mu, log_var = self.model(batch_x)

                # Compute loss
                loss, loss_dict = self.model.compute_loss(
                    batch_x, x_recon, return_mu, return_logvar,
                    mu, log_var, batch_y, beta
                )

                total_loss += loss_dict['total_loss']
                n_batches += 1

        return {'loss': total_loss / n_batches}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            verbose: Whether to show progress bar

        Returns:
            Training history
        """
        train_loader, val_loader = self.create_dataloaders(
            X_train, y_train, X_val, y_val
        )

        logger.info(f"Starting training on {self.device}")
        logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

        pbar = tqdm(range(self.config.max_epochs), disable=not verbose)

        for epoch in pbar:
            self.epoch = epoch

            # Get current beta
            beta = self._get_beta(epoch)

            # Train epoch
            train_metrics = self.train_epoch(train_loader, beta)

            # Validate
            val_metrics = self.validate(val_loader, beta)

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics['loss'])

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_kl_loss'].append(train_metrics['kl_loss'])
            self.history['train_return_loss'].append(train_metrics['return_loss'])
            self.history['learning_rate'].append(current_lr)
            self.history['beta'].append(beta)

            # Update progress bar
            pbar.set_description(
                f"Train: {train_metrics['loss']:.4f} | Val: {val_metrics['loss']:.4f} | "
                f"Beta: {beta:.2f} | LR: {current_lr:.6f}"
            )

            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Save final model
        self.save_checkpoint('final_model.pt')
        self.save_history()

        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': {
                'model_config': self.model.config.__dict__,
                'training_config': self.config.__dict__
            }
        }
        torch.save(checkpoint, self.model_dir / filename)
        logger.debug(f"Saved checkpoint: {filename}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(self.model_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        logger.info(f"Loaded checkpoint: {filename}")

    def save_history(self):
        """Save training history to JSON"""
        history_path = self.model_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.debug(f"Saved training history to {history_path}")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    model_dir: str = "./models",
    verbose: bool = True
) -> Tuple[MarketVAE, Dict[str, List[float]]]:
    """
    Convenience function to train a model

    Args:
        X_train: Training features (N, seq_len, n_features)
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_config: Model configuration
        training_config: Training configuration
        model_dir: Directory to save models
        verbose: Whether to show progress

    Returns:
        Tuple of (trained_model, training_history)
    """
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()

    # Infer input dimensions
    seq_len = X_train.shape[1]
    input_dim = X_train.shape[2]
    model_config.input_dim = input_dim

    # Create model
    model = create_model(model_config, sequence_length=seq_len, device=training_config.device)

    # Create trainer
    trainer = Trainer(model, training_config, model_dir)

    # Train
    history = trainer.train(X_train, y_train, X_val, y_val, verbose)

    # Load best model
    trainer.load_checkpoint('best_model.pt')

    return trainer.model, history


if __name__ == "__main__":
    # Test training with synthetic data
    import sys
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Create synthetic data
    n_samples = 1000
    seq_len = 60
    n_features = 13

    X_train = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y_train = np.random.randn(n_samples).astype(np.float32)
    X_val = np.random.randn(200, seq_len, n_features).astype(np.float32)
    y_val = np.random.randn(200).astype(np.float32)

    # Configure
    model_config = ModelConfig(
        input_dim=n_features,
        latent_dim=8,
        hidden_dims=[64, 32]
    )
    training_config = TrainingConfig(
        batch_size=64,
        max_epochs=10,
        early_stopping_patience=5,
        beta_warmup_epochs=3
    )

    # Train
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        model_config, training_config,
        model_dir="./test_models"
    )

    print(f"\nFinal train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
