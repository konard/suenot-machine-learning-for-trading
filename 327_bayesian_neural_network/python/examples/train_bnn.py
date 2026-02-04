"""
Example: Train a Bayesian Neural Network on cryptocurrency data.

Usage:
    python -m examples.train_bnn --symbol BTC/USDT --days 60
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from loguru import logger

from bnn import BayesianNN, BayesianNNConfig, elbo_loss, get_kl_weight
from data import BybitFetcher
from features import FeatureEngineer


def train_epoch(
    model: BayesianNN,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    kl_weight: float,
    device: str,
    n_samples: int = 1
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        loss, metrics = elbo_loss(
            model, batch_x, batch_y,
            n_samples=n_samples,
            kl_weight=kl_weight
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += metrics['loss']
        total_nll += metrics['nll']
        total_kl += metrics['kl_scaled']
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'nll': total_nll / n_batches,
        'kl': total_kl / n_batches
    }


def validate(
    model: BayesianNN,
    dataloader: DataLoader,
    device: str,
    n_mc_samples: int = 50
) -> dict:
    """Validate model with MC inference."""
    model.train(False)
    all_probs = []
    all_targets = []
    all_uncertainties = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            mean_probs, uncertainty = model.predict_proba(batch_x, n_samples=n_mc_samples)

            all_probs.append(mean_probs)
            all_targets.append(batch_y)
            all_uncertainties.append(uncertainty)

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    uncertainties = torch.cat(all_uncertainties, dim=0)

    # Accuracy
    predictions = probs.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean().item()

    # Mean uncertainty
    mean_uncertainty = uncertainties.mean().item()

    # Accuracy on high-confidence predictions
    median_unc = torch.median(uncertainties)
    high_conf_mask = uncertainties < median_unc
    if high_conf_mask.sum() > 0:
        high_conf_acc = (predictions[high_conf_mask] == targets[high_conf_mask]).float().mean().item()
    else:
        high_conf_acc = accuracy

    return {
        'accuracy': accuracy,
        'mean_uncertainty': mean_uncertainty,
        'high_conf_accuracy': high_conf_acc
    }


def main():
    parser = argparse.ArgumentParser(description='Train Bayesian Neural Network')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe')
    parser.add_argument('--days', type=int, default=60, help='Days of data')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--kl-warmup', type=int, default=10, help='KL warmup epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--save-path', type=str, default='bnn_model.pt', help='Model save path')

    args = parser.parse_args()

    logger.info(f"Training BNN for {args.symbol}")

    # Fetch data
    logger.info("Fetching data from Bybit...")
    fetcher = BybitFetcher()
    ohlcv_data = fetcher.fetch_ohlcv(
        args.symbol,
        timeframe=args.timeframe,
        days=args.days
    )

    logger.info(f"Fetched {len(ohlcv_data.df)} candles")

    # Feature engineering
    logger.info("Creating features...")
    engineer = FeatureEngineer(
        lookback=20,
        prediction_horizon=5,
        return_threshold=0.002
    )
    features = engineer.create_features(ohlcv_data.df)

    logger.info(f"Created {len(features.feature_names)} features")
    logger.info(f"Class distribution: up={np.mean(features.y_class==0):.2%}, "
                f"neutral={np.mean(features.y_class==1):.2%}, "
                f"down={np.mean(features.y_class==2):.2%}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        features.X, features.y_class,
        test_size=0.2,
        shuffle=False  # Keep temporal order
    )

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    config = BayesianNNConfig(
        input_dim=features.X.shape[1],
        hidden_dims=[128, 64, 32],
        num_classes=3,
        dropout_rate=0.1,
        heteroscedastic=True
    )
    model = BayesianNN(config)
    model.to(args.device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # Training loop
    best_val_acc = 0.0

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        # KL annealing
        kl_weight = get_kl_weight(epoch, warmup_epochs=args.kl_warmup)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer,
            kl_weight=kl_weight,
            device=args.device
        )

        # Validate
        val_metrics = validate(model, val_loader, device=args.device)

        # Update scheduler
        scheduler.step(val_metrics['accuracy'])

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'feature_names': features.feature_names,
                'epoch': epoch
            }, args.save_path)

        # Log progress
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2%} | "
                f"High Conf Acc: {val_metrics['high_conf_accuracy']:.2%} | "
                f"Uncertainty: {val_metrics['mean_uncertainty']:.4f} | "
                f"KL Weight: {kl_weight:.2f}"
            )

    logger.info(f"Training complete! Best validation accuracy: {best_val_acc:.2%}")
    logger.info(f"Model saved to {args.save_path}")


if __name__ == '__main__':
    main()
