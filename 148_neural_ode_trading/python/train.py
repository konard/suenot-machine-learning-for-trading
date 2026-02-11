"""
Training Pipeline for Neural ODE Trading Models

Supports training:
1. NeuralODE - Standard Neural ODE for sequence prediction
2. LatentODE - VAE-based model for irregular time series
3. ODE-RNN - Hybrid architecture

Uses the adjoint sensitivity method for memory-efficient backpropagation.
The adjoint method computes gradients by solving an augmented ODE backward
in time, requiring only O(1) memory regardless of the number of solver steps.

Usage:
    python train.py --model neural_ode --symbol BTCUSDT --epochs 100
    python train.py --model latent_ode --symbol ETHUSDT --epochs 200
    python train.py --model ode_rnn --source stock --epochs 150
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.neural_ode import NeuralODE
from python.latent_ode import LatentODE
from python.ode_rnn import ODERNN
from python.data_loader import create_irregular_dataset


def train_neural_ode(
    model: NeuralODE,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    save_dir: str = "checkpoints",
) -> Dict[str, list]:
    """
    Train a NeuralODE model.

    Args:
        model: NeuralODE model instance
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device ('cpu' or 'cuda')
        save_dir: Directory to save checkpoints

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "test_loss": [], "nfe": []}
    best_test_loss = float("inf")

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        total_nfe = 0

        for batch in train_loader:
            x = batch["x"].to(device)        # (batch, seq_len, features)
            target = batch["target"].to(device)  # (batch, 1)

            optimizer.zero_grad()

            pred = model(x)  # (batch, 1)
            loss = criterion(pred, target)

            loss.backward()

            # Gradient clipping for stable ODE training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(loss.item())
            total_nfe += model.ode_func.nfe

        scheduler.step()

        # Evaluation
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                target = batch["target"].to(device)
                pred = model(x)
                loss = criterion(pred, target)
                test_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_test = np.mean(test_losses) if test_losses else float("inf")
        avg_nfe = total_nfe / max(len(train_loader), 1)

        history["train_loss"].append(avg_train)
        history["test_loss"].append(avg_test)
        history["nfe"].append(avg_nfe)

        if avg_test < best_test_loss:
            best_test_loss = avg_test
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_neural_ode.pt"),
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{epochs} | "
                f"Train: {avg_train:.6f} | "
                f"Test: {avg_test:.6f} | "
                f"NFE: {avg_nfe:.0f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    return history


def train_latent_ode(
    model: LatentODE,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 200,
    lr: float = 1e-3,
    kl_anneal_epochs: int = 50,
    device: str = "cpu",
    save_dir: str = "checkpoints",
) -> Dict[str, list]:
    """
    Train a Latent ODE model with KL annealing.

    KL annealing: gradually increase the weight of the KL divergence term
    to prevent posterior collapse (a common VAE training issue).

    Args:
        model: LatentODE model instance
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of epochs
        lr: Learning rate
        kl_anneal_epochs: Epochs over which to anneal KL weight from 0 to 1
        device: Device
        save_dir: Checkpoint directory

    Returns:
        Training history dictionary
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "test_loss": [],
        "recon_loss": [], "kl_loss": [],
    }
    best_test_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        # KL annealing schedule
        kl_weight = min(1.0, epoch / max(kl_anneal_epochs, 1)) * 0.01

        # Training
        model.train()
        epoch_losses = {"total": [], "recon": [], "kl": []}

        for batch in train_loader:
            x = batch["x"].to(device)
            t = batch["t"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            # Forward pass through Latent ODE
            result = model(x, t[0] if t.dim() > 1 else t, mask=mask)

            # Compute VAE loss
            losses = LatentODE.compute_loss(
                x_true=x,
                result=result,
                mask=mask,
                kl_weight=kl_weight,
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses["total"].append(losses["total"].item())
            epoch_losses["recon"].append(losses["recon"].item())
            epoch_losses["kl"].append(losses["kl"].item())

        scheduler.step()

        # Evaluation
        model.eval()
        test_losses_epoch = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                t = batch["t"].to(device)
                mask = batch["mask"].to(device)

                result = model(x, t[0] if t.dim() > 1 else t, mask=mask)
                losses = LatentODE.compute_loss(x, result, mask, kl_weight)
                test_losses_epoch.append(losses["total"].item())

        avg_train = np.mean(epoch_losses["total"])
        avg_test = np.mean(test_losses_epoch) if test_losses_epoch else float("inf")

        history["train_loss"].append(avg_train)
        history["test_loss"].append(avg_test)
        history["recon_loss"].append(np.mean(epoch_losses["recon"]))
        history["kl_loss"].append(np.mean(epoch_losses["kl"]))

        if avg_test < best_test_loss:
            best_test_loss = avg_test
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_latent_ode.pt"),
            )

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{epochs} | "
                f"Total: {avg_train:.6f} | "
                f"Recon: {np.mean(epoch_losses['recon']):.6f} | "
                f"KL: {np.mean(epoch_losses['kl']):.4f} | "
                f"Test: {avg_test:.6f} | "
                f"KL_w: {kl_weight:.4f}"
            )

    return history


def train_ode_rnn(
    model: ODERNN,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    save_dir: str = "checkpoints",
) -> Dict[str, list]:
    """
    Train an ODE-RNN model.

    Args:
        model: ODERNN model instance
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Device
        save_dir: Checkpoint directory

    Returns:
        Training history
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "test_loss": []}
    best_test_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["x"].to(device)
            t = batch["t"].to(device)
            mask_data = batch["mask"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()

            # For ODE-RNN, we need per-timestep mask (any feature observed)
            obs_mask = mask_data.any(dim=-1).float()  # (batch, seq_len)

            # Use common time vector
            t_common = t[0] if t.dim() > 1 else t

            pred = model(x, t_common, mask=obs_mask)
            loss = criterion(pred, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Evaluation
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                t = batch["t"].to(device)
                mask_data = batch["mask"].to(device)
                target = batch["target"].to(device)

                obs_mask = mask_data.any(dim=-1).float()
                t_common = t[0] if t.dim() > 1 else t

                pred = model(x, t_common, mask=obs_mask)
                loss = criterion(pred, target)
                test_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_test = np.mean(test_losses) if test_losses else float("inf")

        history["train_loss"].append(avg_train)
        history["test_loss"].append(avg_test)

        if avg_test < best_test_loss:
            best_test_loss = avg_test
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_ode_rnn.pt"),
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{epochs} | "
                f"Train: {avg_train:.6f} | "
                f"Test: {avg_test:.6f}"
            )

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Neural ODE Trading Models")
    parser.add_argument(
        "--model",
        type=str,
        default="neural_ode",
        choices=["neural_ode", "latent_ode", "ode_rnn"],
        help="Model architecture to train",
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument(
        "--source",
        type=str,
        default="bybit",
        choices=["bybit", "stock", "synthetic"],
        help="Data source",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dim (LatentODE)")
    parser.add_argument("--seq_len", type=int, default=60, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--solver", type=str, default="dopri5", help="ODE solver")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Save directory")

    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"Neural ODE Trading - Training Pipeline")
    print(f"{'=' * 60}")
    print(f"Model:   {args.model}")
    print(f"Symbol:  {args.symbol}")
    print(f"Source:  {args.source}")
    print(f"Solver:  {args.solver}")
    print(f"Device:  {args.device}")
    print(f"{'=' * 60}")

    # Create datasets
    print("\nLoading data...")
    train_ds, test_ds = create_irregular_dataset(
        symbol=args.symbol,
        source=args.source,
        seq_len=args.seq_len,
    )
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    input_dim = 5  # OHLCV

    # Create and train model
    if args.model == "neural_ode":
        model = NeuralODE(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            seq_len=args.seq_len,
            solver=args.solver,
            use_adjoint=True,
            task="regression",
        )
        print(f"\nNeuralODE parameters: {sum(p.numel() for p in model.parameters()):,}")
        history = train_neural_ode(
            model, train_loader, test_loader,
            epochs=args.epochs, lr=args.lr,
            device=args.device, save_dir=args.save_dir,
        )

    elif args.model == "latent_ode":
        model = LatentODE(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            output_dim=input_dim,
            solver=args.solver,
            use_adjoint=True,
        )
        print(f"\nLatentODE parameters: {sum(p.numel() for p in model.parameters()):,}")
        history = train_latent_ode(
            model, train_loader, test_loader,
            epochs=args.epochs, lr=args.lr,
            device=args.device, save_dir=args.save_dir,
        )

    elif args.model == "ode_rnn":
        model = ODERNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            solver=args.solver,
            use_adjoint=True,
            task="regression",
        )
        print(f"\nODE-RNN parameters: {sum(p.numel() for p in model.parameters()):,}")
        history = train_ode_rnn(
            model, train_loader, test_loader,
            epochs=args.epochs, lr=args.lr,
            device=args.device, save_dir=args.save_dir,
        )

    print(f"\nTraining complete!")
    print(f"Best test loss: {min(history['test_loss']):.6f}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")

    return history


if __name__ == "__main__":
    main()
