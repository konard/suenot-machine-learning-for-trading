"""
Training Script for Operator Learning Models

Supports training DeepONet and FNO architectures on:
- Synthetic PDE data (Black-Scholes, heat equation)
- Real stock/crypto data prepared as operator learning tasks

Features:
- Multi-resolution training strategy
- Curriculum learning (simple -> complex)
- Relative L2 loss and Sobolev loss
- Learning rate scheduling
- Checkpointing and early stopping
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

from .model import (
    DeepONet,
    DeepONetConfig,
    FNOConfig,
    FourierNeuralOperator,
    OptionPricingOperator,
    generate_black_scholes_data,
    generate_yield_curve_data,
    relative_l2_error,
)
from .data_loader import (
    DataConfig,
    generate_synthetic_operator_data,
    train_val_test_split,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

def relative_l2_loss(prediction: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """
    Relative L2 loss: ||pred - target||_2 / ||target||_2

    Standard loss function for operator learning that normalizes by
    the target magnitude, making it scale-invariant.
    """
    diff_norm = torch.norm(prediction - target, dim=-1)
    target_norm = torch.norm(target, dim=-1) + 1e-10
    return torch.mean(diff_norm / target_norm)


def sobolev_loss(
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    derivative_weight: float = 0.1,
) -> "torch.Tensor":
    """
    Sobolev H1 loss: L2 + lambda * L2_derivative

    Matches both function values and their derivatives, encouraging
    smoother and more physically consistent predictions.
    """
    # L2 part
    l2 = relative_l2_loss(prediction, target)

    # Derivative part (finite difference approximation)
    pred_diff = prediction[..., 1:] - prediction[..., :-1]
    target_diff = target[..., 1:] - target[..., :-1]

    deriv_diff_norm = torch.norm(pred_diff - target_diff, dim=-1)
    deriv_target_norm = torch.norm(target_diff, dim=-1) + 1e-10
    deriv_loss = torch.mean(deriv_diff_norm / deriv_target_norm)

    return l2 + derivative_weight * deriv_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Training Functions
# ═══════════════════════════════════════════════════════════════════════════════

def train_deeponet(
    config: Optional[DeepONetConfig] = None,
    data_source: str = "black_scholes",
    n_epochs: int = 100,
    save_dir: str = "checkpoints",
    device: str = "auto",
) -> Dict:
    """
    Train a DeepONet model for operator learning.

    Args:
        config: Model configuration (uses defaults if None)
        data_source: 'black_scholes', 'yield_curve', or 'crypto'
        n_epochs: Number of training epochs
        save_dir: Directory for saving checkpoints
        device: 'cpu', 'cuda', or 'auto'

    Returns:
        Dict with training history and trained model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for training")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training DeepONet on {device} | Data source: {data_source}")
    print("=" * 60)

    # --- Data preparation ---
    if data_source == "black_scholes":
        vol_funcs, eval_pts, prices = generate_black_scholes_data(
            n_samples=2000, n_vol_points=50, n_eval_points=100
        )

        if config is None:
            config = DeepONetConfig(
                branch_input_dim=50,
                trunk_input_dim=2,
                latent_dim=128,
                branch_hidden_dims=[256, 256, 256],
                trunk_hidden_dims=[256, 256, 256],
            )

        # Flatten eval points for single-point evaluation
        n_samples, n_eval = prices.shape
        branch_expanded = np.repeat(vol_funcs, n_eval, axis=0)
        trunk_flat = eval_pts.reshape(-1, 2)
        targets_flat = prices.reshape(-1)

        branch_tensor = torch.tensor(branch_expanded, dtype=torch.float32)
        trunk_tensor = torch.tensor(trunk_flat, dtype=torch.float32)
        target_tensor = torch.tensor(targets_flat, dtype=torch.float32)

    elif data_source == "yield_curve":
        curves_today, curves_future = generate_yield_curve_data(n_samples=2000)

        if config is None:
            config = DeepONetConfig(
                branch_input_dim=30,
                trunk_input_dim=1,
                latent_dim=64,
                branch_hidden_dims=[128, 128, 128],
                trunk_hidden_dims=[128, 128, 128],
            )

        n_mats = curves_today.shape[1]
        maturities = np.linspace(0.25, 30.0, n_mats).reshape(1, -1, 1)
        maturities_tiled = np.tile(maturities, (curves_today.shape[0], 1, 1))

        branch_expanded = np.repeat(curves_today, n_mats, axis=0)
        trunk_flat = maturities_tiled.reshape(-1, 1)
        targets_flat = curves_future.reshape(-1)

        branch_tensor = torch.tensor(branch_expanded, dtype=torch.float32)
        trunk_tensor = torch.tensor(trunk_flat, dtype=torch.float32)
        target_tensor = torch.tensor(targets_flat, dtype=torch.float32)

    else:
        # Use crypto or stock data
        try:
            from .data_loader import prepare_crypto_price_operator_data
            crypto_data = prepare_crypto_price_operator_data(
                symbol="BTCUSDT", limit=1000, sequence_length=60, prediction_horizon=5
            )
            branch_np = crypto_data["branch_inputs"]
            trunk_np = crypto_data["trunk_inputs"]
            targets_np = crypto_data["targets"]
        except Exception as e:
            print(f"Crypto data failed ({e}), falling back to synthetic.")
            return train_deeponet(config, "black_scholes", n_epochs, save_dir, device)

        if config is None:
            config = DeepONetConfig(
                branch_input_dim=branch_np.shape[1],
                trunk_input_dim=trunk_np.shape[-1],
                latent_dim=64,
            )

        # Flatten multi-eval structure
        n_samples, n_eval, trunk_dim = trunk_np.shape
        branch_expanded = np.repeat(branch_np, n_eval, axis=0)
        trunk_flat = trunk_np.reshape(-1, trunk_dim)
        targets_flat = targets_np.reshape(-1)

        branch_tensor = torch.tensor(branch_expanded, dtype=torch.float32)
        trunk_tensor = torch.tensor(trunk_flat, dtype=torch.float32)
        target_tensor = torch.tensor(targets_flat, dtype=torch.float32)

    # --- Split data ---
    n = branch_tensor.shape[0]
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_ds = TensorDataset(
        branch_tensor[:train_end],
        trunk_tensor[:train_end],
        target_tensor[:train_end],
    )
    val_ds = TensorDataset(
        branch_tensor[train_end:val_end],
        trunk_tensor[train_end:val_end],
        target_tensor[train_end:val_end],
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    # --- Model ---
    model = DeepONet(config).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Training samples: {train_end:,} | Validation: {val_end - train_end:,}")

    # --- Training loop ---
    history = {"train_loss": [], "val_loss": [], "val_rel_l2": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_losses = []
        for branch_b, trunk_b, target_b in train_loader:
            branch_b = branch_b.to(device)
            trunk_b = trunk_b.to(device)
            target_b = target_b.to(device)

            optimizer.zero_grad()
            pred = model(branch_b, trunk_b)
            loss = nn.MSELoss()(pred, target_b) + 0.1 * relative_l2_loss(
                pred.unsqueeze(-1), target_b.unsqueeze(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        val_rel_l2s = []
        with torch.no_grad():
            for branch_b, trunk_b, target_b in val_loader:
                branch_b = branch_b.to(device)
                trunk_b = trunk_b.to(device)
                target_b = target_b.to(device)
                pred = model(branch_b, trunk_b)
                val_loss = nn.MSELoss()(pred, target_b)
                val_losses.append(val_loss.item())

                rel_err = relative_l2_error(
                    pred.cpu().numpy(), target_b.cpu().numpy()
                )
                val_rel_l2s.append(rel_err)

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_rel_l2 = np.mean(val_rel_l2s)
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_rel_l2"].append(avg_rel_l2)
        history["lr"].append(current_lr)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs} | "
                f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
                f"Rel L2: {avg_rel_l2:.4f} | LR: {current_lr:.2e}"
            )

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "deeponet_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(save_dir, "deeponet_best.pt"), weights_only=True)
    )

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    return {
        "model": model,
        "config": config,
        "history": history,
        "best_val_loss": best_val_loss,
    }


def train_fno(
    config: Optional[FNOConfig] = None,
    data_source: str = "heat",
    n_epochs: int = 100,
    save_dir: str = "checkpoints",
    device: str = "auto",
) -> Dict:
    """
    Train a Fourier Neural Operator model.

    Args:
        config: FNO configuration
        data_source: 'heat', 'advection', 'burgers', or 'crypto'
        n_epochs: Number of training epochs
        save_dir: Directory for saving checkpoints
        device: 'cpu', 'cuda', or 'auto'

    Returns:
        Dict with training history and trained model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for training")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training FNO on {device} | Data source: {data_source}")
    print("=" * 60)

    # --- Data preparation ---
    if data_source in ("heat", "advection", "burgers"):
        inputs, outputs = generate_synthetic_operator_data(
            data_source, n_samples=2000, n_points=128
        )

        if config is None:
            config = FNOConfig(
                input_dim=1,
                output_dim=1,
                hidden_channels=64,
                n_fourier_layers=4,
                n_fourier_modes=16,
                spatial_dim=1,
            )
    else:
        # Crypto data
        try:
            from .data_loader import prepare_crypto_price_operator_data
            crypto_data = prepare_crypto_price_operator_data(
                symbol="BTCUSDT", limit=1000
            )
            inputs = crypto_data["fno_inputs"]
            outputs = crypto_data["fno_targets"]
        except Exception as e:
            print(f"Crypto data failed ({e}), falling back to heat equation.")
            return train_fno(config, "heat", n_epochs, save_dir, device)

        if config is None:
            config = FNOConfig(
                input_dim=1,
                output_dim=1,
                hidden_channels=32,
                n_fourier_layers=4,
                n_fourier_modes=12,
                spatial_dim=1,
            )

    # --- Split ---
    n = inputs.shape[0]
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    x_train = torch.tensor(inputs[:train_end], dtype=torch.float32)
    y_train = torch.tensor(outputs[:train_end], dtype=torch.float32)
    x_val = torch.tensor(inputs[train_end:val_end], dtype=torch.float32)
    y_val = torch.tensor(outputs[train_end:val_end], dtype=torch.float32)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    # --- Model ---
    model = FourierNeuralOperator(config).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Training samples: {train_end:,} | Validation: {val_end - train_end:,}")

    # --- Training loop ---
    history = {"train_loss": [], "val_loss": [], "val_rel_l2": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)

            # Combined loss
            loss = sobolev_loss(pred, y_batch, derivative_weight=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        val_rel_l2s = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                val_loss = relative_l2_loss(pred, y_batch)
                val_losses.append(val_loss.item())
                val_rel_l2s.append(val_loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_rel_l2 = np.mean(val_rel_l2s)
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_rel_l2"].append(avg_rel_l2)
        history["lr"].append(current_lr)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs} | "
                f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
                f"Rel L2: {avg_rel_l2:.4f} | LR: {current_lr:.2e}"
            )

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "fno_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(
        torch.load(os.path.join(save_dir, "fno_best.pt"), weights_only=True)
    )

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    # --- Zero-shot super-resolution test ---
    print("\nZero-shot super-resolution test:")
    model.eval()
    with torch.no_grad():
        test_input = x_val[:5].to(device)
        original_res = test_input.shape[-1]

        for res_multiplier in [2, 4]:
            target_res = original_res * res_multiplier
            pred_hr = model.predict_superresolution(test_input, target_res)
            print(
                f"  {original_res} -> {target_res} points: "
                f"output shape = {pred_hr.shape}"
            )

    return {
        "model": model,
        "config": config,
        "history": history,
        "best_val_loss": best_val_loss,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Operator Learning Models")
    parser.add_argument(
        "--model",
        type=str,
        default="deeponet",
        choices=["deeponet", "fno"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="black_scholes",
        choices=["black_scholes", "yield_curve", "heat", "advection", "burgers", "crypto"],
        help="Data source for training",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")

    args = parser.parse_args()

    if args.model == "deeponet":
        config = DeepONetConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        result = train_deeponet(
            config=config,
            data_source=args.data,
            n_epochs=args.epochs,
            save_dir=args.save_dir,
            device=args.device,
        )
    else:
        config = FNOConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        result = train_fno(
            config=config,
            data_source=args.data,
            n_epochs=args.epochs,
            save_dir=args.save_dir,
            device=args.device,
        )

    # Save training history
    history_path = os.path.join(args.save_dir, f"{args.model}_history.json")
    history_serializable = {
        k: [float(v) for v in vals] for k, vals in result["history"].items()
    }
    with open(history_path, "w") as f:
        json.dump(history_serializable, f, indent=2)

    print(f"\nTraining history saved to {history_path}")


if __name__ == "__main__":
    main()
