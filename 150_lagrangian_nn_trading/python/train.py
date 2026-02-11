"""
Training pipeline for Lagrangian Neural Networks.

Supports:
  - Standard LNN (energy-conserving)
  - Dissipative LNN (with market friction)
  - Forced LNN (with external inputs)
  - Multi-scale LNN (multiple timeframes)

Usage:
    python train.py --model lnn --epochs 500 --lr 3e-4
    python train.py --model dissipative --epochs 500
    python train.py --model forced --epochs 500 --external-dim 3
    python train.py --model multiscale --epochs 500
"""

import argparse
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import (
    LagrangianNN,
    DissipativeLNN,
    ForcedLNN,
    MultiScaleLNN,
    compute_lnn_loss,
    compute_dissipative_loss,
    compute_forced_loss,
    integrate_trajectory,
    compute_energy_along_trajectory,
)
from data_loader import (
    fetch_bybit_extended,
    fetch_yahoo_data,
    construct_config_space,
    construct_multiscale_config_space,
    compute_external_features,
    normalize_config_space,
    train_test_split_sequential,
)


def create_model(
    model_type: str,
    coord_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 4,
    external_dim: int = 3,
    mass_reg: float = 1e-4,
) -> nn.Module:
    """
    Create an LNN model based on type.

    Args:
        model_type: One of "lnn", "dissipative", "forced", "multiscale".
        coord_dim: Dimension of generalized coordinates.
        hidden_dim: Width of hidden layers.
        num_layers: Number of hidden layers.
        external_dim: Dimension of external input (for forced model).
        mass_reg: Regularization for mass matrix.

    Returns:
        PyTorch model.
    """
    if model_type == "lnn":
        return LagrangianNN(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mass_reg=mass_reg,
        )
    elif model_type == "dissipative":
        return DissipativeLNN(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mass_reg=mass_reg,
        )
    elif model_type == "forced":
        return ForcedLNN(
            coord_dim=coord_dim,
            external_dim=external_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mass_reg=mass_reg,
        )
    elif model_type == "multiscale":
        return MultiScaleLNN(
            n_scales=coord_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mass_reg=mass_reg,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model_type: str,
    energy_reg: float = 0.0,
    dissipation_reg: float = 0.01,
    force_reg: float = 0.001,
    grad_clip: float = 1.0,
) -> dict:
    """
    Train for one epoch.

    Returns:
        Dictionary of average metrics.
    """
    model.train()
    total_metrics: dict = {}
    n_batches = 0

    for batch in dataloader:
        if model_type == "forced" and len(batch) == 4:
            q, qdot, qddot_target, external = batch
        else:
            q, qdot, qddot_target = batch[:3]
            external = None

        optimizer.zero_grad()

        if model_type == "lnn" or model_type == "multiscale":
            loss, metrics = compute_lnn_loss(
                model, q, qdot, qddot_target, energy_reg=energy_reg
            )
        elif model_type == "dissipative":
            loss, metrics = compute_dissipative_loss(
                model, q, qdot, qddot_target, dissipation_reg=dissipation_reg
            )
        elif model_type == "forced":
            loss, metrics = compute_forced_loss(
                model, q, qdot, qddot_target, u=external, force_reg=force_reg
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n_batches += 1

    # Average metrics
    return {k: v / n_batches for k, v in total_metrics.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    model_type: str,
) -> dict:
    """
    Evaluate model on validation data.

    Note: Even though we use @no_grad decorator at the outer level,
    we need enable_grad inside for autograd within the model.
    """
    model.eval()
    total_metrics: dict = {}
    n_batches = 0

    for batch in dataloader:
        q, qdot, qddot_target = batch[:3]

        # Need gradients for autograd inside the model
        with torch.enable_grad():
            q = q.requires_grad_(True)
            qdot = qdot.requires_grad_(True)
            qddot_pred = model(q, qdot)

        loss = ((qddot_pred - qddot_target) ** 2).mean()
        metrics = {"loss_total": loss.item()}

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}


def evaluate_energy_conservation(
    model: nn.Module,
    q_test: torch.Tensor,
    qdot_test: torch.Tensor,
    dt: float = 0.1,
    n_steps: int = 100,
) -> dict:
    """
    Evaluate energy conservation over a trajectory.

    Returns:
        Dictionary with energy statistics.
    """
    model.eval()
    # Use first test sample as initial condition
    q0 = q_test[:1]
    qdot0 = qdot_test[:1]

    traj_q, traj_qdot = integrate_trajectory(
        model, q0, qdot0, dt, n_steps, method="rk4"
    )
    energies = compute_energy_along_trajectory(model, traj_q, traj_qdot)

    return {
        "energy_mean": float(energies.mean()),
        "energy_std": float(energies.std()),
        "energy_drift": float(energies[-1].mean() - energies[0].mean()),
        "energy_relative_drift": float(
            (energies[-1].mean() - energies[0].mean())
            / (abs(energies[0].mean()) + 1e-10)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Lagrangian Neural Network")

    # Data arguments
    parser.add_argument(
        "--source", type=str, default="bybit", choices=["bybit", "yahoo"]
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="5")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--ma-window", type=int, default=20)

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="lnn",
        choices=["lnn", "dissipative", "forced", "multiscale"],
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--external-dim", type=int, default=3)
    parser.add_argument("--mass-reg", type=float, default=1e-4)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--energy-reg", type=float, default=0.001)
    parser.add_argument("--dissipation-reg", type=float, default=0.01)
    parser.add_argument("--force-reg", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"]
    )

    # Output
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--save-model", type=str, default="saved_model.pt")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Fetch Data ----
    print(f"Fetching data: {args.source}/{args.symbol}...")
    if args.source == "bybit":
        df = fetch_bybit_extended(
            symbol=args.symbol,
            interval=args.interval,
            total_candles=args.limit,
        )
    else:
        df = fetch_yahoo_data(symbol=args.symbol, period="2y", interval="1d")

    print(f"Loaded {len(df)} candles")

    if len(df) < 100:
        print("ERROR: Not enough data. Need at least 100 candles.")
        sys.exit(1)

    # ---- Construct Configuration Space ----
    print("Constructing configuration space...")
    if args.model == "multiscale":
        q, qdot, qddot = construct_multiscale_config_space(
            df, windows=[5, 20, 50]
        )
    else:
        q, qdot, qddot = construct_config_space(
            df, ma_window=args.ma_window, velocity_method="gradient"
        )

    q_norm, qdot_norm, qddot_norm, stats = normalize_config_space(q, qdot, qddot)

    coord_dim = q_norm.shape[1]
    print(f"Config space: coord_dim={coord_dim}, samples={len(q_norm)}")

    # External features (for forced model)
    external = None
    if args.model == "forced":
        print("Computing external features...")
        external_raw = compute_external_features(df)
        # Align with config space (skip MA warmup rows)
        n_valid = len(q_norm)
        external = external_raw[-n_valid:]
        print(f"External features shape: {external.shape}")

    # ---- Train/Test Split ----
    splits = train_test_split_sequential(q_norm, qdot_norm, qddot_norm)
    q_train, qdot_train, qddot_train = splits[:3]
    q_test, qdot_test, qddot_test = splits[3:]
    print(f"Train: {len(q_train)}, Test: {len(q_test)}")

    # ---- Create DataLoaders ----
    if args.model == "forced" and external is not None:
        split_idx = int(len(q_norm) * 0.8)
        ext_train = external[:split_idx]
        ext_test = external[split_idx:]

        train_dataset = TensorDataset(
            torch.FloatTensor(q_train),
            torch.FloatTensor(qdot_train),
            torch.FloatTensor(qddot_train),
            torch.FloatTensor(ext_train),
        )
    else:
        train_dataset = TensorDataset(
            torch.FloatTensor(q_train),
            torch.FloatTensor(qdot_train),
            torch.FloatTensor(qddot_train),
        )

    test_dataset = TensorDataset(
        torch.FloatTensor(q_test),
        torch.FloatTensor(qdot_test),
        torch.FloatTensor(qddot_test),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ---- Create Model ----
    model = create_model(
        model_type=args.model,
        coord_dim=coord_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        external_dim=args.external_dim,
        mass_reg=args.mass_reg,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model} ({n_params} parameters)")

    # ---- Optimizer and Scheduler ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=0.5
        )

    # ---- Training Loop ----
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float("inf")
    history: list = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            args.model,
            energy_reg=args.energy_reg,
            dissipation_reg=args.dissipation_reg,
            force_reg=args.force_reg,
            grad_clip=args.grad_clip,
        )

        # Evaluate
        val_metrics = evaluate(model, test_loader, args.model)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "time": elapsed,
        }
        history.append(entry)

        # Save best model
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            save_path = os.path.join(args.output_dir, args.save_model)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model,
                    "coord_dim": coord_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "external_dim": args.external_dim,
                    "mass_reg": args.mass_reg,
                    "stats": stats,
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                save_path,
            )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.epochs}: "
                f"train_loss={train_metrics.get('loss_total', 0):.6f}, "
                f"val_loss={val_metrics['loss_total']:.6f}, "
                f"lr={optimizer.param_groups[0]['lr']:.6f}, "
                f"time={elapsed:.2f}s"
            )

    # ---- Final Evaluation ----
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Energy conservation check
    print("\nEnergy conservation evaluation:")
    q_test_t = torch.FloatTensor(q_test)
    qdot_test_t = torch.FloatTensor(qdot_test)
    energy_stats = evaluate_energy_conservation(model, q_test_t, qdot_test_t)
    for k, v in energy_stats.items():
        print(f"  {k}: {v:.8f}")

    # ---- Save History ----
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    # Save stats for inference
    stats_path = os.path.join(args.output_dir, "normalization_stats.json")
    stats_serializable = {k: v.tolist() for k, v in stats.items()}
    with open(stats_path, "w") as f:
        json.dump(stats_serializable, f, indent=2)

    print(f"Model saved to {os.path.join(args.output_dir, args.save_model)}")


if __name__ == "__main__":
    main()
