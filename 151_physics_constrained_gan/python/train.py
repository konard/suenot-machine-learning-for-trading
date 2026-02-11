"""
Training Module for Physics-Constrained GAN

Implements the WGAN-GP training loop with physics constraint penalties
for generating realistic financial time series.

Training procedure:
    1. For each epoch:
        a. Train critic n_critic times (Wasserstein loss + gradient penalty)
        b. Train generator once (adversarial loss + physics penalty)
    2. Log metrics and save checkpoints periodically
    3. Evaluate statistical properties of generated data

Usage:
    python train.py --symbol BTCUSDT --epochs 500 --batch-size 64
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from model import GANConfig, PhysicsConstrainedGAN
from data_loader import (
    BybitDataLoader,
    generate_synthetic_data,
    create_dataloader,
    ReturnWindowDataset,
)


def train_physics_gan(
    gan: PhysicsConstrainedGAN,
    data_dict: Dict,
    epochs: int = 500,
    batch_size: int = 64,
    log_interval: int = 10,
    save_interval: int = 50,
    save_dir: str = "checkpoints",
    eval_interval: int = 25,
) -> Dict[str, List[float]]:
    """
    Train the Physics-Constrained GAN.

    Args:
        gan: PhysicsConstrainedGAN instance
        data_dict: Preprocessed data dict from data_loader
        epochs: Number of training epochs
        batch_size: Batch size
        log_interval: Print metrics every N epochs
        save_interval: Save checkpoint every N epochs
        save_dir: Directory for checkpoints
        eval_interval: Evaluate generated data statistics every N epochs

    Returns:
        Dict with training history (loss curves)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for training")

    os.makedirs(save_dir, exist_ok=True)

    # Set physics targets from real data
    real_returns = torch.tensor(data_dict["windows"], dtype=torch.float32)
    gan.physics_loss.set_targets_from_data(real_returns.to(gan.device))

    # Create data loader
    dataloader = create_dataloader(data_dict, batch_size=batch_size)

    # Training history
    history = {
        "critic_loss": [],
        "wasserstein_distance": [],
        "gen_loss": [],
        "adversarial_loss": [],
        "martingale": [],
        "volatility_clustering": [],
        "kurtosis": [],
        "leverage_effect": [],
        "autocorrelation": [],
        "total_physics": [],
    }

    print("=" * 70)
    print("Physics-Constrained GAN Training")
    print("=" * 70)
    print(f"  Data windows: {len(data_dict['windows'])}")
    print(f"  Sequence length: {gan.config.seq_len}")
    print(f"  Latent dim: {gan.config.latent_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Critic iterations: {gan.config.n_critic}")
    print(f"  Device: {gan.device}")
    print(f"  Physics weights:")
    print(f"    Martingale:  {gan.config.lambda_martingale}")
    print(f"    Volatility:  {gan.config.lambda_volatility}")
    print(f"    Kurtosis:    {gan.config.lambda_kurtosis}")
    print(f"    Leverage:    {gan.config.lambda_leverage}")
    print(f"    Autocorr:    {gan.config.lambda_autocorr}")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_critic_losses = []
        epoch_gen_losses = []
        epoch_wd = []
        epoch_physics = {}

        for batch_idx, (returns, regimes, vol_levels) in enumerate(dataloader):
            returns = returns.to(gan.device)
            regimes = regimes.to(gan.device)
            vol_levels = vol_levels.to(gan.device)

            # ---- Train Critic ----
            for _ in range(gan.config.n_critic):
                critic_metrics = gan.train_critic_step(returns, regimes, vol_levels)
                epoch_critic_losses.append(critic_metrics["critic_loss"])
                epoch_wd.append(critic_metrics["wasserstein_distance"])

            # ---- Train Generator ----
            gen_metrics = gan.train_generator_step(
                batch_size=returns.size(0),
                regime=regimes,
                vol_level=vol_levels,
            )
            epoch_gen_losses.append(gen_metrics["gen_loss"])

            # Accumulate physics losses
            for key in ["martingale", "volatility_clustering", "kurtosis",
                        "leverage_effect", "autocorrelation", "total_physics"]:
                if key not in epoch_physics:
                    epoch_physics[key] = []
                epoch_physics[key].append(gen_metrics.get(key, 0.0))

        # Record epoch averages
        history["critic_loss"].append(np.mean(epoch_critic_losses))
        history["wasserstein_distance"].append(np.mean(epoch_wd))
        history["gen_loss"].append(np.mean(epoch_gen_losses))
        history["adversarial_loss"].append(gen_metrics.get("adversarial_loss", 0.0))

        for key in ["martingale", "volatility_clustering", "kurtosis",
                     "leverage_effect", "autocorrelation", "total_physics"]:
            history[key].append(np.mean(epoch_physics.get(key, [0.0])))

        # ---- Logging ----
        if epoch % log_interval == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"D_loss: {history['critic_loss'][-1]:8.4f} | "
                f"G_loss: {history['gen_loss'][-1]:8.4f} | "
                f"WD: {history['wasserstein_distance'][-1]:8.4f} | "
                f"Physics: {history['total_physics'][-1]:8.4f} | "
                f"Time: {elapsed:.0f}s"
            )

        # ---- Evaluation ----
        if epoch % eval_interval == 0:
            stats = gan.evaluate_statistics(n_samples=200)
            print(f"  Generated stats: mean={stats['mean']:.6f}, "
                  f"std={stats['std']:.4f}, "
                  f"kurtosis={stats['kurtosis']:.2f}, "
                  f"ACF|r|(1)={stats['acf_abs_lag1']:.3f}")

        # ---- Save checkpoint ----
        if epoch % save_interval == 0:
            path = os.path.join(save_dir, f"physics_gan_epoch_{epoch}.pt")
            gan.save(path)

    # Save final model
    final_path = os.path.join(save_dir, "physics_gan_final.pt")
    gan.save(final_path)

    # Save training history
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)
    print(f"Training history saved to {history_path}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} min)")

    return history


def train_with_curriculum(
    gan: PhysicsConstrainedGAN,
    data_dict: Dict,
    epochs: int = 500,
    batch_size: int = 64,
    warmup_epochs: int = 50,
) -> Dict[str, List[float]]:
    """
    Train with curriculum learning: gradually increase physics constraint weights.

    Phase 1 (warmup): Train as standard WGAN-GP (physics weights = 0)
    Phase 2 (ramp-up): Linearly increase physics weights over epochs
    Phase 3 (full): Train with full physics constraints

    Args:
        gan: PhysicsConstrainedGAN instance
        data_dict: Preprocessed data dict
        epochs: Total training epochs
        batch_size: Batch size
        warmup_epochs: Number of warmup epochs (no physics)

    Returns:
        Training history dict
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for training")

    # Store original weights
    original_weights = {
        "lambda_martingale": gan.config.lambda_martingale,
        "lambda_volatility": gan.config.lambda_volatility,
        "lambda_kurtosis": gan.config.lambda_kurtosis,
        "lambda_leverage": gan.config.lambda_leverage,
        "lambda_autocorr": gan.config.lambda_autocorr,
        "lambda_moment": gan.config.lambda_moment,
    }

    ramp_epochs = warmup_epochs  # Same length for ramp-up phase

    print(f"Curriculum training: {warmup_epochs} warmup + {ramp_epochs} ramp + "
          f"{epochs - warmup_epochs - ramp_epochs} full")

    # Set physics weights to zero for warmup
    for key in original_weights:
        setattr(gan.config, key, 0.0)

    # Phase 1: Warmup (standard WGAN-GP)
    print("\n--- Phase 1: Warmup (no physics constraints) ---")
    h1 = train_physics_gan(
        gan, data_dict, epochs=warmup_epochs, batch_size=batch_size,
        log_interval=10, save_interval=warmup_epochs,
    )

    # Phase 2: Ramp up physics constraints
    print("\n--- Phase 2: Ramping up physics constraints ---")
    for epoch_offset in range(ramp_epochs):
        alpha = (epoch_offset + 1) / ramp_epochs
        for key, val in original_weights.items():
            setattr(gan.config, key, val * alpha)

    h2 = train_physics_gan(
        gan, data_dict, epochs=ramp_epochs, batch_size=batch_size,
        log_interval=10, save_interval=ramp_epochs,
    )

    # Phase 3: Full physics constraints
    print("\n--- Phase 3: Full physics constraints ---")
    remaining = epochs - warmup_epochs - ramp_epochs
    if remaining > 0:
        for key, val in original_weights.items():
            setattr(gan.config, key, val)

        h3 = train_physics_gan(
            gan, data_dict, epochs=remaining, batch_size=batch_size,
            log_interval=10, save_interval=50,
        )
    else:
        h3 = {k: [] for k in h1}

    # Combine histories
    combined = {}
    for key in h1:
        combined[key] = h1[key] + h2[key] + h3.get(key, [])

    return combined


def main():
    parser = argparse.ArgumentParser(description="Train Physics-Constrained GAN")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--interval", type=str, default="60",
                        help="Candle interval (default: 60 = 1 hour)")
    parser.add_argument("--n-candles", type=int, default=5000,
                        help="Number of candles to fetch")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--latent-dim", type=int, default=128,
                        help="Latent dimension")
    parser.add_argument("--n-critic", type=int, default=5,
                        help="Critic iterations per generator iteration")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Use synthetic data instead of Bybit")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use curriculum learning")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda/mps)")

    args = parser.parse_args()

    # Load data
    if args.use_synthetic:
        print("Using synthetic Heston model data...")
        data_dict = generate_synthetic_data(seq_len=args.seq_len, n_samples=500)
    else:
        print(f"Fetching {args.symbol} data from Bybit...")
        loader = BybitDataLoader(symbol=args.symbol, interval=args.interval)
        data_dict = loader.fetch_and_preprocess(
            n_candles=args.n_candles,
            seq_len=args.seq_len,
        )

    print(f"Data: {len(data_dict['windows'])} windows of length {args.seq_len}")
    print(f"Real data stats: {data_dict['stats']}")

    # Configure GAN
    config = GANConfig(
        latent_dim=args.latent_dim,
        seq_len=args.seq_len,
        n_critic=args.n_critic,
        lr_gen=args.lr,
        lr_critic=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
    )

    # Initialize and train
    gan = PhysicsConstrainedGAN(config)

    if args.curriculum:
        history = train_with_curriculum(
            gan, data_dict,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        history = train_physics_gan(
            gan, data_dict,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=args.save_dir,
        )

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    stats = gan.evaluate_statistics(n_samples=500)
    for key, val in stats.items():
        print(f"  {key}: {val:.6f}")

    print(f"\nReal data statistics:")
    for key, val in data_dict["stats"].items():
        print(f"  {key}: {val:.6f}")


if __name__ == "__main__":
    main()
