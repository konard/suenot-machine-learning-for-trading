"""
VAE Training Script for Volatility Surfaces.

Generates synthetic surfaces via SABR, trains a VAE, and evaluates
reconstruction quality and arbitrage-freeness.

Usage:
    python -m python.train
    python -m python.train --epochs 200 --latent-dim 12
"""

import argparse
import sys

import numpy as np

from .model import VAEConfig, VAEModel
from .surface_generator import SurfaceGenerator, default_maturity_days, default_moneyness_grid
from .vol_surface import ArbitrageChecker, compute_butterfly, compute_skew


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE on synthetic volatility surfaces")
    parser.add_argument("--n-surfaces", type=int, default=500, help="Number of training surfaces")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for beta-VAE")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    moneyness = default_moneyness_grid()
    maturities = default_maturity_days() / 365.0  # Convert to years

    print("=" * 60)
    print("VAE Volatility Surface — Training")
    print("=" * 60)
    print(f"Grid: {len(moneyness)} moneyness x {len(maturities)} maturities = {len(moneyness) * len(maturities)} points")
    print(f"Latent dim: {args.latent_dim}, Hidden dim: {args.hidden_dim}, Beta: {args.beta}")

    # Generate training data
    print(f"\nGenerating {args.n_surfaces} SABR surfaces...")
    gen = SurfaceGenerator(seed=args.seed)
    surfaces = gen.generate_batch(args.n_surfaces, moneyness, maturities)
    data = [s.flatten() for s in surfaces]
    print(f"  Surface shape: ({len(maturities)}, {len(moneyness)})")
    print(f"  Flat vector length: {len(data[0])}")

    # Create and train VAE
    config = VAEConfig(
        input_dim=len(data[0]),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
        learning_rate=args.lr,
        epochs=args.epochs,
    )
    vae = VAEModel(config)

    # Loss before training
    recon_pre, mu_pre, lv_pre = vae.forward(data[0])
    loss_pre, recon_l_pre, kl_pre = vae.loss(data[0], recon_pre, mu_pre, lv_pre)
    print(f"\nLoss before training: total={loss_pre:.4f} (recon={recon_l_pre:.4f}, kl={kl_pre:.4f})")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    total, recon_l, kl_l = vae.train(data, epochs=args.epochs, lr=args.lr)
    print(f"Loss after training:  total={total:.4f} (recon={recon_l:.4f}, kl={kl_l:.4f})")

    # Evaluate reconstruction
    print("\n" + "-" * 60)
    print("Reconstruction evaluation (first 5 surfaces):")
    for i in range(min(5, len(surfaces))):
        recon, mu, lv = vae.forward(data[i])
        mse = np.mean((data[i] - recon) ** 2)
        print(f"  Surface {i}: MSE={mse:.6f}")

    # Sample new surfaces and check arbitrage
    print("\n" + "-" * 60)
    print("Sampling 5 new surfaces from prior:")
    for i in range(5):
        sample = vae.sample()
        sample_surface = surfaces[0].__class__(
            moneyness=moneyness,
            maturities=maturities,
            ivols=sample.reshape(len(maturities), len(moneyness)),
        )
        bf_count, _ = ArbitrageChecker.check_butterfly(sample_surface)
        cal_count, _ = ArbitrageChecker.check_calendar(sample_surface)
        skew_val = compute_skew(sample_surface, 0)
        bfly_val = compute_butterfly(sample_surface, 0)
        print(f"  Sample {i}: butterfly_violations={bf_count}, calendar_violations={cal_count}, "
              f"skew={skew_val:.4f}, butterfly={bfly_val:.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
