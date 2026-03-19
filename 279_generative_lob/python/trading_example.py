"""
Trading Example: Generative LOB Trading

Demonstrates the full pipeline:
1. Fetching BTCUSDT orderbook from Bybit (or using synthetic data)
2. Training VAE, GAN, and Diffusion models on LOB snapshots
3. Generating synthetic LOB states
4. Comparing real vs synthetic statistics
"""

import sys
import os
import numpy as np

# Allow running as standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.lob_snapshot import LobSnapshot, generate_test_snapshots
from python.vae import LobVae, LobVaeTrainer
from python.gan import LobGan
from python.diffusion import LobDiffusion
from python.generator import LobGenerator
from python.validator import LobValidator
from python.bybit_client import BybitClient


def main():
    print("=== Chapter 279: Generative LOB Trading (Python) ===\n")

    num_levels = 10

    # ------------------------------------------------------------------
    # Step 1: Try to fetch real LOB data from Bybit
    # ------------------------------------------------------------------
    print("Step 1: Fetching BTCUSDT orderbook from Bybit...")

    client = BybitClient()
    try:
        snapshot = client.fetch_orderbook("BTCUSDT", "spot", 50)
        print(f"  Fetched orderbook: {len(snapshot.bids)} bids, {len(snapshot.asks)} asks")
        print(f"  Mid-price: {snapshot.mid_price():.2f}")
        print(f"  Spread: {snapshot.spread():.4f}")

        print("  Collecting more snapshots...")
        real_snapshots = client.fetch_orderbook_sequence(
            "BTCUSDT", "spot", 50, count=30, interval=0.5
        )
        print(f"  Collected {len(real_snapshots)} real snapshots\n")
    except Exception as e:
        print(f"  Could not fetch from Bybit: {e}")
        print("  Using synthetic test data instead.\n")
        real_snapshots = generate_test_snapshots(50, 25, 50000.0)

    # ------------------------------------------------------------------
    # Step 2: Train VAE model
    # ------------------------------------------------------------------
    print("Step 2: Training VAE on LOB snapshots...")

    vae_gen = LobGenerator(
        num_levels=num_levels, hidden_dim=64, latent_dim=8, model_type="vae"
    )
    vae_history = vae_gen.train(
        real_snapshots, epochs=20, batch_size=16, learning_rate=1e-3
    )
    print(f"  VAE final loss: {vae_history[-1]['loss']:.4f}")

    vae_snapshots = vae_gen.generate_snapshots(50)
    print(f"  Generated {len(vae_snapshots)} VAE snapshots")
    print(f"  Sample mid-price: {vae_snapshots[0].mid_price():.2f}")
    print(f"  Sample spread: {vae_snapshots[0].spread():.4f}\n")

    # ------------------------------------------------------------------
    # Step 3: Train GAN model
    # ------------------------------------------------------------------
    print("Step 3: Training WGAN-GP on LOB snapshots...")

    gan_gen = LobGenerator(
        num_levels=num_levels, hidden_dim=64, latent_dim=16, model_type="gan"
    )
    gan_history = gan_gen.train(
        real_snapshots, epochs=20, batch_size=16, learning_rate=1e-4
    )
    print(f"  GAN final g_loss: {gan_history[-1]['g_loss']:.4f}")

    gan_snapshots = gan_gen.generate_snapshots(50)
    print(f"  Generated {len(gan_snapshots)} GAN snapshots")
    print(f"  Sample mid-price: {gan_snapshots[0].mid_price():.2f}")
    print(f"  Sample spread: {gan_snapshots[0].spread():.4f}\n")

    # ------------------------------------------------------------------
    # Step 4: Train Diffusion model
    # ------------------------------------------------------------------
    print("Step 4: Training Diffusion model on LOB snapshots...")

    diff_gen = LobGenerator(
        num_levels=num_levels, hidden_dim=64, latent_dim=8, model_type="diffusion"
    )
    diff_history = diff_gen.train(
        real_snapshots, epochs=20, batch_size=16, learning_rate=1e-3
    )
    print(f"  Diffusion final loss: {diff_history[-1]:.4f}")

    diff_snapshots = diff_gen.generate_snapshots(50)
    print(f"  Generated {len(diff_snapshots)} Diffusion snapshots")
    print(f"  Sample mid-price: {diff_snapshots[0].mid_price():.2f}")
    print(f"  Sample spread: {diff_snapshots[0].spread():.4f}\n")

    # ------------------------------------------------------------------
    # Step 5: Validate all models
    # ------------------------------------------------------------------
    print("Step 5: Comparing real vs synthetic statistics...\n")

    for name, synth in [("VAE", vae_snapshots), ("GAN", gan_snapshots), ("Diffusion", diff_snapshots)]:
        print(f"--- {name} Validation ---")
        report = LobValidator.validate(real_snapshots, synth)
        print(report)
        print()

    # ------------------------------------------------------------------
    # Step 6: VAE-specific features
    # ------------------------------------------------------------------
    print("Step 6: VAE anomaly detection and interpolation...")

    # Anomaly detection
    import torch
    features = torch.tensor(
        np.array([s.to_feature_vector(num_levels) for s in real_snapshots[:5]]),
        dtype=torch.float32,
    )
    scores = vae_gen.model.anomaly_score(features)
    print(f"  Anomaly scores for first 5 snapshots: {[f'{s:.4f}' for s in scores.tolist()]}")

    # Latent space interpolation
    if len(real_snapshots) >= 2:
        x_a = torch.tensor(real_snapshots[0].to_feature_vector(num_levels), dtype=torch.float32)
        x_b = torch.tensor(real_snapshots[1].to_feature_vector(num_levels), dtype=torch.float32)
        interp = vae_gen.model.interpolate(x_a, x_b, steps=5)
        print(f"  Interpolation: {interp.shape[0]} steps between two LOB states")

    print("\n=== Generative LOB Trading Example Complete ===")


if __name__ == "__main__":
    main()
