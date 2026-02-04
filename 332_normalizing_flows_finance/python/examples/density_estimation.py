#!/usr/bin/env python3
"""
Density Estimation with Normalizing Flows

This example demonstrates how to:
1. Fetch cryptocurrency data from Bybit
2. Train a normalizing flow on return data
3. Compare learned density vs Gaussian assumption
4. Visualize the results

Usage:
    python density_estimation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

from flows import RealNVP, create_flow
from data_fetcher import BybitDataFetcher, create_synthetic_returns
from training import FlowTrainer


def main():
    print("=" * 60)
    print("Normalizing Flows for Financial Density Estimation")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Step 1: Fetch or create data
    # -----------------------------------------------------------------
    print("\n[Step 1] Preparing data...")

    try:
        # Try to fetch real data from Bybit
        fetcher = BybitDataFetcher()
        returns = fetcher.prepare_training_data(
            symbol="BTC/USDT",
            timeframe="1d",
            limit=500,
            standardize=False,
            winsorize=0.01
        )
        print(f"  Fetched {len(returns)} days of BTC/USDT returns")
        data_source = "Bybit BTC/USDT"

    except Exception as e:
        print(f"  Could not fetch live data: {e}")
        print("  Using synthetic data with fat tails...")
        returns = create_synthetic_returns(
            n_samples=500,
            distribution="fat_tails"
        )
        data_source = "Synthetic (fat tails)"

    # Print statistics
    returns_flat = returns.flatten()
    print(f"\n  Data source: {data_source}")
    print(f"  Sample size: {len(returns_flat)}")
    print(f"  Mean: {returns_flat.mean():.4f}")
    print(f"  Std: {returns_flat.std():.4f}")
    print(f"  Skewness: {stats.skew(returns_flat):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(returns_flat):.4f}")

    # -----------------------------------------------------------------
    # Step 2: Create and train normalizing flow
    # -----------------------------------------------------------------
    print("\n[Step 2] Training normalizing flow...")

    # Create model
    model = create_flow(
        flow_type="realnvp",
        dim=1,
        n_layers=8,
        hidden_dim=128,
        use_actnorm=True
    )

    # Train
    trainer = FlowTrainer(model)
    history = trainer.train(
        train_data=returns,
        epochs=200,
        batch_size=64,
        early_stopping_patience=20,
        verbose=True
    )

    print(f"\n  Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final validation loss: {history['val_loss'][-1]:.4f}")

    # -----------------------------------------------------------------
    # Step 3: Generate samples and compare distributions
    # -----------------------------------------------------------------
    print("\n[Step 3] Generating samples from learned distribution...")

    model.eval()
    with torch.no_grad():
        flow_samples = model.sample(10000).cpu().numpy().flatten()

    # Fit Gaussian to original data
    mu, sigma = returns_flat.mean(), returns_flat.std()
    gaussian_samples = np.random.normal(mu, sigma, 10000)

    # -----------------------------------------------------------------
    # Step 4: Compute densities and compare
    # -----------------------------------------------------------------
    print("\n[Step 4] Computing density comparison...")

    # Create evaluation points
    x_eval = np.linspace(
        min(returns_flat.min(), -0.15),
        max(returns_flat.max(), 0.15),
        200
    ).reshape(-1, 1).astype(np.float32)

    # Flow density
    with torch.no_grad():
        flow_log_probs = model.log_prob(torch.FloatTensor(x_eval)).cpu().numpy()
    flow_density = np.exp(flow_log_probs)

    # Gaussian density
    gaussian_density = stats.norm.pdf(x_eval.flatten(), mu, sigma)

    # -----------------------------------------------------------------
    # Step 5: Compare tail probabilities
    # -----------------------------------------------------------------
    print("\n[Step 5] Comparing tail probabilities...")

    thresholds = [-0.05, -0.10, -0.15]

    print("\n  Threshold | Historical | Gaussian | Flow")
    print("  " + "-" * 50)

    for threshold in thresholds:
        hist_prob = (returns_flat < threshold).mean()
        gauss_prob = stats.norm.cdf(threshold, mu, sigma)
        flow_prob = (flow_samples < threshold).mean()

        print(f"    {threshold:+.0%}    |   {hist_prob:.4f}   | {gauss_prob:.4f}  | {flow_prob:.4f}")

    # -----------------------------------------------------------------
    # Step 6: Visualization
    # -----------------------------------------------------------------
    print("\n[Step 6] Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Histogram comparison
    ax1 = axes[0, 0]
    ax1.hist(returns_flat, bins=50, density=True, alpha=0.5, label='Historical')
    ax1.hist(flow_samples, bins=50, density=True, alpha=0.5, label='Flow samples')
    ax1.hist(gaussian_samples, bins=50, density=True, alpha=0.3, label='Gaussian')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend()

    # Plot 2: Density curves
    ax2 = axes[0, 1]
    ax2.plot(x_eval.flatten(), flow_density, label='Flow density', linewidth=2)
    ax2.plot(x_eval.flatten(), gaussian_density, label='Gaussian density', linewidth=2)
    ax2.hist(returns_flat, bins=50, density=True, alpha=0.3, label='Historical')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Density')
    ax2.set_title('Learned Density vs Gaussian')
    ax2.legend()

    # Plot 3: QQ plot (Flow vs Historical)
    ax3 = axes[1, 0]
    sorted_hist = np.sort(returns_flat)
    sorted_flow = np.sort(flow_samples[:len(returns_flat)])
    ax3.scatter(sorted_hist, sorted_flow, alpha=0.5, s=5)
    ax3.plot([sorted_hist.min(), sorted_hist.max()],
             [sorted_hist.min(), sorted_hist.max()], 'r--', label='Perfect fit')
    ax3.set_xlabel('Historical quantiles')
    ax3.set_ylabel('Flow quantiles')
    ax3.set_title('QQ Plot: Flow vs Historical')
    ax3.legend()

    # Plot 4: Training loss
    ax4 = axes[1, 1]
    ax4.plot(history['train_loss'], label='Train loss')
    ax4.plot(history['val_loss'], label='Validation loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Negative Log-Likelihood')
    ax4.set_title('Training Progress')
    ax4.legend()

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'density_estimation_results.png')
    plt.savefig(output_path, dpi=150)
    print(f"\n  Saved visualization to: {output_path}")

    plt.show()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The normalizing flow learns a more accurate density than the
Gaussian assumption, particularly in the tails.

Key observations:
1. Flow captures fat tails (higher probability of extreme events)
2. Flow can capture skewness in the distribution
3. QQ plot shows how well flow matches historical data

This has important implications for risk management:
- Gaussian VaR underestimates tail risk
- Flow-based VaR provides more accurate risk estimates
""")


if __name__ == "__main__":
    main()
