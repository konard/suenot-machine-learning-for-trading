#!/usr/bin/env python3
"""Example usage of the Disentangled VAE Trading module (Chapter 238).

This script demonstrates the full pipeline:
  1. Generate synthetic financial data (no API key required).
  2. Build and train a Beta-VAE model.
  3. Visualise latent factor traversals.
  4. Run a simple backtest.
  5. Print a performance summary.

Run:
    python -m 238_disentangled_vae_trading.python.example_usage
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .data import create_synthetic_data, DisentangledVAEDataset
from .model import (
    BetaVAE,
    DisentangledVAEConfig,
    DisentangledVAEForecaster,
    beta_vae_loss,
)
from .strategy import DisentangledVAEStrategy, backtest_disentangled_strategy


def main() -> None:
    """Run the complete example pipeline."""
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Create synthetic data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Chapter 238 -- Disentangled VAE Trading")
    print("=" * 60)
    print("\n[1/5] Generating synthetic market data ...")

    n_samples = 2000
    n_features = 6
    df = create_synthetic_data(n_samples=n_samples, n_features=n_features, seed=42)
    print(f"  Data shape: {df.shape}")
    print(f"  Features: {list(df.columns)}")

    # ------------------------------------------------------------------
    # 2. Prepare dataset and dataloaders
    # ------------------------------------------------------------------
    print("\n[2/5] Preparing dataset ...")

    seq_len = 30
    prediction_horizon = 5
    config = DisentangledVAEConfig(
        input_features=n_features,
        latent_dim=8,
        hidden_dims=[16, 32, 64],
        beta=4.0,
        beta_schedule="warmup",
        disentanglement_method="beta_vae",
        seq_len=seq_len,
        prediction_horizon=prediction_horizon,
    )

    dataset = DisentangledVAEDataset(
        df, seq_len=seq_len, prediction_horizon=prediction_horizon
    )
    print(f"  Dataset size: {len(dataset)} windows")

    # Train / test split (80 / 20)
    split = int(0.8 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, range(split))
    test_set = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # ------------------------------------------------------------------
    # 3. Build and train the Beta-VAE
    # ------------------------------------------------------------------
    print("\n[3/5] Training Beta-VAE ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BetaVAE(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        model.train()
        beta = model.update_beta(epoch, n_epochs)
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            recon, mu, logvar, z = model(batch_x)
            loss, recon_loss, kl_loss = beta_vae_loss(recon, batch_x, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs}  loss={avg_loss:.4f}  beta={beta:.3f}")

    # ------------------------------------------------------------------
    # 4. Latent factor traversals
    # ------------------------------------------------------------------
    print("\n[4/5] Latent factor traversal ...")

    model.eval()
    sample_x, _ = dataset[0]
    sample_x = sample_x.unsqueeze(0).to(device)

    with torch.no_grad():
        mu, logvar = model.encode(sample_x)

    print(f"  Latent mean:    {mu.cpu().numpy().round(3).tolist()}")
    print(f"  Latent logvar:  {logvar.cpu().numpy().round(3).tolist()}")

    # Traverse each dimension and measure reconstruction change
    print("\n  Traversal sensitivity per latent dimension:")
    traversal_range = np.linspace(-3, 3, 7)
    for dim in range(config.latent_dim):
        diffs = []
        for val in traversal_range:
            z_mod = mu.clone()
            z_mod[0, dim] = val
            with torch.no_grad():
                recon = model.decode(z_mod)
            diff = (recon - model.decode(mu)).abs().mean().item()
            diffs.append(diff)
        sensitivity = np.mean(diffs)
        bar = "#" * int(sensitivity * 100)
        print(f"    z_{dim}: sensitivity={sensitivity:.4f}  {bar}")

    # ------------------------------------------------------------------
    # 5. Backtest
    # ------------------------------------------------------------------
    print("\n[5/5] Running backtest ...")

    # Prepare test arrays
    test_values = dataset.data.numpy()
    test_data = test_values[split:]

    # Approximate returns from the first feature (log-return proxy)
    returns = np.diff(test_data[:, 0])
    returns = np.concatenate([[0.0], returns])

    results = backtest_disentangled_strategy(
        model=model,
        config=config,
        test_data=test_data,
        returns=returns,
        initial_capital=100_000.0,
        transaction_cost=0.001,
        signal_threshold=0.2,
    )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Return:   {results['total_return']:>10.2%}")
    print(f"  Sharpe Ratio:   {results['sharpe_ratio']:>10.3f}")
    print(f"  Sortino Ratio:  {results['sortino_ratio']:>10.3f}")
    print(f"  Max Drawdown:   {results['max_drawdown']:>10.2%}")
    print(f"  Win Rate:       {results['win_rate']:>10.2%}")
    print(f"  Profit Factor:  {results['profit_factor']:>10.3f}")
    print(f"  Num Trades:     {results['n_trades']:>10d}")
    print(f"  Final Capital:  ${results['capital_history'][-1]:>12,.2f}")
    print("=" * 60)

    # Factor exposures
    strategy = DisentangledVAEStrategy(model=model, config=config)
    for i in range(min(50, len(test_data) - seq_len)):
        window = test_data[i : i + seq_len].T
        strategy.generate_signals(window)

    exposures = strategy.get_factor_exposures()
    if not exposures.empty:
        print("\nLatent Factor Exposures (last 50 windows):")
        print(exposures.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
