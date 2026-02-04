#!/usr/bin/env python3
"""
Synthetic Data Generation with Normalizing Flows

This example demonstrates how to:
1. Train a flow on historical return data
2. Generate realistic synthetic scenarios
3. Use synthetic data for stress testing
4. Monte Carlo simulation for portfolio risk

Usage:
    python synthetic_generation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

from flows import RealNVP
from data_fetcher import create_synthetic_returns, BybitDataFetcher
from training import FlowTrainer
from risk_metrics import stress_test, compute_var, compute_cvar


def main():
    print("=" * 60)
    print("Synthetic Data Generation with Normalizing Flows")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Step 1: Prepare training data
    # -----------------------------------------------------------------
    print("\n[Step 1] Preparing training data...")

    try:
        fetcher = BybitDataFetcher()
        returns = fetcher.prepare_training_data(
            symbol="BTC/USDT",
            timeframe="1d",
            limit=500,
            standardize=False
        )
        data_source = "BTC/USDT"
    except Exception:
        print("  Using synthetic fat-tailed data...")
        returns = create_synthetic_returns(500, "fat_tails")
        data_source = "Synthetic"

    returns_flat = returns.flatten()
    print(f"  Data source: {data_source}")
    print(f"  Samples: {len(returns_flat)}")

    # -----------------------------------------------------------------
    # Step 2: Train normalizing flow
    # -----------------------------------------------------------------
    print("\n[Step 2] Training normalizing flow...")

    model = RealNVP(dim=1, n_layers=8, hidden_dim=128)
    trainer = FlowTrainer(model)

    trainer.train(
        train_data=returns,
        epochs=150,
        batch_size=64,
        verbose=True
    )

    # -----------------------------------------------------------------
    # Step 3: Generate synthetic scenarios
    # -----------------------------------------------------------------
    print("\n[Step 3] Generating synthetic return scenarios...")

    model.eval()
    with torch.no_grad():
        # Generate different amounts of synthetic data
        synthetic_1k = model.sample(1000).cpu().numpy().flatten()
        synthetic_10k = model.sample(10000).cpu().numpy().flatten()
        synthetic_100k = model.sample(100000).cpu().numpy().flatten()

    print(f"\n  Generated scenarios:")
    print(f"    1,000 samples   - Min: {synthetic_1k.min():.4f}, Max: {synthetic_1k.max():.4f}")
    print(f"    10,000 samples  - Min: {synthetic_10k.min():.4f}, Max: {synthetic_10k.max():.4f}")
    print(f"    100,000 samples - Min: {synthetic_100k.min():.4f}, Max: {synthetic_100k.max():.4f}")

    # Compare statistics
    print("\n  Statistical comparison:")
    print("  Metric      | Historical | Synthetic (100k)")
    print("  " + "-" * 45)
    print(f"  Mean        | {returns_flat.mean():+.4f}     | {synthetic_100k.mean():+.4f}")
    print(f"  Std         | {returns_flat.std():.4f}     | {synthetic_100k.std():.4f}")
    print(f"  Skewness    | {stats.skew(returns_flat):+.4f}     | {stats.skew(synthetic_100k):+.4f}")
    print(f"  Kurtosis    | {stats.kurtosis(returns_flat):+.4f}     | {stats.kurtosis(synthetic_100k):+.4f}")
    print(f"  1% quantile | {np.percentile(returns_flat, 1):+.4f}     | {np.percentile(synthetic_100k, 1):+.4f}")
    print(f"  99% quantile| {np.percentile(returns_flat, 99):+.4f}     | {np.percentile(synthetic_100k, 99):+.4f}")

    # -----------------------------------------------------------------
    # Step 4: Stress testing with synthetic data
    # -----------------------------------------------------------------
    print("\n[Step 4] Stress testing...")

    stress_scenarios = ["2x_vol", "fat_tails", "crash"]

    for scenario in stress_scenarios:
        results = stress_test(model, scenario_type=scenario, n_samples=100000)
        print(f"\n  Scenario: {scenario}")
        print(f"    Base VaR 95%:     {results['VaR_95_base']:+.4f}")
        print(f"    Stressed VaR 95%: {results['VaR_95_stressed']:+.4f}")
        print(f"    Base max loss:    {results['max_loss_base']:+.4f}")
        print(f"    Stressed max loss:{results['max_loss_stressed']:+.4f}")

    # -----------------------------------------------------------------
    # Step 5: Monte Carlo simulation for trading strategy
    # -----------------------------------------------------------------
    print("\n[Step 5] Monte Carlo simulation for trading strategy...")

    # Simple strategy: always long with 100% position
    initial_capital = 10000
    position_size = 1.0  # 100% invested
    n_simulations = 1000
    n_days = 30  # 30-day simulation

    # Generate return paths
    print(f"\n  Simulating {n_simulations} paths of {n_days} days each...")

    with torch.no_grad():
        all_returns = model.sample(n_simulations * n_days).cpu().numpy().flatten()

    return_matrix = all_returns.reshape(n_simulations, n_days)

    # Calculate portfolio values
    cumulative_returns = np.exp(np.cumsum(return_matrix, axis=1))
    final_values = initial_capital * cumulative_returns[:, -1]

    # Statistics
    mean_final = final_values.mean()
    median_final = np.median(final_values)
    worst_case = final_values.min()
    best_case = final_values.max()
    var_95 = np.percentile(final_values, 5)
    var_99 = np.percentile(final_values, 1)

    print(f"\n  Monte Carlo Results (30-day horizon):")
    print(f"    Initial capital:    ${initial_capital:,.0f}")
    print(f"    Mean final value:   ${mean_final:,.0f}")
    print(f"    Median final value: ${median_final:,.0f}")
    print(f"    Best case:          ${best_case:,.0f}")
    print(f"    Worst case:         ${worst_case:,.0f}")
    print(f"    5th percentile:     ${var_95:,.0f}")
    print(f"    1st percentile:     ${var_99:,.0f}")

    # Probability calculations
    prob_loss = (final_values < initial_capital).mean()
    prob_loss_10pct = (final_values < initial_capital * 0.9).mean()
    prob_gain_20pct = (final_values > initial_capital * 1.2).mean()

    print(f"\n  Probability estimates:")
    print(f"    P(loss):          {prob_loss*100:.1f}%")
    print(f"    P(loss > 10%):    {prob_loss_10pct*100:.1f}%")
    print(f"    P(gain > 20%):    {prob_gain_20pct*100:.1f}%")

    # -----------------------------------------------------------------
    # Step 6: Generate extreme scenarios
    # -----------------------------------------------------------------
    print("\n[Step 6] Analyzing extreme scenarios from synthetic data...")

    # Find extreme events in synthetic data
    threshold_negative = np.percentile(synthetic_100k, 0.1)  # 0.1% worst
    threshold_positive = np.percentile(synthetic_100k, 99.9)  # 0.1% best

    extreme_negative = synthetic_100k[synthetic_100k < threshold_negative]
    extreme_positive = synthetic_100k[synthetic_100k > threshold_positive]

    print(f"\n  Extreme negative events (worst 0.1%):")
    print(f"    Count: {len(extreme_negative)}")
    print(f"    Average: {extreme_negative.mean():.4f}")
    print(f"    Worst: {extreme_negative.min():.4f}")

    print(f"\n  Extreme positive events (best 0.1%):")
    print(f"    Count: {len(extreme_positive)}")
    print(f"    Average: {extreme_positive.mean():.4f}")
    print(f"    Best: {extreme_positive.max():.4f}")

    # -----------------------------------------------------------------
    # Step 7: Visualization
    # -----------------------------------------------------------------
    print("\n[Step 7] Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Histogram comparison
    ax1 = axes[0, 0]
    ax1.hist(returns_flat, bins=50, density=True, alpha=0.6, label='Historical')
    ax1.hist(synthetic_100k, bins=100, density=True, alpha=0.4, label='Synthetic (100k)')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend()

    # Plot 2: Monte Carlo paths
    ax2 = axes[0, 1]
    for i in range(min(100, n_simulations)):
        ax2.plot(cumulative_returns[i], alpha=0.1, color='blue')
    ax2.axhline(y=1.0, color='red', linestyle='--', label='Initial')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Cumulative Return')
    ax2.set_title(f'Monte Carlo Simulation ({n_simulations} paths)')
    ax2.legend()

    # Plot 3: Final value distribution
    ax3 = axes[1, 0]
    ax3.hist(final_values, bins=50, density=True, alpha=0.7)
    ax3.axvline(x=initial_capital, color='red', linestyle='--', label='Initial capital')
    ax3.axvline(x=var_95, color='orange', linestyle='--', label='5th percentile')
    ax3.set_xlabel('Final Portfolio Value ($)')
    ax3.set_ylabel('Density')
    ax3.set_title('Final Value Distribution')
    ax3.legend()

    # Plot 4: QQ plot
    ax4 = axes[1, 1]
    percentiles = np.linspace(0.1, 99.9, 100)
    hist_quantiles = np.percentile(returns_flat, percentiles)
    synth_quantiles = np.percentile(synthetic_100k, percentiles)
    ax4.scatter(hist_quantiles, synth_quantiles, alpha=0.6)
    ax4.plot([hist_quantiles.min(), hist_quantiles.max()],
             [hist_quantiles.min(), hist_quantiles.max()], 'r--')
    ax4.set_xlabel('Historical Quantiles')
    ax4.set_ylabel('Synthetic Quantiles')
    ax4.set_title('QQ Plot: Synthetic vs Historical')

    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), 'synthetic_generation_results.png')
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
Key capabilities demonstrated:

1. Synthetic Data Generation:
   - Generate unlimited realistic return scenarios
   - Match historical distribution characteristics
   - Preserve fat tails and skewness

2. Stress Testing:
   - Test portfolio under extreme conditions
   - Double volatility scenarios
   - Market crash simulations

3. Monte Carlo Simulation:
   - Simulate portfolio paths over time
   - Compute probability of various outcomes
   - Estimate VaR and CVaR for longer horizons

4. Practical Applications:
   - Backtest strategies on more data than available
   - Regulatory stress testing (Basel III)
   - Risk budgeting and allocation
   - Options pricing with realistic distributions
""")


if __name__ == "__main__":
    main()
