#!/usr/bin/env python3
"""
VaR and CVaR Calculation with Normalizing Flows

This example demonstrates how to:
1. Train a normalizing flow on return data
2. Compute VaR and CVaR using the learned distribution
3. Compare with parametric (Gaussian) and historical methods
4. Backtest VaR predictions

Usage:
    python var_calculation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats

from flows import RealNVP
from data_fetcher import create_synthetic_returns
from training import FlowTrainer
from risk_metrics import (
    compute_var,
    compute_cvar,
    compute_risk_metrics,
    backtest_var,
    compare_distributions
)


def gaussian_var(returns: np.ndarray, alpha: float) -> float:
    """Compute parametric Gaussian VaR."""
    mu = returns.mean()
    sigma = returns.std()
    return stats.norm.ppf(alpha, mu, sigma)


def historical_var(returns: np.ndarray, alpha: float) -> float:
    """Compute historical VaR."""
    return np.percentile(returns, alpha * 100)


def main():
    print("=" * 60)
    print("VaR and CVaR Calculation with Normalizing Flows")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Step 1: Create synthetic data with fat tails
    # -----------------------------------------------------------------
    print("\n[Step 1] Creating synthetic return data...")

    # Create data with fat tails (more realistic)
    np.random.seed(42)
    returns = create_synthetic_returns(
        n_samples=1000,
        distribution="fat_tails"
    )

    returns_flat = returns.flatten()
    print(f"  Sample size: {len(returns_flat)}")
    print(f"  Mean: {returns_flat.mean():.4f}")
    print(f"  Std: {returns_flat.std():.4f}")
    print(f"  Kurtosis: {stats.kurtosis(returns_flat):.2f} (Gaussian = 0)")

    # Split into train and test
    train_returns = returns[:800]
    test_returns = returns_flat[800:]

    # -----------------------------------------------------------------
    # Step 2: Train normalizing flow
    # -----------------------------------------------------------------
    print("\n[Step 2] Training normalizing flow...")

    model = RealNVP(
        dim=1,
        n_layers=8,
        hidden_dim=128
    )

    trainer = FlowTrainer(model)
    history = trainer.train(
        train_data=train_returns,
        epochs=200,
        batch_size=64,
        verbose=True
    )

    # -----------------------------------------------------------------
    # Step 3: Compute VaR using different methods
    # -----------------------------------------------------------------
    print("\n[Step 3] Computing VaR at different confidence levels...")

    alpha_levels = [0.01, 0.05, 0.10]

    print("\n  Method      |   VaR 99%  |   VaR 95%  |   VaR 90%")
    print("  " + "-" * 55)

    # Gaussian VaR
    gauss_vars = [gaussian_var(train_returns.flatten(), a) for a in alpha_levels]
    print(f"  Gaussian    | {gauss_vars[0]:+.4f}   | {gauss_vars[1]:+.4f}   | {gauss_vars[2]:+.4f}")

    # Historical VaR
    hist_vars = [historical_var(train_returns.flatten(), a) for a in alpha_levels]
    print(f"  Historical  | {hist_vars[0]:+.4f}   | {hist_vars[1]:+.4f}   | {hist_vars[2]:+.4f}")

    # Flow VaR
    flow_var_dict = compute_var(model, alpha_levels, n_samples=100000)
    print(f"  Flow        | {flow_var_dict['VaR_99']:+.4f}   | {flow_var_dict['VaR_95']:+.4f}   | {flow_var_dict['VaR_90']:+.4f}")

    # -----------------------------------------------------------------
    # Step 4: Compute CVaR (Expected Shortfall)
    # -----------------------------------------------------------------
    print("\n[Step 4] Computing CVaR (Expected Shortfall)...")

    print("\n  Method      |  CVaR 95%  |  CVaR 99%")
    print("  " + "-" * 40)

    # Gaussian CVaR (closed form)
    for alpha in [0.05, 0.01]:
        mu = train_returns.flatten().mean()
        sigma = train_returns.flatten().std()
        # E[X | X < VaR] for Gaussian
        gauss_cvar = mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
        print(f"  Gaussian    |  {gauss_cvar:+.4f} ({int((1-alpha)*100)}%)")

    # Historical CVaR
    for alpha in [0.05, 0.01]:
        hist_var = np.percentile(train_returns.flatten(), alpha * 100)
        hist_cvar = train_returns.flatten()[train_returns.flatten() <= hist_var].mean()
        print(f"  Historical  |  {hist_cvar:+.4f} ({int((1-alpha)*100)}%)")

    # Flow CVaR
    for alpha in [0.05, 0.01]:
        flow_var, flow_cvar = compute_cvar(model, alpha, n_samples=100000)
        print(f"  Flow        |  {flow_cvar:+.4f} ({int((1-alpha)*100)}%)")

    # -----------------------------------------------------------------
    # Step 5: Backtest VaR on test data
    # -----------------------------------------------------------------
    print("\n[Step 5] Backtesting VaR predictions...")

    # Generate VaR predictions for test period
    n_test = len(test_returns)
    flow_var_predictions = np.full(n_test, flow_var_dict['VaR_95'])
    gauss_var_predictions = np.full(n_test, gauss_vars[1])
    hist_var_predictions = np.full(n_test, hist_vars[1])

    # Backtest
    print("\n  VaR 95% Backtesting Results (expected violations: 5%)")
    print("  " + "-" * 50)

    for name, predictions in [
        ("Gaussian", gauss_var_predictions),
        ("Historical", hist_var_predictions),
        ("Flow", flow_var_predictions)
    ]:
        results = backtest_var(test_returns, predictions, alpha=0.05)
        print(f"\n  {name}:")
        print(f"    Violations: {results['n_violations']}/{results['n_observations']} "
              f"({results['violation_rate']*100:.1f}%)")
        print(f"    Kupiec test p-value: {results['kupiec_pvalue']:.4f}")

    # -----------------------------------------------------------------
    # Step 6: Comprehensive risk metrics
    # -----------------------------------------------------------------
    print("\n[Step 6] Comprehensive risk metrics from flow...")

    metrics = compute_risk_metrics(model, n_samples=100000)

    print("\n  Statistic           | Value")
    print("  " + "-" * 35)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:20} | {value:+.4f}")

    # -----------------------------------------------------------------
    # Step 7: Distribution comparison
    # -----------------------------------------------------------------
    print("\n[Step 7] Comparing flow samples to historical data...")

    model.eval()
    with torch.no_grad():
        flow_samples = model.sample(10000).cpu().numpy().flatten()

    comparison = compare_distributions(train_returns.flatten(), flow_samples)

    print("\n  Metric              | Value")
    print("  " + "-" * 35)
    for key, value in comparison.items():
        if isinstance(value, float):
            print(f"  {key:20} | {value:+.4f}")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key findings:

1. VaR Comparison:
   - Gaussian VaR typically UNDERESTIMATES risk for fat-tailed data
   - Flow-based VaR captures the true tail risk more accurately
   - Historical VaR depends heavily on sample size

2. CVaR (Expected Shortfall):
   - Shows average loss when VaR is breached
   - Flow CVaR accounts for the true tail shape
   - More informative for risk management

3. Backtesting:
   - Violation rate should match confidence level (5% for 95% VaR)
   - Kupiec test checks if violations are at expected rate
   - Flow-based VaR should have better calibration

4. Practical implications:
   - Use flow-based VaR for more accurate risk estimates
   - Especially important for cryptocurrencies with fat tails
   - Combine with CVaR for complete risk picture
""")


if __name__ == "__main__":
    main()
