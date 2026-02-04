"""
Risk Metrics for Normalizing Flows

This module provides financial risk metric calculations using
normalizing flow models:
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR / Expected Shortfall)
- Tail probabilities
- Distribution comparisons
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Union
from scipy import stats


def compute_var(
    flow,
    alpha_levels: List[float] = [0.01, 0.05, 0.10],
    n_samples: int = 100000,
    return_samples: bool = False
) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray]]:
    """
    Compute Value at Risk at multiple confidence levels using flow samples.

    VaR_alpha is the threshold such that P(X < VaR_alpha) = alpha.

    Args:
        flow: Trained normalizing flow model
        alpha_levels: List of alpha values (e.g., 0.05 for 95% VaR)
        n_samples: Number of Monte Carlo samples
        return_samples: Whether to also return the samples

    Returns:
        Dictionary mapping VaR names to values
        Optionally also returns the generated samples
    """
    flow.eval()
    with torch.no_grad():
        samples = flow.sample(n_samples).cpu().numpy()

    # Handle multi-dimensional case (sum for portfolio)
    if samples.ndim > 1 and samples.shape[1] > 1:
        samples = samples.sum(axis=1)
    else:
        samples = samples.flatten()

    var_results = {}
    for alpha in alpha_levels:
        var = np.percentile(samples, alpha * 100)
        confidence = int((1 - alpha) * 100)
        var_results[f'VaR_{confidence}'] = var

    if return_samples:
        return var_results, samples
    return var_results


def compute_cvar(
    flow,
    alpha: float = 0.05,
    n_samples: int = 100000
) -> Tuple[float, float]:
    """
    Compute Conditional Value at Risk (Expected Shortfall).

    CVaR_alpha = E[X | X <= VaR_alpha]

    Args:
        flow: Trained normalizing flow model
        alpha: Confidence level (e.g., 0.05 for 95% CVaR)
        n_samples: Number of Monte Carlo samples

    Returns:
        var: Value at Risk threshold
        cvar: Conditional VaR (expected shortfall)
    """
    flow.eval()
    with torch.no_grad():
        samples = flow.sample(n_samples).cpu().numpy()

    if samples.ndim > 1 and samples.shape[1] > 1:
        samples = samples.sum(axis=1)
    else:
        samples = samples.flatten()

    var = np.percentile(samples, alpha * 100)
    cvar = samples[samples <= var].mean()

    return var, cvar


def compute_tail_probability(
    flow,
    threshold: float,
    n_samples: int = 100000,
    tail: str = "left"
) -> float:
    """
    Compute probability of returns beyond a threshold.

    Args:
        flow: Trained normalizing flow model
        threshold: Return threshold
        n_samples: Number of Monte Carlo samples
        tail: "left" for P(X < threshold), "right" for P(X > threshold)

    Returns:
        Tail probability
    """
    flow.eval()
    with torch.no_grad():
        samples = flow.sample(n_samples).cpu().numpy().flatten()

    if tail == "left":
        return (samples < threshold).mean()
    else:
        return (samples > threshold).mean()


def compute_risk_metrics(
    flow,
    n_samples: int = 100000
) -> Dict[str, float]:
    """
    Compute comprehensive risk metrics from a flow model.

    Args:
        flow: Trained normalizing flow model
        n_samples: Number of Monte Carlo samples

    Returns:
        Dictionary of risk metrics
    """
    flow.eval()
    with torch.no_grad():
        samples = flow.sample(n_samples).cpu().numpy().flatten()

    metrics = {}

    # Basic statistics
    metrics['mean'] = samples.mean()
    metrics['std'] = samples.std()
    metrics['skewness'] = stats.skew(samples)
    metrics['kurtosis'] = stats.kurtosis(samples)

    # VaR at multiple levels
    for alpha in [0.01, 0.05, 0.10]:
        var = np.percentile(samples, alpha * 100)
        cvar = samples[samples <= var].mean()
        confidence = int((1 - alpha) * 100)
        metrics[f'VaR_{confidence}'] = var
        metrics[f'CVaR_{confidence}'] = cvar

    # Tail probabilities for common thresholds
    for threshold in [-0.05, -0.10, -0.15, -0.20]:
        prob = (samples < threshold).mean()
        metrics[f'P(X<{int(threshold*100)}%)'] = prob

    # Upside metrics
    metrics['P(X>0)'] = (samples > 0).mean()
    metrics['mean_if_positive'] = samples[samples > 0].mean() if (samples > 0).any() else 0
    metrics['mean_if_negative'] = samples[samples < 0].mean() if (samples < 0).any() else 0

    return metrics


def backtest_var(
    returns: np.ndarray,
    var_predictions: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Backtest VaR predictions.

    Args:
        returns: Realized returns
        var_predictions: Predicted VaR values (same length as returns)
        alpha: VaR confidence level

    Returns:
        Backtesting statistics
    """
    n = len(returns)
    violations = returns < var_predictions
    n_violations = violations.sum()

    # Expected violations
    expected_violations = alpha * n

    # Kupiec test (proportion of failures test)
    if n_violations > 0 and n_violations < n:
        lr_pof = -2 * (
            n_violations * np.log(alpha) +
            (n - n_violations) * np.log(1 - alpha) -
            n_violations * np.log(n_violations / n) -
            (n - n_violations) * np.log(1 - n_violations / n)
        )
        kupiec_pvalue = 1 - stats.chi2.cdf(lr_pof, 1)
    else:
        lr_pof = np.nan
        kupiec_pvalue = np.nan

    # Christoffersen test (independence)
    # Count transitions: 00, 01, 10, 11
    t00 = ((~violations[:-1]) & (~violations[1:])).sum()
    t01 = ((~violations[:-1]) & (violations[1:])).sum()
    t10 = ((violations[:-1]) & (~violations[1:])).sum()
    t11 = ((violations[:-1]) & (violations[1:])).sum()

    if t01 + t11 > 0 and t00 + t10 > 0:
        pi01 = t01 / (t00 + t01) if (t00 + t01) > 0 else 0
        pi11 = t11 / (t10 + t11) if (t10 + t11) > 0 else 0
        pi = (t01 + t11) / (n - 1)

        if 0 < pi01 < 1 and 0 < pi11 < 1 and 0 < pi < 1:
            lr_ind = -2 * (
                np.log(1 - pi) * (t00 + t10) +
                np.log(pi) * (t01 + t11) -
                np.log(1 - pi01) * t00 - np.log(pi01) * t01 -
                np.log(1 - pi11) * t10 - np.log(pi11) * t11
            )
            christoffersen_pvalue = 1 - stats.chi2.cdf(lr_ind, 1)
        else:
            lr_ind = np.nan
            christoffersen_pvalue = np.nan
    else:
        lr_ind = np.nan
        christoffersen_pvalue = np.nan

    return {
        'n_observations': n,
        'n_violations': n_violations,
        'expected_violations': expected_violations,
        'violation_rate': n_violations / n,
        'expected_rate': alpha,
        'kupiec_statistic': lr_pof,
        'kupiec_pvalue': kupiec_pvalue,
        'christoffersen_statistic': lr_ind,
        'christoffersen_pvalue': christoffersen_pvalue
    }


def compare_distributions(
    samples1: np.ndarray,
    samples2: np.ndarray
) -> Dict[str, float]:
    """
    Compare two distributions (e.g., flow samples vs historical data).

    Args:
        samples1: First sample array
        samples2: Second sample array

    Returns:
        Dictionary of comparison metrics
    """
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(samples1, samples2)

    # Wasserstein distance (Earth Mover's Distance)
    wasserstein = stats.wasserstein_distance(samples1, samples2)

    # Jensen-Shannon divergence (approximate via histograms)
    bins = np.linspace(
        min(samples1.min(), samples2.min()),
        max(samples1.max(), samples2.max()),
        100
    )
    hist1, _ = np.histogram(samples1, bins=bins, density=True)
    hist2, _ = np.histogram(samples2, bins=bins, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist1 = hist1 + eps
    hist2 = hist2 + eps
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()

    m = (hist1 + hist2) / 2
    js_divergence = 0.5 * (
        stats.entropy(hist1, m) + stats.entropy(hist2, m)
    )

    # Moment comparisons
    moment_diffs = {
        'mean_diff': samples1.mean() - samples2.mean(),
        'std_diff': samples1.std() - samples2.std(),
        'skew_diff': stats.skew(samples1) - stats.skew(samples2),
        'kurtosis_diff': stats.kurtosis(samples1) - stats.kurtosis(samples2)
    }

    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'wasserstein_distance': wasserstein,
        'js_divergence': js_divergence,
        **moment_diffs
    }


def compute_portfolio_var(
    flow,
    weights: np.ndarray,
    alpha: float = 0.05,
    n_samples: int = 100000
) -> Tuple[float, float]:
    """
    Compute VaR and CVaR for a portfolio using multivariate flow.

    Args:
        flow: Trained multivariate normalizing flow
        weights: Portfolio weights
        alpha: Confidence level
        n_samples: Number of Monte Carlo samples

    Returns:
        var: Portfolio VaR
        cvar: Portfolio CVaR
    """
    flow.eval()
    with torch.no_grad():
        # Sample asset returns
        asset_returns = flow.sample(n_samples).cpu().numpy()

    # Compute portfolio returns
    portfolio_returns = asset_returns @ weights

    # VaR and CVaR
    var = np.percentile(portfolio_returns, alpha * 100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()

    return var, cvar


def stress_test(
    flow,
    scenario_type: str = "2x_vol",
    n_samples: int = 100000
) -> Dict[str, float]:
    """
    Perform stress testing using modified flow samples.

    Args:
        flow: Trained normalizing flow
        scenario_type: Type of stress scenario
        n_samples: Number of samples

    Returns:
        Stress test results
    """
    flow.eval()
    with torch.no_grad():
        base_samples = flow.sample(n_samples).cpu().numpy().flatten()

    if scenario_type == "2x_vol":
        # Double the volatility
        mean = base_samples.mean()
        stressed_samples = (base_samples - mean) * 2 + mean

    elif scenario_type == "fat_tails":
        # Apply tail transformation (increase extreme values)
        percentile_5 = np.percentile(base_samples, 5)
        percentile_95 = np.percentile(base_samples, 95)

        stressed_samples = base_samples.copy()
        # Amplify tail events
        tail_mask_left = base_samples < percentile_5
        tail_mask_right = base_samples > percentile_95

        stressed_samples[tail_mask_left] *= 1.5
        stressed_samples[tail_mask_right] *= 1.5

    elif scenario_type == "crash":
        # Shift distribution left (simulate crash)
        stressed_samples = base_samples - 0.05  # 5% shift down

    else:
        stressed_samples = base_samples

    # Compute metrics for stressed scenario
    metrics = {
        'scenario': scenario_type,
        'mean_base': base_samples.mean(),
        'mean_stressed': stressed_samples.mean(),
        'std_base': base_samples.std(),
        'std_stressed': stressed_samples.std(),
        'VaR_95_base': np.percentile(base_samples, 5),
        'VaR_95_stressed': np.percentile(stressed_samples, 5),
        'VaR_99_base': np.percentile(base_samples, 1),
        'VaR_99_stressed': np.percentile(stressed_samples, 1),
        'max_loss_base': base_samples.min(),
        'max_loss_stressed': stressed_samples.min()
    }

    return metrics


def marginal_contribution_to_var(
    flow,
    weights: np.ndarray,
    alpha: float = 0.05,
    n_samples: int = 100000,
    delta: float = 0.01
) -> np.ndarray:
    """
    Compute marginal contribution of each asset to portfolio VaR.

    Uses numerical differentiation.

    Args:
        flow: Trained multivariate flow
        weights: Portfolio weights
        alpha: VaR confidence level
        n_samples: Number of samples
        delta: Perturbation for numerical differentiation

    Returns:
        Array of marginal VaR contributions
    """
    n_assets = len(weights)
    base_var, _ = compute_portfolio_var(flow, weights, alpha, n_samples)

    marginal_vars = np.zeros(n_assets)

    for i in range(n_assets):
        # Perturb weight i
        perturbed_weights = weights.copy()
        perturbed_weights[i] += delta

        # Renormalize
        perturbed_weights = perturbed_weights / perturbed_weights.sum()

        perturbed_var, _ = compute_portfolio_var(
            flow, perturbed_weights, alpha, n_samples
        )

        marginal_vars[i] = (perturbed_var - base_var) / delta

    return marginal_vars
