"""
CIR Model for Credit Risk
===========================

The CIR model is widely used for modeling default intensity (hazard rate)
in credit risk. The default intensity lambda(t) follows:

    d(lambda) = kappa*(theta - lambda)*dt + sigma*sqrt(lambda)*dW

Key applications:
- Survival probability: Q(t,T) = A(T-t) * exp(-B(T-t) * lambda_t)
- CDS spread pricing
- Defaultable bond pricing
- Credit option pricing

The survival probability PDE is IDENTICAL to the CIR bond pricing PDE,
so our PINN can be directly reused.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from .analytical import cir_bond_price, cir_A, cir_B, simulate_cir_exact


# =============================================================================
# Survival Probability
# =============================================================================

def survival_probability(
    lambda_t: float,
    t: float,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> float:
    """
    Compute survival probability Q(t, T) under CIR intensity.

    Q(t, T) = E[exp(-integral_t^T lambda(s) ds) | lambda(t)]
            = A(T-t) * exp(-B(T-t) * lambda_t)

    This is mathematically identical to CIR bond pricing.

    Parameters
    ----------
    lambda_t : float
        Current default intensity.
    t : float
        Current time.
    T : float
        Maturity time.
    kappa, theta, sigma : float
        CIR parameters for the intensity process.

    Returns
    -------
    float
        Survival probability.
    """
    return cir_bond_price(lambda_t, t, T, kappa, theta, sigma)


def default_probability(
    lambda_t: float,
    t: float,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> float:
    """
    Compute default probability P(tau <= T | lambda(t)).

    Parameters
    ----------
    lambda_t : float
        Current default intensity.
    t, T : float
        Current time and maturity.
    kappa, theta, sigma : float
        CIR intensity parameters.

    Returns
    -------
    float
        Probability of default before T.
    """
    return 1.0 - survival_probability(lambda_t, t, T, kappa, theta, sigma)


# =============================================================================
# CDS Pricing
# =============================================================================

def cds_spread(
    lambda_t: float,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
    recovery: float = 0.4,
    risk_free_rate: float = 0.03,
    n_payments: int = 4,
) -> float:
    """
    Compute the par CDS spread under CIR intensity dynamics.

    The CDS spread is determined by equating protection and premium legs:
        s = Protection Leg / Premium Leg

    Parameters
    ----------
    lambda_t : float
        Current default intensity.
    T : float
        CDS maturity (years).
    kappa, theta, sigma : float
        CIR parameters for intensity.
    recovery : float
        Recovery rate (e.g., 0.4 = 40%).
    risk_free_rate : float
        Risk-free interest rate.
    n_payments : int
        Number of premium payments per year.

    Returns
    -------
    float
        Annual CDS spread (e.g., 0.01 = 100 bps).
    """
    total_periods = int(T * n_payments)
    dt_payment = 1.0 / n_payments

    protection_leg = 0.0
    premium_leg = 0.0

    for i in range(1, total_periods + 1):
        t_i = i * dt_payment
        t_prev = (i - 1) * dt_payment

        # Survival probabilities
        Q_prev = survival_probability(lambda_t, 0.0, t_prev, kappa, theta, sigma)
        Q_curr = survival_probability(lambda_t, 0.0, t_i, kappa, theta, sigma)

        # Risk-free discount factor
        df = np.exp(-risk_free_rate * t_i)

        # Protection leg: (1 - R) * PV of expected default payments
        default_prob = Q_prev - Q_curr
        protection_leg += (1.0 - recovery) * default_prob * df

        # Premium leg: PV of expected premium payments (weighted by survival)
        premium_leg += Q_curr * dt_payment * df

    if premium_leg <= 0:
        return 0.0

    return protection_leg / premium_leg


def cds_spread_term_structure(
    lambda_t: float,
    maturities: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    recovery: float = 0.4,
    risk_free_rate: float = 0.03,
) -> np.ndarray:
    """
    Compute CDS spread term structure for multiple maturities.

    Parameters
    ----------
    lambda_t : float
        Current default intensity.
    maturities : ndarray
        Array of CDS maturities (years).
    kappa, theta, sigma : float
        CIR intensity parameters.
    recovery : float
        Recovery rate.
    risk_free_rate : float
        Risk-free rate.

    Returns
    -------
    ndarray
        CDS spreads for each maturity (in bps).
    """
    spreads = np.array([
        cds_spread(lambda_t, T, kappa, theta, sigma, recovery, risk_free_rate)
        for T in maturities
    ])
    return spreads * 10000  # convert to basis points


# =============================================================================
# Defaultable Bond Pricing
# =============================================================================

def defaultable_bond_price(
    lambda_t: float,
    r_t: float,
    T: float,
    intensity_params: Tuple[float, float, float],
    rate_params: Tuple[float, float, float],
    recovery: float = 0.4,
    correlation: float = 0.0,
) -> float:
    """
    Price a defaultable zero-coupon bond.

    Under independence of rates and default:
        P_d(t,T) = recovery * (1 - Q(t,T)) * P_rf(t,T) + Q(t,T) * P_rf(t,T)

    where P_rf is the risk-free bond price and Q is the survival probability.

    Parameters
    ----------
    lambda_t : float
        Current default intensity.
    r_t : float
        Current risk-free rate.
    T : float
        Bond maturity.
    intensity_params : tuple
        (kappa_lambda, theta_lambda, sigma_lambda) for default intensity.
    rate_params : tuple
        (kappa_r, theta_r, sigma_r) for risk-free rate.
    recovery : float
        Recovery rate.
    correlation : float
        Correlation between rate and intensity (ignored in simple model).

    Returns
    -------
    float
        Defaultable bond price.
    """
    kappa_l, theta_l, sigma_l = intensity_params
    kappa_r, theta_r, sigma_r = rate_params

    # Risk-free bond price
    P_rf = cir_bond_price(r_t, 0.0, T, kappa_r, theta_r, sigma_r)

    # Survival probability
    Q = survival_probability(lambda_t, 0.0, T, kappa_l, theta_l, sigma_l)

    # Defaultable bond price (assuming independence)
    P_default = Q * P_rf + recovery * (1.0 - Q) * P_rf

    return P_default


def credit_spread(
    lambda_t: float,
    r_t: float,
    T: float,
    intensity_params: Tuple[float, float, float],
    rate_params: Tuple[float, float, float],
    recovery: float = 0.4,
) -> float:
    """
    Compute credit spread from defaultable bond price.

    credit_spread = yield_defaultable - yield_riskfree

    Parameters
    ----------
    lambda_t, r_t, T : float
        Default intensity, risk-free rate, maturity.
    intensity_params : tuple
        CIR params for intensity.
    rate_params : tuple
        CIR params for risk-free rate.
    recovery : float
        Recovery rate.

    Returns
    -------
    float
        Credit spread (annualized).
    """
    kappa_r, theta_r, sigma_r = rate_params

    P_rf = cir_bond_price(r_t, 0.0, T, kappa_r, theta_r, sigma_r)
    P_def = defaultable_bond_price(
        lambda_t, r_t, T, intensity_params, rate_params, recovery
    )

    if T <= 0 or P_rf <= 0 or P_def <= 0:
        return 0.0

    yield_rf = -np.log(P_rf) / T
    yield_def = -np.log(P_def) / T

    return yield_def - yield_rf


# =============================================================================
# Credit Risk Simulation
# =============================================================================

def simulate_default_time(
    lambda_0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int = 1000,
    n_paths: int = 10000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate default times under CIR intensity.

    Default occurs at the first time the cumulative intensity exceeds
    an independent exponential(1) random variable.

    Parameters
    ----------
    lambda_0 : float
        Initial default intensity.
    kappa, theta, sigma : float
        CIR parameters.
    T : float
        Maximum time horizon.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    seed : int, optional
        Random seed.

    Returns
    -------
    ndarray of shape (n_paths,)
        Simulated default times. Values > T mean no default.
    """
    if seed is not None:
        np.random.seed(seed)

    # Simulate CIR intensity paths
    paths = simulate_cir_exact(kappa, theta, sigma, lambda_0, T, n_steps, n_paths)

    dt = T / n_steps
    default_times = np.full(n_paths, T + 1.0)  # no default by default

    # Independent exponential triggers
    triggers = np.random.exponential(1.0, n_paths)

    # Cumulative intensity
    cum_intensity = np.cumsum(paths[:, :-1] * dt, axis=1)

    for j in range(n_paths):
        # Find first time cumulative intensity exceeds trigger
        exceeded = np.where(cum_intensity[j] >= triggers[j])[0]
        if len(exceeded) > 0:
            default_times[j] = (exceeded[0] + 1) * dt

    return default_times


def compute_loss_distribution(
    lambda_0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    exposure: float = 1.0,
    recovery: float = 0.4,
    n_simulations: int = 50000,
    seed: int = 42,
) -> Dict:
    """
    Compute loss distribution for a single-name credit exposure.

    Parameters
    ----------
    lambda_0 : float
        Initial default intensity.
    kappa, theta, sigma : float
        CIR intensity parameters.
    T : float
        Time horizon (years).
    exposure : float
        Exposure at default.
    recovery : float
        Recovery rate.
    n_simulations : int
        Number of Monte Carlo simulations.
    seed : int
        Random seed.

    Returns
    -------
    Dict
        Loss distribution statistics.
    """
    default_times = simulate_default_time(
        lambda_0, kappa, theta, sigma, T,
        n_steps=500, n_paths=n_simulations, seed=seed,
    )

    # Loss: (1 - recovery) * exposure if default, 0 otherwise
    defaulted = default_times <= T
    losses = defaulted.astype(float) * (1.0 - recovery) * exposure

    results = {
        "default_probability": np.mean(defaulted),
        "expected_loss": np.mean(losses),
        "unexpected_loss": np.std(losses),
        "loss_99": np.percentile(losses, 99),
        "loss_999": np.percentile(losses, 99.9),
        "max_loss": np.max(losses),
        "n_defaults": np.sum(defaulted),
        "mean_default_time": (
            np.mean(default_times[defaulted]) if np.any(defaulted) else np.nan
        ),
    }

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("CIR Credit Risk Demo")
    print("=" * 70)

    # Default intensity parameters
    kappa_l, theta_l, sigma_l = 0.3, 0.02, 0.08
    lambda_0 = 0.015

    # Risk-free rate parameters
    kappa_r, theta_r, sigma_r = 0.5, 0.04, 0.06
    r_0 = 0.035

    print(f"\nIntensity params: kappa={kappa_l}, theta={theta_l}, sigma={sigma_l}")
    print(f"Rate params: kappa={kappa_r}, theta={theta_r}, sigma={sigma_r}")
    print(f"lambda_0={lambda_0}, r_0={r_0}")

    # Survival probabilities
    print("\nSurvival Probabilities:")
    for T in [1, 2, 3, 5, 7, 10]:
        Q = survival_probability(lambda_0, 0.0, T, kappa_l, theta_l, sigma_l)
        PD = 1.0 - Q
        print(f"  T={T:2d}y: Q={Q:.6f}, PD={PD:.6f} ({PD*100:.2f}%)")

    # CDS spread term structure
    print("\nCDS Spread Term Structure:")
    maturities = np.array([1, 2, 3, 5, 7, 10], dtype=float)
    spreads = cds_spread_term_structure(
        lambda_0, maturities, kappa_l, theta_l, sigma_l,
        recovery=0.4, risk_free_rate=r_0,
    )
    for T, s in zip(maturities, spreads):
        print(f"  T={T:.0f}y: {s:.1f} bps")

    # Defaultable bond prices
    print("\nDefaultable Bond Prices:")
    for T in [1, 3, 5, 10]:
        P_rf = cir_bond_price(r_0, 0.0, T, kappa_r, theta_r, sigma_r)
        P_def = defaultable_bond_price(
            lambda_0, r_0, T,
            (kappa_l, theta_l, sigma_l),
            (kappa_r, theta_r, sigma_r),
        )
        cs = credit_spread(
            lambda_0, r_0, T,
            (kappa_l, theta_l, sigma_l),
            (kappa_r, theta_r, sigma_r),
        )
        print(f"  T={T:2d}y: P_rf={P_rf:.6f}, P_def={P_def:.6f}, "
              f"credit_spread={cs*10000:.1f} bps")

    # Loss distribution
    print("\nLoss Distribution (T=5y, 50K simulations):")
    loss_dist = compute_loss_distribution(
        lambda_0, kappa_l, theta_l, sigma_l,
        T=5.0, exposure=1.0, recovery=0.4, n_simulations=50000,
    )
    for key, val in loss_dist.items():
        print(f"  {key}: {val:.6f}")
