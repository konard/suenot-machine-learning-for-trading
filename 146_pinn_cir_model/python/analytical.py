"""
Analytical CIR Model Solutions
===============================

Closed-form solutions for the CIR model:
- Bond pricing: P(r, t) = A(tau) * exp(-B(tau) * r)
- Yield curve: y(r, tau) = -ln(P) / tau
- Forward rate: f(r, t, T) = -d ln(P) / dT
- Transition density: non-central chi-squared
- Moments: conditional mean and variance of r(t)

Reference:
    Cox, Ingersoll, Ross (1985). "A Theory of the Term Structure of Interest Rates."
"""

import numpy as np
from scipy.stats import ncx2
from scipy.special import iv as bessel_iv
from typing import Tuple, Optional, Union

ArrayLike = Union[float, np.ndarray]


# =============================================================================
# CIR Bond Pricing (Analytical)
# =============================================================================

def _cir_gamma(kappa: float, sigma: float) -> float:
    """Compute gamma = sqrt(kappa^2 + 2*sigma^2)."""
    return np.sqrt(kappa ** 2 + 2.0 * sigma ** 2)


def cir_B(tau: ArrayLike, kappa: float, sigma: float) -> ArrayLike:
    """
    Compute B(tau) in the CIR bond price formula P = A*exp(-B*r).

    Parameters
    ----------
    tau : float or ndarray
        Time to maturity (T - t).
    kappa : float
        Mean-reversion speed.
    sigma : float
        Volatility coefficient.

    Returns
    -------
    float or ndarray
        B(tau) values.
    """
    gamma = _cir_gamma(kappa, sigma)
    exp_gamma_tau = np.exp(gamma * tau)
    numerator = 2.0 * (exp_gamma_tau - 1.0)
    denominator = (gamma + kappa) * (exp_gamma_tau - 1.0) + 2.0 * gamma
    return numerator / denominator


def cir_A(tau: ArrayLike, kappa: float, theta: float, sigma: float) -> ArrayLike:
    """
    Compute A(tau) in the CIR bond price formula P = A*exp(-B*r).

    Parameters
    ----------
    tau : float or ndarray
        Time to maturity (T - t).
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run mean rate.
    sigma : float
        Volatility coefficient.

    Returns
    -------
    float or ndarray
        A(tau) values.
    """
    gamma = _cir_gamma(kappa, sigma)
    exp_gamma_tau = np.exp(gamma * tau)
    exp_half = np.exp((kappa + gamma) * tau / 2.0)
    denominator = (gamma + kappa) * (exp_gamma_tau - 1.0) + 2.0 * gamma
    base = 2.0 * gamma * exp_half / denominator
    exponent = 2.0 * kappa * theta / (sigma ** 2)
    return np.power(base, exponent)


def cir_bond_price(
    r: ArrayLike,
    t: float,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> ArrayLike:
    """
    Analytical CIR zero-coupon bond price.

    P(r, t) = A(T-t) * exp(-B(T-t) * r)

    Parameters
    ----------
    r : float or ndarray
        Current short rate(s).
    t : float
        Current time.
    T : float
        Maturity time.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run mean rate.
    sigma : float
        Volatility coefficient.

    Returns
    -------
    float or ndarray
        Bond price(s).
    """
    tau = T - t
    if tau <= 0:
        return np.ones_like(r) if isinstance(r, np.ndarray) else 1.0

    A = cir_A(tau, kappa, theta, sigma)
    B = cir_B(tau, kappa, sigma)
    return A * np.exp(-B * r)


def cir_yield(
    r: ArrayLike,
    tau: ArrayLike,
    kappa: float,
    theta: float,
    sigma: float,
) -> ArrayLike:
    """
    CIR continuously compounded yield.

    y(r, tau) = -ln(P(r,t)) / tau = (B(tau)*r - ln(A(tau))) / tau

    Parameters
    ----------
    r : float or ndarray
        Current short rate(s).
    tau : float or ndarray
        Time to maturity.
    kappa, theta, sigma : float
        CIR parameters.

    Returns
    -------
    float or ndarray
        Yield(s).
    """
    tau = np.maximum(tau, 1e-10)  # avoid division by zero
    A = cir_A(tau, kappa, theta, sigma)
    B = cir_B(tau, kappa, sigma)
    return (B * r - np.log(A)) / tau


def cir_forward_rate(
    r: float,
    t: float,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
    dT: float = 1e-6,
) -> float:
    """
    CIR instantaneous forward rate f(r, t, T) = -d ln(P) / dT.

    Computed via finite difference on ln(P).

    Parameters
    ----------
    r : float
        Current short rate.
    t : float
        Current time.
    T : float
        Forward time.
    kappa, theta, sigma : float
        CIR parameters.
    dT : float
        Finite difference step.

    Returns
    -------
    float
        Forward rate.
    """
    P_minus = cir_bond_price(r, t, T - dT / 2, kappa, theta, sigma)
    P_plus = cir_bond_price(r, t, T + dT / 2, kappa, theta, sigma)
    return -(np.log(P_plus) - np.log(P_minus)) / dT


# =============================================================================
# CIR Transition Density
# =============================================================================

def cir_transition_density(
    r_next: ArrayLike,
    r_prev: float,
    dt: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> ArrayLike:
    """
    CIR transition density using the non-central chi-squared distribution.

    r(t+dt) | r(t) ~ (sigma^2*(1-exp(-kappa*dt))/(4*kappa)) * chi^2(df, nc)

    Parameters
    ----------
    r_next : float or ndarray
        Rate(s) at next time step.
    r_prev : float
        Rate at current time step.
    dt : float
        Time step size.
    kappa, theta, sigma : float
        CIR parameters.

    Returns
    -------
    float or ndarray
        Probability density at r_next.
    """
    c = 2.0 * kappa / (sigma ** 2 * (1.0 - np.exp(-kappa * dt)))
    df = 4.0 * kappa * theta / (sigma ** 2)
    nc = 2.0 * c * r_prev * np.exp(-kappa * dt)
    x = 2.0 * c * r_next

    # PDF of non-central chi-squared, with Jacobian for change of variable
    return 2.0 * c * ncx2.pdf(x, df=df, nc=nc)


def cir_transition_logpdf(
    r_next: ArrayLike,
    r_prev: float,
    dt: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> ArrayLike:
    """
    Log of CIR transition density.

    Parameters
    ----------
    r_next : float or ndarray
        Rate(s) at next time step.
    r_prev : float
        Rate at current time step.
    dt : float
        Time step size.
    kappa, theta, sigma : float
        CIR parameters.

    Returns
    -------
    float or ndarray
        Log-density at r_next.
    """
    c = 2.0 * kappa / (sigma ** 2 * (1.0 - np.exp(-kappa * dt)))
    df = 4.0 * kappa * theta / (sigma ** 2)
    nc = 2.0 * c * r_prev * np.exp(-kappa * dt)
    x = 2.0 * c * r_next

    return np.log(2.0 * c) + ncx2.logpdf(x, df=df, nc=nc)


# =============================================================================
# CIR Moments
# =============================================================================

def cir_conditional_mean(r0: float, t: float, kappa: float, theta: float) -> float:
    """
    E[r(t) | r(0) = r0] = theta + (r0 - theta) * exp(-kappa * t).

    Parameters
    ----------
    r0 : float
        Initial rate.
    t : float
        Time horizon.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run mean.

    Returns
    -------
    float
        Conditional mean of r(t).
    """
    return theta + (r0 - theta) * np.exp(-kappa * t)


def cir_conditional_variance(
    r0: float, t: float, kappa: float, theta: float, sigma: float
) -> float:
    """
    Var[r(t) | r(0) = r0].

    Parameters
    ----------
    r0 : float
        Initial rate.
    t : float
        Time horizon.
    kappa, theta, sigma : float
        CIR parameters.

    Returns
    -------
    float
        Conditional variance of r(t).
    """
    exp_kt = np.exp(-kappa * t)
    term1 = r0 * (sigma ** 2) * exp_kt * (1.0 - exp_kt) / kappa
    term2 = theta * (sigma ** 2) * (1.0 - exp_kt) ** 2 / (2.0 * kappa)
    return term1 + term2


def feller_condition_check(kappa: float, theta: float, sigma: float) -> dict:
    """
    Check the Feller condition: 2*kappa*theta >= sigma^2.

    Parameters
    ----------
    kappa, theta, sigma : float
        CIR parameters.

    Returns
    -------
    dict
        Dictionary with 'satisfied', 'lhs', 'rhs', and 'margin'.
    """
    lhs = 2.0 * kappa * theta
    rhs = sigma ** 2
    return {
        "satisfied": lhs >= rhs,
        "lhs": lhs,
        "rhs": rhs,
        "margin": lhs - rhs,
        "description": (
            "Strong Feller (process never reaches zero)"
            if lhs > rhs
            else "Critical Feller (touches zero)"
            if lhs == rhs
            else "Feller violated (process can reach zero)"
        ),
    }


# =============================================================================
# CIR Path Simulation
# =============================================================================

def simulate_cir_exact(
    kappa: float,
    theta: float,
    sigma: float,
    r0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate CIR paths using the exact non-central chi-squared method.

    This method produces exact samples (no discretization error).

    Parameters
    ----------
    kappa, theta, sigma : float
        CIR parameters.
    r0 : float
        Initial rate.
    T : float
        Total time horizon.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths to simulate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1)
        Simulated rate paths.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0

    c = sigma ** 2 * (1.0 - np.exp(-kappa * dt)) / (4.0 * kappa)
    df = 4.0 * kappa * theta / (sigma ** 2)

    for i in range(n_steps):
        # Non-centrality parameter for each path
        nc = paths[:, i] * np.exp(-kappa * dt) / c
        nc = np.maximum(nc, 1e-10)  # ensure positive

        # Draw from non-central chi-squared and rescale
        paths[:, i + 1] = c * np.random.noncentral_chisquare(df, nc)

    return paths


def simulate_cir_euler(
    kappa: float,
    theta: float,
    sigma: float,
    r0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate CIR paths using the Euler-Maruyama scheme with full truncation.

    The full truncation scheme replaces r with max(r, 0) in both drift and
    diffusion to prevent negative values.

    Parameters
    ----------
    kappa, theta, sigma : float
        CIR parameters.
    r0 : float
        Initial rate.
    T : float
        Total time horizon.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths to simulate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1)
        Simulated rate paths.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0

    for i in range(n_steps):
        r_pos = np.maximum(paths[:, i], 0.0)
        dW = np.random.normal(0, 1, n_paths) * sqrt_dt
        drift = kappa * (theta - r_pos) * dt
        diffusion = sigma * np.sqrt(r_pos) * dW
        paths[:, i + 1] = paths[:, i] + drift + diffusion
        paths[:, i + 1] = np.maximum(paths[:, i + 1], 0.0)

    return paths


# =============================================================================
# Yield Curve Generation
# =============================================================================

def generate_yield_curve(
    r: float,
    maturities: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
) -> np.ndarray:
    """
    Generate the CIR yield curve for a range of maturities.

    Parameters
    ----------
    r : float
        Current short rate.
    maturities : ndarray
        Array of maturities (in years).
    kappa, theta, sigma : float
        CIR parameters.

    Returns
    -------
    ndarray
        Yields for each maturity.
    """
    yields = np.zeros_like(maturities)
    for i, tau in enumerate(maturities):
        if tau > 0:
            yields[i] = cir_yield(r, tau, kappa, theta, sigma)
        else:
            yields[i] = r
    return yields


if __name__ == "__main__":
    # Quick demonstration
    kappa, theta, sigma = 0.5, 0.05, 0.1
    r0 = 0.03

    # Check Feller condition
    feller = feller_condition_check(kappa, theta, sigma)
    print(f"Feller condition: {feller}")

    # Bond price at various rates
    rates = np.linspace(0.01, 0.15, 10)
    prices = cir_bond_price(rates, 0.0, 5.0, kappa, theta, sigma)
    print(f"\nBond prices (T=5, various r):")
    for r, p in zip(rates, prices):
        print(f"  r={r:.4f} -> P={p:.6f}")

    # Yield curve
    maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    yields = generate_yield_curve(r0, maturities, kappa, theta, sigma)
    print(f"\nYield curve (r0={r0}):")
    for tau, y in zip(maturities, yields):
        print(f"  tau={tau:.2f}y -> yield={y*100:.4f}%")

    # Simulate paths
    paths = simulate_cir_exact(kappa, theta, sigma, r0, 10.0, 1000, n_paths=5, seed=42)
    print(f"\nSimulated {paths.shape[0]} paths with {paths.shape[1]-1} steps")
    print(f"Min rate: {paths.min():.6f}, Max rate: {paths.max():.6f}")
    print(f"Mean final rate: {paths[:, -1].mean():.6f} (theta={theta})")
