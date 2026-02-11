"""
CIR Model Calibration via Maximum Likelihood Estimation
=========================================================

Calibrates CIR parameters (kappa, theta, sigma) using:
- Exact MLE based on non-central chi-squared transition density
- Approximate MLE based on conditional moments
- Ordinary Least Squares (OLS) as a fast baseline

The CIR transition density:
    r(t+dt) | r(t) ~ c * chi^2(df, nc)
    where c = sigma^2*(1-exp(-kappa*dt))/(4*kappa)
          df = 4*kappa*theta/sigma^2
          nc = 4*kappa*r(t)*exp(-kappa*dt) / (sigma^2*(1-exp(-kappa*dt)))
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import ncx2
from typing import Tuple, Optional, Dict
import warnings


def cir_log_likelihood(
    params: Tuple[float, float, float],
    rates: np.ndarray,
    dt: float,
) -> float:
    """
    Negative log-likelihood for CIR model using non-central chi-squared density.

    Parameters
    ----------
    params : tuple
        (kappa, theta, sigma) CIR parameters.
    rates : ndarray
        Time series of observed rates.
    dt : float
        Time step between observations (in years).

    Returns
    -------
    float
        Negative log-likelihood (to be minimized).
    """
    kappa, theta, sigma = params

    # Parameter constraints
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return 1e10
    if sigma ** 2 > 100 * kappa * theta:  # avoid numerical overflow
        return 1e10

    try:
        exp_neg_kdt = np.exp(-kappa * dt)
        c = 2.0 * kappa / (sigma ** 2 * (1.0 - exp_neg_kdt))
        df = 4.0 * kappa * theta / (sigma ** 2)

        nll = 0.0
        for i in range(1, len(rates)):
            r_prev = max(rates[i - 1], 1e-10)
            r_curr = max(rates[i], 1e-10)

            nc = 2.0 * c * r_prev * exp_neg_kdt
            x = 2.0 * c * r_curr

            if nc < 0 or x < 0 or df < 0:
                return 1e10

            log_pdf = ncx2.logpdf(x, df=df, nc=nc)

            if not np.isfinite(log_pdf):
                return 1e10

            nll -= log_pdf + np.log(2.0 * c)

        return nll

    except (ValueError, RuntimeWarning, FloatingPointError):
        return 1e10


def cir_moment_log_likelihood(
    params: Tuple[float, float, float],
    rates: np.ndarray,
    dt: float,
) -> float:
    """
    Approximate negative log-likelihood using conditional moments.

    Uses the Gaussian approximation to the CIR transition density:
    r(t+dt) | r(t) ~ N(E[r(t+dt)|r(t)], Var[r(t+dt)|r(t)])

    This is faster but less accurate than the exact ncx2 method.

    Parameters
    ----------
    params : tuple
        (kappa, theta, sigma) CIR parameters.
    rates : ndarray
        Time series of observed rates.
    dt : float
        Time step between observations.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    kappa, theta, sigma = params

    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return 1e10

    try:
        exp_neg_kdt = np.exp(-kappa * dt)

        nll = 0.0
        for i in range(1, len(rates)):
            r_prev = max(rates[i - 1], 1e-10)
            r_curr = rates[i]

            # Conditional mean
            mu = theta + (r_prev - theta) * exp_neg_kdt

            # Conditional variance
            var = (
                r_prev * sigma ** 2 * exp_neg_kdt * (1.0 - exp_neg_kdt) / kappa
                + theta * sigma ** 2 * (1.0 - exp_neg_kdt) ** 2 / (2.0 * kappa)
            )

            if var <= 0:
                return 1e10

            # Gaussian log-likelihood
            nll += 0.5 * np.log(2.0 * np.pi * var) + 0.5 * (r_curr - mu) ** 2 / var

        return nll

    except (ValueError, RuntimeWarning, FloatingPointError):
        return 1e10


def calibrate_cir_ols(
    rates: np.ndarray,
    dt: float,
) -> Tuple[float, float, float]:
    """
    Fast OLS-based calibration of CIR parameters.

    Uses the discrete-time regression:
    r(t+dt) - r(t) = kappa*(theta - r(t))*dt + epsilon

    And the volatility is estimated from residuals.

    Parameters
    ----------
    rates : ndarray
        Time series of observed rates.
    dt : float
        Time step.

    Returns
    -------
    Tuple[float, float, float]
        (kappa, theta, sigma) estimated parameters.
    """
    dr = np.diff(rates)
    r_prev = rates[:-1]

    # Regression: dr = a + b * r_prev + error
    # where a = kappa * theta * dt, b = -kappa * dt
    X = np.column_stack([np.ones_like(r_prev), r_prev])
    y = dr

    # OLS estimation
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    a, b = beta

    kappa = -b / dt
    theta = a / (kappa * dt) if kappa > 0 else np.mean(rates)

    # Estimate sigma from residuals
    residuals = y - X @ beta
    # Under CIR: Var(residual) ~ sigma^2 * r * dt
    sigma_sq = np.mean(residuals ** 2 / (np.maximum(r_prev, 1e-10) * dt))
    sigma = np.sqrt(max(sigma_sq, 1e-10))

    # Ensure positive parameters
    kappa = max(kappa, 0.01)
    theta = max(theta, 1e-6)
    sigma = max(sigma, 1e-6)

    return kappa, theta, sigma


def calibrate_cir_mle(
    rates: np.ndarray,
    dt: float,
    method: str = "exact",
    use_global: bool = False,
) -> Dict:
    """
    Calibrate CIR parameters using Maximum Likelihood Estimation.

    Parameters
    ----------
    rates : ndarray
        Time series of observed rates (must be positive).
    dt : float
        Time step between observations (in years).
    method : str
        'exact' for ncx2-based MLE, 'moment' for moment-based MLE.
    use_global : bool
        If True, use differential evolution for global optimization.

    Returns
    -------
    Dict
        Dictionary with 'kappa', 'theta', 'sigma', 'log_likelihood',
        'feller_satisfied', and 'aic'.
    """
    # Get OLS initial estimates
    kappa_init, theta_init, sigma_init = calibrate_cir_ols(rates, dt)

    print(f"OLS initial estimates: kappa={kappa_init:.4f}, "
          f"theta={theta_init:.6f}, sigma={sigma_init:.4f}")

    # Choose likelihood function
    ll_func = cir_log_likelihood if method == "exact" else cir_moment_log_likelihood

    # Bounds for parameters
    bounds = [
        (0.01, 200.0),   # kappa
        (1e-8, 1.0),     # theta
        (1e-6, 10.0),    # sigma
    ]

    if use_global:
        # Global optimization with differential evolution
        print("Running global optimization (differential evolution)...")
        result = differential_evolution(
            ll_func,
            bounds=bounds,
            args=(rates, dt),
            maxiter=100,
            tol=1e-8,
            seed=42,
        )
    else:
        # Local optimization from OLS starting point
        x0 = [kappa_init, theta_init, sigma_init]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                ll_func,
                x0=x0,
                args=(rates, dt),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-12},
            )

    kappa, theta, sigma = result.x
    nll = result.fun
    n = len(rates) - 1  # number of transitions

    # Feller condition
    feller_lhs = 2.0 * kappa * theta
    feller_rhs = sigma ** 2

    # AIC (Akaike Information Criterion): 2k - 2*ln(L)
    aic = 2 * 3 + 2 * nll  # 3 parameters

    results = {
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "log_likelihood": -nll,
        "aic": aic,
        "feller_satisfied": feller_lhs >= feller_rhs,
        "feller_margin": feller_lhs - feller_rhs,
        "converged": result.success,
        "n_observations": n,
    }

    print(f"\nMLE Results ({method}):")
    print(f"  kappa = {kappa:.6f}")
    print(f"  theta = {theta:.8f}")
    print(f"  sigma = {sigma:.6f}")
    print(f"  Log-likelihood = {-nll:.4f}")
    print(f"  AIC = {aic:.4f}")
    print(f"  Feller: 2*kappa*theta = {feller_lhs:.6f} vs sigma^2 = {feller_rhs:.6f} "
          f"({'SATISFIED' if feller_lhs >= feller_rhs else 'VIOLATED'})")

    return results


def calibrate_cir_from_yield_curve(
    maturities: np.ndarray,
    yields: np.ndarray,
    r0: float,
) -> Dict:
    """
    Calibrate CIR parameters from an observed yield curve.

    Minimizes the sum of squared errors between CIR-implied yields
    and observed yields.

    Parameters
    ----------
    maturities : ndarray
        Array of maturities (in years).
    yields : ndarray
        Array of observed yields.
    r0 : float
        Current short rate.

    Returns
    -------
    Dict
        Calibrated parameters.
    """
    from .analytical import cir_yield

    def objective(params):
        kappa, theta, sigma = params
        if kappa <= 0 or theta <= 0 or sigma <= 0:
            return 1e10
        try:
            model_yields = np.array([
                cir_yield(r0, tau, kappa, theta, sigma)
                for tau in maturities
            ])
            return np.sum((model_yields - yields) ** 2)
        except Exception:
            return 1e10

    # Initial guess
    theta_init = yields[-1]  # long-end yield as theta approximation
    kappa_init = 0.5
    sigma_init = 0.1

    result = minimize(
        objective,
        x0=[kappa_init, theta_init, sigma_init],
        method="L-BFGS-B",
        bounds=[(0.01, 100), (1e-6, 0.5), (1e-6, 5.0)],
    )

    kappa, theta, sigma = result.x

    results = {
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rmse": np.sqrt(result.fun / len(maturities)),
        "converged": result.success,
    }

    print(f"Yield curve calibration:")
    print(f"  kappa = {kappa:.6f}, theta = {theta:.6f}, sigma = {sigma:.6f}")
    print(f"  RMSE = {results['rmse']:.6f}")

    return results


if __name__ == "__main__":
    from .analytical import simulate_cir_exact

    print("=" * 70)
    print("CIR Calibration Demo")
    print("=" * 70)

    # True parameters
    kappa_true, theta_true, sigma_true = 0.5, 0.05, 0.1
    print(f"\nTrue parameters: kappa={kappa_true}, theta={theta_true}, sigma={sigma_true}")

    # Generate synthetic CIR data
    paths = simulate_cir_exact(
        kappa_true, theta_true, sigma_true,
        r0=0.03, T=10.0, n_steps=2520, n_paths=1, seed=42,
    )
    rates = paths[0]
    dt = 10.0 / 2520.0

    # OLS calibration
    print("\n--- OLS Calibration ---")
    k_ols, t_ols, s_ols = calibrate_cir_ols(rates, dt)
    print(f"  kappa={k_ols:.4f} (true: {kappa_true})")
    print(f"  theta={t_ols:.6f} (true: {theta_true})")
    print(f"  sigma={s_ols:.4f} (true: {sigma_true})")

    # Moment-based MLE
    print("\n--- Moment-Based MLE ---")
    results_moment = calibrate_cir_mle(rates, dt, method="moment")

    # Exact MLE
    print("\n--- Exact MLE ---")
    results_exact = calibrate_cir_mle(rates, dt, method="exact")
