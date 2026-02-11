"""
Hull-White Model Calibration

Calibrate Hull-White parameters (a, sigma) to market data:
1. Yield curve fitting (theta(t) determination)
2. Cap/swaption implied volatility fitting
3. Funding rate time series fitting (crypto)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from typing import Optional, Dict, Tuple, List
import pandas as pd

from analytical import HullWhiteAnalytical


def calibrate_to_yield_curve(
    market_maturities: np.ndarray,
    market_rates: np.ndarray,
    a_init: float = 0.1,
    sigma_init: float = 0.01,
    method: str = "Nelder-Mead",
) -> Dict:
    """
    Calibrate Hull-White parameters to match a yield curve.

    The Hull-White model automatically fits the initial yield curve through
    theta(t). This function finds the (a, sigma) that best match the
    yield curve dynamics (i.e., minimize pricing errors for known instruments).

    Args:
        market_maturities: Array of maturities (years)
        market_rates: Array of zero rates
        a_init: Initial guess for mean reversion
        sigma_init: Initial guess for volatility
        method: Optimization method

    Returns:
        Dict with calibrated parameters
    """
    def objective(params):
        a, sigma = params
        if a <= 0 or sigma <= 0:
            return 1e10

        try:
            hw = HullWhiteAnalytical(
                a=a, sigma=sigma,
                market_times=market_maturities,
                market_rates=market_rates,
            )

            # Evaluate bond pricing consistency
            r0 = market_rates[0]
            errors = []
            for i, T in enumerate(market_maturities):
                if T < 0.01:
                    continue
                # Market discount factor
                P_market = np.exp(-market_rates[i] * T)
                # Model discount factor
                P_model = hw.bond_price(r0, 0.0, T)
                errors.append((P_model - P_market) ** 2)

            return np.mean(errors) if errors else 1e10

        except Exception:
            return 1e10

    result = minimize(
        objective,
        x0=[a_init, sigma_init],
        method=method,
        bounds=[(0.001, 5.0), (0.0001, 0.1)] if method in ["L-BFGS-B", "SLSQP"] else None,
        options={"maxiter": 1000},
    )

    a_cal, sigma_cal = result.x

    hw_calibrated = HullWhiteAnalytical(
        a=a_cal, sigma=sigma_cal,
        market_times=market_maturities,
        market_rates=market_rates,
    )

    return {
        "a": float(a_cal),
        "sigma": float(sigma_cal),
        "theta_mean": float(np.mean([hw_calibrated.theta(t) for t in market_maturities])),
        "optimization_success": result.success,
        "optimization_message": result.message,
        "objective_value": float(result.fun),
    }


def calibrate_to_caplets(
    market_maturities: np.ndarray,
    market_rates: np.ndarray,
    caplet_expiries: np.ndarray,
    caplet_vols: np.ndarray,
    strike: float = 0.05,
    a_init: float = 0.1,
    sigma_init: float = 0.01,
) -> Dict:
    """
    Calibrate Hull-White parameters to cap/caplet implied volatilities.

    Under Hull-White, caplet prices have a closed form. We find (a, sigma)
    to best match market caplet implied volatilities.

    Args:
        market_maturities: Yield curve maturities
        market_rates: Yield curve rates
        caplet_expiries: Caplet expiry times
        caplet_vols: Market caplet implied volatilities
        strike: Cap strike rate
        a_init: Initial mean reversion guess
        sigma_init: Initial volatility guess

    Returns:
        Dict with calibrated parameters
    """
    def hw_caplet_vol(a, sigma, t, T, delta=0.25):
        """Implied volatility of a caplet under Hull-White."""
        B = (1 - np.exp(-a * delta)) / a
        sigma_p = sigma * B * np.sqrt((1 - np.exp(-2 * a * t)) / (2 * a))
        return sigma_p

    def objective(params):
        a, sigma = params
        if a <= 0.001 or sigma <= 0.0001:
            return 1e10

        errors = []
        for i, T_exp in enumerate(caplet_expiries):
            model_vol = hw_caplet_vol(a, sigma, T_exp, T_exp + 0.25)
            market_vol = caplet_vols[i]
            errors.append((model_vol - market_vol) ** 2)

        return np.mean(errors)

    result = minimize(
        objective,
        x0=[a_init, sigma_init],
        method="Nelder-Mead",
        options={"maxiter": 2000},
    )

    a_cal, sigma_cal = result.x

    return {
        "a": float(a_cal),
        "sigma": float(sigma_cal),
        "optimization_success": result.success,
        "objective_value": float(result.fun),
    }


def calibrate_hull_white(
    funding_rates_df: pd.DataFrame,
    rate_column: str = "funding_rate",
    dt_hours: float = 8.0,
) -> Dict:
    """
    Calibrate Hull-White parameters from a time series of funding rates.

    Uses Maximum Likelihood Estimation for the Ornstein-Uhlenbeck process:
        dr = (theta - a*r) dt + sigma dW

    The discrete-time MLE for the OU process:
        r_{i+1} = r_i * exp(-a*dt) + theta/a * (1-exp(-a*dt)) + epsilon_i
        where epsilon_i ~ N(0, sigma^2/(2a) * (1-exp(-2*a*dt)))

    Args:
        funding_rates_df: DataFrame with funding rate observations
        rate_column: Column name for the rates
        dt_hours: Time between observations in hours

    Returns:
        Dict with calibrated parameters (a, sigma, theta_mean)
    """
    rates = funding_rates_df[rate_column].values

    if len(rates) < 10:
        raise ValueError("Need at least 10 observations for calibration")

    # Convert dt to yearly units (8 hours = 8/8760 years)
    dt = dt_hours / 8760.0

    # Method 1: OLS regression (quick estimate)
    # dr_i = (theta - a * r_i) * dt + noise
    dr = np.diff(rates)
    r_prev = rates[:-1]

    X = np.column_stack([np.ones(len(r_prev)), r_prev])
    coeffs, residuals, _, _ = np.linalg.lstsq(X, dr, rcond=None)
    c0, c1 = coeffs

    a_ols = -c1 / dt
    theta_ols = c0 / dt
    sigma_ols = np.std(dr - X @ coeffs) / np.sqrt(dt)

    # Method 2: MLE for OU process
    def neg_log_likelihood(params):
        a_p, theta_p, sigma_p = params
        if a_p <= 0 or sigma_p <= 0:
            return 1e10

        exp_adt = np.exp(-a_p * dt)
        mu_cond = r_prev * exp_adt + (theta_p / a_p) * (1 - exp_adt)
        var_cond = (sigma_p**2 / (2 * a_p)) * (1 - np.exp(-2 * a_p * dt))

        if var_cond <= 0:
            return 1e10

        n = len(dr)
        r_next = rates[1:]
        ll = -0.5 * n * np.log(2 * np.pi * var_cond) \
            - 0.5 * np.sum((r_next - mu_cond)**2 / var_cond)

        return -ll  # negative for minimization

    # Use OLS estimates as starting point for MLE
    a_init = max(a_ols, 0.01)
    theta_init = theta_ols if abs(theta_ols) < 1.0 else np.mean(rates) * a_init
    sigma_init = max(sigma_ols, 1e-6)

    try:
        result = minimize(
            neg_log_likelihood,
            x0=[a_init, theta_init, sigma_init],
            method="Nelder-Mead",
            options={"maxiter": 5000},
        )
        a_mle, theta_mle, sigma_mle = result.x
        mle_success = result.success
    except Exception:
        a_mle, theta_mle, sigma_mle = a_init, theta_init, sigma_init
        mle_success = False

    # Annualize parameters
    a_annual = max(a_mle, 0.001)
    sigma_annual = max(sigma_mle, 1e-8)
    theta_annual = theta_mle

    # Long-run mean rate
    long_run_mean = theta_annual / a_annual if a_annual > 0.001 else np.mean(rates)

    # Half-life of mean reversion (in hours and days)
    half_life_years = np.log(2) / a_annual
    half_life_hours = half_life_years * 8760
    half_life_days = half_life_hours / 24

    return {
        "a": float(a_annual),
        "sigma": float(sigma_annual),
        "theta_mean": float(theta_annual),
        "long_run_mean": float(long_run_mean),
        "half_life_hours": float(half_life_hours),
        "half_life_days": float(half_life_days),
        "mle_success": mle_success,
        "n_observations": len(rates),
        "current_rate": float(rates[-1]),
        "mean_rate": float(np.mean(rates)),
        "std_rate": float(np.std(rates)),
    }


def calibrate_piecewise_theta(
    market_maturities: np.ndarray,
    market_rates: np.ndarray,
    a: float,
    sigma: float,
    n_pieces: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrate piecewise-constant theta(t) to match the term structure exactly.

    This is an alternative to the continuous theta(t) formula, useful when
    the market curve is noisy.

    Args:
        market_maturities: Yield curve maturities
        market_rates: Yield curve zero rates
        a: Mean reversion speed
        sigma: Volatility
        n_pieces: Number of piecewise segments

    Returns:
        (theta_times, theta_values) arrays
    """
    T_max = market_maturities[-1]
    theta_times = np.linspace(0, T_max, n_pieces + 1)
    theta_dt = theta_times[1] - theta_times[0]

    def objective(theta_vals):
        """Minimize pricing errors given piecewise theta."""
        errors = []
        for i, T in enumerate(market_maturities):
            P_market = np.exp(-market_rates[i] * T)

            # Numerical bond price with given theta
            P_model = _numerical_bond_price(
                r0=market_rates[0], t=0.0, T=T,
                a=a, sigma=sigma,
                theta_times=theta_times[:-1],
                theta_values=theta_vals,
                theta_dt=theta_dt,
            )
            errors.append((P_model - P_market) ** 2)

        return np.sum(errors)

    # Initial guess: constant theta
    theta_init = np.full(n_pieces, a * market_rates.mean())

    result = minimize(
        objective,
        x0=theta_init,
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )

    return theta_times[:-1] + theta_dt / 2, result.x


def _numerical_bond_price(
    r0: float, t: float, T: float,
    a: float, sigma: float,
    theta_times: np.ndarray,
    theta_values: np.ndarray,
    theta_dt: float,
    n_steps: int = 100,
) -> float:
    """Simple Euler scheme for bond pricing with piecewise theta."""
    dt = (T - t) / n_steps
    r = r0
    integral = 0.0

    for i in range(n_steps):
        ti = t + i * dt
        # Find theta at time ti
        idx = np.searchsorted(theta_times, ti, side="right") - 1
        idx = max(0, min(idx, len(theta_values) - 1))
        theta_i = theta_values[idx]

        integral += r * dt
        dr = (theta_i - a * r) * dt
        r += dr

    return np.exp(-integral)


if __name__ == "__main__":
    from data_loader import load_sample_treasury_curve, generate_synthetic_funding_rates

    print("=" * 60)
    print("Hull-White Calibration")
    print("=" * 60)

    # 1. Calibrate to Treasury yield curve
    print("\n--- Calibration to Treasury Yield Curve ---")
    mat, rates = load_sample_treasury_curve()

    params_yc = calibrate_to_yield_curve(mat, rates)
    print(f"Mean reversion (a): {params_yc['a']:.4f}")
    print(f"Volatility (sigma): {params_yc['sigma']:.6f}")
    print(f"Optimization success: {params_yc['optimization_success']}")

    # 2. Calibrate to crypto funding rates
    print("\n--- Calibration to Crypto Funding Rates ---")
    funding_df = generate_synthetic_funding_rates(n_periods=500)
    params_crypto = calibrate_hull_white(funding_df)

    print(f"Mean reversion (a): {params_crypto['a']:.4f}")
    print(f"Volatility (sigma): {params_crypto['sigma']:.8f}")
    print(f"Long-run mean rate: {params_crypto['long_run_mean']:.6f}")
    print(f"Half-life: {params_crypto['half_life_hours']:.1f} hours ({params_crypto['half_life_days']:.1f} days)")
    print(f"Current rate: {params_crypto['current_rate']:.6f}")
    print(f"MLE success: {params_crypto['mle_success']}")

    # 3. Piecewise theta calibration
    print("\n--- Piecewise Theta Calibration ---")
    theta_t, theta_v = calibrate_piecewise_theta(mat, rates, a=0.1, sigma=0.01, n_pieces=10)
    print("Piecewise theta(t):")
    for t, v in zip(theta_t, theta_v):
        print(f"  t={t:5.2f}: theta={v:.6f}")
