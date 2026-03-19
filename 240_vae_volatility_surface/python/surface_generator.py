"""
SABR-based synthetic volatility surface generator.

Generates training data for the VAE using the SABR implied volatility model
(Hagan et al. 2002).
"""

import math
from typing import List

import numpy as np

from .vol_surface import VolSurface


def default_moneyness_grid() -> np.ndarray:
    """Standard moneyness grid for surface construction."""
    return np.array([0.80, 0.85, 0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10, 1.15, 1.20])


def default_maturity_days() -> np.ndarray:
    """Standard maturity grid (in days)."""
    return np.array([7.0, 14.0, 30.0, 60.0, 90.0, 180.0, 365.0])


class SurfaceGenerator:
    """Generates synthetic implied volatility surfaces using the SABR model."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def sabr_iv(
        alpha: float, beta: float, rho: float, nu: float,
        f: float, k: float, t: float,
    ) -> float:
        """SABR implied volatility approximation (Hagan et al. 2002).

        Args:
            alpha: Vol-of-vol scale.
            beta: CEV exponent.
            rho: Correlation between asset and vol.
            nu: Vol-of-vol.
            f: Forward price.
            k: Strike price.
            t: Time to expiration (years).

        Returns:
            Approximate implied volatility.
        """
        if abs(f - k) < 1e-12:
            # ATM approximation
            fk_mid = f ** (1.0 - beta)
            term1 = alpha / fk_mid
            term2 = 1.0 + (
                (1.0 - beta) ** 2 / 24.0 * alpha ** 2 / f ** (2.0 * (1.0 - beta))
                + 0.25 * rho * beta * nu * alpha / fk_mid
                + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
            ) * t
            return max(term1 * term2, 0.01)

        fk = f * k
        fk_beta = fk ** ((1.0 - beta) / 2.0)
        log_fk = math.log(f / k)

        z = (nu / alpha) * fk_beta * log_fk
        x_z_denom = math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho
        x_z_ratio = x_z_denom / (1.0 - rho)
        if abs(x_z_ratio) < 1e-12:
            x_z = 1.0
        else:
            x_z = z / math.log(x_z_ratio)

        a_term = alpha / (
            fk_beta * (
                1.0
                + (1.0 - beta) ** 2 / 24.0 * log_fk ** 2
                + (1.0 - beta) ** 4 / 1920.0 * log_fk ** 4
            )
        )

        b_term = 1.0 + (
            (1.0 - beta) ** 2 / 24.0 * alpha ** 2 / fk ** (1.0 - beta)
            + 0.25 * rho * beta * nu * alpha / fk_beta
            + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
        ) * t

        return max(a_term * x_z * b_term, 0.01)

    def generate_surface(
        self,
        moneyness: np.ndarray,
        maturities: np.ndarray,
    ) -> VolSurface:
        """Generate a single synthetic volatility surface with random SABR parameters."""
        alpha = self.rng.uniform(0.15, 0.60)
        beta = self.rng.uniform(0.3, 0.9)
        rho = self.rng.uniform(-0.8, -0.1)
        nu = self.rng.uniform(0.2, 0.8)
        forward = 1.0  # Normalized (moneyness = K/F, so F=1)

        n_mat = len(maturities)
        n_mon = len(moneyness)
        ivols = np.zeros((n_mat, n_mon))

        for t_idx, tau in enumerate(maturities):
            for k_idx, m in enumerate(moneyness):
                strike = m * forward
                ivols[t_idx, k_idx] = self.sabr_iv(
                    alpha, beta, rho, nu, forward, strike, tau,
                )

        return VolSurface(moneyness=moneyness.copy(), maturities=maturities.copy(), ivols=ivols)

    def generate_batch(
        self,
        n: int,
        moneyness: np.ndarray,
        maturities: np.ndarray,
    ) -> List[VolSurface]:
        """Generate a batch of synthetic surfaces for training."""
        return [self.generate_surface(moneyness, maturities) for _ in range(n)]
