"""
Volatility Surface representation and arbitrage checking.

Provides:
- VolSurface: discretized implied volatility surface on a moneyness x maturity grid
- ArbitrageChecker: butterfly and calendar spread arbitrage detection
- compute_skew / compute_butterfly: surface metrics
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class VolSurface:
    """A discretized implied volatility surface on a moneyness x maturity grid.

    Attributes:
        moneyness: Moneyness values (K/F) for each column.
        maturities: Time-to-expiration values (in years) for each row.
        ivols: Implied volatility grid, shape (n_maturities, n_moneyness).
    """

    moneyness: np.ndarray
    maturities: np.ndarray
    ivols: np.ndarray

    def __post_init__(self) -> None:
        self.moneyness = np.asarray(self.moneyness, dtype=np.float64)
        self.maturities = np.asarray(self.maturities, dtype=np.float64)
        self.ivols = np.asarray(self.ivols, dtype=np.float64)
        assert self.ivols.shape == (len(self.maturities), len(self.moneyness))

    def flatten(self) -> np.ndarray:
        """Flatten the surface grid into a 1-D vector (row-major)."""
        return self.ivols.flatten()

    @property
    def grid_size(self) -> int:
        """Number of grid points."""
        return self.ivols.size

    def get(self, mat_idx: int, mon_idx: int) -> float:
        """Get implied vol at grid indices (maturity_idx, moneyness_idx)."""
        return float(self.ivols[mat_idx, mon_idx])

    def atm_vol(self, mat_idx: int) -> float:
        """ATM implied vol for a given maturity index (moneyness closest to 1.0)."""
        atm_idx = int(np.argmin(np.abs(self.moneyness - 1.0)))
        return float(self.ivols[mat_idx, atm_idx])


class ArbitrageChecker:
    """Checks no-arbitrage conditions on a volatility surface."""

    @staticmethod
    def check_butterfly(surface: VolSurface) -> Tuple[int, List[Tuple[int, int]]]:
        """Check butterfly arbitrage: total variance must be convex in moneyness.

        Returns:
            Tuple of (violation_count, list of (mat_idx, mon_idx) violation positions).
        """
        violations: List[Tuple[int, int]] = []
        for t in range(len(surface.maturities)):
            tau = surface.maturities[t]
            for k in range(1, len(surface.moneyness) - 1):
                w_left = surface.ivols[t, k - 1] ** 2 * tau
                w_mid = surface.ivols[t, k] ** 2 * tau
                w_right = surface.ivols[t, k + 1] ** 2 * tau
                if w_mid > (w_left + w_right) / 2.0 + 1e-10:
                    violations.append((t, k))
        return len(violations), violations

    @staticmethod
    def check_calendar(surface: VolSurface) -> Tuple[int, List[Tuple[int, int]]]:
        """Check calendar spread arbitrage: total variance must be non-decreasing in maturity.

        Returns:
            Tuple of (violation_count, list of (mat_idx, mon_idx) violation positions).
        """
        violations: List[Tuple[int, int]] = []
        for k in range(len(surface.moneyness)):
            for t in range(1, len(surface.maturities)):
                w_prev = surface.ivols[t - 1, k] ** 2 * surface.maturities[t - 1]
                w_curr = surface.ivols[t, k] ** 2 * surface.maturities[t]
                if w_curr < w_prev - 1e-10:
                    violations.append((t, k))
        return len(violations), violations


def compute_skew(surface: VolSurface, mat_idx: int) -> float:
    """Compute the 25-delta skew: OTM put vol - OTM call vol."""
    n = len(surface.moneyness)
    if n < 3:
        return 0.0
    put_vol = surface.ivols[mat_idx, 1]
    call_vol = surface.ivols[mat_idx, n - 2]
    return float(put_vol - call_vol)


def compute_butterfly(surface: VolSurface, mat_idx: int) -> float:
    """Compute butterfly spread metric: average wing vol - ATM vol."""
    n = len(surface.moneyness)
    if n < 3:
        return 0.0
    atm_vol = surface.atm_vol(mat_idx)
    wing_avg = (surface.ivols[mat_idx, 1] + surface.ivols[mat_idx, n - 2]) / 2.0
    return float(wing_avg - atm_vol)
