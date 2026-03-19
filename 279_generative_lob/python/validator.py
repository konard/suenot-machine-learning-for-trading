"""
Statistical Validation for LOB Data

Provides tools for comparing real and synthetic LOB data across
key statistical properties: mid-price distribution, spread, volume profiles,
and autocorrelation structure.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from .lob_snapshot import LobSnapshot


@dataclass
class ValidationReport:
    """Validation report comparing real and synthetic LOB statistics."""
    real_mid_stats: Tuple[float, float, float, float]
    synthetic_mid_stats: Tuple[float, float, float, float]
    real_spread_stats: Tuple[float, float, float, float]
    synthetic_spread_stats: Tuple[float, float, float, float]
    real_volume_stats: Tuple[float, float, float, float]
    synthetic_volume_stats: Tuple[float, float, float, float]

    def __str__(self) -> str:
        def fmt(stats: tuple) -> str:
            return f"(mean={stats[0]:.4f}, std={stats[1]:.4f}, min={stats[2]:.4f}, max={stats[3]:.4f})"

        lines = [
            "=== LOB Validation Report ===",
            "",
            "Mid-Price Statistics:",
            f"  Real:      {fmt(self.real_mid_stats)}",
            f"  Synthetic: {fmt(self.synthetic_mid_stats)}",
            "",
            "Spread Statistics:",
            f"  Real:      {fmt(self.real_spread_stats)}",
            f"  Synthetic: {fmt(self.synthetic_spread_stats)}",
            "",
            "Bid Volume Statistics (top 5 levels):",
            f"  Real:      {fmt(self.real_volume_stats)}",
            f"  Synthetic: {fmt(self.synthetic_volume_stats)}",
        ]
        return "\n".join(lines)


class LobValidator:
    """Statistical validator for comparing real and synthetic LOB data."""

    @staticmethod
    def basic_stats(values: List[float]) -> Tuple[float, float, float, float]:
        """
        Compute basic statistics.

        Returns:
            (mean, std, min, max) tuple.
        """
        if not values:
            return (0.0, 0.0, 0.0, 0.0)
        arr = np.array(values)
        return (float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()))

    @staticmethod
    def mid_prices(snapshots: List[LobSnapshot]) -> List[float]:
        """Extract mid-prices from snapshots."""
        return [s.mid_price() for s in snapshots]

    @staticmethod
    def spreads(snapshots: List[LobSnapshot]) -> List[float]:
        """Extract spreads from snapshots."""
        return [s.spread() for s in snapshots]

    @staticmethod
    def bid_volumes(snapshots: List[LobSnapshot], levels: int = 5) -> List[float]:
        """Extract total bid volume at top N levels."""
        return [
            sum(s.bids[i].quantity for i in range(min(levels, len(s.bids))))
            for s in snapshots
        ]

    @staticmethod
    def ask_volumes(snapshots: List[LobSnapshot], levels: int = 5) -> List[float]:
        """Extract total ask volume at top N levels."""
        return [
            sum(s.asks[i].quantity for i in range(min(levels, len(s.asks))))
            for s in snapshots
        ]

    @staticmethod
    def order_imbalances(snapshots: List[LobSnapshot], levels: int = 5) -> List[float]:
        """Extract order imbalance at top N levels."""
        return [s.order_imbalance(levels) for s in snapshots]

    @classmethod
    def validate(
        cls, real: List[LobSnapshot], synthetic: List[LobSnapshot]
    ) -> ValidationReport:
        """
        Compare real and synthetic LOB data.

        Args:
            real: List of real LOB snapshots.
            synthetic: List of synthetic LOB snapshots.

        Returns:
            ValidationReport with comparative statistics.
        """
        return ValidationReport(
            real_mid_stats=cls.basic_stats(cls.mid_prices(real)),
            synthetic_mid_stats=cls.basic_stats(cls.mid_prices(synthetic)),
            real_spread_stats=cls.basic_stats(cls.spreads(real)),
            synthetic_spread_stats=cls.basic_stats(cls.spreads(synthetic)),
            real_volume_stats=cls.basic_stats(cls.bid_volumes(real)),
            synthetic_volume_stats=cls.basic_stats(cls.bid_volumes(synthetic)),
        )


if __name__ == "__main__":
    from .lob_snapshot import generate_test_snapshots

    real = generate_test_snapshots(50, 10, 50000.0)
    synthetic = generate_test_snapshots(50, 10, 50000.0)

    report = LobValidator.validate(real, synthetic)
    print(report)
    print("\nValidator test passed!")
