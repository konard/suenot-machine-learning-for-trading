"""
Visualization utilities for volatility surfaces.

Provides functions to plot:
- 3D volatility surfaces
- Surface comparisons (observed vs reconstructed)
- Latent space embeddings
- Backtest PnL curves
"""

from typing import List, Optional

import numpy as np

from .vol_surface import VolSurface

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def plot_surface(
    surface: VolSurface,
    title: str = "Implied Volatility Surface",
    save_path: Optional[str] = None,
) -> None:
    """Plot a 3D volatility surface."""
    if not MPL_AVAILABLE:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    M, T = np.meshgrid(surface.moneyness, surface.maturities)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(M, T, surface.ivols, cmap="viridis", alpha=0.8)
    ax.set_xlabel("Moneyness (K/F)")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_surface_comparison(
    observed: VolSurface,
    reconstructed: VolSurface,
    title: str = "Observed vs Reconstructed",
    save_path: Optional[str] = None,
) -> None:
    """Plot side-by-side comparison of observed and reconstructed surfaces."""
    if not MPL_AVAILABLE:
        print("matplotlib not available.")
        return

    M, T = np.meshgrid(observed.moneyness, observed.maturities)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": "3d"})

    axes[0].plot_surface(M, T, observed.ivols, cmap="viridis", alpha=0.8)
    axes[0].set_title("Observed")
    axes[0].set_xlabel("Moneyness")
    axes[0].set_ylabel("Maturity")

    axes[1].plot_surface(M, T, reconstructed.ivols, cmap="viridis", alpha=0.8)
    axes[1].set_title("Reconstructed")
    axes[1].set_xlabel("Moneyness")
    axes[1].set_ylabel("Maturity")

    diff = observed.ivols - reconstructed.ivols
    axes[2].plot_surface(M, T, diff, cmap="RdBu", alpha=0.8)
    axes[2].set_title("Difference")
    axes[2].set_xlabel("Moneyness")
    axes[2].set_ylabel("Maturity")

    fig.suptitle(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_pnl(
    pnl_series: List[float],
    title: str = "Strategy PnL",
    save_path: Optional[str] = None,
) -> None:
    """Plot cumulative PnL curve."""
    if not MPL_AVAILABLE:
        print("matplotlib not available.")
        return

    cum_pnl = np.cumsum(pnl_series)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    ax1.plot(cum_pnl, color="blue")
    ax1.set_title("Cumulative PnL")
    ax1.set_ylabel("PnL")
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(pnl_series)), pnl_series, color=["green" if p >= 0 else "red" for p in pnl_series], alpha=0.7)
    ax2.set_title("Period PnL")
    ax2.set_xlabel("Period")
    ax2.set_ylabel("PnL")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()
