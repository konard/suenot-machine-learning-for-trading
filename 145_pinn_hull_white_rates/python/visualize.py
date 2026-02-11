"""
Visualization for Hull-White PINN

Plotting functions for:
- Yield curves and forward rate curves
- Bond price surfaces
- Rate distributions and paths
- Training loss curves
- PINN vs analytical comparison
- Funding rate analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from typing import Optional, Dict, List, Tuple
import os


# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "primary": "#2196F3",
    "secondary": "#FF5722",
    "accent": "#4CAF50",
    "warning": "#FFC107",
    "dark": "#263238",
    "light": "#ECEFF1",
}
FIGSIZE = (12, 8)
DPI = 150
SAVE_DIR = "plots"


def _ensure_save_dir():
    """Create plots directory if it does not exist."""
    os.makedirs(SAVE_DIR, exist_ok=True)


def plot_yield_curve(
    maturities: np.ndarray,
    yields: np.ndarray,
    maturities_pinn: Optional[np.ndarray] = None,
    yields_pinn: Optional[np.ndarray] = None,
    title: str = "Yield Curve",
    save_path: Optional[str] = None,
):
    """
    Plot yield curve with optional PINN comparison.

    Args:
        maturities: Market yield curve maturities
        yields: Market zero rates
        maturities_pinn: PINN yield curve maturities
        yields_pinn: PINN zero rates
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    ax.plot(maturities, yields * 100, "o-", color=COLORS["primary"],
            linewidth=2, markersize=6, label="Market / Analytical", zorder=5)

    if maturities_pinn is not None and yields_pinn is not None:
        ax.plot(maturities_pinn, yields_pinn * 100, "--",
                color=COLORS["secondary"], linewidth=2,
                label="PINN", zorder=4)

    ax.set_xlabel("Maturity (years)", fontsize=12)
    ax.set_ylabel("Zero Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        _ensure_save_dir()
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


def plot_bond_surface(
    r_range: Tuple[float, float],
    T_range: Tuple[float, float],
    pricing_fn,
    t: float = 0.0,
    n_r: int = 50,
    n_T: int = 50,
    title: str = "Bond Price Surface P(r, T)",
    save_path: Optional[str] = None,
):
    """
    3D surface plot of bond prices as a function of rate and maturity.

    Args:
        r_range: (r_min, r_max) for the rate axis
        T_range: (T_min, T_max) for the maturity axis
        pricing_fn: Function(r, t, T) -> price
        t: Current time
        n_r: Number of rate grid points
        n_T: Number of maturity grid points
        title: Plot title
        save_path: Path to save figure
    """
    r_vals = np.linspace(r_range[0], r_range[1], n_r)
    T_vals = np.linspace(T_range[0], T_range[1], n_T)
    R, T = np.meshgrid(r_vals, T_vals)

    P = np.zeros_like(R)
    for i in range(n_T):
        for j in range(n_r):
            P[i, j] = pricing_fn(R[i, j], t, T[i, j])

    fig = plt.figure(figsize=(12, 8), dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(R * 100, T, P, cmap=cm.viridis, alpha=0.8,
                           edgecolor="none", antialiased=True)

    ax.set_xlabel("Short Rate (%)", fontsize=11)
    ax.set_ylabel("Maturity (years)", fontsize=11)
    ax.set_zlabel("Bond Price", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

    fig.colorbar(surf, shrink=0.5, aspect=10, label="Price")

    plt.tight_layout()
    if save_path:
        _ensure_save_dir()
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


def plot_rate_paths(
    times: np.ndarray,
    paths: np.ndarray,
    mean_path: Optional[np.ndarray] = None,
    confidence_band: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "Simulated Short Rate Paths",
    save_path: Optional[str] = None,
):
    """
    Plot simulated interest rate paths.

    Args:
        times: Time array
        paths: Rate paths, shape (n_paths, n_steps+1)
        mean_path: Expected rate path
        confidence_band: (lower, upper) confidence bands
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)

    n_paths = min(paths.shape[0], 20)  # Cap at 20 paths
    for i in range(n_paths):
        ax.plot(times, paths[i] * 100, alpha=0.3, linewidth=0.8,
                color=COLORS["primary"])

    if mean_path is not None:
        ax.plot(times, mean_path * 100, color=COLORS["secondary"],
                linewidth=2.5, label="Mean", zorder=5)

    if confidence_band is not None:
        lower, upper = confidence_band
        ax.fill_between(times, lower * 100, upper * 100,
                        alpha=0.15, color=COLORS["warning"],
                        label="95% CI")

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Short Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        _ensure_save_dir()
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


def plot_training_history(
    history: Dict,
    title: str = "PINN Training Progress",
    save_path: Optional[str] = None,
):
    """
    Plot training loss curves.

    Args:
        history: Dict with keys: epoch, total_loss, pde_loss, terminal_loss, etc.
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)

    epochs = history["epoch"]

    # Total loss
    ax = axes[0, 0]
    ax.semilogy(epochs, history["total_loss"], color=COLORS["dark"], linewidth=1.5)
    ax.set_title("Total Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # PDE loss
    ax = axes[0, 1]
    ax.semilogy(epochs, history["pde_loss"], color=COLORS["primary"], linewidth=1.5)
    ax.set_title("PDE Residual Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Terminal + boundary loss
    ax = axes[1, 0]
    ax.semilogy(epochs, history["terminal_loss"], label="Terminal",
                color=COLORS["secondary"], linewidth=1.5)
    ax.semilogy(epochs, history["boundary_loss"], label="Boundary",
                color=COLORS["accent"], linewidth=1.5)
    ax.set_title("Condition Losses", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Max error
    ax = axes[1, 1]
    errors = [e for e in history["max_error"] if e > 0]
    error_epochs = [epochs[i] for i, e in enumerate(history["max_error"]) if e > 0]
    if errors:
        ax.semilogy(error_epochs, errors, "o-", color=COLORS["warning"],
                    markersize=3, linewidth=1)
    ax.set_title("Max Price Error vs Analytical", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Max |Error|")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        _ensure_save_dir()
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


def plot_pinn_vs_analytical(
    maturities: np.ndarray,
    pinn_prices: np.ndarray,
    analytical_prices: np.ndarray,
    r0: float,
    title: str = "PINN vs Analytical Bond Prices",
    save_path: Optional[str] = None,
):
    """
    Compare PINN and analytical bond prices.

    Args:
        maturities: Array of maturities
        pinn_prices: PINN bond prices
        analytical_prices: Analytical bond prices
        r0: Current short rate (for labeling)
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=DPI,
                                     gridspec_kw={"height_ratios": [3, 1]})

    # Price comparison
    ax1.plot(maturities, analytical_prices, "o-", color=COLORS["primary"],
             linewidth=2, markersize=5, label="Analytical", zorder=5)
    ax1.plot(maturities, pinn_prices, "s--", color=COLORS["secondary"],
             linewidth=2, markersize=5, label="PINN", zorder=4)
    ax1.set_ylabel("Bond Price", fontsize=12)
    ax1.set_title(f"{title} (r={r0:.2%})", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Error plot
    errors = pinn_prices - analytical_prices
    ax2.bar(maturities, errors * 10000, width=maturities[-1] / (3 * len(maturities)),
            color=COLORS["accent"], alpha=0.7)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Maturity (years)", fontsize=12)
    ax2.set_ylabel("Error (bps)", fontsize=12)
    ax2.set_title("Pricing Error", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        _ensure_save_dir()
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


def plot_funding_rates(
    timestamps,
    rates: np.ndarray,
    symbol: str = "BTCUSDT",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot Bybit funding rate time series.

    Args:
        timestamps: Array of datetime objects or timestamps
        rates: Array of funding rates
        symbol: Trading pair name
        title: Plot title
        save_path: Path to save figure
    """
    if title is None:
        title = f"{symbol} Funding Rates"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)

    # Time series
    ax = axes[0, 0]
    annualized = rates * 3 * 365 * 100  # to annualized percent
    ax.plot(timestamps, annualized, color=COLORS["primary"], linewidth=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.fill_between(timestamps, 0, annualized,
                    where=annualized > 0, alpha=0.2, color=COLORS["accent"])
    ax.fill_between(timestamps, 0, annualized,
                    where=annualized < 0, alpha=0.2, color=COLORS["secondary"])
    ax.set_title(f"{symbol} Annualized Funding Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Annualized Rate (%)")
    ax.grid(True, alpha=0.3)

    # Distribution
    ax = axes[0, 1]
    ax.hist(rates * 10000, bins=40, color=COLORS["primary"], alpha=0.7,
            edgecolor="white", density=True)
    ax.axvline(x=np.mean(rates) * 10000, color=COLORS["secondary"],
               linewidth=2, label=f"Mean: {np.mean(rates)*10000:.2f} bps")
    ax.set_title("Funding Rate Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Funding Rate (bps)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Autocorrelation
    ax = axes[1, 0]
    max_lag = min(50, len(rates) // 2)
    autocorrs = np.array([
        np.corrcoef(rates[:-lag], rates[lag:])[0, 1]
        for lag in range(1, max_lag + 1)
    ])
    ax.bar(range(1, max_lag + 1), autocorrs, color=COLORS["primary"], alpha=0.7)
    ax.set_title("Autocorrelation", fontsize=12, fontweight="bold")
    ax.set_xlabel("Lag (8h periods)")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True, alpha=0.3)

    # Rolling statistics
    ax = axes[1, 1]
    window = min(30, len(rates) // 3)
    if len(rates) >= window:
        rolling_mean = np.convolve(rates, np.ones(window) / window, mode="valid") * 10000
        rolling_std = np.array([
            np.std(rates[max(0, i - window):i]) * 10000
            for i in range(window, len(rates))
        ])
        x_vals = range(window, len(rates))
        ax.plot(list(x_vals), rolling_mean[:len(list(x_vals))],
                color=COLORS["primary"], label="Rolling Mean")
        ax.fill_between(list(x_vals),
                        rolling_mean[:len(list(x_vals))] - 2 * rolling_std[:len(list(x_vals))],
                        rolling_mean[:len(list(x_vals))] + 2 * rolling_std[:len(list(x_vals))],
                        alpha=0.2, color=COLORS["primary"])
    ax.set_title(f"Rolling Stats (window={window})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Period")
    ax.set_ylabel("Rate (bps)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        _ensure_save_dir()
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


def plot_greeks_surface(
    r_range: Tuple[float, float],
    T_range: Tuple[float, float],
    greeks_fn,
    greek_name: str = "duration",
    n_r: int = 40,
    n_T: int = 40,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Heatmap of a bond Greek as a function of rate and maturity.

    Args:
        r_range: (r_min, r_max)
        T_range: (T_min, T_max)
        greeks_fn: Function(r, t, T) -> dict with Greek values
        greek_name: Which Greek to plot
        n_r: Grid resolution for rate
        n_T: Grid resolution for maturity
        title: Plot title
        save_path: Path to save figure
    """
    r_vals = np.linspace(r_range[0], r_range[1], n_r)
    T_vals = np.linspace(T_range[0], T_range[1], n_T)

    Z = np.zeros((n_T, n_r))
    for i, T in enumerate(T_vals):
        for j, r in enumerate(r_vals):
            greeks = greeks_fn(r, 0.0, T)
            Z[i, j] = greeks.get(greek_name, 0.0)

    if title is None:
        title = f"Bond {greek_name.capitalize()} Surface"

    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    c = ax.pcolormesh(r_vals * 100, T_vals, Z, cmap="RdYlBu_r", shading="auto")
    fig.colorbar(c, ax=ax, label=greek_name.capitalize())
    ax.set_xlabel("Short Rate (%)", fontsize=12)
    ax.set_ylabel("Maturity (years)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        _ensure_save_dir()
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


def create_summary_dashboard(
    maturities: np.ndarray,
    market_rates: np.ndarray,
    pinn_rates: Optional[np.ndarray],
    analytical_rates: np.ndarray,
    history: Optional[Dict],
    r0: float,
    funding_rates: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """
    Create a comprehensive summary dashboard.

    Args:
        maturities: Yield curve maturities
        market_rates: Market zero rates
        pinn_rates: PINN-implied zero rates (optional)
        analytical_rates: Analytical zero rates
        history: Training history dict
        r0: Current short rate
        funding_rates: Bybit funding rate array (optional)
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12), dpi=DPI)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # 1. Yield curve comparison
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(maturities, market_rates * 100, "o-", color=COLORS["primary"],
             label="Market", linewidth=2, markersize=5)
    ax1.plot(maturities, analytical_rates * 100, "s--", color=COLORS["accent"],
             label="Analytical", linewidth=1.5)
    if pinn_rates is not None:
        ax1.plot(maturities, pinn_rates * 100, "^:", color=COLORS["secondary"],
                 label="PINN", linewidth=1.5)
    ax1.set_title("Yield Curve Comparison", fontweight="bold")
    ax1.set_xlabel("Maturity (years)")
    ax1.set_ylabel("Rate (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Model info
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    info_text = (
        f"Hull-White PINN\n\n"
        f"Short Rate: r0 = {r0:.2%}\n"
        f"Maturities: {len(maturities)}\n"
        f"Min Rate: {market_rates.min():.2%}\n"
        f"Max Rate: {market_rates.max():.2%}\n"
    )
    if history:
        info_text += (
            f"\nTraining:\n"
            f"Epochs: {len(history['epoch'])}\n"
            f"Final Loss: {history['total_loss'][-1]:.2e}\n"
        )
    ax2.text(0.1, 0.5, info_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor=COLORS["light"], alpha=0.8))
    ax2.set_title("Model Summary", fontweight="bold")

    # 3. Training loss (if available)
    if history:
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.semilogy(history["epoch"], history["total_loss"],
                     color=COLORS["dark"], linewidth=1)
        ax3.set_title("Training Loss", fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.semilogy(history["epoch"], history["pde_loss"],
                     color=COLORS["primary"], linewidth=1, label="PDE")
        ax4.semilogy(history["epoch"], history["data_loss"],
                     color=COLORS["secondary"], linewidth=1, label="Data")
        ax4.set_title("Loss Components", fontweight="bold")
        ax4.set_xlabel("Epoch")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    # 4. Pricing errors
    if pinn_rates is not None:
        ax5 = fig.add_subplot(gs[1, 2])
        errors = (pinn_rates - analytical_rates) * 10000
        ax5.bar(maturities, errors, width=0.3, color=COLORS["accent"], alpha=0.7)
        ax5.axhline(y=0, color="black", linewidth=0.5)
        ax5.set_title("PINN Error (bps)", fontweight="bold")
        ax5.set_xlabel("Maturity")
        ax5.grid(True, alpha=0.3)

    # 5. Funding rates (if available)
    if funding_rates is not None:
        ax6 = fig.add_subplot(gs[2, :])
        ann_rates = funding_rates * 3 * 365 * 100
        ax6.plot(ann_rates, color=COLORS["primary"], linewidth=0.8)
        ax6.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax6.set_title("Bybit Funding Rates (Annualized %)", fontweight="bold")
        ax6.set_xlabel("Period (8h intervals)")
        ax6.grid(True, alpha=0.3)

    fig.suptitle("Hull-White PINN Dashboard", fontsize=16, fontweight="bold", y=1.01)

    if save_path:
        _ensure_save_dir()
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    from analytical import FlatCurveHullWhite
    from data_loader import generate_synthetic_funding_rates

    print("Generating sample visualizations...")

    hw = FlatCurveHullWhite(a=0.1, sigma=0.01, flat_rate=0.03)
    r0 = 0.03

    # Yield curve
    mats, ylds = hw.yield_curve(r0)
    plot_yield_curve(mats, ylds, title="Hull-White Yield Curve (Flat)")

    # Bond surface
    plot_bond_surface(
        r_range=(-0.01, 0.08), T_range=(0.5, 10.0),
        pricing_fn=hw.bond_price,
        title="Hull-White Bond Price Surface"
    )

    # Rate paths
    times, paths = hw.simulate_rate_path(r0, T=5.0, n_steps=1260, n_paths=50, seed=42)
    mean_path = np.mean(paths, axis=0)
    lower = np.percentile(paths, 2.5, axis=0)
    upper = np.percentile(paths, 97.5, axis=0)
    plot_rate_paths(times, paths, mean_path, (lower, upper))

    # Funding rates
    funding_df = generate_synthetic_funding_rates(n_periods=500)
    plot_funding_rates(
        funding_df["datetime"].values,
        funding_df["funding_rate"].values,
        symbol="BTCUSDT"
    )

    print("Done.")
