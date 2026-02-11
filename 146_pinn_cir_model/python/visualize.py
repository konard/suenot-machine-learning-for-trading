"""
Visualization for CIR PINN
============================

Plotting utilities for:
- Bond price surfaces (3D and heatmap)
- Yield curves at different rates
- CIR rate paths
- PINN vs analytical comparison
- PDE residual heatmaps
- Training loss curves
- Credit risk term structures
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, List, Dict, Tuple

from .analytical import (
    cir_bond_price,
    cir_yield,
    generate_yield_curve,
    simulate_cir_exact,
    simulate_cir_euler,
    feller_condition_check,
)


def plot_bond_surface(
    kappa: float,
    theta: float,
    sigma: float,
    T: float = 5.0,
    r_range: Tuple[float, float] = (0.001, 0.15),
    n_r: int = 50,
    n_t: int = 50,
    pinn=None,
    save_path: Optional[str] = None,
):
    """
    Plot 3D bond price surface P(r, tau).

    Parameters
    ----------
    kappa, theta, sigma : float
        CIR parameters.
    T : float
        Maximum maturity.
    r_range : tuple
        (r_min, r_max) for the rate axis.
    n_r, n_t : int
        Grid resolution.
    pinn : CIRPINN, optional
        If provided, also plot PINN surface.
    save_path : str, optional
        Path to save figure.
    """
    r_grid = np.linspace(r_range[0], r_range[1], n_r)
    tau_grid = np.linspace(0.01, T, n_t)
    R, TAU = np.meshgrid(r_grid, tau_grid)

    # Analytical surface
    P_analytical = np.zeros_like(R)
    for i in range(n_t):
        for j in range(n_r):
            P_analytical[i, j] = cir_bond_price(
                R[i, j], 0.0, TAU[i, j], kappa, theta, sigma
            )

    fig = plt.figure(figsize=(14, 5))

    # Analytical surface
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(R, TAU, P_analytical, cmap=cm.viridis, alpha=0.8)
    ax1.set_xlabel("Rate r")
    ax1.set_ylabel("Time to Maturity tau")
    ax1.set_zlabel("Bond Price P(r, tau)")
    ax1.set_title("CIR Bond Price (Analytical)")

    # PINN surface (if provided)
    if pinn is not None:
        import torch
        P_pinn = np.zeros_like(R)
        for i in range(n_t):
            for j in range(n_r):
                r_t = torch.tensor([R[i, j]], dtype=torch.float32)
                t_t = torch.tensor([T - TAU[i, j]], dtype=torch.float32)
                P_pinn[i, j] = pinn.predict(r_t, t_t).item()

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.plot_surface(R, TAU, P_pinn, cmap=cm.plasma, alpha=0.8)
        ax2.set_xlabel("Rate r")
        ax2.set_ylabel("Time to Maturity tau")
        ax2.set_zlabel("Bond Price P(r, tau)")
        ax2.set_title("CIR Bond Price (PINN)")
    else:
        # Heatmap view
        ax2 = fig.add_subplot(122)
        im = ax2.imshow(
            P_analytical, extent=[r_range[0], r_range[1], 0.01, T],
            aspect="auto", origin="lower", cmap=cm.viridis,
        )
        plt.colorbar(im, ax=ax2, label="Bond Price")
        ax2.set_xlabel("Rate r")
        ax2.set_ylabel("Time to Maturity tau")
        ax2.set_title("CIR Bond Price Heatmap")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_yield_curves(
    kappa: float,
    theta: float,
    sigma: float,
    rates: List[float] = None,
    max_maturity: float = 30.0,
    n_maturities: int = 100,
    save_path: Optional[str] = None,
):
    """
    Plot CIR yield curves at different short rates.

    Parameters
    ----------
    kappa, theta, sigma : float
        CIR parameters.
    rates : list of float
        Short rates to plot curves for.
    max_maturity : float
        Maximum maturity (years).
    n_maturities : int
        Number of maturity points.
    save_path : str, optional
        Path to save figure.
    """
    if rates is None:
        rates = [0.01, 0.03, 0.05, 0.07, 0.10]

    maturities = np.linspace(0.1, max_maturity, n_maturities)

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in rates:
        yields = generate_yield_curve(r, maturities, kappa, theta, sigma) * 100
        ax.plot(maturities, yields, label=f"r = {r:.1%}", linewidth=2)

    # Long-run yield
    gamma = np.sqrt(kappa ** 2 + 2 * sigma ** 2)
    y_inf = 2 * kappa * theta / (kappa + gamma) * 100
    ax.axhline(y=y_inf, color="gray", linestyle="--", alpha=0.7,
               label=f"Long-run yield = {y_inf:.2f}%")

    ax.axhline(y=theta * 100, color="red", linestyle=":", alpha=0.5,
               label=f"theta = {theta:.2%}")

    ax.set_xlabel("Maturity (years)", fontsize=12)
    ax.set_ylabel("Yield (%)", fontsize=12)
    ax.set_title(
        f"CIR Yield Curves (kappa={kappa}, theta={theta:.2%}, sigma={sigma})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_rate_paths(
    kappa: float,
    theta: float,
    sigma: float,
    r0: float = 0.03,
    T: float = 10.0,
    n_steps: int = 2520,
    n_paths: int = 10,
    method: str = "exact",
    save_path: Optional[str] = None,
):
    """
    Plot simulated CIR rate paths.

    Parameters
    ----------
    kappa, theta, sigma : float
        CIR parameters.
    r0 : float
        Initial rate.
    T : float
        Time horizon.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths to plot.
    method : str
        'exact' for non-central chi-squared, 'euler' for Euler-Maruyama.
    save_path : str, optional
        Path to save figure.
    """
    if method == "exact":
        paths = simulate_cir_exact(kappa, theta, sigma, r0, T, n_steps, n_paths, seed=42)
    else:
        paths = simulate_cir_euler(kappa, theta, sigma, r0, T, n_steps, n_paths, seed=42)

    time_grid = np.linspace(0, T, n_steps + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    # Rate paths
    for i in range(n_paths):
        ax1.plot(time_grid, paths[i] * 100, alpha=0.6, linewidth=0.8)

    ax1.axhline(y=theta * 100, color="red", linestyle="--", linewidth=2,
                label=f"theta = {theta:.2%}")
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Feller check
    feller = feller_condition_check(kappa, theta, sigma)
    feller_str = "Satisfied" if feller["satisfied"] else "VIOLATED"

    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Rate (%)")
    ax1.set_title(
        f"CIR Rate Paths ({method.title()}) | "
        f"kappa={kappa}, theta={theta:.2%}, sigma={sigma} | "
        f"Feller: {feller_str}",
        fontsize=12,
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Distribution of final rates
    final_rates = paths[:, -1] * 100
    ax2.hist(final_rates, bins=30, density=True, alpha=0.7, color="steelblue",
             edgecolor="white")
    ax2.axvline(x=theta * 100, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Final Rate (%)")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of r(T)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pinn_vs_analytical(
    pinn,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_points: int = 200,
    save_path: Optional[str] = None,
):
    """
    Plot PINN predictions vs analytical CIR bond prices.

    Parameters
    ----------
    pinn : CIRPINN
        Trained PINN model.
    kappa, theta, sigma, T : float
        CIR parameters.
    n_points : int
        Number of test points.
    save_path : str, optional
        Path to save figure.
    """
    import torch

    r_test = np.linspace(0.001, 0.15, n_points)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Different time slices
    time_fractions = [0.0, 0.25, 0.5, 0.75]
    titles = ["t = 0", "t = T/4", "t = T/2", "t = 3T/4"]

    for ax, t_frac, title in zip(axes.flatten(), time_fractions, titles):
        t_val = t_frac * T

        # Analytical
        P_analytical = np.array([
            cir_bond_price(r, t_val, T, kappa, theta, sigma)
            for r in r_test
        ])

        # PINN
        r_tensor = torch.tensor(r_test, dtype=torch.float32)
        t_tensor = torch.full_like(r_tensor, t_val)
        P_pinn = pinn.predict(r_tensor, t_tensor).numpy()

        error = np.abs(P_pinn - P_analytical)

        ax.plot(r_test * 100, P_analytical, "b-", linewidth=2, label="Analytical")
        ax.plot(r_test * 100, P_pinn, "r--", linewidth=2, label="PINN")

        ax_err = ax.twinx()
        ax_err.fill_between(r_test * 100, error * 1e4, alpha=0.2, color="green")
        ax_err.set_ylabel("Error (x1e-4)", color="green")

        ax.set_xlabel("Rate (%)")
        ax.set_ylabel("Bond Price")
        ax.set_title(f"{title} (tau = {T - t_val:.1f}y)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"PINN vs Analytical CIR Bond Prices "
        f"(kappa={kappa}, theta={theta:.2%}, sigma={sigma})",
        fontsize=13,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_loss(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """
    Plot training loss curves.

    Parameters
    ----------
    history : dict
        Loss history from train_pinn().
    save_path : str, optional
        Path to save figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total loss
    ax1 = axes[0]
    ax1.semilogy(history["total"], linewidth=1.5, color="black", label="Total")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (log scale)")
    ax1.set_title("Total Training Loss")
    ax1.grid(True, alpha=0.3)

    # Individual components
    ax2 = axes[1]
    components = {
        "pde": ("PDE Residual", "blue"),
        "terminal": ("Terminal Condition", "red"),
        "lower_bc": ("Lower BC", "green"),
        "upper_bc": ("Upper BC", "orange"),
        "analytical": ("Analytical", "purple"),
    }

    for key, (label, color) in components.items():
        if key in history and len(history[key]) > 0:
            vals = np.array(history[key])
            if np.any(vals > 0):
                ax2.semilogy(vals, linewidth=1, color=color, alpha=0.8, label=label)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (log scale)")
    ax2.set_title("Loss Components")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(
    kappa: float = 0.5,
    theta: float = 0.05,
    sigma_cir: float = 0.1,
    sigma_vasicek: float = 0.02,
    r0: float = 0.03,
    T: float = 10.0,
    n_paths: int = 1000,
    save_path: Optional[str] = None,
):
    """
    Compare CIR vs Vasicek paths side by side.

    Parameters
    ----------
    kappa, theta : float
        Common mean-reversion parameters.
    sigma_cir, sigma_vasicek : float
        Volatility for CIR and Vasicek respectively.
    r0 : float
        Initial rate.
    T : float
        Time horizon.
    n_paths : int
        Number of simulation paths.
    save_path : str, optional
        Path to save figure.
    """
    n_steps = int(T * 252)

    # CIR paths (exact)
    cir_paths = simulate_cir_exact(kappa, theta, sigma_cir, r0, T, n_steps, n_paths, seed=42)

    # Vasicek paths (Euler)
    np.random.seed(42)
    vasicek_paths = np.zeros((n_paths, n_steps + 1))
    vasicek_paths[:, 0] = r0
    dt = T / n_steps
    for i in range(n_steps):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        vasicek_paths[:, i + 1] = (
            vasicek_paths[:, i]
            + kappa * (theta - vasicek_paths[:, i]) * dt
            + sigma_vasicek * dW
        )

    time_grid = np.linspace(0, T, n_steps + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # CIR sample paths
    ax1 = axes[0, 0]
    for i in range(min(20, n_paths)):
        ax1.plot(time_grid, cir_paths[i] * 100, alpha=0.4, linewidth=0.5)
    ax1.axhline(y=theta * 100, color="red", linestyle="--")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_title(f"CIR Paths (sigma={sigma_cir})")
    ax1.set_ylabel("Rate (%)")
    ax1.grid(True, alpha=0.3)

    # Vasicek sample paths
    ax2 = axes[0, 1]
    for i in range(min(20, n_paths)):
        ax2.plot(time_grid, vasicek_paths[i] * 100, alpha=0.4, linewidth=0.5)
    ax2.axhline(y=theta * 100, color="red", linestyle="--")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_title(f"Vasicek Paths (sigma={sigma_vasicek})")
    ax2.set_ylabel("Rate (%)")
    ax2.grid(True, alpha=0.3)

    # Distribution comparison at final time
    ax3 = axes[1, 0]
    ax3.hist(cir_paths[:, -1] * 100, bins=50, density=True, alpha=0.6,
             color="blue", label="CIR", edgecolor="white")
    ax3.hist(vasicek_paths[:, -1] * 100, bins=50, density=True, alpha=0.6,
             color="orange", label="Vasicek", edgecolor="white")
    ax3.axvline(x=0, color="black", linewidth=1)
    ax3.set_xlabel("Rate (%)")
    ax3.set_ylabel("Density")
    ax3.set_title(f"Distribution at t={T}")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Negative rate fraction over time
    ax4 = axes[1, 1]
    cir_neg = np.mean(cir_paths < 0, axis=0) * 100
    vas_neg = np.mean(vasicek_paths < 0, axis=0) * 100
    ax4.plot(time_grid, cir_neg, label="CIR", linewidth=2, color="blue")
    ax4.plot(time_grid, vas_neg, label="Vasicek", linewidth=2, color="orange")
    ax4.set_xlabel("Time (years)")
    ax4.set_ylabel("% Negative Paths")
    ax4.set_title("Fraction of Negative Rate Paths")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle("CIR vs Vasicek Model Comparison", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("Generating CIR visualizations...")

    # 1. Bond surface
    plot_bond_surface(kappa=0.5, theta=0.05, sigma=0.1, T=5.0)

    # 2. Yield curves
    plot_yield_curves(kappa=0.5, theta=0.05, sigma=0.1)

    # 3. Rate paths
    plot_rate_paths(kappa=0.5, theta=0.05, sigma=0.1, r0=0.03, T=10.0, n_paths=20)

    # 4. Model comparison
    plot_model_comparison()
