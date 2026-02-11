"""
Visualization Module for Neural ODE Trading

Provides visualization utilities for:
1. ODE trajectories in latent space
2. Phase portraits of continuous dynamics
3. Prediction vs actual comparisons
4. Latent space evolution
5. NFE (Number of Function Evaluations) analysis

All plots are saved to the 'plots/' directory within this chapter.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def ensure_plot_dir():
    """Create plots directory if it doesn't exist."""
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_trajectories(
    t: np.ndarray,
    trajectories: np.ndarray,
    true_values: Optional[np.ndarray] = None,
    title: str = "Neural ODE Trajectories",
    labels: Optional[List[str]] = None,
    save_name: str = "trajectories.png",
):
    """
    Plot ODE trajectories over time.

    Args:
        t: Time points, shape (n_time,)
        trajectories: Predicted trajectories, shape (n_time, n_dims) or (n_samples, n_time, n_dims)
        true_values: Ground truth, shape (n_time, n_dims)
        title: Plot title
        labels: Dimension labels
        save_name: Filename to save
    """
    ensure_plot_dir()
    fig, axes = plt.subplots(
        min(trajectories.shape[-1], 4), 1,
        figsize=(12, 3 * min(trajectories.shape[-1], 4)),
        sharex=True,
    )
    if not hasattr(axes, "__len__"):
        axes = [axes]

    n_dims = min(trajectories.shape[-1], len(axes))

    for dim in range(n_dims):
        ax = axes[dim]
        label = labels[dim] if labels else f"Dim {dim}"

        if trajectories.ndim == 3:
            # Multiple samples: plot mean and confidence interval
            mean = trajectories[:, :, dim].mean(axis=0)
            std = trajectories[:, :, dim].std(axis=0)
            ax.plot(t, mean, "b-", linewidth=1.5, label="Predicted (mean)")
            ax.fill_between(
                t, mean - 2 * std, mean + 2 * std,
                alpha=0.2, color="blue", label="95% CI",
            )
        else:
            ax.plot(t, trajectories[:, dim], "b-", linewidth=1.5, label="Predicted")

        if true_values is not None:
            ax.plot(t[:len(true_values)], true_values[:, dim], "r--",
                    linewidth=1.0, label="Actual", alpha=0.8)

        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {os.path.join(PLOT_DIR, save_name)}")


def plot_latent_space(
    z_trajectory: np.ndarray,
    title: str = "Latent Space Evolution",
    dim_pairs: Optional[List[Tuple[int, int]]] = None,
    save_name: str = "latent_space.png",
):
    """
    Plot the evolution of latent states z(t).

    Args:
        z_trajectory: Latent trajectory, shape (n_time, batch_size, latent_dim)
                      or (n_time, latent_dim)
        title: Plot title
        dim_pairs: Pairs of dimensions to plot against each other
        save_name: Filename to save
    """
    ensure_plot_dir()

    if z_trajectory.ndim == 3:
        # Average over batch
        z = z_trajectory[:, 0, :]  # Take first sample
    else:
        z = z_trajectory

    latent_dim = z.shape[-1]

    if dim_pairs is None:
        n_pairs = min(latent_dim // 2, 4)
        dim_pairs = [(2 * i, 2 * i + 1) for i in range(n_pairs)]
        if not dim_pairs:
            dim_pairs = [(0, min(1, latent_dim - 1))]

    fig, axes = plt.subplots(1, len(dim_pairs), figsize=(5 * len(dim_pairs), 5))
    if not hasattr(axes, "__len__"):
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, len(z)))

    for idx, (d1, d2) in enumerate(dim_pairs):
        ax = axes[idx]
        for i in range(len(z) - 1):
            ax.plot(
                z[i : i + 2, d1], z[i : i + 2, d2],
                color=colors[i], linewidth=0.8,
            )

        # Mark start and end
        ax.scatter(z[0, d1], z[0, d2], color="green", s=100, zorder=5,
                   marker="o", label="Start")
        ax.scatter(z[-1, d1], z[-1, d2], color="red", s=100, zorder=5,
                   marker="s", label="End")

        ax.set_xlabel(f"z_{d1}")
        ax.set_ylabel(f"z_{d2}")
        ax.legend(fontsize=8)
        ax.set_title(f"Dims ({d1}, {d2})")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {os.path.join(PLOT_DIR, save_name)}")


def plot_phase_portrait(
    model: torch.nn.Module,
    hidden_dim: int = 2,
    grid_range: Tuple[float, float] = (-3.0, 3.0),
    n_points: int = 20,
    title: str = "Phase Portrait of ODE Dynamics",
    save_name: str = "phase_portrait.png",
):
    """
    Plot the vector field (phase portrait) of the ODE dynamics.

    Shows dh/dt as arrows in a 2D slice of the state space,
    revealing fixed points, limit cycles, and flow structure.

    Args:
        model: The ODEFunc network
        hidden_dim: Total hidden dimension (we plot first 2 dims)
        grid_range: Range for the grid
        n_points: Number of grid points per axis
        title: Plot title
        save_name: Filename
    """
    ensure_plot_dir()

    x = np.linspace(grid_range[0], grid_range[1], n_points)
    y = np.linspace(grid_range[0], grid_range[1], n_points)
    X, Y = np.meshgrid(x, y)

    # Compute vector field
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    model.eval()
    with torch.no_grad():
        for i in range(n_points):
            for j in range(n_points):
                h = torch.zeros(1, hidden_dim)
                h[0, 0] = X[i, j]
                if hidden_dim > 1:
                    h[0, 1] = Y[i, j]

                t_val = torch.tensor(0.5)
                dh = model(t_val, h)

                U[i, j] = dh[0, 0].item()
                V[i, j] = dh[0, 1].item() if hidden_dim > 1 else 0

    # Normalize arrows for visibility
    speed = np.sqrt(U ** 2 + V ** 2)
    speed_max = speed.max() + 1e-8
    U_norm = U / speed_max
    V_norm = V / speed_max

    fig, ax = plt.subplots(figsize=(8, 8))
    strm = ax.streamplot(
        X, Y, U, V,
        color=speed, cmap="coolwarm",
        linewidth=1.5, density=1.5, arrowsize=1.5,
    )
    fig.colorbar(strm.lines, label="Speed |dh/dt|")

    ax.set_xlabel("h_0")
    ax.set_ylabel("h_1")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(grid_range)
    ax.set_ylim(grid_range)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {os.path.join(PLOT_DIR, save_name)}")


def plot_predictions(
    timestamps: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    upper_bound: Optional[np.ndarray] = None,
    lower_bound: Optional[np.ndarray] = None,
    title: str = "Neural ODE Price Predictions",
    save_name: str = "predictions.png",
):
    """
    Plot actual vs predicted prices with optional confidence intervals.

    Args:
        timestamps: Time points (datetime or numeric)
        actual: Actual prices, shape (n,)
        predicted: Predicted prices, shape (n,)
        upper_bound: Upper confidence bound
        lower_bound: Lower confidence bound
        title: Plot title
        save_name: Filename
    """
    ensure_plot_dir()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Price plot
    ax1 = axes[0]
    ax1.plot(timestamps, actual, "k-", linewidth=1.0, label="Actual", alpha=0.9)
    ax1.plot(timestamps, predicted, "b-", linewidth=1.0, label="Predicted", alpha=0.8)

    if upper_bound is not None and lower_bound is not None:
        ax1.fill_between(
            timestamps, lower_bound, upper_bound,
            alpha=0.2, color="blue", label="95% CI",
        )

    ax1.set_ylabel("Price")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")

    # Error plot
    ax2 = axes[1]
    error = predicted - actual
    ax2.bar(timestamps, error, width=0.8, color=np.where(error > 0, "green", "red"), alpha=0.6)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("Prediction Error")
    ax2.set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {os.path.join(PLOT_DIR, save_name)}")


def plot_training_history(
    history: Dict[str, list],
    title: str = "Training History",
    save_name: str = "training_history.png",
):
    """
    Plot training and test loss curves.

    Args:
        history: Dictionary with 'train_loss', 'test_loss', and optionally 'nfe'
        title: Plot title
        save_name: Filename
    """
    ensure_plot_dir()

    n_plots = 1 + ("nfe" in history) + ("kl_loss" in history)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if not hasattr(axes, "__len__"):
        axes = [axes]

    # Loss curves
    ax = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax.plot(epochs, history["test_loss"], "r-", label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.set_yscale("log")

    plot_idx = 1

    # NFE plot (if available)
    if "nfe" in history:
        ax = axes[plot_idx]
        ax.plot(epochs, history["nfe"], "g-", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg NFE")
        ax.set_title("Function Evaluations (ODE Solver)")
        plot_idx += 1

    # KL loss (for Latent ODE)
    if "kl_loss" in history:
        ax = axes[plot_idx]
        ax.plot(epochs, history["recon_loss"], "b-", label="Reconstruction")
        ax.plot(epochs, history["kl_loss"], "r-", label="KL Divergence")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("VAE Loss Components")
        ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {os.path.join(PLOT_DIR, save_name)}")


def plot_irregular_timestamps(
    timestamps: np.ndarray,
    values: np.ndarray,
    title: str = "Irregularly-Sampled Time Series",
    save_name: str = "irregular_timestamps.png",
):
    """
    Visualize the irregularity of time series data.

    Shows:
    - The time series with points at irregular intervals
    - Distribution of time gaps
    - Comparison with regular grid

    Args:
        timestamps: Time points (seconds or datetime)
        values: Observed values
        title: Plot title
        save_name: Filename
    """
    ensure_plot_dir()

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)

    # Time series with irregular points
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(timestamps, values, s=8, alpha=0.6, c="steelblue", zorder=3)
    ax1.plot(timestamps, values, "b-", alpha=0.2, linewidth=0.5)
    ax1.set_ylabel("Value")
    ax1.set_title(f"{title} ({len(timestamps)} observations)")

    # Time gap distribution
    if isinstance(timestamps[0], (int, float, np.integer, np.floating)):
        time_gaps = np.diff(timestamps)
    else:
        time_gaps = np.array([(timestamps[i+1] - timestamps[i]).total_seconds()
                              for i in range(len(timestamps) - 1)])

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(time_gaps, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Time Gap (seconds)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Time Gaps")
    ax2.axvline(np.median(time_gaps), color="red", linestyle="--",
                label=f"Median: {np.median(time_gaps):.1f}s")
    ax2.legend()

    # Time gaps over time
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(range(len(time_gaps)), time_gaps, s=3, alpha=0.4, c="steelblue")
    ax3.set_xlabel("Observation Index")
    ax3.set_ylabel("Gap to Next (seconds)")
    ax3.set_title("Time Gap Sequence")
    ax3.axhline(np.mean(time_gaps), color="red", linestyle="--",
                label=f"Mean: {np.mean(time_gaps):.1f}s")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {os.path.join(PLOT_DIR, save_name)}")


def plot_ode_solver_comparison(
    results: Dict[str, Dict],
    title: str = "ODE Solver Comparison",
    save_name: str = "solver_comparison.png",
):
    """
    Compare different ODE solvers (accuracy, speed, NFE).

    Args:
        results: Dict mapping solver name -> {'loss': float, 'nfe': float, 'time': float}
        title: Plot title
        save_name: Filename
    """
    ensure_plot_dir()

    solvers = list(results.keys())
    losses = [results[s]["loss"] for s in solvers]
    nfes = [results[s]["nfe"] for s in solvers]
    times = [results[s]["time"] for s in solvers]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(solvers)))

    # Loss comparison
    axes[0].bar(solvers, losses, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Test Loss")
    axes[0].set_title("Accuracy")
    axes[0].tick_params(axis="x", rotation=45)

    # NFE comparison
    axes[1].bar(solvers, nfes, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Avg NFE per batch")
    axes[1].set_title("Solver Complexity")
    axes[1].tick_params(axis="x", rotation=45)

    # Time comparison
    axes[2].bar(solvers, times, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].set_ylabel("Time per epoch (sec)")
    axes[2].set_title("Speed")
    axes[2].tick_params(axis="x", rotation=45)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {os.path.join(PLOT_DIR, save_name)}")
