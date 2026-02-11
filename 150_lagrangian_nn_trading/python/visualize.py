"""
Visualization utilities for Lagrangian Neural Network trading.

Includes:
  - Configuration space (q, q-dot) phase portraits
  - Lagrangian landscape visualization
  - Energy conservation plots
  - Trajectory prediction comparison
  - Training history curves
  - Mass matrix visualization
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, List


def plot_config_space(
    q: np.ndarray,
    qdot: np.ndarray,
    title: str = "Configuration Space (q, q-dot)",
    color_by: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    save_path: Optional[str] = None,
):
    """
    Plot 2D configuration space (q vs q-dot).

    Args:
        q: Generalized coordinates (N,) or (N, 1).
        qdot: Generalized velocities (N,) or (N, 1).
        title: Plot title.
        color_by: Optional array for coloring points (e.g., time, energy).
        cmap: Colormap name.
        save_path: Path to save figure.
    """
    q_flat = q.ravel() if q.ndim > 1 else q
    qdot_flat = qdot.ravel() if qdot.ndim > 1 else qdot

    fig, ax = plt.subplots(figsize=(10, 8))

    if color_by is not None:
        sc = ax.scatter(
            q_flat, qdot_flat, c=color_by, cmap=cmap,
            s=3, alpha=0.6, rasterized=True
        )
        plt.colorbar(sc, ax=ax, label="Color value")
    else:
        ax.scatter(
            q_flat, qdot_flat, c="steelblue",
            s=3, alpha=0.5, rasterized=True
        )

    ax.set_xlabel("q (position / price deviation)", fontsize=12)
    ax.set_ylabel("q-dot (velocity)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_lagrangian_landscape(
    model,
    q_range: tuple = (-3, 3),
    qdot_range: tuple = (-3, 3),
    n_points: int = 100,
    title: str = "Learned Lagrangian L(q, q-dot)",
    save_path: Optional[str] = None,
):
    """
    Visualize the learned Lagrangian as a 2D heatmap.

    Args:
        model: LNN model with lagrangian() method.
        q_range: Range for q axis.
        qdot_range: Range for q-dot axis.
        n_points: Resolution per axis.
        title: Plot title.
        save_path: Path to save figure.
    """
    q_grid = np.linspace(q_range[0], q_range[1], n_points)
    qdot_grid = np.linspace(qdot_range[0], qdot_range[1], n_points)
    Q, QDOT = np.meshgrid(q_grid, qdot_grid)

    q_flat = torch.FloatTensor(Q.ravel()).unsqueeze(1)
    qdot_flat = torch.FloatTensor(QDOT.ravel()).unsqueeze(1)

    with torch.no_grad():
        L = model.lagrangian(q_flat, qdot_flat)

    L_grid = L.numpy().reshape(n_points, n_points)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Heatmap
    im = axes[0].imshow(
        L_grid, extent=[q_range[0], q_range[1], qdot_range[0], qdot_range[1]],
        origin="lower", cmap="RdBu_r", aspect="auto"
    )
    axes[0].set_xlabel("q (position)", fontsize=12)
    axes[0].set_ylabel("q-dot (velocity)", fontsize=12)
    axes[0].set_title(title, fontsize=14)
    plt.colorbar(im, ax=axes[0], label="L(q, q-dot)")

    # Contour
    axes[1].contour(Q, QDOT, L_grid, levels=30, cmap="RdBu_r")
    axes[1].set_xlabel("q (position)", fontsize=12)
    axes[1].set_ylabel("q-dot (velocity)", fontsize=12)
    axes[1].set_title("Lagrangian Contours", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_energy_conservation(
    energies: np.ndarray,
    title: str = "Energy Conservation: E = q-dot * dL/dq-dot - L",
    save_path: Optional[str] = None,
):
    """
    Plot energy along a trajectory to verify conservation.

    Args:
        energies: Array of energy values along trajectory.
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Absolute energy
    axes[0].plot(energies, "b-", linewidth=1, alpha=0.8)
    axes[0].axhline(y=energies[0], color="r", linestyle="--", alpha=0.5, label="Initial E")
    axes[0].set_ylabel("Energy E", fontsize=12)
    axes[0].set_title(title, fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Relative drift
    drift = (energies - energies[0]) / (abs(energies[0]) + 1e-10) * 100
    axes[1].plot(drift, "r-", linewidth=1, alpha=0.8)
    axes[1].axhline(y=0, color="k", linewidth=0.5)
    axes[1].set_ylabel("Relative Drift (%)", fontsize=12)
    axes[1].set_xlabel("Integration Step", fontsize=12)
    axes[1].set_title(f"Energy Drift (final: {drift[-1]:.4f}%)", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_trajectory_comparison(
    q_true: np.ndarray,
    qdot_true: np.ndarray,
    q_pred: np.ndarray,
    qdot_pred: np.ndarray,
    title: str = "Trajectory: True vs. LNN Predicted",
    save_path: Optional[str] = None,
):
    """
    Compare true and predicted trajectories in configuration space.

    Args:
        q_true: True coordinates.
        qdot_true: True velocities.
        q_pred: Predicted coordinates.
        qdot_pred: Predicted velocities.
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Configuration space portrait
    ax = axes[0]
    ax.plot(q_true, qdot_true, "b-", linewidth=1.5, alpha=0.7, label="True")
    ax.plot(q_pred, qdot_pred, "r--", linewidth=1.5, alpha=0.7, label="LNN Predicted")
    ax.plot(q_true[0], qdot_true[0], "go", markersize=10, label="Start")
    ax.set_xlabel("q", fontsize=12)
    ax.set_ylabel("q-dot", fontsize=12)
    ax.set_title("Configuration Space", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # q(t)
    ax = axes[1]
    t = np.arange(len(q_true))
    ax.plot(t, q_true, "b-", linewidth=1.5, alpha=0.7, label="True")
    ax.plot(t[:len(q_pred)], q_pred, "r--", linewidth=1.5, alpha=0.7, label="Predicted")
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("q (position)", fontsize=12)
    ax.set_title("Position Over Time", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # q-dot(t)
    ax = axes[2]
    ax.plot(t, qdot_true, "b-", linewidth=1.5, alpha=0.7, label="True")
    ax.plot(t[:len(qdot_pred)], qdot_pred, "r--", linewidth=1.5, alpha=0.7, label="Predicted")
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("q-dot (velocity)", fontsize=12)
    ax.set_title("Velocity Over Time", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_training_history(
    history: list,
    title: str = "Training History",
    save_path: Optional[str] = None,
):
    """
    Plot training metrics over epochs.

    Args:
        history: List of dicts with 'epoch', 'train', 'val' keys.
        title: Plot title.
        save_path: Path to save figure.
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"].get("loss_total", h["train"].get("loss_accel", 0)) for h in history]
    val_loss = [h["val"].get("loss_total", 0) for h in history]
    lrs = [h.get("lr", 0) for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Loss
    axes[0].semilogy(epochs, train_loss, "b-", alpha=0.7, label="Train Loss")
    axes[0].semilogy(epochs, val_loss, "r-", alpha=0.7, label="Val Loss")
    axes[0].set_ylabel("Loss (log scale)", fontsize=12)
    axes[0].set_title(title, fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Learning rate
    axes[1].plot(epochs, lrs, "g-", alpha=0.7)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Learning Rate", fontsize=12)
    axes[1].set_title("Learning Rate Schedule", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_mass_matrix_spectrum(
    model,
    q_data: np.ndarray,
    qdot_data: np.ndarray,
    n_samples: int = 200,
    title: str = "Mass Matrix Eigenvalue Spectrum",
    save_path: Optional[str] = None,
):
    """
    Plot eigenvalue spectrum of the learned mass matrix M = d^2L/dq-dot^2.

    The mass matrix must be positive-definite for well-defined dynamics.
    """
    from model import compute_mass_matrix

    indices = np.random.choice(len(q_data), min(n_samples, len(q_data)), replace=False)
    all_eigenvalues = []

    for i in indices:
        q_i = torch.FloatTensor(q_data[i:i+1])
        qdot_i = torch.FloatTensor(qdot_data[i:i+1])

        with torch.enable_grad():
            M = compute_mass_matrix(model.l_net, q_i, qdot_i)
        eigenvalues = torch.linalg.eigvalsh(M).detach().numpy().ravel()
        all_eigenvalues.extend(eigenvalues)

    all_eigenvalues = np.array(all_eigenvalues)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    axes[0].hist(all_eigenvalues, bins=50, color="steelblue", alpha=0.7, edgecolor="k")
    axes[0].axvline(x=0, color="r", linestyle="--", linewidth=2, label="Zero line")
    axes[0].set_xlabel("Eigenvalue", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Eigenvalue Distribution", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sorted eigenvalues
    sorted_eigs = np.sort(all_eigenvalues)
    axes[1].plot(sorted_eigs, "b-", linewidth=0.5)
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Index (sorted)", fontsize=12)
    axes[1].set_ylabel("Eigenvalue", fontsize=12)
    axes[1].set_title("Sorted Eigenvalues", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    n_negative = (all_eigenvalues < 0).sum()
    n_total = len(all_eigenvalues)
    fig.suptitle(
        f"{title}\n"
        f"({n_negative}/{n_total} negative eigenvalues, "
        f"min={all_eigenvalues.min():.4f}, max={all_eigenvalues.max():.4f})",
        fontsize=14
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_lnn_vs_hnn_comparison(
    lnn_energies: np.ndarray,
    hnn_energies: np.ndarray,
    ode_energies: Optional[np.ndarray] = None,
    title: str = "Energy Conservation: LNN vs HNN vs Neural ODE",
    save_path: Optional[str] = None,
):
    """
    Compare energy conservation across different physics-informed models.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    n_steps = max(len(lnn_energies), len(hnn_energies))
    t = np.arange(n_steps)

    # Relative drift
    lnn_drift = (lnn_energies - lnn_energies[0]) / (abs(lnn_energies[0]) + 1e-10) * 100
    hnn_drift = (hnn_energies - hnn_energies[0]) / (abs(hnn_energies[0]) + 1e-10) * 100

    ax.plot(t[:len(lnn_drift)], lnn_drift, "b-", linewidth=2, alpha=0.8, label="LNN (This Chapter)")
    ax.plot(t[:len(hnn_drift)], hnn_drift, "g-", linewidth=2, alpha=0.8, label="HNN (Chapter 149)")

    if ode_energies is not None:
        ode_drift = (ode_energies - ode_energies[0]) / (abs(ode_energies[0]) + 1e-10) * 100
        ax.plot(t[:len(ode_drift)], ode_drift, "r-", linewidth=2, alpha=0.8, label="Neural ODE")

    ax.axhline(y=0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Integration Step", fontsize=12)
    ax.set_ylabel("Energy Drift (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig
