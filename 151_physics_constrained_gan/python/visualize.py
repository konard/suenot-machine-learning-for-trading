"""
Visualization Module for Physics-Constrained GAN

Provides comprehensive comparison plots between real and synthetic financial data,
including distribution plots, ACF analysis, QQ-plots, and training diagnostics.

Visualizations:
    - Return distribution comparison (histogram + KDE)
    - Sample path comparison
    - Autocorrelation function of returns, |returns|, and returns^2
    - QQ-plot for tail behavior
    - Training loss curves (adversarial + physics)
    - Stylized facts comparison table
    - Conditional generation visualization by regime
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False
    print("matplotlib not available. Install with: pip install matplotlib")


def _check_plt():
    if not PLT_AVAILABLE:
        raise RuntimeError("matplotlib required for visualization")


def plot_real_vs_synthetic(
    real_returns: np.ndarray,
    synthetic_returns: np.ndarray,
    title: str = "Physics-Constrained GAN: Real vs Synthetic",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Comprehensive comparison of real and synthetic return distributions.

    Creates a 2x3 grid:
        - Return distributions (histogram)
        - Sample paths (cumulative returns)
        - ACF of absolute returns
        - QQ-plot
        - ACF of squared returns
        - Summary statistics

    Args:
        real_returns: Real return arrays [n_samples, seq_len] or [seq_len]
        synthetic_returns: Synthetic return arrays [n_samples, seq_len] or [seq_len]
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    _check_plt()

    if real_returns.ndim == 1:
        real_returns = real_returns.reshape(1, -1)
    if synthetic_returns.ndim == 1:
        synthetic_returns = synthetic_returns.reshape(1, -1)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    real_flat = real_returns.flatten()
    synth_flat = synthetic_returns.flatten()

    # 1. Return distributions
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(
        min(real_flat.min(), synth_flat.min()),
        max(real_flat.max(), synth_flat.max()),
        80,
    )
    ax1.hist(real_flat, bins=bins, alpha=0.5, density=True, label="Real", color="steelblue")
    ax1.hist(synth_flat, bins=bins, alpha=0.5, density=True, label="Synthetic", color="coral")
    ax1.set_title("Return Distribution")
    ax1.set_xlabel("Log Return")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Sample paths (cumulative returns)
    ax2 = fig.add_subplot(gs[0, 1])
    n_show = min(5, len(real_returns), len(synthetic_returns))
    for i in range(n_show):
        real_cum = np.cumprod(1 + real_returns[i])
        synth_cum = np.cumprod(1 + synthetic_returns[i])
        ax2.plot(real_cum, color="steelblue", alpha=0.6,
                 label="Real" if i == 0 else None)
        ax2.plot(synth_cum, color="coral", alpha=0.6, linestyle="--",
                 label="Synthetic" if i == 0 else None)
    ax2.set_title("Sample Price Paths")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. ACF of absolute returns
    ax3 = fig.add_subplot(gs[0, 2])
    max_lag = 30
    real_acf_abs = _compute_avg_acf(np.abs(real_returns), max_lag)
    synth_acf_abs = _compute_avg_acf(np.abs(synthetic_returns), max_lag)
    lags = np.arange(1, max_lag + 1)
    ax3.bar(lags - 0.15, real_acf_abs, width=0.3, alpha=0.7, label="Real", color="steelblue")
    ax3.bar(lags + 0.15, synth_acf_abs, width=0.3, alpha=0.7, label="Synthetic", color="coral")
    ax3.set_title("ACF of |Returns|")
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("Autocorrelation")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. QQ-plot
    ax4 = fig.add_subplot(gs[1, 0])
    _qq_plot(ax4, real_flat, synth_flat)
    ax4.set_title("QQ-Plot (Real vs Synthetic)")
    ax4.grid(True, alpha=0.3)

    # 5. ACF of squared returns
    ax5 = fig.add_subplot(gs[1, 1])
    real_acf_sq = _compute_avg_acf(real_returns ** 2, max_lag)
    synth_acf_sq = _compute_avg_acf(synthetic_returns ** 2, max_lag)
    ax5.bar(lags - 0.15, real_acf_sq, width=0.3, alpha=0.7, label="Real", color="steelblue")
    ax5.bar(lags + 0.15, synth_acf_sq, width=0.3, alpha=0.7, label="Synthetic", color="coral")
    ax5.set_title("ACF of Returns^2")
    ax5.set_xlabel("Lag")
    ax5.set_ylabel("Autocorrelation")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    stats_table = _compute_comparison_stats(real_flat, synth_flat)
    table = ax6.table(
        cellText=stats_table["data"],
        colLabels=stats_table["columns"],
        rowLabels=stats_table["rows"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax6.set_title("Summary Statistics", fontweight="bold", pad=20)

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot training loss curves including physics constraint decomposition.

    Args:
        history: Dict from train_physics_gan with loss lists
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    _check_plt()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Physics-Constrained GAN Training History", fontsize=14, fontweight="bold")

    epochs = range(1, len(history.get("critic_loss", [])) + 1)

    # Critic loss
    ax = axes[0, 0]
    if "critic_loss" in history:
        ax.plot(epochs, history["critic_loss"], color="steelblue", linewidth=0.8)
    ax.set_title("Critic Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # Wasserstein distance
    ax = axes[0, 1]
    if "wasserstein_distance" in history:
        ax.plot(epochs, history["wasserstein_distance"], color="darkgreen", linewidth=0.8)
    ax.set_title("Wasserstein Distance")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # Generator loss
    ax = axes[0, 2]
    if "gen_loss" in history:
        ax.plot(epochs, history["gen_loss"], color="coral", linewidth=0.8)
    ax.set_title("Generator Loss (total)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # Physics losses
    ax = axes[1, 0]
    physics_keys = ["martingale", "volatility_clustering", "kurtosis"]
    colors = ["navy", "darkgreen", "darkred"]
    for key, color in zip(physics_keys, colors):
        if key in history:
            ax.plot(epochs, history[key], label=key, linewidth=0.8, color=color)
    ax.set_title("Physics Losses (1)")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    physics_keys2 = ["leverage_effect", "autocorrelation"]
    colors2 = ["purple", "darkorange"]
    for key, color in zip(physics_keys2, colors2):
        if key in history:
            ax.plot(epochs, history[key], label=key, linewidth=0.8, color=color)
    ax.set_title("Physics Losses (2)")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Total physics loss
    ax = axes[1, 2]
    if "total_physics" in history:
        ax.plot(epochs, history["total_physics"], color="black", linewidth=0.8)
    ax.set_title("Total Physics Penalty")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_conditional_generation(
    gan,
    n_paths: int = 20,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Visualize conditional generation across different regimes and volatility levels.

    Args:
        gan: PhysicsConstrainedGAN instance
        n_paths: Number of paths per condition
        save_path: Path to save figure
        show: Whether to display
    """
    _check_plt()

    regimes = ["bull", "bear", "sideways", "crisis"]
    vol_levels = ["low", "medium", "high", "extreme"]

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle("Conditional Generation: Regime x Volatility Level",
                 fontsize=14, fontweight="bold")

    for i, regime in enumerate(regimes):
        for j, vol in enumerate(vol_levels):
            ax = axes[i][j]
            paths = gan.generate(
                n_paths=n_paths,
                condition={"regime": regime, "volatility": vol},
                as_prices=True,
                initial_price=100.0,
            )

            for k in range(min(n_paths, 10)):
                ax.plot(paths[k], alpha=0.5, linewidth=0.5)

            ax.set_title(f"{regime} / {vol}", fontsize=9)
            if i == 3:
                ax.set_xlabel("Time")
            if j == 0:
                ax.set_ylabel("Price")
            ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_physics_diagnostics(
    real_returns: np.ndarray,
    synthetic_returns: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Detailed physics constraint diagnostics.

    Shows how well each constraint is satisfied:
        - Martingale: mean return distribution
        - Vol clustering: rolling volatility comparison
        - Fat tails: tail probability comparison (log scale)
        - Leverage effect: scatter of r_t vs |r_{t+1}|
    """
    _check_plt()

    if real_returns.ndim == 1:
        real_returns = real_returns.reshape(1, -1)
    if synthetic_returns.ndim == 1:
        synthetic_returns = synthetic_returns.reshape(1, -1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Physics Constraint Diagnostics", fontsize=14, fontweight="bold")

    # 1. Martingale: distribution of path mean returns
    ax = axes[0, 0]
    real_means = real_returns.mean(axis=1)
    synth_means = synthetic_returns.mean(axis=1)
    ax.hist(real_means, bins=40, alpha=0.5, density=True, label="Real", color="steelblue")
    ax.hist(synth_means, bins=40, alpha=0.5, density=True, label="Synthetic", color="coral")
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Martingale: Mean Return Distribution")
    ax.set_xlabel("Mean Return per Path")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Volatility clustering: rolling vol
    ax = axes[0, 1]
    window = 20
    idx_real = 0
    idx_synth = 0
    real_vol = _rolling_std(real_returns[idx_real], window)
    synth_vol = _rolling_std(synthetic_returns[idx_synth], window)
    ax.plot(real_vol, label="Real", color="steelblue", alpha=0.8)
    ax.plot(synth_vol, label="Synthetic", color="coral", alpha=0.8)
    ax.set_title(f"Rolling Volatility (window={window})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Fat tails: log-scale tail probabilities
    ax = axes[1, 0]
    real_flat = real_returns.flatten()
    synth_flat = synthetic_returns.flatten()
    thresholds = np.linspace(0.5, 4.0, 50)
    real_std = np.std(real_flat)
    synth_std = np.std(synth_flat)

    real_probs = [np.mean(np.abs(real_flat / real_std) > t) for t in thresholds]
    synth_probs = [np.mean(np.abs(synth_flat / synth_std) > t) for t in thresholds]

    ax.semilogy(thresholds, real_probs, label="Real", color="steelblue", linewidth=2)
    ax.semilogy(thresholds, synth_probs, label="Synthetic", color="coral", linewidth=2)
    # Normal reference
    from scipy.stats import norm
    normal_probs = [2 * (1 - norm.cdf(t)) for t in thresholds]
    ax.semilogy(thresholds, normal_probs, label="Normal", color="gray",
                linestyle="--", linewidth=1)
    ax.set_title("Tail Probabilities P(|r/sigma| > threshold)")
    ax.set_xlabel("Threshold (in std units)")
    ax.set_ylabel("Probability (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Leverage effect scatter
    ax = axes[1, 1]
    # Use first few paths for scatter
    n_scatter = min(5, len(real_returns), len(synthetic_returns))
    for i in range(n_scatter):
        r_real = real_returns[i, :-1]
        abs_next_real = np.abs(real_returns[i, 1:])
        ax.scatter(r_real, abs_next_real, alpha=0.1, s=2, color="steelblue",
                   label="Real" if i == 0 else None)

        r_synth = synthetic_returns[i, :-1]
        abs_next_synth = np.abs(synthetic_returns[i, 1:])
        ax.scatter(r_synth, abs_next_synth, alpha=0.1, s=2, color="coral",
                   label="Synthetic" if i == 0 else None)

    ax.set_title("Leverage Effect: r_t vs |r_{t+1}|")
    ax.set_xlabel("Return r_t")
    ax.set_ylabel("|r_{t+1}|")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()

    plt.close()


# ---- Helper functions ----

def _compute_avg_acf(data: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute average autocorrelation across sample paths."""
    if data.ndim == 1:
        data = data.reshape(1, -1)
    acf_values = np.zeros(max_lag)
    n_paths = min(len(data), 100)
    for i in range(n_paths):
        for k in range(max_lag):
            acf_values[k] += _acf_single(data[i], k + 1)
    return acf_values / n_paths


def _acf_single(x: np.ndarray, lag: int) -> float:
    """Compute autocorrelation at given lag."""
    n = len(x)
    if lag >= n:
        return 0.0
    mean = np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return 0.0
    cov = np.mean((x[:-lag] - mean) * (x[lag:] - mean))
    return cov / var


def _qq_plot(ax, real: np.ndarray, synthetic: np.ndarray):
    """Create QQ-plot comparing real and synthetic quantiles."""
    percentiles = np.linspace(0.5, 99.5, 200)
    real_q = np.percentile(real, percentiles)
    synth_q = np.percentile(synthetic, percentiles)

    ax.scatter(real_q, synth_q, s=5, alpha=0.5, color="steelblue")
    # 45-degree reference line
    min_val = min(real_q.min(), synth_q.min())
    max_val = max(real_q.max(), synth_q.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Real Quantiles")
    ax.set_ylabel("Synthetic Quantiles")


def _compute_comparison_stats(
    real: np.ndarray,
    synthetic: np.ndarray,
) -> Dict:
    """Compute comparison statistics table."""
    def _stats(x):
        return {
            "Mean": f"{np.mean(x):.6f}",
            "Std": f"{np.std(x):.6f}",
            "Skew": f"{_skew(x):.3f}",
            "Kurt": f"{_kurt(x):.2f}",
            "Min": f"{np.min(x):.4f}",
            "Max": f"{np.max(x):.4f}",
            "5%": f"{np.percentile(x, 5):.4f}",
            "95%": f"{np.percentile(x, 95):.4f}",
        }

    real_stats = _stats(real)
    synth_stats = _stats(synthetic)

    rows = list(real_stats.keys())
    data = [[real_stats[r], synth_stats[r]] for r in rows]

    return {
        "rows": rows,
        "columns": ["Real", "Synthetic"],
        "data": data,
    }


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling standard deviation."""
    n = len(x)
    result = np.zeros(max(n - window + 1, 0))
    for i in range(len(result)):
        result[i] = np.std(x[i:i + window])
    return result


def _skew(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurt(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


if __name__ == "__main__":
    # Demo with synthetic data
    from data_loader import generate_synthetic_data

    print("Generating synthetic data for visualization demo...")
    data = generate_synthetic_data(seq_len=256, n_samples=200)

    real_returns = data["windows"]

    # Create "fake" synthetic data with slightly different parameters
    fake_synth = generate_synthetic_data(
        seq_len=256, n_samples=200,
        mu=0.08, xi=0.25, rho=-0.5,
    )
    synthetic_returns = fake_synth["windows"]

    print("Creating comparison plots...")
    plot_real_vs_synthetic(
        real_returns, synthetic_returns,
        title="Demo: Real vs Synthetic Returns",
        save_path="outputs/demo_comparison.png",
    )

    print("Creating physics diagnostics...")
    plot_physics_diagnostics(
        real_returns, synthetic_returns,
        save_path="outputs/demo_diagnostics.png",
    )

    # Demo training history
    fake_history = {
        "critic_loss": list(np.random.randn(100).cumsum() * 0.01),
        "wasserstein_distance": list(np.abs(np.random.randn(100).cumsum() * 0.01)),
        "gen_loss": list(np.random.randn(100).cumsum() * 0.01 + 1),
        "martingale": list(np.exp(-np.arange(100) * 0.03) * 0.5),
        "volatility_clustering": list(np.exp(-np.arange(100) * 0.02) * 0.3),
        "kurtosis": list(np.exp(-np.arange(100) * 0.01) * 2.0),
        "leverage_effect": list(np.exp(-np.arange(100) * 0.015) * 0.2),
        "autocorrelation": list(np.exp(-np.arange(100) * 0.025) * 0.1),
        "total_physics": list(np.exp(-np.arange(100) * 0.015) * 3.0),
    }

    plot_training_history(
        fake_history,
        save_path="outputs/demo_training.png",
    )

    print("Visualizations saved to outputs/")
