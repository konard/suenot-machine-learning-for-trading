"""
Visualization Module for Operator Learning in Finance

Provides plotting utilities for:
- Training curves (loss, relative L2 error)
- Input/output function comparisons
- Operator prediction vs ground truth
- Super-resolution demonstrations
- Yield curve and volatility surface visualizations
- Order book operator predictions
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize

    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False
    print("matplotlib not available. Install with: pip install matplotlib")

try:
    import seaborn as sns

    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False


def setup_plot_style():
    """Configure matplotlib for publication-quality figures."""
    if not PLT_AVAILABLE:
        return
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
    if SNS_AVAILABLE:
        sns.set_palette("husl")


# ═══════════════════════════════════════════════════════════════════════════════
# Training Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Operator Learning Training History",
    save_path: Optional[str] = None,
):
    """
    Plot training and validation loss curves.

    Args:
        history: Dict with 'train_loss', 'val_loss', 'val_rel_l2', 'lr'
        title: Plot title
        save_path: If provided, save figure to this path
    """
    if not PLT_AVAILABLE:
        print("Cannot plot: matplotlib not available")
        return

    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax = axes[0, 0]
    ax.semilogy(epochs, history["train_loss"], "b-", alpha=0.8, label="Train Loss")
    ax.semilogy(epochs, history["val_loss"], "r-", alpha=0.8, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative L2 error
    ax = axes[0, 1]
    ax.plot(epochs, history["val_rel_l2"], "g-", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("Validation Relative L2 Error")
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 0]
    ax.semilogy(epochs, history["lr"], "m-", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)

    # Gap between train and val (overfitting indicator)
    ax = axes[1, 1]
    gap = np.array(history["val_loss"]) - np.array(history["train_loss"])
    ax.plot(epochs, gap, "orange", alpha=0.8)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.fill_between(epochs, 0, gap, alpha=0.2, color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val - Train Loss")
    ax.set_title("Generalization Gap")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved training plot to {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Operator Prediction Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_operator_predictions(
    inputs: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_samples: int = 6,
    title: str = "Operator Predictions vs Ground Truth",
    x_label: str = "x",
    y_label: str = "Function value",
    save_path: Optional[str] = None,
):
    """
    Plot side-by-side input functions, predicted outputs, and ground truth.

    Args:
        inputs: Input functions (N, n_points)
        predictions: Predicted output functions (N, n_points)
        targets: Ground truth output functions (N, n_points)
        n_samples: Number of samples to show
        title: Overall title
        x_label, y_label: Axis labels
        save_path: Save path
    """
    if not PLT_AVAILABLE:
        return

    setup_plot_style()
    n_samples = min(n_samples, len(inputs))
    fig, axes = plt.subplots(n_samples, 3, figsize=(16, 3 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Determine x grids
        n_in = inputs[i].shape[0] if inputs[i].ndim == 1 else inputs[i].shape[-1]
        n_out = predictions[i].shape[0] if predictions[i].ndim == 1 else predictions[i].shape[-1]
        x_in = np.linspace(0, 1, n_in)
        x_out = np.linspace(0, 1, n_out)

        inp = inputs[i].flatten() if inputs[i].ndim > 1 else inputs[i]
        pred = predictions[i].flatten() if predictions[i].ndim > 1 else predictions[i]
        tgt = targets[i].flatten() if targets[i].ndim > 1 else targets[i]

        # Input
        axes[i, 0].plot(x_in, inp, "b-", linewidth=1.5)
        axes[i, 0].set_title(f"Input u(x) [{i}]")
        axes[i, 0].set_xlabel(x_label)
        axes[i, 0].set_ylabel(y_label)
        axes[i, 0].grid(True, alpha=0.3)

        # Prediction vs target
        axes[i, 1].plot(x_out, tgt, "k-", linewidth=1.5, label="Ground Truth")
        axes[i, 1].plot(x_out, pred, "r--", linewidth=1.5, label="Prediction")
        axes[i, 1].set_title(f"Output s(x) [{i}]")
        axes[i, 1].set_xlabel(x_label)
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

        # Error
        error = np.abs(pred[:len(tgt)] - tgt[:len(pred)])
        axes[i, 2].fill_between(
            x_out[:len(error)], 0, error, alpha=0.5, color="red"
        )
        axes[i, 2].set_title(f"Abs Error [{i}] (max={error.max():.4f})")
        axes[i, 2].set_xlabel(x_label)
        axes[i, 2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_superresolution(
    input_coarse: np.ndarray,
    pred_coarse: np.ndarray,
    pred_fine: np.ndarray,
    target_fine: Optional[np.ndarray] = None,
    title: str = "Zero-Shot Super-Resolution",
    save_path: Optional[str] = None,
):
    """
    Visualize FNO zero-shot super-resolution capability.

    Args:
        input_coarse: Input on coarse grid (n_coarse,)
        pred_coarse: Prediction on coarse grid (n_coarse,)
        pred_fine: Prediction on fine grid (n_fine,)
        target_fine: Ground truth on fine grid (optional)
        title: Title
        save_path: Save path
    """
    if not PLT_AVAILABLE:
        return

    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    n_coarse = len(input_coarse)
    n_fine = len(pred_fine)
    x_coarse = np.linspace(0, 1, n_coarse)
    x_fine = np.linspace(0, 1, n_fine)

    # Input
    axes[0].plot(x_coarse, input_coarse, "b-o", markersize=3, label=f"Input ({n_coarse} pts)")
    axes[0].set_title("Input Function (Coarse Grid)")
    axes[0].set_xlabel("x")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Coarse prediction
    axes[1].plot(x_coarse, pred_coarse, "r-o", markersize=3, label=f"Prediction ({n_coarse} pts)")
    axes[1].set_title(f"Prediction at Training Resolution ({n_coarse} pts)")
    axes[1].set_xlabel("x")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Fine prediction
    axes[2].plot(x_fine, pred_fine, "r-", linewidth=1.5, label=f"Prediction ({n_fine} pts)")
    if target_fine is not None:
        axes[2].plot(x_fine, target_fine, "k--", linewidth=1.5, label="Ground Truth")
    axes[2].set_title(f"Zero-Shot Super-Resolution ({n_fine} pts)")
    axes[2].set_xlabel("x")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Financial Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_yield_curve_operator(
    curves_today: np.ndarray,
    curves_predicted: np.ndarray,
    curves_actual: np.ndarray,
    maturities: Optional[np.ndarray] = None,
    n_samples: int = 4,
    title: str = "Yield Curve Operator Predictions",
    save_path: Optional[str] = None,
):
    """
    Plot yield curve evolution predictions.

    Args:
        curves_today: Today's yield curves (N, M)
        curves_predicted: Predicted future curves (N, M)
        curves_actual: Actual future curves (N, M)
        maturities: Maturity values for x-axis
        n_samples: Number of curves to plot
        save_path: Save path
    """
    if not PLT_AVAILABLE:
        return

    setup_plot_style()
    n_samples = min(n_samples, len(curves_today))
    n_mats = curves_today.shape[1]

    if maturities is None:
        maturities = np.linspace(0.25, 30, n_mats)

    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 5))
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        ax = axes[i]
        ax.plot(maturities, curves_today[i] * 100, "b-", linewidth=2, label="Today")
        ax.plot(
            maturities,
            curves_actual[i] * 100,
            "k--",
            linewidth=2,
            label="Actual Future",
        )
        ax.plot(
            maturities,
            curves_predicted[i] * 100,
            "r-.",
            linewidth=2,
            label="Predicted Future",
        )
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Yield (%)")
        ax.set_title(f"Sample {i+1}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_option_pricing_operator(
    vol_functions: np.ndarray,
    predicted_prices: np.ndarray,
    true_prices: np.ndarray,
    s_k_ratios: np.ndarray,
    n_samples: int = 4,
    title: str = "Option Pricing Operator",
    save_path: Optional[str] = None,
):
    """
    Plot option pricing operator results.

    Args:
        vol_functions: Input volatility functions (N, n_vol_points)
        predicted_prices: Predicted prices (N, n_eval)
        true_prices: Ground truth prices (N, n_eval)
        s_k_ratios: S/K ratios for x-axis (N, n_eval)
        n_samples: Number of samples to plot
        save_path: Save path
    """
    if not PLT_AVAILABLE:
        return

    setup_plot_style()
    n_samples = min(n_samples, len(vol_functions))

    fig, axes = plt.subplots(2, n_samples, figsize=(5 * n_samples, 8))
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    s_grid = np.linspace(0.5, 2.0, vol_functions.shape[1])

    for i in range(n_samples):
        # Volatility function
        axes[0, i].plot(s_grid, vol_functions[i], "b-", linewidth=2)
        axes[0, i].set_xlabel("S/K")
        axes[0, i].set_ylabel("Volatility")
        axes[0, i].set_title(f"Vol Function {i+1}")
        axes[0, i].grid(True, alpha=0.3)

        # Price comparison
        sort_idx = np.argsort(s_k_ratios[i])
        s_sorted = s_k_ratios[i][sort_idx]
        axes[1, i].scatter(s_sorted, true_prices[i][sort_idx], c="k", s=10, label="True", alpha=0.6)
        axes[1, i].scatter(
            s_sorted, predicted_prices[i][sort_idx], c="r", s=10, label="Predicted", alpha=0.6
        )
        axes[1, i].set_xlabel("S/K")
        axes[1, i].set_ylabel("Option Price")
        axes[1, i].set_title(f"Prices {i+1}")
        axes[1, i].legend(fontsize=9)
        axes[1, i].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_vol_surface_evolution(
    vol_today: np.ndarray,
    vol_predicted: np.ndarray,
    vol_actual: Optional[np.ndarray] = None,
    strikes: Optional[np.ndarray] = None,
    maturities: Optional[np.ndarray] = None,
    title: str = "Volatility Surface Evolution",
    save_path: Optional[str] = None,
):
    """
    Plot implied volatility surface predictions as 3D surfaces.

    Args:
        vol_today: Today's vol surface (n_strikes, n_mats)
        vol_predicted: Predicted future surface (n_strikes, n_mats)
        vol_actual: Actual future surface (optional)
        strikes: Strike values
        maturities: Maturity values
        save_path: Save path
    """
    if not PLT_AVAILABLE:
        return

    setup_plot_style()

    n_strikes, n_mats = vol_today.shape
    if strikes is None:
        strikes = np.linspace(0.8, 1.2, n_strikes)
    if maturities is None:
        maturities = np.linspace(0.1, 2.0, n_mats)

    K, T = np.meshgrid(strikes, maturities, indexing="ij")

    n_plots = 3 if vol_actual is not None else 2
    fig = plt.figure(figsize=(7 * n_plots, 6))

    ax1 = fig.add_subplot(1, n_plots, 1, projection="3d")
    ax1.plot_surface(K, T, vol_today * 100, cmap="viridis", alpha=0.8)
    ax1.set_xlabel("K/S")
    ax1.set_ylabel("T (years)")
    ax1.set_zlabel("IV (%)")
    ax1.set_title("Today")

    ax2 = fig.add_subplot(1, n_plots, 2, projection="3d")
    ax2.plot_surface(K, T, vol_predicted * 100, cmap="plasma", alpha=0.8)
    ax2.set_xlabel("K/S")
    ax2.set_ylabel("T (years)")
    ax2.set_zlabel("IV (%)")
    ax2.set_title("Predicted Future")

    if vol_actual is not None:
        ax3 = fig.add_subplot(1, n_plots, 3, projection="3d")
        ax3.plot_surface(K, T, vol_actual * 100, cmap="coolwarm", alpha=0.8)
        ax3.set_xlabel("K/S")
        ax3.set_ylabel("T (years)")
        ax3.set_zlabel("IV (%)")
        ax3.set_title("Actual Future")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_crypto_operator_signals(
    prices: np.ndarray,
    predictions: np.ndarray,
    signals: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    title: str = "Crypto Operator Trading Signals (Bybit)",
    save_path: Optional[str] = None,
):
    """
    Plot crypto price with operator-based trading signals.

    Args:
        prices: Price series (N,)
        predictions: Predicted next-step values (N,)
        signals: Trading signals: +1 (buy), -1 (sell), 0 (hold)
        timestamps: Optional datetime values
        save_path: Save path
    """
    if not PLT_AVAILABLE:
        return

    setup_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    x = timestamps if timestamps is not None else np.arange(len(prices))

    # Price + predictions
    axes[0].plot(x, prices, "b-", linewidth=1.2, label="Actual Price")
    axes[0].plot(x, predictions, "r--", linewidth=1, alpha=0.7, label="Operator Prediction")
    axes[0].set_ylabel("Price")
    axes[0].set_title("Price vs Operator Prediction")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Prediction error
    error = predictions - prices
    axes[1].fill_between(x, 0, error, where=error >= 0, color="green", alpha=0.4, label="Over-predict")
    axes[1].fill_between(x, 0, error, where=error < 0, color="red", alpha=0.4, label="Under-predict")
    axes[1].set_ylabel("Prediction Error")
    axes[1].set_title("Operator Prediction Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Signals
    buy_mask = signals == 1
    sell_mask = signals == -1
    axes[2].plot(x, prices, "gray", linewidth=0.8, alpha=0.5)
    if np.any(buy_mask):
        axes[2].scatter(
            x[buy_mask], prices[buy_mask], c="green", marker="^", s=40, label="Buy"
        )
    if np.any(sell_mask):
        axes[2].scatter(
            x[sell_mask], prices[sell_mask], c="red", marker="v", s=40, label="Sell"
        )
    axes[2].set_ylabel("Price")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Trading Signals")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_method_comparison(
    methods: List[str],
    inference_times_ms: List[float],
    errors: List[float],
    title: str = "Neural Operator vs Traditional Methods",
    save_path: Optional[str] = None,
):
    """
    Bar chart comparing neural operators with traditional methods.

    Args:
        methods: List of method names
        inference_times_ms: Inference time in milliseconds for each method
        errors: RMSE or relative L2 error for each method
        save_path: Save path
    """
    if not PLT_AVAILABLE:
        return

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    bar_colors = colors[:len(methods)]

    # Inference time
    bars = axes[0].bar(methods, inference_times_ms, color=bar_colors, alpha=0.8)
    axes[0].set_ylabel("Inference Time (ms)")
    axes[0].set_title("Inference Speed Comparison")
    axes[0].set_yscale("log")
    for bar, val in zip(bars, inference_times_ms):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0].grid(True, alpha=0.3, axis="y")

    # Error
    bars = axes[1].bar(methods, errors, color=bar_colors, alpha=0.8)
    axes[1].set_ylabel("Error (RMSE)")
    axes[1].set_title("Accuracy Comparison")
    for bar, val in zip(bars, errors):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    print("Visualize module for Operator Learning - Chapter 152")
    print("Run train.py first to generate data, then use these plotting functions.")

    # Demo with synthetic data
    print("\nDemo: Method Comparison Chart")
    plot_method_comparison(
        methods=["Finite Diff", "Monte Carlo", "Standard NN", "DeepONet", "FNO"],
        inference_times_ms=[100.0, 500.0, 0.1, 0.2, 0.15],
        errors=[0.0001, 0.01, 0.005, 0.003, 0.002],
        title="Option Pricing: Neural Operators vs Traditional Methods",
    )
