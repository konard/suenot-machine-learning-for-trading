"""
Visualization Module for EfficientNet Trading

This module provides visualization utilities for:
- Time series to image transformations
- Grad-CAM attention heatmaps
- Training metrics
- Backtest results
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not available. Install with: pip install matplotlib")


def plot_gaf_comparison(
    prices: np.ndarray,
    gasf: np.ndarray,
    gadf: np.ndarray,
    mtf: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Plot comparison of different image transformations.

    Args:
        prices: Original price series
        gasf: Gramian Angular Summation Field
        gadf: Gramian Angular Difference Field
        mtf: Markov Transition Field
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original prices
    ax1 = axes[0, 0]
    ax1.plot(prices, "b-", linewidth=1)
    ax1.set_title("Original Price Series")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)

    # GASF
    ax2 = axes[0, 1]
    im2 = ax2.imshow(gasf, cmap="RdBu_r", aspect="auto")
    ax2.set_title("Gramian Angular Summation Field (GASF)")
    plt.colorbar(im2, ax=ax2)

    # GADF
    ax3 = axes[1, 0]
    im3 = ax3.imshow(gadf, cmap="RdBu_r", aspect="auto")
    ax3.set_title("Gramian Angular Difference Field (GADF)")
    plt.colorbar(im3, ax=ax3)

    # MTF
    ax4 = axes[1, 1]
    im4 = ax4.imshow(mtf, cmap="viridis", aspect="auto")
    ax4.set_title("Markov Transition Field (MTF)")
    plt.colorbar(im4, ax=ax4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_multi_timeframe_image(
    image: np.ndarray,
    channel_names: List[str] = ["1m GASF", "5m GASF", "15m GASF"],
    save_path: Optional[str] = None,
):
    """
    Plot multi-timeframe RGB image with channel breakdown.

    Args:
        image: RGB image of shape (H, W, 3)
        channel_names: Names for each channel
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Combined RGB
    axes[0].imshow(image)
    axes[0].set_title("Combined Multi-Timeframe Image")
    axes[0].axis("off")

    # Individual channels
    for i, (ax, name) in enumerate(zip(axes[1:], channel_names)):
        channel = image[:, :, i]
        ax.imshow(channel, cmap="RdBu_r")
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_attention_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    prediction: Dict,
    alpha: float = 0.5,
    save_path: Optional[str] = None,
):
    """
    Plot Grad-CAM attention heatmap overlay.

    Args:
        original_image: Original input image (H, W, 3)
        heatmap: Attention heatmap (H, W)
        prediction: Prediction dictionary with 'signal' and 'confidence'
        alpha: Overlay transparency
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap
    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Attention Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(original_image)
    axes[2].imshow(heatmap, cmap="jet", alpha=alpha)
    signal = prediction.get("signal", "UNKNOWN")
    confidence = prediction.get("confidence", 0.0)
    axes[2].set_title(f"Overlay - Prediction: {signal} ({confidence:.1%})")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """
    Plot training history.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_prediction_distribution(
    predictions: List[Dict],
    save_path: Optional[str] = None,
):
    """
    Plot distribution of predictions.

    Args:
        predictions: List of prediction dictionaries
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    # Count signals
    signals = [p.get("signal", "HOLD") for p in predictions]
    signal_counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    for s in signals:
        if s in signal_counts:
            signal_counts[s] += 1

    # Get confidences
    confidences = [p.get("confidence", 0.0) for p in predictions if p.get("signal") != "HOLD"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Signal distribution
    ax1 = axes[0]
    colors = ["green", "gray", "red"]
    ax1.bar(signal_counts.keys(), signal_counts.values(), color=colors)
    ax1.set_title("Signal Distribution")
    ax1.set_xlabel("Signal Type")
    ax1.set_ylabel("Count")

    # Confidence distribution
    ax2 = axes[1]
    if confidences:
        ax2.hist(confidences, bins=20, color="blue", alpha=0.7, edgecolor="black")
    ax2.set_title("Confidence Distribution (BUY/SELL only)")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_efficientnet_variants(
    save_path: Optional[str] = None,
):
    """
    Plot comparison of EfficientNet variants.

    Shows the trade-off between model size, speed, and accuracy.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    variants = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
    params = [5.3, 7.8, 9.2, 12, 19, 30, 43, 66]  # Millions
    accuracy = [77.1, 79.1, 80.1, 81.6, 82.9, 83.6, 84.0, 84.3]  # Top-1 %
    latency = [5, 8, 10, 15, 25, 40, 60, 100]  # ms (approximate)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Parameters
    ax1 = axes[0]
    bars1 = ax1.bar(variants, params, color="steelblue")
    ax1.set_title("Model Parameters (Millions)")
    ax1.set_xlabel("Variant")
    ax1.set_ylabel("Parameters (M)")
    for bar, val in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val}M", ha="center", va="bottom", fontsize=8)

    # Accuracy
    ax2 = axes[1]
    bars2 = ax2.bar(variants, accuracy, color="forestgreen")
    ax2.set_title("ImageNet Top-1 Accuracy (%)")
    ax2.set_xlabel("Variant")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(75, 86)
    for bar, val in zip(bars2, accuracy):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val}%", ha="center", va="bottom", fontsize=8)

    # Latency
    ax3 = axes[2]
    bars3 = ax3.bar(variants, latency, color="coral")
    ax3.set_title("Inference Latency (ms, GPU)")
    ax3.set_xlabel("Variant")
    ax3.set_ylabel("Latency (ms)")
    for bar, val in zip(bars3, latency):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val}ms", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def create_trading_dashboard(
    backtest_result,
    predictions: List[Dict],
    sample_attention: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    save_path: Optional[str] = None,
):
    """
    Create comprehensive trading dashboard.

    Args:
        backtest_result: BacktestResult object
        predictions: List of prediction dictionaries
        sample_attention: Optional tuple of (image, heatmap)
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Equity Curve (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(backtest_result.timestamps, backtest_result.equity_curve, "b-", linewidth=1)
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True, alpha=0.3)

    # 2. Key Metrics
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    metrics_text = (
        f"Total Return: {backtest_result.total_return:.2f}%\n"
        f"Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}\n"
        f"Max Drawdown: {backtest_result.max_drawdown:.2f}%\n"
        f"Win Rate: {backtest_result.win_rate:.2f}%\n"
        f"Profit Factor: {backtest_result.profit_factor:.2f}\n"
        f"Total Trades: {backtest_result.total_trades}"
    )
    ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes,
             fontsize=12, verticalalignment="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))
    ax2.set_title("Key Metrics")

    # 3. Drawdown
    ax3 = fig.add_subplot(gs[1, :2])
    equity = np.array(backtest_result.equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    ax3.fill_between(backtest_result.timestamps, drawdown, 0, color="red", alpha=0.3)
    ax3.set_title("Drawdown")
    ax3.set_ylabel("Drawdown (%)")
    ax3.grid(True, alpha=0.3)

    # 4. Signal Distribution
    ax4 = fig.add_subplot(gs[1, 2])
    signals = [p.get("signal", "HOLD") for p in predictions]
    signal_counts = {"BUY": signals.count("BUY"),
                    "HOLD": signals.count("HOLD"),
                    "SELL": signals.count("SELL")}
    colors = ["green", "gray", "red"]
    ax4.bar(signal_counts.keys(), signal_counts.values(), color=colors)
    ax4.set_title("Signal Distribution")

    # 5. Trade P&L
    ax5 = fig.add_subplot(gs[2, :2])
    pnls = [t.pnl for t in backtest_result.trades if t.pnl is not None]
    if pnls:
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax5.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax5.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax5.set_title("Trade P&L")
    ax5.set_xlabel("Trade #")
    ax5.set_ylabel("P&L ($)")
    ax5.grid(True, alpha=0.3)

    # 6. Sample Attention (if provided)
    ax6 = fig.add_subplot(gs[2, 2])
    if sample_attention is not None:
        image, heatmap = sample_attention
        ax6.imshow(image)
        ax6.imshow(heatmap, cmap="jet", alpha=0.5)
        ax6.set_title("Sample Attention")
    else:
        ax6.text(0.5, 0.5, "No attention\nmap provided",
                ha="center", va="center", transform=ax6.transAxes)
        ax6.set_title("Sample Attention")
    ax6.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Dashboard saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Testing Visualization Module...")

    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping visualization tests")
    else:
        # Generate sample data
        np.random.seed(42)

        # Sample price series
        prices = 100 + np.cumsum(np.random.randn(120) * 0.5)

        # Sample transformations (simulated)
        gasf = np.random.randn(64, 64)
        gadf = np.random.randn(64, 64)
        mtf = np.random.randn(64, 64)

        print("1. Plotting GAF comparison...")
        # Uncomment to display:
        # plot_gaf_comparison(prices, gasf, gadf, mtf)

        print("2. Plotting EfficientNet variants...")
        # Uncomment to display:
        # plot_efficientnet_variants()

        print("3. Plotting training history...")
        history = {
            "train_loss": [0.8, 0.6, 0.5, 0.45, 0.4],
            "val_loss": [0.85, 0.65, 0.55, 0.5, 0.48],
            "train_acc": [0.5, 0.6, 0.65, 0.68, 0.7],
            "val_acc": [0.48, 0.58, 0.62, 0.64, 0.65],
        }
        # Uncomment to display:
        # plot_training_history(history)

        print("4. Plotting prediction distribution...")
        predictions = [
            {"signal": "BUY", "confidence": 0.7},
            {"signal": "HOLD", "confidence": 0.0},
            {"signal": "SELL", "confidence": 0.65},
        ] * 50
        # Uncomment to display:
        # plot_prediction_distribution(predictions)

        print("\nVisualization tests completed!")
        print("Uncomment the plot functions to display figures.")
