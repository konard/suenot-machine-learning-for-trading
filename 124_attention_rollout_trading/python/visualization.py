"""
visualization.py - Visualization tools for attention rollout

This module provides visualization utilities for attention rollout
analysis in trading applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple


def plot_attention_rollout(
    attention: np.ndarray,
    timestamps: Optional[List[str]] = None,
    title: str = "Attention Rollout",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot attention rollout heatmap.

    Args:
        attention: Attention matrix (seq_len, seq_len)
        timestamps: Optional timestamp labels
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn not installed. Using matplotlib only.")
        sns = None

    plt.figure(figsize=figsize)

    if timestamps is None:
        timestamps = [f"t-{i}" for i in range(attention.shape[0]-1, -1, -1)]

    if sns:
        sns.heatmap(
            attention,
            xticklabels=timestamps,
            yticklabels=timestamps,
            cmap="YlOrRd",
            annot=False,
            fmt=".2f"
        )
    else:
        plt.imshow(attention, cmap="YlOrRd", aspect="auto")
        plt.colorbar()

    plt.title(title)
    plt.xlabel("Input Position")
    plt.ylabel("Output Position")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_input_attribution(
    attribution: np.ndarray,
    timestamps: Optional[List[str]] = None,
    title: str = "Input Attribution Scores",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot bar chart of input attribution scores.

    Args:
        attribution: Attribution scores for each input position
        timestamps: Optional timestamp labels
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)

    if timestamps is None:
        timestamps = [f"t-{i}" for i in range(len(attribution)-1, -1, -1)]

    colors = plt.cm.YlOrRd(attribution / attribution.max())

    plt.bar(range(len(attribution)), attribution, color=colors)
    plt.xticks(range(len(attribution)), timestamps, rotation=45, ha='right')
    plt.xlabel("Time Step")
    plt.ylabel("Attribution Score")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_attention_comparison(
    winning_attention: np.ndarray,
    losing_attention: np.ndarray,
    timestamps: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Compare attention patterns between winning and losing trades.

    Args:
        winning_attention: Average attention for winning trades
        losing_attention: Average attention for losing trades
        timestamps: Optional timestamp labels
        figsize: Figure size
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)

    if timestamps is None:
        n = len(winning_attention)
        timestamps = [f"t-{i}" for i in range(n-1, -1, -1)]

    x = np.arange(len(timestamps))
    width = 0.35

    plt.bar(x - width/2, winning_attention, width, label='Winning Trades',
            color='green', alpha=0.7)
    plt.bar(x + width/2, losing_attention, width, label='Losing Trades',
            color='red', alpha=0.7)

    plt.xlabel('Time Step')
    plt.ylabel('Average Attention')
    plt.title('Attention Pattern Comparison: Winning vs Losing Trades')
    plt.xticks(x, timestamps, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_equity_curve(
    equity_curve: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot equity curve with optional benchmark comparison.

    Args:
        equity_curve: Strategy equity values
        benchmark: Optional benchmark equity values
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)

    plt.plot(equity_curve, label='Strategy', linewidth=2)

    if benchmark is not None:
        plt.plot(benchmark, label='Benchmark', linewidth=2, linestyle='--')

    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_regime_detection(
    regimes: List[str],
    timestamps: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None
) -> None:
    """
    Plot detected market regimes over time.

    Args:
        regimes: List of detected regimes ('momentum', 'mean_reversion', 'mixed')
        timestamps: Optional timestamp labels
        figsize: Figure size
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)

    regime_colors = {
        'momentum': 'green',
        'mean_reversion': 'blue',
        'mixed': 'orange'
    }

    regime_values = {
        'momentum': 2,
        'mean_reversion': 0,
        'mixed': 1
    }

    values = [regime_values.get(r, 1) for r in regimes]
    colors = [regime_colors.get(r, 'gray') for r in regimes]

    plt.scatter(range(len(regimes)), values, c=colors, s=50, alpha=0.7)
    plt.yticks([0, 1, 2], ['Mean Reversion', 'Mixed', 'Momentum'])
    plt.xlabel('Time')
    plt.ylabel('Detected Regime')
    plt.title('Market Regime Detection via Attention Analysis')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
