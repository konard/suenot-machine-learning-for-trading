"""Attention visualization tools for interpreting Transformer predictions.

Provides functions and classes for visualizing attention weights from
Transformer models, including heatmaps, attention flow, and aggregation methods.

References:
    Vig (2019) - "A Multiscale Visualization of Attention in the Transformer Model"
    Abnar & Zuidema (2020) - "Quantifying Attention Flow in Transformers"
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention distributions.

    Low entropy = focused attention (high confidence)
    High entropy = diffuse attention (low confidence)

    Args:
        attention_weights: Attention tensor of shape (..., seq_len)

    Returns:
        Entropy tensor with one less dimension
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-9
    entropy = -torch.sum(
        attention_weights * torch.log(attention_weights + eps), dim=-1
    )
    return entropy


def compute_attention_confidence(
    attention_weights: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """Compute confidence score based on attention focus.

    Args:
        attention_weights: Attention tensor of shape (batch, heads, seq_len, seq_len)
        normalize: Whether to normalize to [0, 1]

    Returns:
        Confidence scores of shape (batch,)
    """
    # Compute entropy
    entropy = compute_attention_entropy(attention_weights)  # (batch, heads, seq_len)

    # Average over heads and positions
    mean_entropy = entropy.mean(dim=(1, 2))  # (batch,)

    if normalize:
        # Maximum entropy for uniform distribution
        seq_len = attention_weights.shape[-1]
        max_entropy = math.log(seq_len)
        # Invert so that focused = high confidence
        confidence = 1 - (mean_entropy / max_entropy)
        return confidence.clamp(0, 1)

    return -mean_entropy  # Negative entropy = confidence


def attention_rollout(
    attention_weights: Dict[str, torch.Tensor],
    add_residual: bool = True,
) -> torch.Tensor:
    """Compute attention rollout across layers.

    Multiplies attention matrices from each layer to get the total
    attention flow from input to output.

    Args:
        attention_weights: Dict mapping layer names to attention tensors
            Each tensor has shape (batch, heads, seq_len, seq_len)
        add_residual: Whether to account for residual connections

    Returns:
        Rolled-out attention of shape (batch, seq_len, seq_len)
    """
    # Sort layers by index
    layer_names = sorted(attention_weights.keys(), key=lambda x: int(x.split("_")[1]))

    # Average over heads for each layer
    layer_attentions = []
    for name in layer_names:
        attn = attention_weights[name].mean(dim=1)  # (batch, seq_len, seq_len)
        if add_residual:
            # Add identity for residual connection
            eye = torch.eye(attn.shape[-1], device=attn.device)
            attn = 0.5 * attn + 0.5 * eye
        layer_attentions.append(attn)

    # Multiply attention matrices
    rollout = layer_attentions[0]
    for attn in layer_attentions[1:]:
        rollout = torch.bmm(attn, rollout)

    # Normalize rows to sum to 1
    rollout = rollout / rollout.sum(dim=-1, keepdim=True)

    return rollout


def get_position_importance(
    attention_weights: Dict[str, torch.Tensor],
    method: str = "rollout",
) -> torch.Tensor:
    """Compute importance score for each input position.

    Args:
        attention_weights: Dict of attention tensors from each layer
        method: Aggregation method ('rollout', 'last_layer', 'mean')

    Returns:
        Position importance of shape (batch, seq_len)
    """
    if method == "rollout":
        rollout = attention_rollout(attention_weights)
        # Importance = attention received by each position from the last position
        importance = rollout[:, -1, :]  # (batch, seq_len)

    elif method == "last_layer":
        # Use only the last layer's attention
        last_layer = max(attention_weights.keys(), key=lambda x: int(x.split("_")[1]))
        attn = attention_weights[last_layer].mean(dim=1)  # Average over heads
        importance = attn[:, -1, :]  # Attention from last position

    elif method == "mean":
        # Average attention across all layers
        all_attn = torch.stack([
            attention_weights[k].mean(dim=1) for k in attention_weights
        ], dim=0)
        mean_attn = all_attn.mean(dim=0)  # (batch, seq_len, seq_len)
        importance = mean_attn[:, -1, :]

    else:
        raise ValueError(f"Unknown method: {method}")

    return importance


def analyze_head_patterns(
    attention_weights: torch.Tensor,
) -> Dict[str, float]:
    """Analyze attention patterns for a single layer.

    Args:
        attention_weights: Attention tensor of shape (batch, heads, seq_len, seq_len)

    Returns:
        Dictionary with pattern metrics for each head
    """
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    attn = attention_weights.mean(dim=0)  # Average over batch: (heads, seq_len, seq_len)

    metrics = {}

    for h in range(n_heads):
        head_attn = attn[h]  # (seq_len, seq_len)

        # Diagonal dominance (local attention)
        diag_mass = torch.diag(head_attn).sum().item()
        metrics[f"head_{h}_diagonal"] = diag_mass / seq_len

        # Off-diagonal mass (long-range attention)
        off_diag_mask = 1 - torch.eye(seq_len, device=head_attn.device)
        off_diag_mass = (head_attn * off_diag_mask).sum().item()
        metrics[f"head_{h}_off_diagonal"] = off_diag_mass / (seq_len * (seq_len - 1))

        # Entropy (focus vs. diffuse)
        entropy = compute_attention_entropy(head_attn).mean().item()
        max_entropy = math.log(seq_len)
        metrics[f"head_{h}_entropy_normalized"] = entropy / max_entropy

        # Sparsity (percentage of near-zero weights)
        sparsity = (head_attn < 0.1 / seq_len).float().mean().item()
        metrics[f"head_{h}_sparsity"] = sparsity

    return metrics


class AttentionVisualizer:
    """Visualization tools for attention weights.

    Provides methods for creating heatmaps, flow diagrams, and analysis plots.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """Initialize visualizer.

        Args:
            figsize: Default figure size for plots
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
        self.figsize = figsize

    def plot_attention_heatmap(
        self,
        attention: torch.Tensor,
        title: str = "Attention Weights",
        xlabel: str = "Key Position",
        ylabel: str = "Query Position",
        timestamps: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
        cmap: str = "Blues",
    ) -> plt.Figure:
        """Plot attention weights as a heatmap.

        Args:
            attention: Attention tensor of shape (seq_len, seq_len)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            timestamps: Optional list of timestamp labels
            ax: Optional matplotlib axes to plot on
            cmap: Colormap name

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        attn_np = attention.detach().cpu().numpy()

        im = ax.imshow(attn_np, cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if timestamps is not None:
            ax.set_xticks(range(len(timestamps)))
            ax.set_xticklabels(timestamps, rotation=45, ha="right")
            ax.set_yticks(range(len(timestamps)))
            ax.set_yticklabels(timestamps)

        plt.colorbar(im, ax=ax, label="Attention Weight")

        return fig

    def plot_multi_head_attention(
        self,
        attention_weights: torch.Tensor,
        layer_name: str = "Layer 0",
    ) -> plt.Figure:
        """Plot attention for all heads in a layer.

        Args:
            attention_weights: Attention tensor of shape (batch, heads, seq_len, seq_len)
            layer_name: Name of the layer for title

        Returns:
            matplotlib Figure with subplots for each head
        """
        # Average over batch
        attn = attention_weights.mean(dim=0)  # (heads, seq_len, seq_len)
        n_heads = attn.shape[0]

        # Create subplot grid
        cols = min(4, n_heads)
        rows = (n_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten()

        for h in range(n_heads):
            self.plot_attention_heatmap(
                attn[h],
                title=f"{layer_name} - Head {h}",
                ax=axes[h],
            )

        # Hide empty subplots
        for i in range(n_heads, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_attention_rollout(
        self,
        attention_weights: Dict[str, torch.Tensor],
        timestamps: Optional[List[str]] = None,
    ) -> plt.Figure:
        """Plot attention rollout across layers.

        Args:
            attention_weights: Dict of attention tensors from each layer
            timestamps: Optional list of timestamp labels

        Returns:
            matplotlib Figure
        """
        rollout = attention_rollout(attention_weights)
        rollout_avg = rollout.mean(dim=0)  # Average over batch

        fig, ax = plt.subplots(figsize=self.figsize)
        self.plot_attention_heatmap(
            rollout_avg,
            title="Attention Rollout (All Layers Combined)",
            timestamps=timestamps,
            ax=ax,
            cmap="Purples",
        )

        return fig

    def plot_position_importance(
        self,
        attention_weights: Dict[str, torch.Tensor],
        timestamps: Optional[List[str]] = None,
        method: str = "rollout",
    ) -> plt.Figure:
        """Plot importance scores for each input position.

        Args:
            attention_weights: Dict of attention tensors
            timestamps: Optional list of timestamp labels
            method: Importance computation method

        Returns:
            matplotlib Figure
        """
        importance = get_position_importance(attention_weights, method)
        importance_avg = importance.mean(dim=0).detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=self.figsize)

        x = range(len(importance_avg))
        if timestamps is not None:
            x = timestamps

        ax.bar(x, importance_avg, color="steelblue", alpha=0.7)
        ax.set_title(f"Input Position Importance ({method})")
        ax.set_xlabel("Position")
        ax.set_ylabel("Importance Score")

        if timestamps is not None:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        return fig

    def plot_head_analysis(
        self,
        attention_weights: torch.Tensor,
        layer_name: str = "Layer 0",
    ) -> plt.Figure:
        """Plot analysis of attention head patterns.

        Args:
            attention_weights: Attention tensor of shape (batch, heads, seq_len, seq_len)
            layer_name: Name of the layer

        Returns:
            matplotlib Figure
        """
        metrics = analyze_head_patterns(attention_weights)
        n_heads = attention_weights.shape[1]

        # Extract metrics for plotting
        diagonals = [metrics[f"head_{h}_diagonal"] for h in range(n_heads)]
        entropies = [metrics[f"head_{h}_entropy_normalized"] for h in range(n_heads)]
        sparsities = [metrics[f"head_{h}_sparsity"] for h in range(n_heads)]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        heads = range(n_heads)

        axes[0].bar(heads, diagonals, color="coral")
        axes[0].set_title(f"{layer_name}: Diagonal Dominance (Local Attention)")
        axes[0].set_xlabel("Head")
        axes[0].set_ylabel("Score")

        axes[1].bar(heads, entropies, color="teal")
        axes[1].set_title(f"{layer_name}: Normalized Entropy (Focus)")
        axes[1].set_xlabel("Head")
        axes[1].set_ylabel("Score")

        axes[2].bar(heads, sparsities, color="purple")
        axes[2].set_title(f"{layer_name}: Sparsity")
        axes[2].set_xlabel("Head")
        axes[2].set_ylabel("Score")

        plt.tight_layout()
        return fig


def plot_attention_heatmap(
    attention: torch.Tensor,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
) -> None:
    """Convenience function to plot and optionally save an attention heatmap.

    Args:
        attention: Attention tensor of shape (seq_len, seq_len)
        title: Plot title
        save_path: Optional path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    visualizer = AttentionVisualizer()
    fig = visualizer.plot_attention_heatmap(attention, title=title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention heatmap to {save_path}")
    else:
        plt.show()

    plt.close(fig)
