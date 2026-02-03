"""
Saliency map computation for trading models.

This module provides various gradient-based methods for computing saliency maps
that explain which input features most influence model predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt


class SaliencyComputer:
    """
    Compute saliency maps for trading models using various methods.

    Args:
        model: Trained PyTorch model
        device: Device to run computations on
    """

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def vanilla_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute vanilla gradient saliency.

        The simplest saliency method: compute the gradient of the output
        with respect to the input.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)

        Returns:
            Saliency map with same shape as input
        """
        x = x.clone().to(self.device).requires_grad_(True)
        output = self.model(x)
        output.backward(torch.ones_like(output))
        return x.grad.abs().cpu()

    def gradient_x_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient times input saliency.

        Multiplies the gradient by the input value, ensuring that features
        with zero input have zero saliency.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)

        Returns:
            Saliency map with same shape as input
        """
        x = x.clone().to(self.device).requires_grad_(True)
        output = self.model(x)
        output.backward(torch.ones_like(output))
        saliency = (x.grad * x).abs()
        return saliency.cpu()

    def integrated_gradients(
        self,
        x: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> torch.Tensor:
        """
        Compute integrated gradients.

        Integrates gradients along a path from baseline to input,
        providing more robust attributions.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)
            baseline: Baseline tensor (default: zeros)
            steps: Number of interpolation steps

        Returns:
            Saliency map with same shape as input
        """
        x = x.to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(self.device)

        scaled_inputs = [
            baseline + (float(i) / steps) * (x - baseline)
            for i in range(steps + 1)
        ]

        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.clone().requires_grad_(True)
            output = self.model(scaled_input)
            output.backward(torch.ones_like(output))
            gradients.append(scaled_input.grad.clone())

        gradients = torch.stack(gradients)
        avg_gradients = gradients.mean(dim=0)

        integrated_grad = (x - baseline) * avg_gradients
        return integrated_grad.abs().cpu()

    def smoothgrad(
        self,
        x: torch.Tensor,
        noise_level: float = 0.1,
        num_samples: int = 50
    ) -> torch.Tensor:
        """
        Compute SmoothGrad saliency.

        Averages gradients over noisy versions of the input to reduce noise.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)
            noise_level: Standard deviation of noise (as fraction of input range)
            num_samples: Number of noisy samples to average

        Returns:
            Saliency map with same shape as input
        """
        x = x.to(self.device)
        stdev = noise_level * (x.max() - x.min())

        total_gradients = torch.zeros_like(x)

        for _ in range(num_samples):
            noise = torch.randn_like(x) * stdev
            noisy_input = (x + noise).requires_grad_(True)
            output = self.model(noisy_input)
            output.backward(torch.ones_like(output))
            total_gradients += noisy_input.grad.abs()

        return (total_gradients / num_samples).cpu()

    def compute_all(
        self,
        x: torch.Tensor,
        methods: Optional[List[str]] = None
    ) -> dict:
        """
        Compute saliency maps using multiple methods.

        Args:
            x: Input tensor
            methods: List of methods to use (default: all)

        Returns:
            Dictionary mapping method names to saliency maps
        """
        if methods is None:
            methods = ['vanilla', 'gradient_x_input', 'integrated', 'smoothgrad']

        results = {}
        for method in methods:
            if method == 'vanilla':
                results['vanilla'] = self.vanilla_gradient(x)
            elif method == 'gradient_x_input':
                results['gradient_x_input'] = self.gradient_x_input(x)
            elif method == 'integrated':
                results['integrated'] = self.integrated_gradients(x)
            elif method == 'smoothgrad':
                results['smoothgrad'] = self.smoothgrad(x)

        return results


def aggregate_temporal_importance(
    saliency: torch.Tensor,
    aggregation: str = 'mean'
) -> torch.Tensor:
    """
    Aggregate saliency across features to get temporal importance.

    Args:
        saliency: Saliency map of shape (batch, sequence_length, features)
        aggregation: 'mean', 'max', or 'sum'

    Returns:
        Temporal importance of shape (batch, sequence_length)
    """
    if aggregation == 'mean':
        return saliency.mean(dim=-1)
    elif aggregation == 'max':
        return saliency.max(dim=-1)[0]
    elif aggregation == 'sum':
        return saliency.sum(dim=-1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def aggregate_feature_importance(
    saliency: torch.Tensor,
    aggregation: str = 'mean'
) -> torch.Tensor:
    """
    Aggregate saliency across time to get feature importance.

    Args:
        saliency: Saliency map of shape (batch, sequence_length, features)
        aggregation: 'mean', 'max', or 'sum'

    Returns:
        Feature importance of shape (batch, features)
    """
    if aggregation == 'mean':
        return saliency.mean(dim=1)
    elif aggregation == 'max':
        return saliency.max(dim=1)[0]
    elif aggregation == 'sum':
        return saliency.sum(dim=1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def compute_saliency_concentration(saliency: torch.Tensor) -> torch.Tensor:
    """
    Compute the concentration of saliency (inverse entropy).

    High concentration means the model focuses on few features.
    Low concentration means attention is spread across many features.

    Args:
        saliency: Saliency map of shape (batch, sequence_length, features)

    Returns:
        Concentration score per batch sample, shape (batch,)
    """
    flat_saliency = saliency.view(saliency.shape[0], -1)
    probs = flat_saliency / (flat_saliency.sum(dim=-1, keepdim=True) + 1e-10)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    max_entropy = np.log(flat_saliency.shape[-1])
    concentration = 1 - (entropy / max_entropy)
    return concentration


def plot_saliency_heatmap(
    saliency: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    title: str = "Saliency Map",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot saliency map as a heatmap.

    Args:
        saliency: Saliency map of shape (sequence_length, features)
        feature_names: Names of features for y-axis labels
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if saliency.dim() == 3:
        saliency = saliency[0]

    saliency_np = saliency.numpy()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(saliency_np.T, aspect='auto', cmap='hot')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature')
    ax.set_title(title)

    if feature_names is not None:
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)

    plt.colorbar(im, ax=ax, label='Saliency')
    plt.tight_layout()

    return fig


def plot_temporal_importance(
    saliency: torch.Tensor,
    title: str = "Temporal Importance",
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot aggregated temporal importance.

    Args:
        saliency: Saliency map of shape (batch, sequence_length, features)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    temporal = aggregate_temporal_importance(saliency)
    if temporal.dim() == 2:
        temporal = temporal[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(temporal)), temporal.numpy())
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Importance')
    ax.set_title(title)
    plt.tight_layout()

    return fig


def plot_feature_importance(
    saliency: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot aggregated feature importance.

    Args:
        saliency: Saliency map of shape (batch, sequence_length, features)
        feature_names: Names of features
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    feature_imp = aggregate_feature_importance(saliency)
    if feature_imp.dim() == 2:
        feature_imp = feature_imp[0]

    feature_imp_np = feature_imp.numpy()

    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(feature_imp_np))]

    sorted_idx = np.argsort(feature_imp_np)[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        range(len(feature_imp_np)),
        feature_imp_np[sorted_idx]
    )
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()

    return fig
