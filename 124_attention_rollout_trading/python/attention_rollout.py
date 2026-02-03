"""
attention_rollout.py - Core implementation of Attention Rollout for trading

This module provides attention rollout computation for transformer models,
enabling interpretable trading decisions by tracking attention flow through layers.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn


class AttentionRollout:
    """
    Compute attention rollout for transformer models.

    Attention rollout tracks how attention flows through transformer layers,
    providing interpretable explanations for model predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layer_name: str = "attn",
        head_fusion: str = "mean",
        discard_ratio: float = 0.0
    ):
        """
        Initialize AttentionRollout.

        Args:
            model: PyTorch transformer model
            attention_layer_name: Name pattern for attention layers
            head_fusion: Method to combine heads ('mean', 'max', 'min')
            discard_ratio: Fraction of lowest attention weights to discard
        """
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions: List[torch.Tensor] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks to capture attention weights."""
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name:
                module.register_forward_hook(self._attention_hook)

    def _attention_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: Tuple
    ) -> None:
        """Hook function to capture attention weights."""
        if isinstance(output, tuple):
            attention = output[1] if len(output) > 1 else output[0]
        else:
            attention = output
        self.attentions.append(attention.detach())

    def _fuse_heads(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Fuse multiple attention heads into single attention matrix.

        Args:
            attention: Tensor of shape (batch, heads, seq_len, seq_len)

        Returns:
            Fused attention of shape (batch, seq_len, seq_len)
        """
        if self.head_fusion == "mean":
            return attention.mean(dim=1)
        elif self.head_fusion == "max":
            return attention.max(dim=1)[0]
        elif self.head_fusion == "min":
            return attention.min(dim=1)[0]
        else:
            raise ValueError(f"Unknown head fusion method: {self.head_fusion}")

    def _discard_low_attention(
        self,
        attention: torch.Tensor
    ) -> torch.Tensor:
        """Discard lowest attention weights based on discard_ratio."""
        if self.discard_ratio <= 0:
            return attention

        flat = attention.flatten()
        threshold = torch.quantile(flat, self.discard_ratio)
        attention = torch.where(
            attention > threshold,
            attention,
            torch.zeros_like(attention)
        )
        # Re-normalize
        attention = attention / attention.sum(dim=-1, keepdim=True)
        return attention

    def compute_rollout(
        self,
        input_tensor: torch.Tensor,
        start_layer: int = 0
    ) -> np.ndarray:
        """
        Compute attention rollout for given input.

        Args:
            input_tensor: Input tensor for the model
            start_layer: Layer to start rollout computation

        Returns:
            Rollout matrix of shape (seq_len, seq_len)
        """
        self.attentions = []

        # Forward pass to collect attention weights
        with torch.no_grad():
            _ = self.model(input_tensor)

        if not self.attentions:
            raise RuntimeError("No attention weights captured. Check layer name.")

        # Process attention matrices
        batch_size = self.attentions[0].shape[0]
        seq_len = self.attentions[0].shape[-1]

        # Initialize rollout with identity matrix
        rollout = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1)
        rollout = rollout.to(self.attentions[0].device)

        for i, attention in enumerate(self.attentions[start_layer:]):
            # Fuse attention heads
            attention = self._fuse_heads(attention)

            # Discard low attention weights
            attention = self._discard_low_attention(attention)

            # Add residual connection (identity matrix)
            identity = torch.eye(seq_len).unsqueeze(0).to(attention.device)
            attention = 0.5 * attention + 0.5 * identity

            # Accumulate rollout
            rollout = torch.bmm(attention, rollout)

        # Normalize rows
        rollout = rollout / rollout.sum(dim=-1, keepdim=True)

        return rollout.cpu().numpy()

    def get_input_attribution(
        self,
        input_tensor: torch.Tensor,
        output_position: int = -1
    ) -> np.ndarray:
        """
        Get attribution scores for input positions.

        Args:
            input_tensor: Input tensor
            output_position: Position to get attribution for (-1 for last)

        Returns:
            Attribution scores for each input position
        """
        rollout = self.compute_rollout(input_tensor)
        attribution = rollout[0, output_position, :]
        return attribution


class TradingAttentionRollout(AttentionRollout):
    """
    Specialized attention rollout for trading applications.

    Extends base AttentionRollout with trading-specific features:
    - Feature importance analysis
    - Temporal pattern detection
    - Multi-asset correlation analysis
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize TradingAttentionRollout.

        Args:
            model: Trading transformer model
            feature_names: Names of input features for interpretation
            **kwargs: Arguments passed to AttentionRollout
        """
        super().__init__(model, **kwargs)
        self.feature_names = feature_names

    def analyze_temporal_importance(
        self,
        input_tensor: torch.Tensor,
        timestamps: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Analyze which time periods are most important for prediction.

        Args:
            input_tensor: Input tensor with shape (batch, seq_len, features)
            timestamps: Optional timestamp labels

        Returns:
            Dictionary mapping timestamps to importance scores
        """
        attribution = self.get_input_attribution(input_tensor)

        if timestamps is None:
            timestamps = [f"t-{i}" for i in range(len(attribution)-1, -1, -1)]

        return dict(zip(timestamps, attribution))

    def detect_attention_regime(
        self,
        input_tensor: torch.Tensor,
        threshold_recent: float = 0.6
    ) -> str:
        """
        Detect market regime based on attention pattern.

        Args:
            input_tensor: Input tensor
            threshold_recent: Threshold for recent attention concentration

        Returns:
            Detected regime: 'momentum', 'mean_reversion', or 'mixed'
        """
        attribution = self.get_input_attribution(input_tensor)
        seq_len = len(attribution)

        recent_window = seq_len // 4
        recent_attention = attribution[-recent_window:].sum()

        if recent_attention > threshold_recent:
            return "momentum"
        elif recent_attention < 1 - threshold_recent:
            return "mean_reversion"
        else:
            return "mixed"

    def compute_feature_importance(
        self,
        input_tensor: torch.Tensor,
        n_features: int
    ) -> Dict[str, float]:
        """
        Compute importance of each feature type across time.

        Args:
            input_tensor: Input tensor (batch, seq_len, features)
            n_features: Number of features per time step

        Returns:
            Feature importance scores
        """
        rollout = self.compute_rollout(input_tensor)
        seq_len = rollout.shape[-1]
        time_steps = seq_len // n_features

        importance = {}
        for f in range(n_features):
            feature_positions = list(range(f, seq_len, n_features))
            feature_attention = rollout[0, -1, feature_positions].sum()

            feature_name = (
                self.feature_names[f]
                if self.feature_names and f < len(self.feature_names)
                else f"feature_{f}"
            )
            importance[feature_name] = float(feature_attention)

        return importance
