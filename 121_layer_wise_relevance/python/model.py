"""
Layer-wise Relevance Propagation (LRP) Model Implementation

This module provides LRP-enabled neural network layers and models
for explainable trading predictions.

Features:
    - Multiple LRP rules: LRP-0, LRP-epsilon, LRP-gamma, LRP-alpha-beta
    - Composite rule support for different layers
    - Conservation-preserving relevance propagation
    - Support for classification and regression tasks

Example:
    >>> config = LRPConfig(epsilon=0.01, gamma=0.25)
    >>> model = LRPNetwork(input_dim=100, hidden_dims=[64, 32], output_dim=2, config=config)
    >>> output = model(input_tensor)
    >>> relevance = model.explain(input_tensor)
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class LRPRule(Enum):
    """Available LRP propagation rules."""
    ZERO = "zero"
    EPSILON = "epsilon"
    GAMMA = "gamma"
    ALPHA_BETA = "alpha_beta"


@dataclass
class LRPConfig:
    """
    Configuration for LRP analysis.

    Attributes:
        epsilon: Stabilization term for LRP-epsilon rule (default: 0.01)
        gamma: Emphasis factor for positive weights in LRP-gamma (default: 0.25)
        alpha: Positive contribution weight for LRP-alpha-beta (default: 2.0)
        beta: Negative contribution weight for LRP-alpha-beta (default: 1.0)
        default_rule: Default rule to use if not specified (default: EPSILON)
    """
    epsilon: float = 0.01
    gamma: float = 0.25
    alpha: float = 2.0
    beta: float = 1.0
    default_rule: LRPRule = LRPRule.EPSILON


class LRPLinear(nn.Module):
    """
    Linear layer with Layer-wise Relevance Propagation support.

    This layer wraps nn.Linear and stores activations during forward pass
    for use in backward relevance propagation.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: If True, adds learnable bias (default: True)
        config: LRP configuration (default: None, uses default config)

    Example:
        >>> layer = LRPLinear(64, 32, config=LRPConfig())
        >>> output = layer(input)
        >>> relevance = layer.lrp(next_layer_relevance)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[LRPConfig] = None
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.config = config or LRPConfig()
        self.activations: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with activation storage.

        Args:
            x: Input tensor of shape [batch, in_features]

        Returns:
            Output tensor of shape [batch, out_features]
        """
        self.activations = x.detach().clone()
        return self.linear(x)

    def lrp(
        self,
        R: torch.Tensor,
        rule: Optional[LRPRule] = None
    ) -> torch.Tensor:
        """
        Propagate relevance backwards through this layer.

        Args:
            R: Relevance from next layer [batch, out_features]
            rule: LRP rule to apply (default: uses config.default_rule)

        Returns:
            Relevance for previous layer [batch, in_features]

        Raises:
            ValueError: If no activations stored (forward not called)
        """
        if self.activations is None:
            raise ValueError("No activations stored. Call forward() first.")

        rule = rule or self.config.default_rule
        a = self.activations
        w = self.linear.weight.T  # [in_features, out_features]

        if rule == LRPRule.ZERO:
            return self._lrp_zero(a, w, R)
        elif rule == LRPRule.EPSILON:
            return self._lrp_epsilon(a, w, R)
        elif rule == LRPRule.GAMMA:
            return self._lrp_gamma(a, w, R)
        elif rule == LRPRule.ALPHA_BETA:
            return self._lrp_alpha_beta(a, w, R)
        else:
            raise ValueError(f"Unknown LRP rule: {rule}")

    def _lrp_zero(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
        R: torch.Tensor
    ) -> torch.Tensor:
        """
        LRP-0: Basic rule without stabilization.

        R_i = Σ_j (a_i * w_ij / Σ_k a_k * w_kj) * R_j
        """
        # z[batch, in, out] = contribution of each input to each output
        z = a.unsqueeze(-1) * w.unsqueeze(0)

        # z_sum[batch, 1, out] = total contribution to each output
        z_sum = z.sum(dim=1, keepdim=True)

        # Avoid division by zero
        z_sum = z_sum + 1e-9 * (z_sum == 0).float()

        # s[batch, in, out] = proportion of contribution
        s = z / z_sum

        # Distribute relevance
        R_prev = (s * R.unsqueeze(1)).sum(dim=-1)

        return R_prev

    def _lrp_epsilon(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
        R: torch.Tensor
    ) -> torch.Tensor:
        """
        LRP-epsilon: Stabilized rule with epsilon term.

        R_i = Σ_j (a_i * w_ij / (Σ_k a_k * w_kj + ε * sign(z_j))) * R_j

        The epsilon term absorbs some relevance, providing numerical stability
        at the cost of weak conservation.
        """
        z = a.unsqueeze(-1) * w.unsqueeze(0)
        z_sum = z.sum(dim=1, keepdim=True)

        # Add epsilon with sign preservation
        z_sum_stabilized = z_sum + self.config.epsilon * torch.sign(z_sum)

        # Handle exact zeros
        z_sum_stabilized = torch.where(
            z_sum == 0,
            torch.ones_like(z_sum) * self.config.epsilon,
            z_sum_stabilized
        )

        s = z / z_sum_stabilized
        R_prev = (s * R.unsqueeze(1)).sum(dim=-1)

        return R_prev

    def _lrp_gamma(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
        R: torch.Tensor
    ) -> torch.Tensor:
        """
        LRP-gamma: Rule emphasizing positive contributions.

        Uses modified weights: w' = w + γ * max(w, 0)
        This increases the importance of positive (excitatory) evidence.
        """
        w_positive = torch.clamp(w, min=0)
        w_modified = w + self.config.gamma * w_positive

        z = a.unsqueeze(-1) * w_modified.unsqueeze(0)
        z_sum = z.sum(dim=1, keepdim=True) + 1e-9

        s = z / z_sum
        R_prev = (s * R.unsqueeze(1)).sum(dim=-1)

        return R_prev

    def _lrp_alpha_beta(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
        R: torch.Tensor
    ) -> torch.Tensor:
        """
        LRP-alpha-beta: Separate treatment of positive and negative contributions.

        R_i = α * (a_i * w_ij)+ / Σ_k (a_k * w_kj)+ * R_j
            - β * (a_i * w_ij)- / Σ_k (a_k * w_kj)- * R_j

        Constraint: α - β = 1 for conservation
        Common choices: α=2, β=1 or α=1, β=0
        """
        alpha = self.config.alpha
        beta = self.config.beta

        z = a.unsqueeze(-1) * w.unsqueeze(0)

        # Separate positive and negative contributions
        z_pos = torch.clamp(z, min=0)
        z_neg = torch.clamp(z, max=0)

        # Sum for normalization
        z_pos_sum = z_pos.sum(dim=1, keepdim=True) + 1e-9
        z_neg_sum = z_neg.sum(dim=1, keepdim=True) - 1e-9  # Negative sum

        # Proportions
        s_pos = alpha * z_pos / z_pos_sum
        s_neg = beta * z_neg / z_neg_sum

        R_prev = ((s_pos + s_neg) * R.unsqueeze(1)).sum(dim=-1)

        return R_prev


class LRPNetwork(nn.Module):
    """
    Neural network with built-in LRP explanation capability.

    Architecture:
        - Multiple LRP-enabled linear layers
        - ReLU activations between layers
        - Dropout for regularization
        - Configurable hidden dimensions

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (e.g., 2 for binary classification)
        dropout: Dropout probability (default: 0.1)
        config: LRP configuration (default: None, uses default config)

    Example:
        >>> model = LRPNetwork(
        ...     input_dim=420,
        ...     hidden_dims=[128, 64, 32],
        ...     output_dim=2,
        ...     dropout=0.2
        ... )
        >>> output = model(batch_x)
        >>> relevance = model.explain(batch_x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        config: Optional[LRPConfig] = None
    ):
        super().__init__()
        self.config = config or LRPConfig()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(
                LRPLinear(dims[i], dims[i + 1], config=self.config)
            )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Assign composite rules
        self._assign_composite_rules()

    def _assign_composite_rules(self):
        """
        Assign different LRP rules to different layers.

        Strategy:
            - Lower layers (near input): LRP-gamma (focus on positive evidence)
            - Middle layers: LRP-epsilon (stable)
            - Upper layers (near output): LRP-0 (precise)
        """
        num_layers = len(self.layers)
        self.layer_rules: Dict[int, LRPRule] = {}

        for i in range(num_layers):
            position = i / num_layers
            if position < 0.33:
                self.layer_rules[i] = LRPRule.GAMMA
            elif position < 0.66:
                self.layer_rules[i] = LRPRule.EPSILON
            else:
                self.layer_rules[i] = LRPRule.ZERO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass storing activations for LRP.

        Args:
            x: Input tensor [batch, seq_len, features] or [batch, features]

        Returns:
            Output tensor [batch, output_dim]
        """
        batch_size = x.shape[0]

        # Flatten if 3D input
        x = x.view(batch_size, -1)

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Final layer (no activation)
        x = self.layers[-1](x)

        return x

    def explain(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute LRP explanations for predictions.

        Args:
            x: Input tensor [batch, ...] (same format as forward)
            target_class: Class to explain (default: predicted class)

        Returns:
            Relevance scores [batch, input_dim] reshaped to match input
        """
        original_shape = x.shape
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        # Forward pass to get predictions
        output = self.forward(x)

        # Initialize relevance at output
        R = torch.zeros_like(output)

        if target_class is not None:
            # Explain specific class
            R[:, target_class] = output[:, target_class]
        else:
            # Explain predicted class for each sample
            pred = output.argmax(dim=1)
            R[torch.arange(batch_size), pred] = output[torch.arange(batch_size), pred]

        # Backward propagation through layers
        for i, layer in enumerate(reversed(self.layers)):
            layer_idx = len(self.layers) - 1 - i
            rule = self.layer_rules.get(layer_idx, self.config.default_rule)
            R = layer.lrp(R, rule=rule)

        # Reshape to original input shape
        return R.view(original_shape)

    def get_layer_relevances(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Get relevance at each layer during backward propagation.

        Useful for debugging and understanding information flow.

        Args:
            x: Input tensor
            target_class: Class to explain

        Returns:
            List of relevance tensors, from output layer to input
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        # Forward pass
        output = self.forward(x)

        # Initialize
        R = torch.zeros_like(output)
        if target_class is not None:
            R[:, target_class] = output[:, target_class]
        else:
            pred = output.argmax(dim=1)
            R[torch.arange(batch_size), pred] = output[torch.arange(batch_size), pred]

        relevances = [R.clone()]

        # Backward pass
        for i, layer in enumerate(reversed(self.layers)):
            layer_idx = len(self.layers) - 1 - i
            rule = self.layer_rules.get(layer_idx, self.config.default_rule)
            R = layer.lrp(R, rule=rule)
            relevances.append(R.clone())

        return relevances


class LRPClassifier(LRPNetwork):
    """
    LRP-enabled classifier for trading signal prediction.

    Convenience wrapper around LRPNetwork for binary classification
    (BUY vs SELL/HOLD) with softmax output.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        config: Optional[LRPConfig] = None
    ):
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=2,  # Binary classification
            dropout=dropout,
            config=config
        )

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.

        Args:
            x: Input tensor

        Returns:
            Probabilities [batch, 2] for [SELL/HOLD, BUY]
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted classes.

        Args:
            x: Input tensor

        Returns:
            Predicted classes [batch] (0=SELL/HOLD, 1=BUY)
        """
        return self.forward(x).argmax(dim=1)


def verify_conservation(
    model: LRPNetwork,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    tolerance: float = 0.1
) -> Tuple[bool, float]:
    """
    Verify that LRP conserves relevance.

    The sum of input relevances should equal the output value (within tolerance).

    Args:
        model: LRP-enabled network
        x: Input tensor
        target_class: Class to verify
        tolerance: Acceptable relative error

    Returns:
        Tuple of (conservation_holds, relative_error)
    """
    model.eval()

    with torch.no_grad():
        output = model.forward(x)
        relevance = model.explain(x, target_class=target_class)

        if target_class is not None:
            output_val = output[:, target_class]
        else:
            pred = output.argmax(dim=1)
            output_val = output[torch.arange(len(pred)), pred]

        # Sum input relevances
        input_relevance_sum = relevance.view(x.shape[0], -1).sum(dim=1)

        # Calculate relative error
        relative_error = torch.abs(input_relevance_sum - output_val) / (torch.abs(output_val) + 1e-9)
        avg_error = relative_error.mean().item()

        return avg_error < tolerance, avg_error
