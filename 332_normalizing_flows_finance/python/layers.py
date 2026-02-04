"""
Normalizing Flow Layers

This module provides the building blocks for normalizing flows:
- Affine Coupling Layers (RealNVP style)
- Masked Linear layers (for autoregressive flows)
- Permutation layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class CouplingNetwork(nn.Module):
    """
    Neural network for computing scale and translation in coupling layers.

    This network takes the "unchanged" part of the input and computes
    the parameters for transforming the other part.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
        activation: str = "relu"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build network
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self._get_activation(activation))

        # Hidden layers
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))

        self.net = nn.Sequential(*layers)

        # Output layers for scale and translation
        self.scale_net = nn.Linear(hidden_dim, output_dim)
        self.translation_net = nn.Linear(hidden_dim, output_dim)

        # Initialize to identity transform (important for training stability)
        self._initialize_identity()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }
        return activations.get(activation, nn.ReLU())

    def _initialize_identity(self):
        """Initialize weights so that the initial transform is identity."""
        nn.init.zeros_(self.scale_net.weight)
        nn.init.zeros_(self.scale_net.bias)
        nn.init.zeros_(self.translation_net.weight)
        nn.init.zeros_(self.translation_net.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing scale and translation parameters.

        Args:
            x: Input tensor of shape [batch, input_dim]

        Returns:
            scale: Scale parameters [batch, output_dim]
            translation: Translation parameters [batch, output_dim]
        """
        h = self.net(x)

        # Scale is clamped to prevent numerical issues
        scale = torch.tanh(self.scale_net(h)) * 2.0  # Clamp to [-2, 2]
        translation = self.translation_net(h)

        return scale, translation


class AffineCouplingLayer(nn.Module):
    """
    Affine Coupling Layer as used in RealNVP.

    Splits input into two parts. One part remains unchanged and
    parameterizes an affine transform applied to the other part.

    Transform: y = x * exp(s(x_mask)) + t(x_mask)
    where x_mask is the masked (unchanged) part.
    """

    def __init__(
        self,
        dim: int,
        mask: torch.Tensor,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2
    ):
        super().__init__()

        self.dim = dim
        self.register_buffer('mask', mask)

        # Count dimensions for each part
        n_masked = int(mask.sum().item())
        n_unmasked = dim - n_masked

        # Coupling network: masked -> parameters for unmasked
        self.coupling_net = CouplingNetwork(
            input_dim=n_masked,
            output_dim=n_unmasked,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: data space -> latent space.

        Args:
            x: Input tensor [batch, dim]

        Returns:
            y: Transformed tensor [batch, dim]
            log_det: Log determinant of Jacobian [batch]
        """
        # Split into masked and unmasked parts
        x_masked = x[:, self.mask.bool()]
        x_unmasked = x[:, ~self.mask.bool()]

        # Get transformation parameters from masked part
        scale, translation = self.coupling_net(x_masked)

        # Apply affine transform to unmasked part
        y_unmasked = x_unmasked * torch.exp(scale) + translation

        # Reconstruct full output
        y = torch.zeros_like(x)
        y[:, self.mask.bool()] = x_masked
        y[:, ~self.mask.bool()] = y_unmasked

        # Log determinant is sum of scales
        log_det = scale.sum(dim=-1)

        return y, log_det

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass: latent space -> data space.

        Args:
            y: Input tensor [batch, dim]

        Returns:
            x: Transformed tensor [batch, dim]
            log_det: Log determinant of Jacobian [batch] (negative of forward)
        """
        # Split
        y_masked = y[:, self.mask.bool()]
        y_unmasked = y[:, ~self.mask.bool()]

        # Get transformation parameters (same as forward, since masked part is unchanged)
        scale, translation = self.coupling_net(y_masked)

        # Inverse affine transform
        x_unmasked = (y_unmasked - translation) * torch.exp(-scale)

        # Reconstruct
        x = torch.zeros_like(y)
        x[:, self.mask.bool()] = y_masked
        x[:, ~self.mask.bool()] = x_unmasked

        # Log determinant (negative of forward)
        log_det = -scale.sum(dim=-1)

        return x, log_det


class MaskedLinear(nn.Module):
    """
    Masked Linear layer for autoregressive flows.

    Implements a linear layer where certain connections are masked
    to enforce autoregressive structure.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply masked linear transformation."""
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (MADE).

    Used as the building block for autoregressive flows like MAF.
    Each output dimension depends only on previous input dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 256],
        output_dim_multiplier: int = 2,  # For scale and translation
        natural_ordering: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim_multiplier = output_dim_multiplier

        # Assign random or natural ordering to inputs
        if natural_ordering:
            self.ordering = torch.arange(input_dim)
        else:
            self.ordering = torch.randperm(input_dim)

        # Build masks for autoregressive property
        self.masks = self._create_masks(hidden_dims)

        # Build network with masked layers
        layers = []
        dims = [input_dim] + hidden_dims + [input_dim * output_dim_multiplier]

        for i in range(len(dims) - 1):
            layers.append(MaskedLinear(dims[i], dims[i+1], self.masks[i]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def _create_masks(self, hidden_dims: list) -> list:
        """Create masks enforcing autoregressive property."""
        masks = []

        # Assign orderings to hidden units
        hidden_orderings = []
        for dim in hidden_dims:
            # Assign each hidden unit to one of the input dimensions
            ordering = torch.randint(0, self.input_dim - 1, (dim,))
            hidden_orderings.append(ordering)

        # Create mask for first layer (input -> first hidden)
        m = self.ordering.unsqueeze(0) <= hidden_orderings[0].unsqueeze(1)
        masks.append(m.float().t())

        # Create masks for hidden layers
        for i in range(len(hidden_dims) - 1):
            m = hidden_orderings[i].unsqueeze(0) <= hidden_orderings[i+1].unsqueeze(1)
            masks.append(m.float().t())

        # Create mask for output layer
        # Output depends on inputs < its dimension
        output_ordering = self.ordering.repeat(self.output_dim_multiplier)
        m = hidden_orderings[-1].unsqueeze(0) < output_ordering.unsqueeze(1)
        masks.append(m.float().t())

        return masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MADE.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Output tensor [batch, input_dim * output_dim_multiplier]
        """
        return self.net(x)


class ActNorm(nn.Module):
    """
    Activation Normalization layer.

    Data-dependent initialization that normalizes activations
    using learned scale and bias parameters.
    """

    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim
        self.log_scale = nn.Parameter(torch.zeros(1, dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        self.initialized = False

    def initialize(self, x: torch.Tensor):
        """Initialize parameters based on first batch of data."""
        with torch.no_grad():
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + 1e-6

            self.bias.data = -mean
            self.log_scale.data = -torch.log(std)
            self.initialized = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with learned normalization.

        Args:
            x: Input tensor [batch, dim]

        Returns:
            y: Normalized tensor [batch, dim]
            log_det: Log determinant [batch]
        """
        if not self.initialized:
            self.initialize(x)

        y = (x + self.bias) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum() * torch.ones(x.shape[0], device=x.device)

        return y, log_det

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse normalization."""
        x = y * torch.exp(-self.log_scale) - self.bias
        log_det = -self.log_scale.sum() * torch.ones(y.shape[0], device=y.device)

        return x, log_det


class Permutation(nn.Module):
    """
    Permutation layer for shuffling dimensions.

    Can be fixed, random, or learned (1x1 convolution).
    """

    def __init__(self, dim: int, mode: str = "shuffle"):
        super().__init__()

        self.dim = dim
        self.mode = mode

        if mode == "shuffle":
            # Random fixed permutation
            perm = torch.randperm(dim)
            inv_perm = torch.argsort(perm)
            self.register_buffer('perm', perm)
            self.register_buffer('inv_perm', inv_perm)
        elif mode == "reverse":
            # Simple reversal
            perm = torch.arange(dim - 1, -1, -1)
            self.register_buffer('perm', perm)
            self.register_buffer('inv_perm', perm)  # Reverse is its own inverse
        elif mode == "learned":
            # Learned permutation via 1x1 conv (LU decomposition)
            W = torch.randn(dim, dim)
            Q, _ = torch.linalg.qr(W)
            self.W = nn.Parameter(Q)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply permutation."""
        if self.mode in ["shuffle", "reverse"]:
            y = x[:, self.perm]
            log_det = torch.zeros(x.shape[0], device=x.device)
        else:  # learned
            y = F.linear(x, self.W)
            log_det = torch.slogdet(self.W)[1] * torch.ones(x.shape[0], device=x.device)

        return y, log_det

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply inverse permutation."""
        if self.mode in ["shuffle", "reverse"]:
            x = y[:, self.inv_perm]
            log_det = torch.zeros(y.shape[0], device=y.device)
        else:  # learned
            x = F.linear(y, torch.inverse(self.W))
            log_det = -torch.slogdet(self.W)[1] * torch.ones(y.shape[0], device=y.device)

        return x, log_det


class BatchNormFlow(nn.Module):
    """
    Batch Normalization as a flow layer.

    Uses running statistics for inference, learnable parameters.
    """

    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()

        self.dim = dim
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters
        self.log_gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

        # Running statistics
        self.register_buffer('running_mean', torch.zeros(1, dim))
        self.register_buffer('running_var', torch.ones(1, dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply batch normalization in forward direction."""
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True) + self.eps

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var)

        # Scale and shift
        y = x_norm * torch.exp(self.log_gamma) + self.beta

        # Log determinant
        log_det = (self.log_gamma - 0.5 * torch.log(var)).sum() * torch.ones(x.shape[0], device=x.device)

        return y, log_det

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply inverse batch normalization."""
        mean = self.running_mean
        var = self.running_var

        # Inverse scale and shift
        x_norm = (y - self.beta) * torch.exp(-self.log_gamma)

        # Inverse normalize
        x = x_norm * torch.sqrt(var) + mean

        # Log determinant
        log_det = (-self.log_gamma + 0.5 * torch.log(var)).sum() * torch.ones(y.shape[0], device=y.device)

        return x, log_det
