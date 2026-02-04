"""
Normalizing Flow Models

This module provides complete normalizing flow implementations:
- NormalizingFlow: Base class for all flows
- RealNVP: Real-valued Non-Volume Preserving flow
- MAF: Masked Autoregressive Flow
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Union
from .layers import (
    AffineCouplingLayer,
    ActNorm,
    Permutation,
    BatchNormFlow,
    MADE
)


class NormalizingFlow(nn.Module):
    """
    Base class for Normalizing Flow models.

    Provides common interface for density estimation and sampling.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()

        # Base distribution (standard normal)
        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_std', torch.ones(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base distribution -> data distribution.

        Args:
            z: Samples from base distribution [batch, dim]

        Returns:
            x: Samples from learned distribution [batch, dim]
        """
        x = z
        for layer in self.layers:
            x, _ = layer.inverse(x)
        return x

    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass: data distribution -> base distribution.

        Args:
            x: Data samples [batch, dim]

        Returns:
            z: Latent samples [batch, dim]
            log_det: Total log determinant [batch]
        """
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)

        for layer in reversed(self.layers):
            z, log_det = layer(z)
            total_log_det += log_det

        return z, total_log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of data under the learned distribution.

        Args:
            x: Data samples [batch, dim]

        Returns:
            log_prob: Log probability [batch]
        """
        z, log_det = self.inverse(x)

        # Log prob under base distribution (standard normal)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)

        # Total log prob via change of variables
        return log_pz + log_det

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples from the learned distribution.

        Args:
            n_samples: Number of samples to generate

        Returns:
            samples: Generated samples [n_samples, dim]
        """
        device = next(self.parameters()).device
        z = torch.randn(n_samples, self.dim, device=device)
        return self.forward(z)

    def nll_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for training.

        Args:
            x: Training data [batch, dim]

        Returns:
            loss: Scalar loss value
        """
        return -self.log_prob(x).mean()


class RealNVP(NormalizingFlow):
    """
    Real-valued Non-Volume Preserving (RealNVP) flow.

    Uses alternating affine coupling layers with checkerboard or
    channel-wise masking patterns.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 8,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
        use_actnorm: bool = True,
        use_batchnorm: bool = False,
        permutation: str = "shuffle"
    ):
        super().__init__(dim)

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Build flow layers
        for i in range(n_layers):
            # Create alternating masks
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[:dim // 2] = 1.0
            else:
                mask[dim // 2:] = 1.0

            # Optional ActNorm
            if use_actnorm:
                self.layers.append(ActNorm(dim))

            # Coupling layer
            self.layers.append(
                AffineCouplingLayer(
                    dim=dim,
                    mask=mask,
                    hidden_dim=hidden_dim,
                    n_hidden_layers=n_hidden_layers
                )
            )

            # Optional BatchNorm
            if use_batchnorm:
                self.layers.append(BatchNormFlow(dim))

            # Permutation (except last layer)
            if i < n_layers - 1:
                self.layers.append(Permutation(dim, mode=permutation))


class MAF(NormalizingFlow):
    """
    Masked Autoregressive Flow (MAF).

    Uses MADE networks to compute autoregressive transformations.
    Fast density estimation, slow sampling.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 5,
        hidden_dims: List[int] = [256, 256],
        use_actnorm: bool = True
    ):
        super().__init__(dim)

        self.n_layers = n_layers

        for i in range(n_layers):
            # Optional ActNorm
            if use_actnorm:
                self.layers.append(ActNorm(dim))

            # MAF layer
            self.layers.append(
                MAFLayer(
                    dim=dim,
                    hidden_dims=hidden_dims,
                    reverse=(i % 2 == 1)  # Alternate ordering
                )
            )


class MAFLayer(nn.Module):
    """
    Single layer of Masked Autoregressive Flow.

    Each dimension is transformed based on previous dimensions.
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: List[int] = [256, 256],
        reverse: bool = False
    ):
        super().__init__()

        self.dim = dim
        self.reverse = reverse

        # MADE network outputs scale and translation for each dimension
        self.made = MADE(
            input_dim=dim,
            hidden_dims=hidden_dims,
            output_dim_multiplier=2,  # scale and translation
            natural_ordering=not reverse
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: data -> latent.

        Args:
            x: Input tensor [batch, dim]

        Returns:
            z: Transformed tensor [batch, dim]
            log_det: Log determinant [batch]
        """
        # Get autoregressive parameters
        params = self.made(x)
        scale = params[:, :self.dim]
        translation = params[:, self.dim:]

        # Apply transform
        z = (x - translation) * torch.exp(-scale)

        # Log determinant
        log_det = -scale.sum(dim=-1)

        return z, log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass: latent -> data.

        Note: This is slow (sequential) for MAF.
        """
        x = torch.zeros_like(z)

        for i in range(self.dim):
            # Get parameters using current partial x
            params = self.made(x)
            scale_i = params[:, i]
            translation_i = params[:, self.dim + i]

            # Compute x_i
            x[:, i] = z[:, i] * torch.exp(scale_i) + translation_i

        # Log determinant
        params = self.made(x)
        scale = params[:, :self.dim]
        log_det = scale.sum(dim=-1)

        return x, log_det


class ConditionalRealNVP(NormalizingFlow):
    """
    Conditional RealNVP flow.

    Allows conditioning the distribution on external features
    (e.g., market regime, volatility level).
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        n_layers: int = 8,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2
    ):
        super().__init__(dim)

        self.cond_dim = cond_dim

        # Conditioning encoder
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Build conditional coupling layers
        for i in range(n_layers):
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[:dim // 2] = 1.0
            else:
                mask[dim // 2:] = 1.0

            self.layers.append(ActNorm(dim))
            self.layers.append(
                ConditionalAffineCouplingLayer(
                    dim=dim,
                    cond_dim=hidden_dim,
                    mask=mask,
                    hidden_dim=hidden_dim,
                    n_hidden_layers=n_hidden_layers
                )
            )

            if i < n_layers - 1:
                self.layers.append(Permutation(dim, mode="shuffle"))

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass with conditioning."""
        cond_encoded = self.cond_encoder(cond)
        x = z

        for layer in self.layers:
            if isinstance(layer, ConditionalAffineCouplingLayer):
                x, _ = layer.inverse(x, cond_encoded)
            else:
                x, _ = layer.inverse(x)

        return x

    def inverse(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass with conditioning."""
        cond_encoded = self.cond_encoder(cond)
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)

        for layer in reversed(self.layers):
            if isinstance(layer, ConditionalAffineCouplingLayer):
                z, log_det = layer(z, cond_encoded)
            else:
                z, log_det = layer(z)
            total_log_det += log_det

        return z, total_log_det

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Compute conditional log probability."""
        z, log_det = self.inverse(x, cond)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det

    def sample(self, n_samples: int, cond: torch.Tensor) -> torch.Tensor:
        """Generate conditional samples."""
        device = next(self.parameters()).device
        z = torch.randn(n_samples, self.dim, device=device)
        return self.forward(z, cond)


class ConditionalAffineCouplingLayer(nn.Module):
    """
    Affine coupling layer with conditioning.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        mask: torch.Tensor,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2
    ):
        super().__init__()

        self.dim = dim
        self.register_buffer('mask', mask)

        n_masked = int(mask.sum().item())
        n_unmasked = dim - n_masked

        # Coupling network takes both masked input and conditioning
        input_dim = n_masked + cond_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.scale_net = nn.Linear(hidden_dim, n_unmasked)
        self.translation_net = nn.Linear(hidden_dim, n_unmasked)

        # Initialize to identity
        nn.init.zeros_(self.scale_net.weight)
        nn.init.zeros_(self.scale_net.bias)
        nn.init.zeros_(self.translation_net.weight)
        nn.init.zeros_(self.translation_net.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with conditioning."""
        x_masked = x[:, self.mask.bool()]
        x_unmasked = x[:, ~self.mask.bool()]

        # Concatenate masked input with conditioning
        net_input = torch.cat([x_masked, cond], dim=-1)
        h = self.net(net_input)

        scale = torch.tanh(self.scale_net(h)) * 2.0
        translation = self.translation_net(h)

        y_unmasked = x_unmasked * torch.exp(scale) + translation

        y = torch.zeros_like(x)
        y[:, self.mask.bool()] = x_masked
        y[:, ~self.mask.bool()] = y_unmasked

        log_det = scale.sum(dim=-1)

        return y, log_det

    def inverse(self, y: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse with conditioning."""
        y_masked = y[:, self.mask.bool()]
        y_unmasked = y[:, ~self.mask.bool()]

        net_input = torch.cat([y_masked, cond], dim=-1)
        h = self.net(net_input)

        scale = torch.tanh(self.scale_net(h)) * 2.0
        translation = self.translation_net(h)

        x_unmasked = (y_unmasked - translation) * torch.exp(-scale)

        x = torch.zeros_like(y)
        x[:, self.mask.bool()] = y_masked
        x[:, ~self.mask.bool()] = x_unmasked

        log_det = -scale.sum(dim=-1)

        return x, log_det


def create_flow(
    flow_type: str = "realnvp",
    dim: int = 1,
    n_layers: int = 8,
    hidden_dim: int = 256,
    **kwargs
) -> NormalizingFlow:
    """
    Factory function to create normalizing flow models.

    Args:
        flow_type: Type of flow ("realnvp", "maf")
        dim: Input dimension
        n_layers: Number of flow layers
        hidden_dim: Hidden layer dimension
        **kwargs: Additional arguments for specific flow types

    Returns:
        Normalizing flow model
    """
    if flow_type.lower() == "realnvp":
        return RealNVP(
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            **kwargs
        )
    elif flow_type.lower() == "maf":
        return MAF(
            dim=dim,
            n_layers=n_layers,
            hidden_dims=[hidden_dim, hidden_dim],
            **kwargs
        )
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")
