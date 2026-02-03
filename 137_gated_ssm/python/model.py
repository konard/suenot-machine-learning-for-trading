"""
Gated State Space Model for trading.

This module provides:
- DiagonalSSMLayer: Diagonal state space model layer with O(N) per-step computation
- GatedSSMBlock: SSM with output gating (GSS-style)
- GatedSSMModel: Full model with stacked gated SSM blocks and residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiagonalSSMLayer(nn.Module):
    """
    Diagonal State Space Model layer.

    Uses diagonal state matrix A for O(N) computation per step.
    Discretization via zero-order hold (ZOH).
    State matrix A is parameterized in log-space for training stability.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # State matrix A (log-space parameterization ensures negative real parts)
        log_A_real = torch.log(0.5 * torch.ones(d_model, d_state))
        self.log_A_real = nn.Parameter(log_A_real)

        # Input matrix B
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

        # Output matrix C
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

        # Discretization step (log-space)
        log_dt = torch.rand(d_model) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # Skip connection D
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the diagonal SSM.

        Args:
            u: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = u.shape

        # Compute discretized parameters
        dt = torch.exp(self.log_dt)  # (d_model,)
        A = -torch.exp(self.log_A_real)  # (d_model, d_state)

        # ZOH discretization
        dtA = dt.unsqueeze(-1) * A  # (d_model, d_state)
        A_bar = torch.exp(dtA)  # (d_model, d_state)
        B_bar = self.B * dt.unsqueeze(-1)  # (d_model, d_state)

        # Sequential scan (recurrence)
        x = torch.zeros(batch, self.d_model, self.d_state, device=u.device)
        outputs = []

        for t in range(seq_len):
            x = A_bar * x + B_bar * u[:, t, :].unsqueeze(-1)
            y_t = (self.C * x).sum(dim=-1)  # (batch, d_model)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        y = y + u * self.D  # Skip connection
        return y


class GatedSSMBlock(nn.Module):
    """
    Gated State Space Model block (GSS-style).

    Architecture:
        z = SSM(Linear_V(u))
        g = GELU(Linear_G(u))
        y = Linear_O(LayerNorm(z) * g)

    The gate path provides input-dependent modulation of the SSM output,
    allowing selective emphasis on relevant time steps.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_inner = d_model * expand

        # Value and gate projections
        self.linear_v = nn.Linear(d_model, d_inner)
        self.linear_g = nn.Linear(d_model, d_inner)

        # SSM layer
        self.ssm = DiagonalSSMLayer(d_inner, d_state)

        # Layer normalization (applied after SSM, before gating)
        self.norm = nn.LayerNorm(d_inner)

        # Output projection
        self.linear_o = nn.Linear(d_inner, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through gated SSM block.

        Args:
            u: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # SSM path
        v = self.linear_v(u)
        z = self.ssm(v)
        z = self.norm(z)

        # Gate path (GELU activation for smoother gradients)
        g = F.gelu(self.linear_g(u))

        # Merge: element-wise gating
        y = z * g
        y = self.dropout(y)
        y = self.linear_o(y)
        return y


class GatedSSMModel(nn.Module):
    """
    Full Gated SSM model for sequence prediction.

    Architecture:
        Input Projection -> [GatedSSMBlock + Residual + LayerNorm] x N -> Output Head

    Supports both regression (num_classes=None) and classification tasks.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                GatedSSMBlock(d_model, d_state, expand, dropout)
            )
            self.norms.append(nn.LayerNorm(d_model))

        self.final_norm = nn.LayerNorm(d_model)

        # Output head
        if num_classes is not None:
            self.head = nn.Linear(d_model, num_classes)
        else:
            self.head = nn.Linear(d_model, 1)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full Gated SSM model.

        Args:
            x: Input features (batch, seq_len, input_size)

        Returns:
            Predictions (batch, 1) for regression or (batch, num_classes) for classification
        """
        h = self.input_proj(x)

        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = norm(h)
            h = layer(h)
            h = h + residual

        h = self.final_norm(h)
        h = h[:, -1, :]  # Take last time step
        return self.head(h)


if __name__ == "__main__":
    # Quick test
    batch, seq_len, input_size = 8, 60, 4
    model = GatedSSMModel(
        input_size=input_size,
        d_model=64,
        d_state=32,
        n_layers=3,
        expand=2,
        dropout=0.1,
    )

    x = torch.randn(batch, seq_len, input_size)
    y = model(x)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {y.shape}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
